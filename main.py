#!/usr/bin/env python3
"""
Low-Rank Tensorized 代理网络训练框架的主程序入口。

这个程序可以：
1. 读取数据（支持多种格式：.npy, .csv, .txt, .mat）
2. 处理和归一化数据
3. 训练 U_Network 代理网络
4. 评估模型性能
5. 保存模型和结果
"""

import argparse
import json
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Optional, List
import sys

import torch

from network import U_Network
from train import train_u_network, train_u_network_with_validation, load_u_network, save_u_network
from test import evaluate_on_dataset, predict, compute_derivatives, plot_predictions_vs_targets, test_on_regular_grid
from measure import compute_all_metrics, print_metrics, model_complexity_metrics


def set_random_seed(seed: int) -> None:
    """
    设置所有随机数生成器的种子，确保可重现性。
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 cuDNN 的可重现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def load_data(
    input_file: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    从 .npy 文件读取数据。
    
    数据格式要求：
    - 最后一列：u值（解）
    - 倒数第二列：t值（时间）
    - 前面的列：空间坐标（x, y, z 等）
    
    示例：
    - 1D+时间：[x, t, u] -> coords shape=(n, 2), input_dim=2
    - 2D+时间：[x, y, t, u] -> coords shape=(n, 3), input_dim=3
    - 3D+时间：[x, y, z, t, u] -> coords shape=(n, 4), input_dim=4
    
    Args:
        input_file: .npy 文件路径
        
    Returns:
        (coords, values, input_dim) 元组，其中：
        - coords: shape=(n_samples, n_spatial_dims + 1) 的坐标数组 [x, y, ..., t]
        - values: shape=(n_samples,) 的目标值数组
        - input_dim: 输入维度 = n_spatial_dims + 1
    """
    file_path = Path(input_file)
    
    if file_path.suffix != ".npy":
        raise ValueError(f"Only .npy files are supported. Got: {file_path.suffix}")
    
    data = np.load(input_file)
    
    # 检查数据维度
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}")
    
    n_samples, n_cols = data.shape
    
    if n_cols < 3:
        raise ValueError(f"Expected at least 3 columns (spatial_coords + t + u), got {n_cols}")
    
    # 分解数据：最后一列是 u，倒数第二列是 t，前面是空间坐标
    coords = data[:, :-1]  # 所有列除了最后一列 [x, y, ..., t]
    values = data[:, -1]   # 最后一列 [u]
    
    # 计算输入维度
    input_dim = n_cols - 1  # n_spatial_dims + 1 (for time)
    n_spatial_dims = input_dim - 1
    
    print(f"\nLoaded .npy file:")
    print(f"  Total samples: {n_samples}")
    print(f"  Spatial dimensions: {n_spatial_dims}")
    print(f"  Input dimension (including time): {input_dim}")
    print(f"  Data shape: {data.shape}")
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Spatial coords range: [{coords[:, :-1].min():.4f}, {coords[:, :-1].max():.4f}]")
    print(f"  Time range: [{coords[:, -1].min():.4f}, {coords[:, -1].max():.4f}]")
    print(f"  Value range: [{values.min():.4f}, {values.max():.4f}]")
    
    return coords, values, input_dim


def preprocess_data(
    coords: np.ndarray,
    values: np.ndarray,
    noise_std: float = 0.0,
    test_split: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]], dict]:
    """
    预处理数据：清理、标准化、添加噪声、划分数据集。
    
    Args:
        coords: 输入坐标
        values: 目标值
        noise_std: 添加的高斯噪声标准差
        test_split: 测试集比例（0 表示不划分）
        
    Returns:
        如果 test_split > 0：(coords_train, values_train, (coords_test, values_test))
        否则：(coords_train, values_train, None)
    """
    # 移除 NaN 和 Inf
    valid_mask = np.isfinite(coords).all(axis=1) & np.isfinite(values)
    coords = coords[valid_mask]
    values = values[valid_mask]
    print(f"Removed invalid data: {(~valid_mask).sum()} samples")
    
    # 添加噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=values.shape)
        values = values + noise
        print(f"Added Gaussian noise (std={noise_std}) to values")
    
    # 划分数据集
    test_data = None
    if test_split > 0:
        n_samples = len(values)
        n_test = int(n_samples * test_split)
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        coords_test = coords[test_indices]
        values_test = values[test_indices]
        coords = coords[train_indices]
        values = values[train_indices]
        
        test_data = (coords_test, values_test)
        print(f"Split data: train={len(values)}, test={len(values_test)}")
    
    print(f"Final dataset: coords shape={coords.shape}, values shape={values.shape}")
    return coords, values, test_data


def main() -> None:
    """主程序入口。"""
    parser = argparse.ArgumentParser(
        description="Low-Rank Tensorized 代理网络训练框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ===== 数据相关参数 =====
    parser.add_argument("--input", required=True, help="输入 .npy 文件路径（必需）")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--noise-std", type=float, default=0.0, help="添加的噪声标准差")
    parser.add_argument("--test-split", type=float, default=0.2, help="测试集比例")
    
    # ===== 网络架构参数 =====
    parser.add_argument("--num-hidden-layers", type=int, default=5, help="隐藏层数")
    parser.add_argument("--neurons-per-layer", type=int, default=50, help="每层神经元数")
    parser.add_argument("--activation", default="Rat", choices=["Rat", "Tanh", "Sin"],
                       help="激活函数类型")
    
    # ===== 训练参数 =====
    parser.add_argument("--epochs", type=int, default=5000, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="L2 正则化系数")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="计算设备")
    
    # ===== 随机种子 =====
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    
    # ===== 输出参数 =====
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--save-model", action="store_true", help="保存模型")
    parser.add_argument("--plot", action="store_true", help="绘制结果图表")
    
    args = parser.parse_args()
    
    # ===== 设置随机种子 =====
    print("="*60)
    print("Setting Random Seeds")
    print("="*60)
    set_random_seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # ===== 步骤 1: 读取数据 =====
    print("\n" + "="*60)
    print("Step 1: Loading Data")
    print("="*60)
    coords, values, input_dim = load_data(args.input)
    print(f"  Automatically detected input_dim: {input_dim}")
    
    # ===== 步骤 2: 预处理数据 =====
    print("\n" + "="*60)
    print("Step 2: Preprocessing Data")
    print("="*60)
    coords_train, values_train, test_data = preprocess_data(
        coords, values,
        noise_std=args.noise_std,
        test_split=args.test_split,
    )
    
    # ===== 步骤 3: 训练网络 =====
    print("\n" + "="*60)
    print("Step 3: Training U_Network")
    print("="*60)
    
    if test_data is not None:
        coords_test, values_test = test_data
        print(f"Training with validation set...")
        model, train_log = train_u_network(
            coords=coords_train,
            targets=values_train,
            input_dim=input_dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_hidden_layers=args.num_hidden_layers,
            neurons_per_layer=args.neurons_per_layer,
            activation_function=args.activation,
            device=args.device,
            verbose=args.verbose or True,
            coords_test=coords_test,
            targets_test=values_test,
            lambda_reg=args.lambda_reg,
        )
    else:
        print(f"Training without validation set...")
        model, train_log = train_u_network(
            coords=coords_train,
            targets=values_train,
            input_dim=input_dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_hidden_layers=args.num_hidden_layers,
            neurons_per_layer=args.neurons_per_layer,
            activation_function=args.activation,
            device=args.device,
            verbose=args.verbose or True,
            lambda_reg=args.lambda_reg,
        )
    
    # ===== 步骤 4: 评估模型 =====
    print("\n" + "="*60)
    print("Step 4: Model Evaluation")
    print("="*60)
    
    # 在训练集上评估
    eval_train = evaluate_on_dataset(model, coords_train, values_train)
    metrics_train = compute_all_metrics(eval_train["predictions"], values_train)
    print("\nTraining Set Metrics:")
    print_metrics(metrics_train)
    
    # 在测试集上评估
    if test_data is not None:
        coords_test, values_test = test_data
        eval_test = evaluate_on_dataset(model, coords_test, values_test)
        metrics_test = compute_all_metrics(eval_test["predictions"], values_test)
        print("\nTest Set Metrics:")
        print_metrics(metrics_test)
    
    # ===== 步骤 5: 保存结果 =====
    print("\n" + "="*60)
    print("Step 5: Saving Results")
    print("="*60)
    
    # 保存模型
    if args.save_model:
        model_path = output_dir / "u_network_model.pt"
        save_u_network(model, str(model_path))
    
    # 保存训练日志
    train_log_path = output_dir / "training_log.json"
    np.save(output_dir / "train_losses.npy", train_log["train_losses"])
    if train_log.get("test_losses"):
        np.save(output_dir / "test_losses.npy", train_log["test_losses"])
    print(f"Training log saved to {train_log_path}")
    
    # 保存指标
    metrics_path = output_dir / "metrics.json"
    metrics_summary = {
        "train": metrics_train,
    }
    if test_data is not None:
        metrics_summary["test"] = metrics_test
    
    with open(metrics_path, "w") as f:
        # 转换为可序列化的格式
        metrics_json = {}
        for split, metrics in metrics_summary.items():
            metrics_json[split] = {k: float(v) if not np.isnan(v) and not np.isinf(v) else str(v) 
                                   for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # 保存模型复杂度
    complexity = model_complexity_metrics(model)
    print("\nModel Complexity:")
    print_metrics(complexity, prefix="Model Statistics")
    
    # ===== 步骤 6: 绘图（可选） =====
    if args.plot:
        print("\n" + "="*60)
        print("Step 6: Plotting Results")
        print("="*60)
        
        # 绘制预测 vs 真实值
        plot_path = output_dir / "predictions_vs_targets_train.png"
        plot_predictions_vs_targets(eval_train["predictions"], values_train, str(plot_path))
        
        if test_data is not None:
            plot_path = output_dir / "predictions_vs_targets_test.png"
            plot_predictions_vs_targets(eval_test["predictions"], values_test, str(plot_path))
        
        print("Plots saved to output directory")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
