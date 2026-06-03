#!/usr/bin/env python3
"""
Low-Rank Tensorized 代理网络训练框架的主程序入口。

这个程序可以：
1. 读取 .npy 数据
2. 处理和归一化数据
3. 训练 U_Network 代理网络
4. 评估模型性能
5. 执行连续稀疏低秩张量 PDE 发现
6. 保存模型和结果
"""

import argparse
import json
import numpy as np
import random
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from train import discover_pde, train_u_network, save_u_network, load_u_network
from measure import evaluate_on_dataset, print_metrics


def set_random_seed(seed: int, device: str) -> None:
    """
    设置所有随机数生成器的种子，确保可重现性。
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 cuDNN 的可重现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def parse_rank_by_order(rank_string: Optional[str]) -> Optional[Dict[int, int]]:
    """解析形如 "2:8,3:8,4:4" 的每阶最大 rank 配置。"""
    if rank_string is None or rank_string.strip() == "":
        return None

    result = {}
    for item in rank_string.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid rank specification '{item}', expected order:rank")

        order = int(parts[0].strip())
        rank = int(parts[1].strip())
        if order < 2:
            raise ValueError("PDE interaction order must be >= 2.")
        if rank < 1:
            raise ValueError("Rank must be >= 1.")
        result[order] = rank

    return result or None


def preprocessing_matches(saved_info: Dict, current_info: Dict) -> bool:
    """检查已有模型的标准化参数是否和本次数据预处理一致。"""
    array_keys = ("coord_mean", "coord_std")
    scalar_keys = ("value_mean", "value_std")

    for key in array_keys:
        if key not in saved_info or key not in current_info:
            return False
        if not np.allclose(
            np.asarray(saved_info[key], dtype=np.float64),
            np.asarray(current_info[key], dtype=np.float64),
            rtol=1e-10,
            atol=1e-12,
        ):
            return False

    for key in scalar_keys:
        if key not in saved_info or key not in current_info:
            return False
        if not np.isclose(
            float(saved_info[key]),
            float(current_info[key]),
            rtol=1e-10,
            atol=1e-12,
        ):
            return False

    return saved_info.get("standardization") == current_info.get("standardization")


def build_reproduce_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    entrypoint: str = "main.py",
    python_executable: Optional[str] = None,
) -> str:
    """生成包含当前 argparse 参数的可复制复现实验命令。"""
    command_parts: List[str] = [python_executable or sys.executable or "python", "-u", entrypoint]

    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        option = next((opt for opt in action.option_strings if opt.startswith("--")), None)
        if option is None:
            continue

        value = getattr(args, action.dest)
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command_parts.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not value:
                command_parts.append(option)
            continue
        if value is None:
            continue

        command_parts.extend([option, str(value)])

    return " ".join(shlex.quote(part) for part in command_parts)


def build_run_metadata(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    device: str,
    input_file: Path,
    output_dir: Path,
    model_path: Path,
) -> Dict[str, Any]:
    """收集本次运行的全部参数和复现命令。"""
    return {
        "arguments": vars(args),
        "device": device,
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "python_executable": sys.executable,
        "reproduce_command": build_reproduce_command(parser, args),
    }


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
    预处理数据：清理、添加噪声、划分数据集、用训练集统计量标准化。
    
    Args:
        coords: 输入坐标
        values: 目标值
        noise_std: 添加的高斯噪声标准差
        test_split: 测试集比例（0 表示不划分）
        
    Returns:
        如果 test_split > 0：(coords_train, values_train, (coords_test, values_test), preprocessing_info)
        否则：(coords_train, values_train, None, preprocessing_info)
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

    coords = coords.astype(np.float64)
    values = values.astype(np.float64)

    def fit_standardizer(array: np.ndarray, axis=None) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(array, axis=axis)
        std = np.std(array, axis=axis)
        std = np.maximum(std, 1e-12)
        return mean, std

    def apply_standardizer(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (array - mean) / std
    
    # 划分数据集
    test_data = None
    if test_split > 0:
        n_samples = len(values)
        n_test = int(n_samples * test_split)
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        coords_train = coords[train_indices]
        values_train = values[train_indices]
        coords_test = coords[test_indices]
        values_test = values[test_indices]
    else:
        coords_train = coords
        values_train = values

    coord_mean, coord_std = fit_standardizer(coords_train, axis=0)
    value_mean, value_std = fit_standardizer(values_train)

    coords_train = apply_standardizer(coords_train, coord_mean.reshape(1, -1), coord_std.reshape(1, -1))
    values_train = apply_standardizer(values_train, value_mean, value_std)

    if test_split > 0:
        coords_test = apply_standardizer(coords_test, coord_mean.reshape(1, -1), coord_std.reshape(1, -1))
        values_test = apply_standardizer(values_test, value_mean, value_std)
        test_data = (coords_test, values_test)
        print(f"Split data: train={len(values_train)}, test={len(values_test)}")

    preprocessing_info = {
        "coord_mean": coord_mean.tolist(),
        "coord_std": coord_std.tolist(),
        "value_mean": float(value_mean),
        "value_std": float(value_std),
        "standardization": "z_score",
        "fit_on": "train_split",
    }
    print("Standardized coordinates, time, and values using training-set statistics")
    
    print(f"Final dataset: coords shape={coords_train.shape}, values shape={values_train.shape}")
    return coords_train, values_train, test_data, preprocessing_info


def main() -> None:
    """主程序入口。"""
    parser = argparse.ArgumentParser(
        description="Low-Rank Tensorized 代理网络训练框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ===== 数据相关参数 =====
    parser.add_argument("--input_path", type=str, default="data", help="输入 .npy 数据目录")
    parser.add_argument("--data", type=str, default="burgers_sine", help="输入数据集名称（不含 .npy 后缀）")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认保存到 ./output/<输入文件名>/",
    )
    parser.add_argument("--noise-std", type=float, default=0.0, help="添加的噪声标准差")
    parser.add_argument("--test-split", type=float, default=0.2, help="测试集比例")
    
    # ===== 网络架构参数 =====
    parser.add_argument("--num-hidden-layers", type=int, default=5, help="隐藏层数")
    parser.add_argument("--neurons-per-layer", type=int, default=50, help="每层神经元数")
    parser.add_argument("--activation", default="Rat", choices=["Rat", "Tanh", "Sin"],
                       help="激活函数类型")
    
    # ===== 训练参数 =====
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="L2 正则化系数")
    # ===== PDE 发现参数 =====
    parser.add_argument("--discover-pde", action="store_true",default=True, help="训练代理网络后执行 PDE 发现")
    parser.add_argument("--pde-max-order", type=int, default=3, help="低秩 PDE 最大交互阶数")
    parser.add_argument("--pde-rank", type=int, default=4, help="每个交互阶数的最大低秩分量数")
    parser.add_argument(
        "--pde-ranks",
        default="2:8,3:8",
        help='每个交互阶数的最大 rank，例如 "2:8,3:8,4:4"；优先于 --pde-rank',
    )
    parser.add_argument("--pde-dense-epochs", type=int, default=1000, help="PDE dense fitting 轮数")
    parser.add_argument("--pde-sparse-epochs", type=int, default=2000, help="PDE sparse fitting 轮数")
    parser.add_argument("--pde-learning-rate", type=float, default=1e-3, help="PDE 模型学习率")
    parser.add_argument("--pde-lambda-ridge", type=float, default=1e-6, help="PDE dense 阶段 ridge 正则")
    parser.add_argument("--pde-lambda-g", type=float, default=1e-4, help="PDE 门控稀疏正则")
    parser.add_argument("--pde-lambda-alpha", type=float, default=1e-5, help="PDE alpha 稀疏正则")
    parser.add_argument("--pde-lambda-b", type=float, default=1e-5, help="PDE 线性项稀疏正则")
    parser.add_argument("--pde-lambda-w", type=float, default=1e-5, help="PDE 低秩因子稀疏正则")
    parser.add_argument("--pde-lambda-binary", type=float, default=1e-4, help="PDE 门控二值化正则")
    parser.add_argument(
        "--pde-sparse-gate-init",
        type=float,
        default=0.5,
        help="dense 阶段结束后 sparse 阶段开始前重置 gate 的目标值，并补偿 alpha",
    )
    parser.add_argument("--pde-gate-threshold", type=float, default=0.2, help="门控剪枝阈值")
    parser.add_argument("--pde-alpha-threshold", type=float, default=1e-6, help="alpha 剪枝阈值")
    parser.add_argument("--pde-w-threshold", type=float, default=1e-2, help="低秩因子元素剪枝阈值")
    parser.add_argument("--pde-term-threshold", type=float, default=1e-6, help="展开单项式剪枝阈值")
    parser.add_argument("--pde-coefficient-threshold", type=float, default=1e-6, help="重拟合系数剪枝阈值")
    parser.add_argument(
        "--pde-derivative-method",
        choices=["autograd", "finite_difference"],
        default="autograd",
        help="PDE 发现导数来源：代理网络自动微分或原始网格有限差分",
    )
    parser.add_argument(
        "--pde-fd-boundary",
        type=int,
        default=3,
        help="finite_difference 模式下裁掉的空间/时间边界点数",
    )
    parser.add_argument(
        "--true-pde-json",
        default='{"u*u_x": -1.0, "u_xx": 0.1}',
        help='可选真实 PDE 系数字典，例如 {"u*u_x": -1.0, "u_xx": 0.1}',
    )
    
    # ===== 随机种子 =====
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    
    # ===== 输出参数 =====
    parser.add_argument(
        "--model-path",
        default=None,
        help="代理网络模型路径；默认使用输出目录下的 u_network_model.pt",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="即使模型文件已存在，也重新训练代理网络并覆盖保存",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="兼容旧参数；代理网络训练完成后现在会自动保存",
    )
    args = parser.parse_args()
    
    # ===== 设置随机种子 =====
    print("="*60)
    print("Setting Random Seeds")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_random_seed(args.seed, device)
    
    # 创建输出目录。未显式指定时按数据集文件名分目录，避免多次运行互相覆盖。
    if args.output_dir is None:
        output_dir = Path("./output") / args.data
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    model_path = Path(args.model_path) if args.model_path else output_dir / "u_network_model.pt"
    preprocessing_path = output_dir / "preprocessing.json"
    
    # ===== 步骤 1: 读取数据 =====
    print("\n" + "="*60)
    print("Step 1: Loading Data")
    print("="*60)
    input_file = Path(args.input_path) / f"{args.data}.npy"
    run_metadata = build_run_metadata(
        parser=parser,
        args=args,
        device=device,
        input_file=input_file,
        output_dir=output_dir,
        model_path=model_path,
    )
    coords, values, input_dim = load_data(str(input_file))
    run_metadata["input_dim"] = input_dim
    print(f"  Automatically detected input_dim: {input_dim}")
    
    # ===== 步骤 2: 预处理数据 =====
    print("\n" + "="*60)
    print("Step 2: Preprocessing Data")
    print("="*60)
    coords_train, values_train, test_data, preprocessing_info = preprocess_data(
        coords, values,
        noise_std=args.noise_std,
        test_split=args.test_split,
    )
    
    # ===== 步骤 3: 训练网络 =====
    print("\n" + "="*60)
    print("Step 3: Training U_Network")
    print("="*60)

    model_loaded_from_disk = False
    training_log_path = output_dir / "training_log.json"
    if model_path.exists() and not args.force_retrain:
        print(f"Existing U_Network model found: {model_path}")
        print("Loading existing model and skipping U_Network training.")
        model = load_u_network(
            str(model_path),
            input_dim=input_dim,
            num_hidden_layers=args.num_hidden_layers,
            neurons_per_layer=args.neurons_per_layer,
            activation_function=args.activation,
            device=device,
        )
        if model.input_dim != input_dim:
            raise ValueError(
                f"Loaded model input_dim={model.input_dim}, but data input_dim={input_dim}. "
                "Use --force-retrain to train a compatible model."
            )
        saved_preprocessing_info = getattr(model, "preprocessing_info", None)
        if saved_preprocessing_info is None and preprocessing_path.exists():
            with open(preprocessing_path, "r") as f:
                saved_preprocessing_info = json.load(f)
        if saved_preprocessing_info is not None:
            if not preprocessing_matches(saved_preprocessing_info, preprocessing_info):
                raise ValueError(
                    "Existing model was trained with different preprocessing statistics. "
                    "Use --force-retrain to retrain it for the current data split/settings."
                )
        else:
            print("Warning: existing model has no preprocessing metadata; using it as-is.")
        model_loaded_from_disk = True
        if training_log_path.exists():
            with open(training_log_path, "r") as f:
                train_log = json.load(f)
        else:
            train_log = {}
        train_log["loaded_existing_model"] = True
        train_log["training_skipped"] = True
        train_log["model_path"] = str(model_path)
    else:
        if args.force_retrain and model_path.exists():
            print(f"--force-retrain enabled; ignoring existing model: {model_path}")

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
                device=device,
                verbose=True,
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
                device=device,
                verbose=True,
                lambda_reg=args.lambda_reg,
            )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_u_network(model, str(model_path), preprocessing_info=preprocessing_info)
        train_log["loaded_existing_model"] = False
        train_log["training_skipped"] = False
        train_log["model_path"] = str(model_path)
    
    # ===== 步骤 4: 评估模型 =====
    print("\n" + "="*60)
    print("Step 4: Model Evaluation")
    print("="*60)
    
    # 在训练集上评估
    eval_train = evaluate_on_dataset(model, coords_train, values_train)
    metrics_train = {key: value for key, value in eval_train.items() if key != "predictions"}
    print("\nTraining Set Metrics:")
    print_metrics(metrics_train)
    
    # 在测试集上评估
    if test_data is not None:
        coords_test, values_test = test_data
        eval_test = evaluate_on_dataset(model, coords_test, values_test)
        metrics_test = {key: value for key, value in eval_test.items() if key != "predictions"}
        print("\nTest Set Metrics:")
        print_metrics(metrics_test)

    pde_result = None
    if args.discover_pde:
        print("\n" + "="*60)
        print("Step 5: PDE Discovery")
        print("="*60)
        if input_dim != 2:
            raise ValueError("PDE discovery currently supports only 1D data with columns [x, t, u].")

        rank_by_order = parse_rank_by_order(args.pde_ranks)
        pde_max_order = args.pde_max_order
        if rank_by_order is not None:
            pde_max_order = max(pde_max_order, max(rank_by_order))

        coords_val = test_data[0] if test_data is not None else None
        true_coefficients = json.loads(args.true_pde_json) if args.true_pde_json else None
        pde_result = discover_pde(
            model=model,
            coords=coords_train,
            preprocessing_info=preprocessing_info,
            coords_val=coords_val,
            raw_coords=coords,
            raw_values=values,
            derivative_method=args.pde_derivative_method,
            fd_boundary=args.pde_fd_boundary,
            max_order=pde_max_order,
            rank=args.pde_rank,
            rank_by_order=rank_by_order,
            dense_epochs=args.pde_dense_epochs,
            sparse_epochs=args.pde_sparse_epochs,
            learning_rate=args.pde_learning_rate,
            lambda_ridge=args.pde_lambda_ridge,
            lambda_g=args.pde_lambda_g,
            lambda_alpha=args.pde_lambda_alpha,
            lambda_b=args.pde_lambda_b,
            lambda_w=args.pde_lambda_w,
            lambda_binary=args.pde_lambda_binary,
            sparse_gate_init=args.pde_sparse_gate_init,
            gate_threshold=args.pde_gate_threshold,
            alpha_threshold=args.pde_alpha_threshold,
            w_threshold=args.pde_w_threshold,
            term_threshold=args.pde_term_threshold,
            coefficient_threshold=args.pde_coefficient_threshold,
            true_coefficients=true_coefficients,
            verbose=True,
        )
        print("\nDiscovered PDE:")
        print(pde_result["pde_string"])
        print(f"PDE residual MSE: {pde_result['residual_mse']:.6e}")
        if pde_result["validation_residual_mse"] is not None:
            print(f"Validation residual MSE: {pde_result['validation_residual_mse']:.6e}")
        if pde_result.get("true_pde_metrics") is not None:
            print("Term identification:")
            print(json.dumps(pde_result["true_pde_metrics"], indent=2))
    
    # ===== 保存结果 =====
    save_step = 6 if args.discover_pde else 5
    print("\n" + "="*60)
    print(f"Step {save_step}: Saving Results")
    print("="*60)
    
    # 模型在训练完成后已自动保存；若本次加载已有模型，这里只记录使用的路径。
    if model_loaded_from_disk:
        print(f"Using existing model: {model_path}")
    
    # 保存训练日志
    train_log_path = training_log_path
    train_log["run_metadata"] = run_metadata
    train_log["run_arguments"] = run_metadata["arguments"]
    train_log["reproduce_command"] = run_metadata["reproduce_command"]
    with open(train_log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"Training log saved to {train_log_path}")

    reproduce_command_path = output_dir / "reproduce_command.txt"
    with open(reproduce_command_path, "w") as f:
        f.write(run_metadata["reproduce_command"] + "\n")
    print(f"Reproduce command saved to {reproduce_command_path}")
    
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

    with open(preprocessing_path, "w") as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"Preprocessing parameters saved to {preprocessing_path}")

    if pde_result is not None:
        pde_path = output_dir / "pde_discovery.json"
        with open(pde_path, "w") as f:
            json.dump(pde_result, f, indent=2)
        print(f"PDE discovery result saved to {pde_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
