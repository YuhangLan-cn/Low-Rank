"""
代理网络测试和推理模块。
"""
import torch
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from network import U_Network, evaluate_u_derivatives
from train import normalize_coords_with_model


def evaluate_on_dataset(
    model: U_Network,
    coords: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """在数据集上评估模型性能。
    
    Args:
        model: U_Network 模型
        coords: 输入坐标
        targets: 目标值
        
    Returns:
        包含 MSE、MAE、RMSE 等指标的字典
    """
    device = model.device
    dtype = model.data_type
    
    coords_norm = normalize_coords_with_model(model, coords)
    coords_tensor = torch.from_numpy(coords_norm).to(dtype=dtype, device=device)
    targets_tensor = torch.from_numpy(targets.astype(np.float64)).to(dtype=dtype, device=device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(coords_tensor).view(-1).cpu().numpy()
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # 相对误差
    target_norm = np.linalg.norm(targets)
    if target_norm > 1e-12:
        relative_error = np.linalg.norm(predictions - targets) / target_norm
    else:
        relative_error = np.inf
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "relative_error": relative_error,
        "predictions": predictions,
    }


def predict(
    model: U_Network,
    coords: np.ndarray,
) -> np.ndarray:
    """使用模型进行预测。
    
    Args:
        model: U_Network 模型
        coords: 输入坐标
        
    Returns:
        预测值
    """
    device = model.device
    dtype = model.data_type
    
    coords_norm = normalize_coords_with_model(model, coords)
    coords_tensor = torch.from_numpy(coords_norm).to(dtype=dtype, device=device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(coords_tensor).view(-1).cpu().numpy()
    
    return predictions


def compute_derivatives(
    model: U_Network,
    coords: np.ndarray,
    num_spatial_dims: int = 1,
    max_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用自动微分计算导数。
    
    Args:
        model: U_Network 模型
        coords: 输入坐标
        num_spatial_dims: 空间维度数
        max_order: 最大导数阶数
        
    Returns:
        (u, atoms, ut) 元组
        - u: 函数值
        - atoms: 所有空间导数原子
        - ut: 时间导数
    """
    device = model.device
    dtype = model.data_type
    
    coords_norm = normalize_coords_with_model(model, coords)
    coords_tensor = torch.from_numpy(coords_norm).to(dtype=dtype, device=device)
    
    derivative_scales = getattr(model, "coord_derivative_scales", None)
    
    u, atoms, ut = evaluate_u_derivatives(
        model,
        coords_tensor,
        num_spatial_dims=num_spatial_dims,
        max_order=max_order,
        derivative_scales=derivative_scales,
    )
    
    return (
        u.detach().cpu().numpy(),
        atoms.detach().cpu().numpy(),
        ut.detach().cpu().numpy(),
    )


def plot_1d_predictions(
    model: U_Network,
    x_range: Tuple[float, float],
    t_values: list,
    n_points: int = 100,
    save_path: Optional[str] = None,
) -> None:
    """绘制 1D 问题的预测结果。
    
    Args:
        model: U_Network 模型
        x_range: x 的范围 (x_min, x_max)
        t_values: 要绘制的时间点列表
        n_points: 空间采样点数
        save_path: 保存路径
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)
    
    fig, axes = plt.subplots(len(t_values), 1, figsize=(10, 4*len(t_values)))
    if len(t_values) == 1:
        axes = [axes]
    
    for idx, t in enumerate(t_values):
        coords = np.column_stack([x, np.full_like(x, t)])
        u = predict(model, coords)
        
        axes[idx].plot(x, u, 'b-', linewidth=2)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('u(x, t)')
        axes[idx].set_title(f'Prediction at t={t}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_predictions_vs_targets(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """绘制预测值与真实值的对比图。
    
    Args:
        predictions: 预测值
        targets: 真实值
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 散点图
    axes[0].scatter(targets, predictions, alpha=0.5, s=10)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Target values')
    axes[0].set_ylabel('Predicted values')
    axes[0].set_title('Predictions vs Targets')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = predictions - targets
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_error_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """绘制详细的误差分析图。
    
    Args:
        predictions: 预测值
        targets: 真实值
        save_path: 保存路径
    """
    errors = np.abs(predictions - targets)
    relative_errors = errors / (np.abs(targets) + 1e-12)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 绝对误差
    axes[0, 0].semilogy(errors, 'o', markersize=3, alpha=0.5)
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].set_title('Absolute Errors')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 相对误差
    axes[0, 1].semilogy(relative_errors, 'o', markersize=3, alpha=0.5, color='orange')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].set_title('Relative Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 误差直方图
    axes[1, 0].hist(np.log10(errors + 1e-16), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('log10(Absolute Error)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Log Error Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 统计信息
    axes[1, 1].axis('off')
    stats_text = f"""
    Mean Absolute Error: {np.mean(errors):.4e}
    Std Absolute Error: {np.std(errors):.4e}
    Max Absolute Error: {np.max(errors):.4e}
    Min Absolute Error: {np.min(errors):.4e}
    
    Mean Relative Error: {np.mean(relative_errors):.4e}
    Std Relative Error: {np.std(relative_errors):.4e}
    Max Relative Error: {np.max(relative_errors):.4e}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()


def test_on_regular_grid(
    model: U_Network,
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    nx: int = 64,
    nt: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在规则网格上进行测试。
    
    Args:
        model: U_Network 模型
        x_range: x 的范围 (x_min, x_max)
        t_range: t 的范围 (t_min, t_max)
        nx: x 方向采样点数
        nt: t 方向采样点数
        
    Returns:
        (x, t, u) 元组
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    xx, tt = np.meshgrid(x, t)
    
    coords = np.column_stack([xx.ravel(), tt.ravel()])
    u = predict(model, coords)
    u = u.reshape(nt, nx)
    
    return x, t, u
