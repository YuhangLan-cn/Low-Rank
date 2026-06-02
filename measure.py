"""
代理网络的评估和度量模块。
"""
import numpy as np
import torch
from typing import Dict


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算均方误差 (MSE)。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        MSE 值
    """
    return np.mean((predictions - targets) ** 2)


def root_mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算均方根误差 (RMSE)。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        RMSE 值
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mean_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算平均绝对误差 (MAE)。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        MAE 值
    """
    return np.mean(np.abs(predictions - targets))


def relative_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算相对误差。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        相对误差
    """
    target_norm = np.linalg.norm(targets)
    if target_norm < 1e-12:
        return np.inf
    
    return np.linalg.norm(predictions - targets) / target_norm


def r_squared(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算决定系数 R²。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        R² 值
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    
    return 1.0 - (ss_res / ss_tot)


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算代理网络拟合质量的核心指标。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        包含核心指标的字典
    """
    return {
        "mse": mean_squared_error(predictions, targets),
        "rmse": root_mean_squared_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets),
        "relative_error": relative_error(predictions, targets),
        "r_squared": r_squared(predictions, targets),
    }


def compute_tensor_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """把张量预测转换为 numpy 后计算核心指标。"""
    metrics = compute_all_metrics(
        predictions.detach().cpu().numpy(),
        targets.detach().cpu().numpy(),
    )
    return {
        key: float(value) if np.isfinite(value) else str(value)
        for key, value in metrics.items()
    }


def tensor_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算张量形式预测和目标之间的 MSE，用于训练过程中的评估记录。"""
    return float(torch.mean((predictions - targets) ** 2).detach().cpu().item())


def evaluate_on_dataset(model, coords: np.ndarray, targets: np.ndarray) -> Dict:
    """在数据集上评估代理网络，并返回预测值和核心指标。"""
    device = model.device
    dtype = model.data_type
    coords_tensor = torch.from_numpy(coords.astype(np.float64)).to(dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        predictions = model(coords_tensor).view(-1).cpu().numpy()

    metrics = compute_all_metrics(predictions, targets)
    metrics["predictions"] = predictions
    return metrics


def pde_residual_mse(ut: np.ndarray, rhs: np.ndarray) -> float:
    """计算 PDE residual MSE: mean((u_t - rhs)^2)。"""
    return mean_squared_error(np.asarray(rhs).reshape(-1), np.asarray(ut).reshape(-1))


def zero_rhs_residual_mse(ut: np.ndarray) -> float:
    """没有候选项时，以 0 右端作为基线 residual。"""
    return float(np.mean(np.asarray(ut).reshape(-1) ** 2))


def summarize_derivative_data(ut: np.ndarray) -> Dict[str, float]:
    """汇总自动微分得到的时间导数信息。"""
    ut = np.asarray(ut).reshape(-1)
    return {
        "num_points": int(ut.shape[0]),
        "ut_mean": float(np.mean(ut)),
        "ut_std": float(np.std(ut)),
    }


def evaluate_discovered_terms(
    discovered_coefficients: Dict[str, float],
    true_coefficients: Dict[str, float],
) -> Dict:
    """评估发现出的 PDE 项：系数相对误差、TP/FP/FN。"""
    discovered_terms = {term for term, coeff in discovered_coefficients.items() if abs(coeff) > 0}
    true_terms = {term for term, coeff in true_coefficients.items() if abs(coeff) > 0}
    all_terms = sorted(discovered_terms | true_terms)

    discovered_vector = np.asarray(
        [float(discovered_coefficients.get(term, 0.0)) for term in all_terms],
        dtype=np.float64,
    )
    true_vector = np.asarray(
        [float(true_coefficients.get(term, 0.0)) for term in all_terms],
        dtype=np.float64,
    )
    true_norm = np.linalg.norm(true_vector)
    coefficient_error = (
        float(np.linalg.norm(discovered_vector - true_vector) / true_norm)
        if true_norm > 1e-12
        else None
    )

    return {
        "coefficient_relative_error": coefficient_error,
        "true_positive": sorted(discovered_terms & true_terms),
        "false_positive": sorted(discovered_terms - true_terms),
        "false_negative": sorted(true_terms - discovered_terms),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """打印格式化的指标。
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    if prefix:
        print(f"\n{prefix}")
        print("-" * 50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if np.isfinite(value):
                if abs(value) < 1e-4 or abs(value) > 1e4:
                    print(f"  {key:20s}: {value:12.4e}")
                else:
                    print(f"  {key:20s}: {value:12.6f}")
            else:
                print(f"  {key:20s}: {value}")
        else:
            print(f"  {key:20s}: {value}")
