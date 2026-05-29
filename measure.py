"""
代理网络的评估和度量模块。
"""
import numpy as np
import torch
from typing import Dict, Tuple
from scipy.stats import pearsonr, spearmanr


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


def mean_absolute_percentage_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算平均绝对百分比误差 (MAPE)。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        MAPE 值（百分比）
    """
    mask = np.abs(targets) > 1e-12
    if not np.any(mask):
        return np.inf
    
    errors = np.abs((predictions[mask] - targets[mask]) / targets[mask])
    return np.mean(errors) * 100


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


def pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """计算 Pearson 相关系数。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        (相关系数, p-value) 元组
    """
    return pearsonr(predictions, targets)


def spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """计算 Spearman 秩相关系数。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        (相关系数, p-value) 元组
    """
    return spearmanr(predictions, targets)


def max_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算最大误差。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        最大误差
    """
    return np.max(np.abs(predictions - targets))


def median_absolute_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算中位数绝对误差。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        中位数绝对误差
    """
    return np.median(np.abs(predictions - targets))


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """一次性计算所有评估指标。
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        包含所有指标的字典
    """
    metrics = {
        "mse": mean_squared_error(predictions, targets),
        "rmse": root_mean_squared_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets),
        "mape": mean_absolute_percentage_error(predictions, targets),
        "max_error": max_error(predictions, targets),
        "median_ae": median_absolute_error(predictions, targets),
        "relative_error": relative_error(predictions, targets),
        "r_squared": r_squared(predictions, targets),
    }
    
    try:
        pearson_r, pearson_p = pearson_correlation(predictions, targets)
        metrics["pearson_r"] = pearson_r
        metrics["pearson_p"] = pearson_p
    except Exception:
        pass
    
    try:
        spearman_r, spearman_p = spearman_correlation(predictions, targets)
        metrics["spearman_r"] = spearman_r
        metrics["spearman_p"] = spearman_p
    except Exception:
        pass
    
    return metrics


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


def compute_derivative_error(
    computed_derivatives: Dict[str, np.ndarray],
    reference_derivatives: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """计算导数的误差。
    
    Args:
        computed_derivatives: 计算出的导数字典
        reference_derivatives: 参考导数字典
        
    Returns:
        导数误差字典
    """
    errors = {}
    
    for key in computed_derivatives:
        if key not in reference_derivatives:
            continue
        
        computed = computed_derivatives[key]
        reference = reference_derivatives[key]
        
        mse = mean_squared_error(computed, reference)
        mae = mean_absolute_error(computed, reference)
        rel_error = relative_error(computed, reference)
        
        errors[f"{key}_mse"] = mse
        errors[f"{key}_mae"] = mae
        errors[f"{key}_rel"] = rel_error
    
    return errors


def model_complexity_metrics(model: torch.nn.Module) -> Dict[str, int]:
    """计算模型的复杂度指标。
    
    Args:
        model: PyTorch 模型
        
    Returns:
        复杂度指标字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算浮点运算数（FLOPs）的粗略估计
    # 对于全连接网络：每层的 FLOPs ≈ 2 * 输入维度 * 输出维度
    total_flops = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            total_flops += 2 * m.in_features * m.out_features
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_flops": total_flops,
    }


def compute_prediction_confidence(
    predictions: np.ndarray,
    uncertainty: np.ndarray = None,
) -> Dict[str, float]:
    """计算预测的置信度。
    
    Args:
        predictions: 预测值
        uncertainty: 不确定性估计（可选）
        
    Returns:
        置信度指标字典
    """
    metrics = {
        "mean_prediction": float(np.mean(predictions)),
        "std_prediction": float(np.std(predictions)),
        "min_prediction": float(np.min(predictions)),
        "max_prediction": float(np.max(predictions)),
    }
    
    if uncertainty is not None:
        metrics["mean_uncertainty"] = float(np.mean(uncertainty))
        metrics["std_uncertainty"] = float(np.std(uncertainty))
        metrics["max_uncertainty"] = float(np.max(uncertainty))
        metrics["signal_to_noise_ratio"] = float(
            np.mean(np.abs(predictions)) / (np.mean(uncertainty) + 1e-12)
        )
    
    return metrics
