"""
代理网络训练的损失函数定义模块。
"""
import torch
from typing import Dict


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """均方误差损失函数。
    
    Args:
        predictions: 模型预测值，形状为 (batch_size,) 或 (batch_size, 1)
        targets: 目标值，形状为 (batch_size,) 或 (batch_size, 1)
        
    Returns:
        MSE 损失值（标量张量）
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    return torch.mean((predictions - targets) ** 2)


def weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """加权均方误差损失函数。
    
    Args:
        predictions: 模型预测值，形状为 (batch_size,) 或 (batch_size, 1)
        targets: 目标值，形状为 (batch_size,) 或 (batch_size, 1)
        weights: 样本权重，形状为 (batch_size,) 或 (batch_size, 1)
        
    Returns:
        加权 MSE 损失值（标量张量）
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    if weights is None:
        return torch.mean((predictions - targets) ** 2)
    
    weights = weights.view(-1)
    weighted_sq_error = weights * ((predictions - targets) ** 2)
    return torch.sum(weighted_sq_error) / torch.sum(weights).clamp_min(1e-12)


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """平均绝对误差损失函数。
    
    Args:
        predictions: 模型预测值，形状为 (batch_size,) 或 (batch_size, 1)
        targets: 目标值，形状为 (batch_size,) 或 (batch_size, 1)
        
    Returns:
        MAE 损失值（标量张量）
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    return torch.mean(torch.abs(predictions - targets))


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber 损失函数（鲁棒的损失函数，对异常值更不敏感）。
    
    Args:
        predictions: 模型预测值
        targets: 目标值
        delta: Huber 损失的阈值参数
        
    Returns:
        Huber 损失值（标量张量）
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    errors = predictions - targets
    return torch.mean(torch.where(
        torch.abs(errors) < delta,
        0.5 * errors ** 2,
        delta * (torch.abs(errors) - 0.5 * delta)
    ))


def regularized_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    model: torch.nn.Module = None,
    lambda_reg: float = 1e-4,
    reg_type: str = "L2",
) -> Dict[str, torch.Tensor]:
    """带正则化的损失函数。
    
    Args:
        predictions: 模型预测值
        targets: 目标值
        model: 网络模型（用于计算正则化项）
        lambda_reg: 正则化系数
        reg_type: 正则化类型 ("L1" 或 "L2")
        
    Returns:
        包含总损失和各项损失的字典
    """
    pred_loss = mse_loss(predictions, targets)
    
    if model is None or lambda_reg == 0:
        return {"total": pred_loss, "pred": pred_loss, "reg": torch.tensor(0.0)}
    
    # 计算正则化项
    reg_loss = torch.tensor(0.0, dtype=pred_loss.dtype, device=pred_loss.device)
    
    if reg_type == "L2":
        for param in model.parameters():
            reg_loss = reg_loss + torch.sum(param ** 2)
    elif reg_type == "L1":
        for param in model.parameters():
            reg_loss = reg_loss + torch.sum(torch.abs(param))
    
    reg_loss = reg_loss * lambda_reg
    total_loss = pred_loss + reg_loss
    
    return {
        "total": total_loss,
        "pred": pred_loss,
        "reg": reg_loss,
    }
