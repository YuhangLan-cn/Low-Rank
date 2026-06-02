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
        return {"total": pred_loss, "pred": pred_loss, "reg": pred_loss.new_tensor(0.0)}
    
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
