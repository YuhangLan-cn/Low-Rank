"""
代理网络测试和推理模块。
"""
import torch
import numpy as np

from network import U_Network


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
    
    coords_tensor = torch.from_numpy(coords.astype(np.float64)).to(dtype=dtype, device=device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(coords_tensor).view(-1).cpu().numpy()
    
    return predictions
