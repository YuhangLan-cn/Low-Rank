"""
代理网络训练模块，包含网络训练的完整流程。
"""
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm

from network import U_Network, u_mse_loss
from loss import weighted_mse_loss, regularized_loss


def _coord_normalization_from_bounds(
    coord_mins: np.ndarray,
    coord_maxs: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 coords_norm = (coords - offset) * scale 的参数。
    
    Args:
        coord_mins: 坐标的最小值
        coord_maxs: 坐标的最大值
        eps: 防止除以零的小值
        
    Returns:
        (offset, scale) 元组
    """
    coord_mins = np.asarray(coord_mins, dtype=np.float64).reshape(-1)
    coord_maxs = np.asarray(coord_maxs, dtype=np.float64).reshape(-1)
    ranges = np.maximum(coord_maxs - coord_mins, eps)
    scale = 2.0 / ranges
    offset = 0.5 * (coord_mins + coord_maxs)
    return offset, scale


def normalize_coords_with_model(model: U_Network, coords: np.ndarray) -> np.ndarray:
    """使用保存在模型上的坐标归一化参数，把物理坐标映射到网络输入尺度。
    
    Args:
        model: U_Network 模型
        coords: 物理坐标
        
    Returns:
        归一化后的坐标
    """
    coords = np.asarray(coords, dtype=np.float64)
    offset = getattr(model, "coord_offset_np", None)
    scale = getattr(model, "coord_scale_np", None)
    if offset is None or scale is None:
        return coords
    return (coords - offset.reshape(1, -1)) * scale.reshape(1, -1)


def train_u_network(
    coords: np.ndarray,
    targets: np.ndarray,
    input_dim: int,
    num_epochs: int = 5000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    num_hidden_layers: int = 5,
    neurons_per_layer: int = 50,
    activation_function: str = "Rat",
    device: str = "cpu",
    verbose: bool = True,
    coords_test: Optional[np.ndarray] = None,
    targets_test: Optional[np.ndarray] = None,
    coord_mins: Optional[np.ndarray] = None,
    coord_maxs: Optional[np.ndarray] = None,
    lambda_reg: float = 0.0,
) -> Tuple[U_Network, Dict]:
    """
    训练 U_Network 来学习时空场（接收散点坐标和值）。
    
    Args:
        coords: shape=(num_samples, input_dim) 的散点坐标，其中 input_dim = num_spatial_dims + 1
        targets: shape=(num_samples,) 的目标值
        input_dim: 输入维度（空间维数 + 时间）
        num_epochs: 训练的总轮数，默认为 5000
        batch_size: 每批次的样本数，默认为 256
        learning_rate: Adam 优化器的学习率，默认为 1e-3
        num_hidden_layers: 网络隐藏层数量，默认为 5
        neurons_per_layer: 每层神经元数量，默认为 50
        activation_function: 激活函数类型 ("Rat", "Tanh", "Sin")，默认为 "Rat"
        device: 计算设备 ("cpu" 或 "cuda")，默认为 "cpu"
        verbose: 是否打印训练进度，默认为 True
        coords_test: 可选，测试集坐标，shape=(num_test_samples, input_dim)
        targets_test: 可选，测试集目标值，shape=(num_test_samples,)
        coord_mins: 坐标的最小值，用于归一化。若为 None，自动从 coords 计算
        coord_maxs: 坐标的最大值，用于归一化。若为 None，自动从 coords 计算
        lambda_reg: 正则化系数，默认为 0（无正则化）
        
    Returns:
        (model, train_log) 元组，其中：
        - model: 训练完成的 U_Network 模型
        - train_log: 包含训练信息的字典（epoch、loss等）
    """
    device_obj = torch.device(device)
    dtype = torch.float64
    
    # 创建网络
    model = U_Network(
        input_dim=input_dim,
        num_hidden_layers=num_hidden_layers,
        neurons_per_layer=neurons_per_layer,
        activation_function=activation_function,
        data_type=dtype,
        device=device_obj,
    )
    
    # 将物理坐标归一化到大致 [-1, 1]
    coords = coords.astype(np.float64)
    targets = targets.astype(np.float64)
    if coord_mins is None:
        coord_mins = coords.min(axis=0)
    if coord_maxs is None:
        coord_maxs = coords.max(axis=0)
    
    coord_offset, coord_scale = _coord_normalization_from_bounds(coord_mins, coord_maxs)
    coords_norm = (coords - coord_offset.reshape(1, -1)) * coord_scale.reshape(1, -1)
    
    # 保存归一化参数到模型
    model.coord_offset_np = coord_offset
    model.coord_scale_np = coord_scale
    model.coord_derivative_scales = torch.from_numpy(coord_scale).to(dtype=dtype, device=device_obj)
    
    coords_tensor = torch.from_numpy(coords_norm).to(dtype=dtype, device=device_obj)
    targets_tensor = torch.from_numpy(targets).to(dtype=dtype, device=device_obj)
    
    # 准备测试数据张量（如果提供）
    has_test = coords_test is not None and targets_test is not None
    if has_test:
        coords_test = coords_test.astype(np.float64)
        targets_test = targets_test.astype(np.float64)
        coords_test_norm = (coords_test - coord_offset.reshape(1, -1)) * coord_scale.reshape(1, -1)
        coords_test_tensor = torch.from_numpy(coords_test_norm).to(dtype=dtype, device=device_obj)
        targets_test_tensor = torch.from_numpy(targets_test).to(dtype=dtype, device=device_obj)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    total_samples = len(coords_tensor)
    train_log = {
        "epochs": [],
        "train_losses": [],
        "test_losses": [],
    }
    
    pbar = tqdm(range(num_epochs), disable=not verbose, desc="Training U_Network")
    
    for epoch in pbar:
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 随机打乱样本
        indices = torch.randperm(total_samples, device=device_obj)
        coords_shuffled = coords_tensor[indices]
        targets_shuffled = targets_tensor[indices]
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_coords = coords_shuffled[batch_start:batch_end]
            batch_targets = targets_shuffled[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            if lambda_reg > 0:
                losses = regularized_loss(
                    model(batch_coords).view(-1),
                    batch_targets,
                    model=model,
                    lambda_reg=lambda_reg,
                )
                loss = losses["total"]
            else:
                loss = u_mse_loss(model, batch_coords, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        
        # 测试阶段
        model.eval()
        with torch.no_grad():
            train_pred = model(coords_tensor).view(-1)
            train_loss = u_mse_loss(model, coords_tensor, targets_tensor).item()
            
            if has_test:
                test_pred = model(coords_test_tensor).view(-1)
                test_loss = u_mse_loss(model, coords_test_tensor, targets_test_tensor).item()
            else:
                test_loss = None
        
        train_log["epochs"].append(epoch)
        train_log["train_losses"].append(train_loss)
        if has_test:
            train_log["test_losses"].append(test_loss)
        
        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            msg = f"Epoch {epoch:4d}: train_loss={train_loss:.6e}"
            if has_test:
                msg += f", test_loss={test_loss:.6e}"
            pbar.set_postfix_str(msg)
    
    if verbose:
        print(f"Training completed. Final train loss: {train_loss:.6e}")
    
    return model, train_log


def train_u_network_with_validation(
    coords: np.ndarray,
    targets: np.ndarray,
    input_dim: int,
    train_ratio: float = 0.8,
    num_epochs: int = 5000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    num_hidden_layers: int = 5,
    neurons_per_layer: int = 50,
    activation_function: str = "Rat",
    device: str = "cpu",
    verbose: bool = True,
    coord_mins: Optional[np.ndarray] = None,
    coord_maxs: Optional[np.ndarray] = None,
    lambda_reg: float = 0.0,
    early_stopping_patience: int = 100,
) -> Tuple[U_Network, Dict]:
    """
    带验证集和早停机制的 U_Network 训练。
    
    Args:
        coords: 输入坐标
        targets: 目标值
        input_dim: 输入维度
        train_ratio: 训练集占比，默认 0.8
        num_epochs: 最大训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        num_hidden_layers: 隐藏层数
        neurons_per_layer: 每层神经元数
        activation_function: 激活函数名称
        device: 计算设备
        verbose: 是否打印日志
        coord_mins: 坐标最小值
        coord_maxs: 坐标最大值
        lambda_reg: 正则化系数
        early_stopping_patience: 早停耐心值
        
    Returns:
        (model, train_log) 元组
    """
    # 划分训练集和验证集
    n_samples = len(targets)
    n_train = int(n_samples * train_ratio)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    coords_train = coords[train_indices]
    targets_train = targets[train_indices]
    coords_val = coords[val_indices]
    targets_val = targets[val_indices]
    
    return train_u_network(
        coords=coords_train,
        targets=targets_train,
        input_dim=input_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_hidden_layers=num_hidden_layers,
        neurons_per_layer=neurons_per_layer,
        activation_function=activation_function,
        device=device,
        verbose=verbose,
        coords_test=coords_val,
        targets_test=targets_val,
        coord_mins=coord_mins,
        coord_maxs=coord_maxs,
        lambda_reg=lambda_reg,
    )


def save_u_network(model: U_Network, output_path: str) -> None:
    """保存 U_Network 模型。
    
    Args:
        model: U_Network 模型
        output_path: 保存路径
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "num_hidden_layers": model.num_hidden_layers,
        "neurons_per_layer": model.neurons_per_layer,
        "activation_name": model.activation_name,
        "data_type": model.data_type,
        "coord_offset_np": getattr(model, "coord_offset_np", None),
        "coord_scale_np": getattr(model, "coord_scale_np", None),
    }
    torch.save(checkpoint, output_path)
    print(f"Model saved to {output_path}")


def load_u_network(
    model_path: str,
    input_dim: int = 2,
    num_hidden_layers: int = 5,
    neurons_per_layer: int = 50,
    activation_function: str = "Rat",
    device: str = "cpu",
) -> U_Network:
    """加载 U_Network 模型。
    
    Args:
        model_path: 模型路径
        input_dim: 输入维度
        num_hidden_layers: 隐藏层数
        neurons_per_layer: 每层神经元数
        activation_function: 激活函数名称
        device: 计算设备
        
    Returns:
        加载的 U_Network 模型
    """
    device_obj = torch.device(device)
    dtype = torch.float64
    
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    model = U_Network(
        input_dim=checkpoint.get("input_dim", input_dim),
        num_hidden_layers=checkpoint.get("num_hidden_layers", num_hidden_layers),
        neurons_per_layer=checkpoint.get("neurons_per_layer", neurons_per_layer),
        activation_function=checkpoint.get("activation_name", activation_function),
        data_type=dtype,
        device=device_obj,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if checkpoint.get("coord_offset_np") is not None:
        model.coord_offset_np = checkpoint["coord_offset_np"]
    if checkpoint.get("coord_scale_np") is not None:
        model.coord_scale_np = checkpoint["coord_scale_np"]
        model.coord_derivative_scales = torch.from_numpy(checkpoint["coord_scale_np"]).to(dtype=dtype, device=device_obj)
    
    model.eval()
    print(f"Model loaded from {model_path}")
    return model
