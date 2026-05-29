"""
代理网络定义模块，包括 U_Network 及其相关的激活函数。
"""
import torch
import torch.nn as nn
from typing import Tuple
import torch.func as F


class Rational(torch.nn.Module):
    """可训练的有理激活函数，用于 PDE-READ 方法。
    
    这是一个自定义的激活函数，形式为 (a0 + a1*x + a2*x^2 + a3*x^3) / (b0 + b1*x + b2*x^2)
    其中 a 和 b 的系数在训练过程中是可学习的参数。
    """

    def __init__(
        self,
        data_type: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.register_parameter("numerator_0", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))
        self.register_parameter("numerator_1", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))
        self.register_parameter("numerator_2", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))
        self.register_parameter("numerator_3", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))
        self.register_parameter("denominator_0", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))
        self.register_parameter("denominator_1", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))
        self.register_parameter("denominator_2", torch.nn.Parameter(torch.tensor(0.5, dtype=data_type, device=device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = (
            self.numerator_0
            + self.numerator_1 * x
            + self.numerator_2 * (x**2)
            + self.numerator_3 * (x**3)
        )
        denominator = (
            self.denominator_0
            + self.denominator_1 * x
            + self.denominator_2 * (x**2)
        )
        return numerator / (denominator + 1e-8)


class Sin(torch.nn.Module):
    """正弦激活函数的包装类。"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class U_Network(torch.nn.Module):
    """独立的代理网络模型，用于近似 U(...spatial_coords..., t)。

    输入张量形状为 (batch_size, n_dim)，其中列按 [x1, x2, ..., xn, t] 的顺序排列。
    输出形状为 (batch_size, 1)。
    
    支持多维空间（1D、2D、3D 或更高维）。
    
    这个网络通常用于 PINN (Physics-Informed Neural Networks) 中，
    用来学习 PDE 的解。
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_hidden_layers: int = 5,
        neurons_per_layer: int = 50,
        activation_function: str = "Rat",
        data_type: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation_function
        self.data_type = data_type
        self.device = device

        layers = []
        layers.append(nn.Linear(input_dim, neurons_per_layer, dtype=data_type, device=device))
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer, dtype=data_type, device=device))
        layers.append(nn.Linear(neurons_per_layer, 1, dtype=data_type, device=device))

        self.layers = nn.ModuleList(layers)
        self.activation = self._make_activation()
        self._initialize_parameters()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    def _make_activation(self) -> torch.nn.Module:
        if self.activation_name == "Rat":
            return Rational(data_type=self.data_type, device=self.device)
        elif self.activation_name == "Tanh":
            return nn.Tanh()
        elif self.activation_name == "Sin":
            return Sin()
        else:
            return nn.Tanh()

    def _initialize_parameters(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


def u_mse_loss(
    u_network: U_Network,
    coords: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算 U 网络的 MSE 损失函数。
    
    该函数用于将神经网络的预测值与观测数据进行拟合。
    
    Args:
        u_network: U_Network 模型实例
        coords: 输入坐标，形状为 (batch_size, 2)，按 [x, t] 顺序排列
        targets: 目标值（真实数据），形状为 (batch_size,)
        
    Returns:
        MSE 损失值（标量张量）
    """
    predictions = u_network(coords).view(-1)
    return torch.mean((predictions - targets.view(-1)) ** 2)


def _generate_derivative_orders(num_dims: int, max_order: int) -> list:
    """生成所有 max_order 以内的混合偏导数阶数组合。
    
    例如，对于 2D 和 max_order=2：
    [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
    
    Args:
        num_dims: 维度数
        max_order: 最大阶数
        
    Returns:
        list of tuple，每个 tuple 表示一个混合偏导数
    """
    if num_dims == 0:
        return [[]]
    if max_order == 0:
        return [[0] * num_dims]
    
    orders = []
    
    def generate(current_order, remaining_dims, remaining_order):
        if remaining_dims == 0:
            orders.append(tuple(current_order))
            return
        if remaining_order == 0:
            orders.append(tuple(current_order + [0] * remaining_dims))
            return
        for order in range(remaining_order + 1):
            generate(current_order + [order], remaining_dims - 1, remaining_order - order)
    
    generate([], num_dims, max_order)
    return orders


def _generate_spatial_derivative_orders(num_spatial_dims: int, max_order: int) -> list:
    """生成仅包含空间导数的阶数组合（不包含时间导数）。
    
    时间维度的导数阶数固定为 0（即不对时间求导）。
    例如，对于 num_spatial_dims=1, max_order=2：
    [(0, 0), (1, 0), (2, 0)]  其中最后一个 0 代表时间维度不求导
    
    Args:
        num_spatial_dims: 空间维度数
        max_order: 最大阶数（仅应用于空间导数）
        
    Returns:
        list of tuple，每个 tuple 长度为 (num_spatial_dims + 1)，最后一个元素固定为 0
    """
    spatial_orders = _generate_derivative_orders(num_spatial_dims, max_order)
    orders_with_time_zero = [order + (0,) for order in spatial_orders]
    return orders_with_time_zero


def evaluate_u_derivatives(
    u_network: U_Network,
    coords: torch.Tensor,
    num_spatial_dims: int = 1,
    max_order: int = 3,
    derivative_scales: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用自动微分计算 U 及其关于空间坐标的混合偏导数（最高三阶）。

    坐标必须按 [x1, x2, ..., xn, t] 顺序排列，其中最后一列是时间。
    
    **重要**：原子库中**仅包含空间导数**，**不包含任何时间导数项**。
    
    返回 (u, all_atoms, ut)，其中：
    - all_atoms: 形状为 (batch_size, num_atoms)，包含所有空间导数项
      - 包括 u, u², u_x, u_xx, u_xxx（仅空间导数，最高3阶）
      - **不包括任何时间导数项**（u_t, u_tt, u_xt等都被排除）
    - ut: 形状为 (batch_size, 1)，u_t（关于 t 的一阶偏导数，作为方程左侧）
    
    Args:
        u_network: U_Network 模型实例
        coords: 输入坐标，形状为 (batch_size, n_dim)，按 [x1, x2, ..., xn, t] 顺序排列。
            如果网络使用了坐标归一化，这里应传入归一化后的坐标。
        num_spatial_dims: 空间维度数（例如，1D 空间时为 1，2D 空间时为 2）
        max_order: 空间混合导数的最大总阶数，默认为 3
        derivative_scales: 链式法则缩放因子 d(normalized_coord)/d(physical_coord)。
            若提供，返回的空间导数和 u_t 会被还原到物理坐标尺度。
        
    Returns:
        元组 (u, all_atoms, ut)，其中：
        - u: 网络输出，形状为 (batch_size,)
        - all_atoms: 形状为 (batch_size, num_atoms)，所有空间导数项
        - ut: 形状为 (batch_size, 1)，一阶时间导数
        
    Raises:
        ValueError: 如果参数无效
    """
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    if num_spatial_dims < 1:
        raise ValueError(f"num_spatial_dims must be >= 1, got {num_spatial_dims}")
    
    total_dims = num_spatial_dims + 1
    if coords.shape[1] != total_dims:
        raise ValueError(
            f"coords shape {coords.shape} does not match (batch_size, {total_dims})"
        )
    if derivative_scales is None:
        derivative_scales = torch.ones(total_dims, dtype=u_network.data_type, device=u_network.device)
    else:
        derivative_scales = derivative_scales.to(dtype=u_network.data_type, device=u_network.device)

    coords = coords.clone().detach().to(
        dtype=u_network.data_type,
        device=u_network.device,
    )
    coords.requires_grad_(True)

    u = u_network(coords).view(-1)
    
    def compute_mixed_derivative(tensor, derivative_order):
        """递归计算混合偏导数。"""
        result = tensor
        for dim, order in enumerate(derivative_order):
            for _ in range(order):
                grads = torch.autograd.grad(
                    result.sum(),
                    coords,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                result = grads[:, dim]
        return result
    
    all_derivative_orders = _generate_spatial_derivative_orders(num_spatial_dims, max_order)
    
    atoms_list = [u, u**2]
    
    for deriv_order in all_derivative_orders:
        if deriv_order == tuple([0] * (num_spatial_dims + 1)):
            continue
        
        deriv_tensor = compute_mixed_derivative(u, deriv_order)
        if derivative_scales is not None:
            scale = 1.0
            for dim, order in enumerate(deriv_order):
                scale *= (derivative_scales[dim] ** order)
            deriv_tensor = deriv_tensor * scale
        atoms_list.append(deriv_tensor)
    
    all_atoms = torch.stack(atoms_list, dim=1)
    
    ut_order = tuple([0] * num_spatial_dims) + (1,)
    ut = compute_mixed_derivative(u, ut_order)
    if derivative_scales is not None:
        ut = ut * derivative_scales[-1]
    
    return u, all_atoms, ut.view(-1, 1)
