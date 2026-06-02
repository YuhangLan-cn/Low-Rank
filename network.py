"""
代理网络定义模块，包括 U_Network 及其相关的激活函数。
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


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


class LowRankPDE(torch.nn.Module):
    """连续稀疏低秩张量 PDE 右端模型。

    输入为标准化后的基础原子 psi=[u, u_x, u_xx, u_xxx]，输出为 PDE 右端 F(psi)。
    常数项和线性项单独建模，非线性交互从二阶开始使用 CP 低秩形式。
    """

    def __init__(
        self,
        atom_dim: int,
        max_order: int = 3,
        rank: Optional[int] = None,
        rank_by_order: Optional[Dict[int, int]] = None,
        data_type: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        gate_init: float = 4,
    ) -> None:
        super().__init__()
        if max_order < 2:
            raise ValueError(f"max_order must be >= 2, got {max_order}")

        base_rank = 4 if rank is None else int(rank)
        if base_rank < 1:
            raise ValueError(f"rank must be >= 1, got {base_rank}")

        explicit_ranks = {}
        if rank_by_order is not None:
            for order, order_rank in rank_by_order.items():
                order = int(order)
                order_rank = int(order_rank)
                if order < 2:
                    raise ValueError(f"PDE interaction order must be >= 2, got {order}")
                if order > max_order:
                    raise ValueError(
                        f"rank_by_order contains order {order}, but max_order is {max_order}"
                    )
                if order_rank < 1:
                    raise ValueError(f"Rank must be >= 1, got {order_rank} for order {order}")
                explicit_ranks[order] = order_rank

        self.atom_dim = atom_dim
        self.max_order = max_order
        self.rank_by_order = {
            order: explicit_ranks.get(order, base_rank)
            for order in range(2, max_order + 1)
        }
        self.rank = max(self.rank_by_order.values())
        self.data_type = data_type
        self.device = device

        self.c = nn.Parameter(torch.zeros((), dtype=data_type, device=device))
        self.b = nn.Parameter(torch.zeros(atom_dim, dtype=data_type, device=device))
        self.alphas = nn.ParameterDict()
        self.gate_logits = nn.ParameterDict()
        self.factors = nn.ParameterDict()

        for order in range(2, max_order + 1):
            key = str(order)
            order_rank = self.rank_by_order[order]
            self.alphas[key] = nn.Parameter(
                0.05 * torch.randn(order_rank, dtype=data_type, device=device)
            )
            self.gate_logits[key] = nn.Parameter(
                torch.full((order_rank,), gate_init, dtype=data_type, device=device)
            )
            self.factors[key] = nn.Parameter(
                torch.randn(order_rank, order, atom_dim, dtype=data_type, device=device)
            )

        self.normalize_factors_()

    def forward(self, atoms: torch.Tensor, force_gates_one: bool = False) -> torch.Tensor:
        rhs = self.c + atoms @ self.b

        for order in range(2, self.max_order + 1):
            key = str(order)
            factors = self.factors[key]
            projections = torch.einsum("nm,rpm->nrp", atoms, factors)
            products = torch.prod(projections, dim=2)
            gates = torch.ones_like(self.alphas[key]) if force_gates_one else self.gates(order)
            rhs = rhs + products @ (gates * self.alphas[key])

        return rhs.view(-1)

    def gates(self, order: int) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits[str(order)])

    @torch.no_grad()
    def reset_gates_for_sparse_(self, target_gate: float = 0.5) -> None:
        """Reset gates before sparse fitting while preserving current PDE output.

        Dense fitting uses force_gates_one=True, so the learned nonlinear weight is
        effectively alpha. Sparse fitting uses gate * alpha. Setting each gate to
        target_gate and scaling alpha by 1 / target_gate keeps gate * alpha equal
        to the dense-stage value at the sparse-stage start.
        """
        if not 0.0 < target_gate < 1.0:
            raise ValueError(f"target_gate must be between 0 and 1, got {target_gate}")

        gate_value = torch.tensor(target_gate, dtype=self.data_type, device=self.device)
        gate_logit = torch.log(gate_value / (1.0 - gate_value))
        for order in range(2, self.max_order + 1):
            key = str(order)
            self.gate_logits[key].fill_(gate_logit)
            self.alphas[key].div_(gate_value)

    def ridge_penalty(self) -> torch.Tensor:
        penalty = torch.sum(self.b**2)
        for order in range(2, self.max_order + 1):
            key = str(order)
            penalty = penalty + torch.sum(self.alphas[key] ** 2)
            penalty = penalty + torch.sum(self.factors[key] ** 2)
        return penalty

    def sparsity_penalty(
        self,
        lambda_g: float,
        lambda_alpha: float,
        lambda_b: float,
        lambda_w: float,
        lambda_binary: float,
    ) -> torch.Tensor:
        penalty = lambda_b * torch.sum(torch.abs(self.b))
        for order in range(2, self.max_order + 1):
            key = str(order)
            gates = self.gates(order)
            penalty = penalty + lambda_g * torch.sum(torch.abs(gates))
            penalty = penalty + lambda_alpha * torch.sum(torch.abs(self.alphas[key]))
            penalty = penalty + lambda_w * torch.sum(torch.abs(self.factors[key]))
            penalty = penalty + lambda_binary * torch.sum(gates * (1.0 - gates))
        return penalty

    @torch.no_grad()
    def normalize_factors_(self, eps: float = 1e-12) -> None:
        """归一化低秩因子，并把缩放量补偿回 alpha，保持模型输出不变。"""
        for order in range(2, self.max_order + 1):
            key = str(order)
            factors = self.factors[key]
            norms = torch.linalg.norm(factors, dim=2, keepdim=True).clamp_min(eps)
            scale = torch.prod(norms.squeeze(-1), dim=1)
            factors.div_(norms)
            self.alphas[key].mul_(scale)

    def effective_rank_by_order(
        self,
        gate_threshold: float = 0.2,
        alpha_threshold: float = 1e-6,
    ) -> Dict[int, int]:
        effective_rank = {}
        for order in range(2, self.max_order + 1):
            key = str(order)
            gates = self.gates(order).detach()
            alphas = self.alphas[key].detach()
            active = (gates >= gate_threshold) & (torch.abs(alphas) >= alpha_threshold)
            effective_rank[order] = int(torch.sum(active).cpu().item())
        return effective_rank

    def rank_summary(
        self,
        gate_threshold: float = 0.2,
        alpha_threshold: float = 1e-6,
    ) -> List[Dict]:
        summary = []
        for order in range(2, self.max_order + 1):
            key = str(order)
            gates = self.gates(order).detach().cpu().numpy()
            alphas = self.alphas[key].detach().cpu().numpy()
            for rank_idx in range(len(alphas)):
                active = bool(
                    gates[rank_idx] >= gate_threshold
                    and abs(alphas[rank_idx]) >= alpha_threshold
                )
                summary.append(
                    {
                        "order": order,
                        "component": rank_idx,
                        "gate": float(gates[rank_idx]),
                        "alpha": float(alphas[rank_idx]),
                        "active": active,
                    }
                )
        return summary

    def component_summary(self) -> List[Dict]:
        summary = self.rank_summary(gate_threshold=0.5, alpha_threshold=1e-8)
        for item in summary:
            item["rank"] = item["component"]
        return summary


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用自动微分计算 U 及其关于空间坐标的混合偏导数（最高三阶）。

    坐标必须按 [x1, x2, ..., xn, t] 顺序排列，其中最后一列是时间。
    
    **重要**：这里返回的是网络输入尺度上的导数。如果 coords 是标准化坐标，
    导数也是关于标准化坐标的导数，不能直接作为物理尺度 PDE 系数重拟合的数据。
    PDE 发现请使用 train.py 中的 compute_physical_derivative_data。
    
    原子库中**仅包含空间导数**，**不包含任何时间导数项**。
    
    返回 (u, all_atoms, ut)，其中：
    - all_atoms: 形状为 (batch_size, num_atoms)，包含所有空间导数项
      - 包括 u, u_x, u_xx, u_xxx（仅空间导数，最高3阶）
      - **不包括任何时间导数项**（u_t, u_tt, u_xt等都被排除）
    - ut: 形状为 (batch_size, 1)，u_t（关于 t 的一阶偏导数，作为方程左侧）
    
    Args:
        u_network: U_Network 模型实例
        coords: 输入坐标，形状为 (batch_size, n_dim)，按 [x1, x2, ..., xn, t] 顺序排列。
            这里应传入已经由 preprocess_data 处理好的坐标。
        num_spatial_dims: 空间维度数（例如，1D 空间时为 1，2D 空间时为 2）
        max_order: 空间混合导数的最大总阶数，默认为 3
        
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
    
    atoms_list = [u]
    
    for deriv_order in all_derivative_orders:
        if deriv_order == tuple([0] * (num_spatial_dims + 1)):
            continue
        
        deriv_tensor = compute_mixed_derivative(u, deriv_order)
        atoms_list.append(deriv_tensor)
    
    all_atoms = torch.stack(atoms_list, dim=1)
    
    ut_order = tuple([0] * num_spatial_dims) + (1,)
    ut = compute_mixed_derivative(u, ut_order)
    
    return u, all_atoms, ut.view(-1, 1)
