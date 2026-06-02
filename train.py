"""
代理网络训练模块，包含网络训练的完整流程。
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional, Dict, List

from network import LowRankPDE, U_Network
from loss import regularized_loss
from measure import (
    compute_tensor_metrics,
    evaluate_discovered_terms,
    pde_residual_mse,
    summarize_derivative_data,
    tensor_mean_squared_error,
    zero_rhs_residual_mse,
)


ATOM_NAMES = ["u", "u_x", "u_xx", "u_xxx"]


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
        lambda_reg: 正则化系数，默认为 0（无正则化）
        
    Returns:
        (model, train_log) 元组，其中：
        - model: 训练完成的 U_Network 模型
        - train_log: 包含训练信息的字典（epoch、核心评估指标等）
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
    
    # 训练阶段接收已经由 preprocess_data 处理好的数据。
    coords = coords.astype(np.float64)
    targets = targets.astype(np.float64)
    coords_tensor = torch.from_numpy(coords).to(dtype=dtype, device=device_obj)
    targets_tensor = torch.from_numpy(targets).to(dtype=dtype, device=device_obj)
    
    # 准备测试数据张量（如果提供）
    has_test = coords_test is not None and targets_test is not None
    if has_test:
        coords_test = coords_test.astype(np.float64)
        targets_test = targets_test.astype(np.float64)
        coords_test_tensor = torch.from_numpy(coords_test).to(dtype=dtype, device=device_obj)
        targets_test_tensor = torch.from_numpy(targets_test).to(dtype=dtype, device=device_obj)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    total_samples = len(coords_tensor)
    train_log = {
        "epochs": [],
        "train_metrics": [],
        "test_metrics": [],
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        
        # 随机打乱样本
        indices = torch.randperm(total_samples, device=device_obj)
        coords_shuffled = coords_tensor[indices]
        targets_shuffled = targets_tensor[indices]
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_coords = coords_shuffled[batch_start:batch_end]
            batch_targets = targets_shuffled[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            predictions = model(batch_coords).view(-1)
            losses = regularized_loss(
                predictions,
                batch_targets,
                model=model,
                lambda_reg=lambda_reg,
            )
            loss = losses["total"]
            
            loss.backward()
            optimizer.step()

        should_report = epoch == 0 or (epoch + 1) % 100 == 0 or epoch == num_epochs - 1
        if not should_report:
            continue
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            train_pred = model(coords_tensor).view(-1)
            train_metrics = compute_tensor_metrics(train_pred, targets_tensor)
            
            if has_test:
                test_pred = model(coords_test_tensor).view(-1)
                test_metrics = compute_tensor_metrics(test_pred, targets_test_tensor)
            else:
                test_metrics = None
        
        train_log["epochs"].append(epoch + 1)
        train_log["train_metrics"].append(train_metrics)
        if has_test:
            train_log["test_metrics"].append(test_metrics)
        
        if verbose:
            msg = f"Epoch {epoch + 1:4d}/{num_epochs}: train_mse={train_metrics['mse']:.6e}"
            if has_test:
                msg += f", test_mse={test_metrics['mse']:.6e}"
            print(msg)
    
    return model, train_log


def save_u_network(
    model: U_Network,
    output_path: str,
    preprocessing_info: Optional[Dict] = None,
) -> None:
    """保存 U_Network 模型。
    
    Args:
        model: U_Network 模型
        output_path: 保存路径
        preprocessing_info: 可选，训练该模型时使用的数据标准化参数
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "num_hidden_layers": model.num_hidden_layers,
        "neurons_per_layer": model.neurons_per_layer,
        "activation_name": model.activation_name,
        "data_type": model.data_type,
    }
    if preprocessing_info is not None:
        checkpoint["preprocessing_info"] = preprocessing_info
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
    model.preprocessing_info = checkpoint.get("preprocessing_info")
    
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def _should_report(epoch: int, num_epochs: int) -> bool:
    return epoch == 0 or (epoch + 1) % 100 == 0 or epoch == num_epochs - 1


def compute_physical_derivative_data(
    model: U_Network,
    coords: np.ndarray,
    preprocessing_info: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """在标准化坐标上自动微分，并还原为物理尺度的 PDE 原子和 u_t。

    返回 psi=[u, u_x, u_xx, u_xxx] 和 u_t，均为原始物理尺度。
    第一版仅支持一维空间坐标 [x, t]。
    """
    if model.input_dim != 2:
        raise ValueError("PDE discovery currently supports only 1D space with coordinates [x, t].")

    coord_std = np.asarray(preprocessing_info["coord_std"], dtype=np.float64)
    value_mean = float(preprocessing_info["value_mean"])
    value_std = float(preprocessing_info["value_std"])
    x_std = float(coord_std[0])
    t_std = float(coord_std[-1])

    device = model.device
    dtype = model.data_type
    coords_tensor = torch.from_numpy(coords.astype(np.float64)).to(dtype=dtype, device=device)
    coords_tensor = coords_tensor.clone().detach().requires_grad_(True)

    model.eval()
    u_hat = model(coords_tensor).view(-1)
    grads = torch.autograd.grad(
        u_hat.sum(),
        coords_tensor,
        create_graph=True,
        retain_graph=True,
    )[0]
    ux_hat = grads[:, 0]
    ut_hat = grads[:, -1]

    uxx_hat = torch.autograd.grad(
        ux_hat.sum(),
        coords_tensor,
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]
    uxxx_hat = torch.autograd.grad(
        uxx_hat.sum(),
        coords_tensor,
        create_graph=False,
        retain_graph=False,
    )[0][:, 0]

    u = u_hat * value_std + value_mean
    ux = ux_hat * (value_std / x_std)
    uxx = uxx_hat * (value_std / (x_std**2))
    uxxx = uxxx_hat * (value_std / (x_std**3))
    ut = ut_hat * (value_std / t_std)

    psi = torch.stack([u, ux, uxx, uxxx], dim=1)
    return psi.detach().cpu().numpy(), ut.detach().cpu().numpy()


def standardize_atoms(
    atoms: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms = np.asarray(atoms, dtype=np.float64)
    if mean is None:
        mean = atoms.mean(axis=0)
    if std is None:
        std = atoms.std(axis=0)
    std = np.maximum(std, 1e-12)
    return (atoms - mean.reshape(1, -1)) / std.reshape(1, -1), mean, std


def standardize_values(
    values: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if mean is None:
        mean = float(np.mean(values))
    if std is None:
        std = float(np.std(values))
    std = max(float(std), 1e-12)
    return (values - mean) / std, float(mean), std


def _train_pde_stage(
    pde_model: LowRankPDE,
    atoms: torch.Tensor,
    ut: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    stage: str,
    atoms_val: Optional[torch.Tensor],
    ut_val: Optional[torch.Tensor],
    lambda_ridge: float = 0.0,
    lambda_g: float = 0.0,
    lambda_alpha: float = 0.0,
    lambda_b: float = 0.0,
    lambda_w: float = 0.0,
    lambda_binary: float = 0.0,
    gate_threshold: float = 0.2,
    alpha_threshold: float = 1e-6,
    verbose: bool = True,
) -> List[Dict]:
    optimizer = torch.optim.Adam(pde_model.parameters(), lr=learning_rate)
    logs = []

    for epoch in range(num_epochs):
        pde_model.train()
        optimizer.zero_grad()

        force_gates_one = stage == "dense"
        prediction = pde_model(atoms, force_gates_one=force_gates_one)
        fit_loss = torch.mean((prediction - ut) ** 2)

        ridge_loss = lambda_ridge * pde_model.ridge_penalty()
        if stage == "dense":
            sparsity_loss = torch.zeros((), dtype=atoms.dtype, device=atoms.device)
        else:
            ramp = min(1.0, float(epoch + 1) / max(1, num_epochs))
            binary_weight = lambda_binary if ramp > 0.5 else 0.0
            sparsity_loss = pde_model.sparsity_penalty(
                lambda_g=lambda_g * ramp,
                lambda_alpha=lambda_alpha * ramp,
                lambda_b=lambda_b * ramp,
                lambda_w=lambda_w * ramp,
                lambda_binary=binary_weight,
            )

        loss = fit_loss + ridge_loss + sparsity_loss
        loss.backward()
        optimizer.step()
        pde_model.normalize_factors_()

        if not _should_report(epoch, num_epochs):
            continue

        pde_model.eval()
        validation_residual = None
        if atoms_val is not None and ut_val is not None:
            with torch.no_grad():
                val_pred = pde_model(atoms_val, force_gates_one=force_gates_one)
                validation_residual = tensor_mean_squared_error(val_pred, ut_val)

        components = pde_model.rank_summary(
            gate_threshold=gate_threshold,
            alpha_threshold=alpha_threshold,
        )
        effective_rank = pde_model.effective_rank_by_order(
            gate_threshold=gate_threshold,
            alpha_threshold=alpha_threshold,
        )
        active_components = sum(effective_rank.values())
        gate_values = [item["gate"] for item in components]
        regularization_loss = ridge_loss + sparsity_loss
        log_item = {
            "epoch": epoch + 1,
            "loss": float(loss.detach().cpu().item()),
            "fit_loss": float(fit_loss.detach().cpu().item()),
            "ridge_loss": float(ridge_loss.detach().cpu().item()),
            "sparsity_loss": float(sparsity_loss.detach().cpu().item()),
            "regularization_loss": float(regularization_loss.detach().cpu().item()),
            "validation_residual": validation_residual,
            "active_components": active_components,
            "effective_rank_by_order": effective_rank,
            "mean_gate": float(np.mean(gate_values)) if gate_values else None,
            "max_gate": float(np.max(gate_values)) if gate_values else None,
            "min_gate": float(np.min(gate_values)) if gate_values else None,
            "components": components,
        }
        logs.append(log_item)

        if verbose:
            msg = (
                f"{stage} epoch {epoch + 1:4d}/{num_epochs}: "
                f"fit={log_item['fit_loss']:.6e}, "
                f"sparse={log_item['sparsity_loss']:.6e}, "
                f"active={active_components}"
            )
            if validation_residual is not None:
                msg += f", val={validation_residual:.6e}"
            print(msg)

    return logs


def _multiply_polynomials(
    left: Dict[Tuple[int, ...], float],
    right: Dict[Tuple[int, ...], float],
) -> Dict[Tuple[int, ...], float]:
    result = defaultdict(float)
    for left_key, left_coeff in left.items():
        for right_key, right_coeff in right.items():
            key = tuple(sorted(left_key + right_key))
            result[key] += left_coeff * right_coeff
    return dict(result)


def _standardized_key_to_physical_terms(
    key: Tuple[int, ...],
    atom_mean: np.ndarray,
    atom_std: np.ndarray,
) -> Dict[Tuple[int, ...], float]:
    polynomial = {(): 1.0}
    for atom_idx in key:
        factor = {
            (): -float(atom_mean[atom_idx] / atom_std[atom_idx]),
            (atom_idx,): float(1.0 / atom_std[atom_idx]),
        }
        polynomial = _multiply_polynomials(polynomial, factor)
    return polynomial


def _term_name(key: Tuple[int, ...], atom_names: List[str]) -> str:
    if len(key) == 0:
        return "1"

    powers = []
    for atom_idx in sorted(set(key)):
        count = key.count(atom_idx)
        name = atom_names[atom_idx]
        powers.append(name if count == 1 else f"{name}^{count}")
    return "*".join(powers)


def _format_pde(coefficients: Dict[Tuple[int, ...], float], atom_names: List[str]) -> str:
    if not coefficients:
        return "u_t = 0"

    pieces = []
    for key, coeff in sorted(
        coefficients.items(),
        key=lambda item: (-abs(item[1]), len(item[0]), item[0]),
    ):
        sign = "-" if coeff < 0 else "+"
        term = _term_name(key, atom_names)
        magnitude = abs(coeff)
        body = f"{magnitude:.6g}" if term == "1" else f"{magnitude:.6g} * {term}"
        pieces.append((sign, body))

    first_sign, first_body = pieces[0]
    rhs = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in pieces[1:]:
        rhs += f" {sign} {body}"
    return f"u_t = {rhs}"


def expand_low_rank_model(
    pde_model: LowRankPDE,
    atom_mean: np.ndarray,
    atom_std: np.ndarray,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    gate_threshold: float = 0.2,
    alpha_threshold: float = 1e-6,
    b_threshold: float = 1e-8,
    w_threshold: float = 1e-3,
    term_threshold: float = 1e-8,
) -> Tuple[Dict[Tuple[int, ...], float], List[Dict], List[Dict]]:
    """展开低秩模型，返回原始尺度单项式候选系数、保留分量和展开项。"""
    atom_mean = np.asarray(atom_mean, dtype=np.float64)
    atom_std = np.asarray(atom_std, dtype=np.float64)
    target_mean = float(target_mean)
    target_std = float(target_std)
    physical_coefficients = defaultdict(float)
    retained_components = []

    c = float(pde_model.c.detach().cpu().item())
    if abs(c) >= term_threshold:
        physical_coefficients[()] += c

    b = pde_model.b.detach().cpu().numpy()
    for atom_idx, coeff in enumerate(b):
        if abs(coeff) < b_threshold:
            continue
        for phys_key, phys_coeff in _standardized_key_to_physical_terms(
            (atom_idx,), atom_mean, atom_std
        ).items():
            physical_coefficients[phys_key] += float(coeff) * phys_coeff

    for order in range(2, pde_model.max_order + 1):
        key = str(order)
        gates = pde_model.gates(order).detach().cpu().numpy()
        alphas = pde_model.alphas[key].detach().cpu().numpy()
        factors = pde_model.factors[key].detach().cpu().numpy()

        for rank_idx in range(len(alphas)):
            gate = float(gates[rank_idx])
            alpha = float(alphas[rank_idx])
            if gate < gate_threshold or abs(alpha) < alpha_threshold:
                continue

            pruned_factors = factors[rank_idx].copy()
            pruned_factors[np.abs(pruned_factors) < w_threshold] = 0.0
            if np.any(np.all(pruned_factors == 0.0, axis=1)):
                continue

            standardized_poly = {(): gate * alpha}
            for factor_idx in range(order):
                factor_terms = {
                    (atom_idx,): float(weight)
                    for atom_idx, weight in enumerate(pruned_factors[factor_idx])
                    if abs(weight) >= w_threshold
                }
                standardized_poly = _multiply_polynomials(standardized_poly, factor_terms)

            for std_key, std_coeff in standardized_poly.items():
                for phys_key, phys_coeff in _standardized_key_to_physical_terms(
                    std_key, atom_mean, atom_std
                ).items():
                    physical_coefficients[phys_key] += std_coeff * phys_coeff

            retained_components.append(
                {
                    "order": order,
                    "rank": rank_idx,
                    "gate": gate,
                    "alpha": alpha,
                    "factors": pruned_factors.tolist(),
                }
            )

    for key in list(physical_coefficients.keys()):
        physical_coefficients[key] *= target_std
    if abs(target_mean) >= term_threshold or () in physical_coefficients:
        physical_coefficients[()] += target_mean

    pruned = {
        key: float(coeff)
        for key, coeff in physical_coefficients.items()
        if abs(coeff) >= term_threshold
    }
    expanded_terms = [
        {
            "term": _term_name(key, ATOM_NAMES),
            "coefficient_estimate": coeff,
        }
        for key, coeff in sorted(pruned.items(), key=lambda item: (len(item[0]), item[0]))
    ]
    return pruned, retained_components, expanded_terms


def build_candidate_matrix(atoms: np.ndarray, support: List[Tuple[int, ...]]) -> np.ndarray:
    columns = []
    for key in support:
        if len(key) == 0:
            columns.append(np.ones(atoms.shape[0], dtype=np.float64))
        else:
            column = np.ones(atoms.shape[0], dtype=np.float64)
            for atom_idx in key:
                column = column * atoms[:, atom_idx]
            columns.append(column)
    if not columns:
        return np.zeros((atoms.shape[0], 0), dtype=np.float64)
    return np.column_stack(columns)


def refit_pde_coefficients(
    atoms: np.ndarray,
    ut: np.ndarray,
    support: List[Tuple[int, ...]],
    coefficient_threshold: float = 1e-8,
) -> Tuple[Dict[Tuple[int, ...], float], float]:
    if not support:
        return {}, zero_rhs_residual_mse(ut)

    theta = build_candidate_matrix(atoms, support)
    coeffs, *_ = np.linalg.lstsq(theta, ut.reshape(-1), rcond=None)
    kept = [idx for idx, coeff in enumerate(coeffs) if abs(coeff) >= coefficient_threshold]
    if not kept:
        return {}, zero_rhs_residual_mse(ut)

    support = [support[idx] for idx in kept]
    theta = theta[:, kept]
    coeffs, *_ = np.linalg.lstsq(theta, ut.reshape(-1), rcond=None)
    final_kept = [idx for idx, coeff in enumerate(coeffs) if abs(coeff) >= coefficient_threshold]
    if not final_kept:
        return {}, zero_rhs_residual_mse(ut)

    support = [support[idx] for idx in final_kept]
    theta = theta[:, final_kept]
    coeffs, *_ = np.linalg.lstsq(theta, ut.reshape(-1), rcond=None)
    predictions = theta @ coeffs
    residual = pde_residual_mse(ut, predictions)
    coefficients = {key: float(coeff) for key, coeff in zip(support, coeffs)}
    return coefficients, residual


def discover_pde(
    model: U_Network,
    coords: np.ndarray,
    preprocessing_info: Dict,
    coords_val: Optional[np.ndarray] = None,
    max_order: int = 3,
    rank: Optional[int] = 4,
    rank_by_order: Optional[Dict[int, int]] = None,
    dense_epochs: int = 1000,
    sparse_epochs: int = 2000,
    learning_rate: float = 1e-3,
    lambda_ridge: float = 1e-6,
    lambda_g: float = 1e-4,
    lambda_alpha: float = 1e-5,
    lambda_b: float = 1e-5,
    lambda_w: float = 1e-5,
    lambda_binary: float = 1e-4,
    sparse_gate_init: float = 0.5,
    gate_threshold: float = 0.2,
    alpha_threshold: float = 1e-6,
    b_threshold: float = 1e-8,
    w_threshold: float = 1e-3,
    term_threshold: float = 1e-8,
    coefficient_threshold: float = 1e-8,
    true_coefficients: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Dict:
    """完整 PDE 发现流程：导数、低秩训练、展开、剪枝和最小二乘重拟合。"""
    atoms_phys, ut_phys = compute_physical_derivative_data(model, coords, preprocessing_info)
    atoms_std, atom_mean, atom_std = standardize_atoms(atoms_phys)
    ut_std_target, ut_mean, ut_std = standardize_values(ut_phys)

    atoms_val_std = None
    ut_val_std_target = None
    atoms_val_phys = None
    ut_val_phys = None
    if coords_val is not None:
        atoms_val_phys, ut_val_phys = compute_physical_derivative_data(
            model, coords_val, preprocessing_info
        )
        atoms_val_std, _, _ = standardize_atoms(atoms_val_phys, atom_mean, atom_std)
        ut_val_std_target, _, _ = standardize_values(ut_val_phys, ut_mean, ut_std)

    device = model.device
    dtype = model.data_type
    atoms_tensor = torch.from_numpy(atoms_std).to(dtype=dtype, device=device)
    ut_tensor = torch.from_numpy(ut_std_target).to(dtype=dtype, device=device)
    atoms_val_tensor = None
    ut_val_tensor = None
    if atoms_val_std is not None and ut_val_std_target is not None:
        atoms_val_tensor = torch.from_numpy(atoms_val_std).to(dtype=dtype, device=device)
        ut_val_tensor = torch.from_numpy(ut_val_std_target).to(dtype=dtype, device=device)

    pde_model = LowRankPDE(
        atom_dim=len(ATOM_NAMES),
        max_order=max_order,
        rank=rank,
        rank_by_order=rank_by_order,
        data_type=dtype,
        device=device,
    )

    if verbose:
        print("\nDense fitting low-rank PDE model")
    dense_log = _train_pde_stage(
        pde_model=pde_model,
        atoms=atoms_tensor,
        ut=ut_tensor,
        num_epochs=dense_epochs,
        learning_rate=learning_rate,
        stage="dense",
        atoms_val=atoms_val_tensor,
        ut_val=ut_val_tensor,
        lambda_ridge=lambda_ridge,
        gate_threshold=gate_threshold,
        alpha_threshold=alpha_threshold,
        verbose=verbose,
    )

    if verbose:
        print("\nSparse fitting low-rank PDE model")
        print(f"Resetting gates to {sparse_gate_init:.6g} before sparse fitting")
    pde_model.reset_gates_for_sparse_(target_gate=sparse_gate_init)
    sparse_log = _train_pde_stage(
        pde_model=pde_model,
        atoms=atoms_tensor,
        ut=ut_tensor,
        num_epochs=sparse_epochs,
        learning_rate=learning_rate,
        stage="sparse",
        atoms_val=atoms_val_tensor,
        ut_val=ut_val_tensor,
        lambda_g=lambda_g,
        lambda_alpha=lambda_alpha,
        lambda_b=lambda_b,
        lambda_w=lambda_w,
        lambda_binary=lambda_binary,
        gate_threshold=gate_threshold,
        alpha_threshold=alpha_threshold,
        verbose=verbose,
    )

    candidate_coefficients, retained_components, expanded_terms = expand_low_rank_model(
        pde_model=pde_model,
        atom_mean=atom_mean,
        atom_std=atom_std,
        target_mean=ut_mean,
        target_std=ut_std,
        gate_threshold=gate_threshold,
        alpha_threshold=alpha_threshold,
        b_threshold=b_threshold,
        w_threshold=w_threshold,
        term_threshold=term_threshold,
    )
    support = sorted(candidate_coefficients.keys(), key=lambda key: (len(key), key))
    coefficients, residual_mse = refit_pde_coefficients(
        atoms=atoms_phys,
        ut=ut_phys,
        support=support,
        coefficient_threshold=coefficient_threshold,
    )

    validation_residual = None
    if atoms_val_phys is not None and ut_val_phys is not None and coefficients:
        val_support = sorted(coefficients.keys(), key=lambda key: (len(key), key))
        theta_val = build_candidate_matrix(atoms_val_phys, val_support)
        val_coeffs = np.asarray([coefficients[key] for key in val_support])
        validation_residual = pde_residual_mse(ut_val_phys, theta_val @ val_coeffs)

    pde_string = _format_pde(coefficients, ATOM_NAMES)
    coefficients_by_name = {
        _term_name(key, ATOM_NAMES): float(coeff)
        for key, coeff in sorted(coefficients.items(), key=lambda item: (len(item[0]), item[0]))
    }
    effective_rank = pde_model.effective_rank_by_order(
        gate_threshold=gate_threshold,
        alpha_threshold=alpha_threshold,
    )
    rank_summary = pde_model.rank_summary(
        gate_threshold=gate_threshold,
        alpha_threshold=alpha_threshold,
    )
    result = {
        "pde_string": pde_string,
        "atom_names": ATOM_NAMES,
        "rank_by_order_max": pde_model.rank_by_order,
        "effective_rank_by_order": effective_rank,
        "rank_summary": rank_summary,
        "sparse_gate_init": sparse_gate_init,
        "atom_statistics": {
            "mean": atom_mean.tolist(),
            "std": atom_std.tolist(),
        },
        "target_statistics": {
            "name": "u_t",
            "mean": ut_mean,
            "std": ut_std,
            "standardization": "z_score",
            "fit_on": "pde_training_points",
        },
        "derivative_summary": summarize_derivative_data(ut_phys),
        "training_log": {
            "dense": dense_log,
            "sparse": sparse_log,
        },
        "retained_components": retained_components,
        "expanded_terms": expanded_terms,
        "candidate_support": [_term_name(key, ATOM_NAMES) for key in support],
        "final_support": list(coefficients_by_name.keys()),
        "coefficients": coefficients_by_name,
        "residual_mse": residual_mse,
        "validation_residual_mse": validation_residual,
    }
    if true_coefficients is not None:
        result["true_pde_metrics"] = evaluate_discovered_terms(
            coefficients_by_name,
            true_coefficients,
        )
    return result
