"""
Low-Rank Tensorized 代理网络训练框架。

这个包提供了完整的神经网络代理训练、测试和评估功能。

主要模块：
- network: 网络架构定义
- train: 训练函数
- test: 测试和推理函数
- loss: 损失函数
- measure: 评估指标
- main: 命令行程序入口
"""

from .network import LowRankPDE, U_Network, Rational, Sin, evaluate_u_derivatives
from .train import (
    discover_pde,
    compute_physical_derivative_data,
    train_u_network,
    save_u_network,
    load_u_network,
)
from .test import predict
from .loss import mse_loss, regularized_loss
from .measure import (
    compute_all_metrics,
    compute_tensor_metrics,
    evaluate_discovered_terms,
    evaluate_on_dataset,
    mean_squared_error,
    mean_absolute_error,
    pde_residual_mse,
    root_mean_squared_error,
    r_squared,
    summarize_derivative_data,
)

__version__ = "0.1.0"
__author__ = "PDE Discovery Project"

__all__ = [
    "U_Network",
    "LowRankPDE",
    "Rational",
    "Sin",
    "evaluate_u_derivatives",
    "discover_pde",
    "compute_physical_derivative_data",
    "train_u_network",
    "save_u_network",
    "load_u_network",
    "evaluate_on_dataset",
    "predict",
    "mse_loss",
    "regularized_loss",
    "compute_all_metrics",
    "compute_tensor_metrics",
    "evaluate_discovered_terms",
    "mean_squared_error",
    "mean_absolute_error",
    "pde_residual_mse",
    "root_mean_squared_error",
    "r_squared",
    "summarize_derivative_data",
]
