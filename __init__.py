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

from .network import U_Network, Rational, Sin, evaluate_u_derivatives, u_mse_loss
from .train import (
    train_u_network,
    train_u_network_with_validation,
    save_u_network,
    load_u_network,
    normalize_coords_with_model,
)
from .test import (
    evaluate_on_dataset,
    predict,
    compute_derivatives,
    plot_1d_predictions,
    plot_predictions_vs_targets,
)
from .loss import mse_loss, weighted_mse_loss, mae_loss, huber_loss, regularized_loss
from .measure import (
    compute_all_metrics,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r_squared,
    model_complexity_metrics,
)

__version__ = "0.1.0"
__author__ = "PDE Discovery Project"

__all__ = [
    "U_Network",
    "Rational",
    "Sin",
    "evaluate_u_derivatives",
    "u_mse_loss",
    "train_u_network",
    "train_u_network_with_validation",
    "save_u_network",
    "load_u_network",
    "normalize_coords_with_model",
    "evaluate_on_dataset",
    "predict",
    "compute_derivatives",
    "plot_1d_predictions",
    "plot_predictions_vs_targets",
    "mse_loss",
    "weighted_mse_loss",
    "mae_loss",
    "huber_loss",
    "regularized_loss",
    "compute_all_metrics",
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r_squared",
    "model_complexity_metrics",
]
