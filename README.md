# Low-Rank Tensorized 代理网络训练框架

这是一个基于代理网络（U_Network）的 PDE 发现框架。它先训练神经网络近似时空观测数据，再通过自动微分和连续稀疏低秩张量模型发现显式 PDE。

## 功能特性

- **完整的训练流程**：从数据加载、预处理、代理网络训练到 PDE 发现
- **灵活的网络架构**：支持自定义隐藏层数、神经元数、激活函数等
- **多种激活函数**：支持 Rational、Tanh、Sin 等激活函数
- **自动微分支持**：通过 PyTorch 自动微分计算各阶导数
- **核心评估指标**：MSE、RMSE、MAE、相对误差、R²
- **低秩 PDE 发现**：支持符号展开、剪枝和固定支持集最小二乘重拟合
- **数据格式**：命令行入口目前支持 `.npy`

## 项目结构

```
Low-Rank_Tensorized/
├── main.py              # 主程序入口
├── train.py             # 训练相关函数
├── test.py              # 测试和推理函数
├── loss.py              # 损失函数定义
├── network.py           # 网络架构定义
├── measure.py           # 评估和度量函数
├── README.md            # 本文件
└── config.py            # 配置参数（可选）
```

## 各模块说明

### network.py
- `U_Network`: 主网络类，用于近似 u(x, t)
- `Rational`: 可训练的有理激活函数
- `Sin`: 正弦激活函数
- `evaluate_u_derivatives()`: 计算网络输入尺度上的函数及导数

### train.py
- `train_u_network()`: 基础训练函数
- `compute_physical_derivative_data()`: 计算并恢复到物理尺度的 PDE 原子和 `u_t`
- `discover_pde()`: 连续稀疏低秩张量 PDE 发现
- `save_u_network()`: 保存模型
- `load_u_network()`: 加载模型

### test.py
- `predict()`: 预测

### loss.py
- `mse_loss()`: 均方误差
- `regularized_loss()`: 带正则化的损失

### measure.py
- `compute_all_metrics()`: 计算核心拟合指标
- `mean_squared_error()`: MSE
- `root_mean_squared_error()`: RMSE
- `mean_absolute_error()`: MAE
- `relative_error()`: 相对误差
- `r_squared()`: R² 值

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy
```

### 2. 基本使用

```bash
# 最简单的用法
python main.py burgers_sine --input_path data
# 默认读取 data/burgers_sine.npy，输出到 ./output/burgers_sine/

# 指定输出目录和评估
python main.py burgers_sine --input_path data --output-dir ./results

# 训练代理网络后执行 PDE 发现
python main.py burgers_sine --input_path data --output-dir ./results --discover-pde

# 每一阶使用统一最大 rank
python main.py burgers_sine --input_path data --discover-pde --pde-rank 6

# 每一阶设置不同最大 rank
python main.py burgers_sine --input_path data --discover-pde --pde-ranks "2:8,3:8,4:4"

# 自定义网络和训练参数
python main.py \
    burgers_sine \
    --input_path data \
    --epochs 3000 \
    --batch-size 128 \
    --learning-rate 1e-2 \
    --num-hidden-layers 6 \
    --neurons-per-layer 64 \
    --activation Tanh \
    --output-dir ./results
```

### 3. 在 Python 中使用

```python
import torch

from train import train_u_network
from measure import evaluate_on_dataset

# 加载数据（这里假设你已经有 coords 和 values）
# coords: shape=(n_samples, 2) 的坐标 [x, t]
# values: shape=(n_samples,) 的目标值

# 训练网络
model, train_log = train_u_network(
    coords=coords,
    targets=values,
    input_dim=2,
    num_epochs=5000,
    batch_size=256,
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 评估
eval_result = evaluate_on_dataset(model, coords, values)
print(f"MSE: {eval_result['mse']:.6e}")
print(f"R²: {eval_result['r_squared']:.6f}")
```

### 4. 计算导数

```python
from train import compute_physical_derivative_data

# coords 应该是 preprocess_data 返回的标准化坐标
# preprocessing_info 也是 preprocess_data 返回的标准化参数
atoms, ut = compute_physical_derivative_data(
    model,
    coords,
    preprocessing_info,
)

# atoms 是物理尺度的 [u, u_x, u_xx, u_xxx, u_xxxx]
# ut 是物理尺度的 u_t
```

## 命令行参数详解

### 数据相关
- `data`: 输入数据集名称，不含 `.npy` 后缀；默认 `burgers_sine`
- `--input_path`: 输入 `.npy` 数据目录，默认 `data`
- `--output-dir`: 输出目录（默认：`./output/<数据集名称>/`）
- `--noise-std`: 添加的高斯噪声标准差
- `--test-split`: 测试集比例（0-1，默认 0.2）

### 网络架构
- `--num-hidden-layers`: 隐藏层数，默认 5
- `--neurons-per-layer`: 每层神经元数，默认 50
- `--activation`: 激活函数，选项：Rat, Tanh, Sin

### 训练参数
- `--epochs`: 训练轮数，默认 5000
- `--batch-size`: 批大小，默认 256
- `--learning-rate`: 学习率，默认 1e-3
- `--lambda-reg`: L2 正则化系数，默认 0

### PDE 发现参数
- `--discover-pde`: 启用连续稀疏低秩张量 PDE 发现
- `--pde-max-order`: 最大非线性交互阶数，默认 3
- `--pde-rank`: 每阶最大低秩分量数，默认 4
- `--pde-ranks`: 每阶最大 rank，例如 `"2:8,3:8,4:4"`，优先于 `--pde-rank`
- `--pde-dense-epochs`: dense fitting 轮数，默认 1000
- `--pde-sparse-epochs`: sparse fitting 轮数，默认 2000
- `--pde-learning-rate`: PDE 模型学习率，默认 1e-3
- `--true-pde-json`: 可选真实 PDE 系数字典，用于计算项识别和系数误差

### 输出选项
- `--save-model`: 保存模型权重

## 输出文件

运行完成后，在输出目录中生成。未指定 `--output-dir` 时，会自动按输入数据文件名保存到 `./output/<输入文件名>/`：

- `u_network_model.pt`: 保存的模型（如果使用 --save-model）
- `training_log.json`: 训练日志
- `metrics.json`: 评估指标
- `preprocessing.json`: 坐标和 u 的标准化参数
- `pde_discovery.json`: PDE 发现结果（如果使用 `--discover-pde`），包含 `rank_by_order_max` 和 `effective_rank_by_order`

## 数据格式说明

### .npy 格式
需要一个 shape=(n_samples, 3) 的数组，其中：
- 第 0 列：x 坐标
- 第 1 列：t 坐标
- 第 2 列：u 值

## 常见问题

### Q: 如何加快训练？
- 减少 `--epochs`
- 增加 `--batch-size`
- 减少 `--num-hidden-layers` 或 `--neurons-per-layer`
- 如果当前 PyTorch 环境能检测到 CUDA，程序会自动使用 GPU

### Q: 模型性能不好怎么办？
- 调整学习率（尝试 1e-4 到 1e-1）
- 增加隐藏层数或神经元数
- 尝试不同的激活函数（Rat 通常效果较好）
- 增加训练轮数
- 尝试正则化：`--lambda-reg 1e-4`

### Q: 如何计算导数？
```python
from train import compute_physical_derivative_data
atoms, ut = compute_physical_derivative_data(model, coords, preprocessing_info)
# atoms 是物理尺度的 [u, u_x, u_xx, u_xxx, u_xxxx]
```

### Q: 如何保存和加载模型？
```python
from train import save_u_network, load_u_network

# 保存
save_u_network(model, "my_model.pt")

# 加载
model = load_u_network("my_model.pt", input_dim=2)
```

## 性能提示

1. **数据标准化**: 程序会先划分训练/测试集，再用训练集统计量对坐标、时间和 u 做标准化
2. **批大小**: 对于小数据集（<10K），较小的批大小（32-64）通常更好
3. **学习率**: 建议从 1e-3 开始，根据损失曲线调整
4. **验证集**: 建议留出 10-20% 的数据用于验证
5. **正则化**: 对于过拟合的情况，增加 `--lambda-reg`

## 参考

这个框架基于以下核心思想：
- U_Network: PINN (Physics-Informed Neural Networks) 中的代理网络
- 自动微分: PyTorch 的自动求导功能
- 低秩结构: 用于 PDE 发现和系统识别

## 许可证

本项目为研究用途。

## 联系方式

如有问题或建议，欢迎反馈。
