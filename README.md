# Low-Rank Tensorized 代理网络训练框架

这是一个独立的代理网络（U_Network）训练框架，从 PDE 发现项目中提取而来。该框架可以独立使用，用于训练神经网络代理来近似时空数据。

## 功能特性

- **完整的训练流程**：从数据加载、预处理、网络训练到模型评估
- **灵活的网络架构**：支持自定义隐藏层数、神经元数、激活函数等
- **多种激活函数**：支持 Rational、Tanh、Sin 等激活函数
- **自动微分支持**：通过 PyTorch 自动微分计算各阶导数
- **详细的评估指标**：MSE、MAE、RMSE、R²、相关系数等
- **多种数据格式支持**：.npy、.csv、.txt、.mat

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
- `evaluate_u_derivatives()`: 计算函数及其导数

### train.py
- `train_u_network()`: 基础训练函数
- `train_u_network_with_validation()`: 带验证集的训练
- `normalize_coords_with_model()`: 坐标归一化
- `save_u_network()`: 保存模型
- `load_u_network()`: 加载模型

### test.py
- `evaluate_on_dataset()`: 在数据集上评估
- `predict()`: 预测
- `compute_derivatives()`: 计算导数
- `plot_1d_predictions()`: 绘制 1D 预测
- `test_on_regular_grid()`: 在规则网格上测试

### loss.py
- `mse_loss()`: 均方误差
- `weighted_mse_loss()`: 加权 MSE
- `mae_loss()`: 平均绝对误差
- `huber_loss()`: Huber 损失
- `regularized_loss()`: 带正则化的损失

### measure.py
- `compute_all_metrics()`: 计算所有评估指标
- `mean_squared_error()`: MSE
- `root_mean_squared_error()`: RMSE
- `mean_absolute_error()`: MAE
- `r_squared()`: R² 值
- `pearson_correlation()`: Pearson 相关系数
- `model_complexity_metrics()`: 模型复杂度指标

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy scipy tqdm matplotlib
```

### 2. 基本使用

```bash
# 最简单的用法
python main.py --input data.npy

# 指定输出目录和评估
python main.py --input data.npy --output-dir ./results --plot

# 使用 MATLAB 文件
python main.py --input data.mat --mat-key usol --output-dir ./results

# 自定义网络和训练参数
python main.py \
    --input data.npy \
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
from train import train_u_network
from test import evaluate_on_dataset
from measure import compute_all_metrics

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
    device="cuda",
)

# 评估
eval_result = evaluate_on_dataset(model, coords, values)
metrics = compute_all_metrics(eval_result["predictions"], values)
print(f"MSE: {metrics['mse']:.6e}")
print(f"R²: {metrics['r_squared']:.6f}")
```

### 4. 计算导数

```python
from test import compute_derivatives

# 计算 u 及其导数
u, atoms, ut = compute_derivatives(
    model,
    coords,
    num_spatial_dims=1,
    max_order=3,
)

# atoms 包含：u, u^2, u_x, u_xx, u_xxx
```

## 命令行参数详解

### 数据相关
- `--input`: 输入文件路径（必需）
- `--mat-key`: MATLAB 文件中的变量名
- `--output-dir`: 输出目录（默认：./output）
- `--noise-std`: 添加的高斯噪声标准差
- `--test-split`: 测试集比例（0-1，默认 0.2）

### 网络架构
- `--input-dim`: 输入维度，默认 2（x 和 t）
- `--num-hidden-layers`: 隐藏层数，默认 5
- `--neurons-per-layer`: 每层神经元数，默认 50
- `--activation`: 激活函数，选项：Rat, Tanh, Sin

### 训练参数
- `--epochs`: 训练轮数，默认 5000
- `--batch-size`: 批大小，默认 256
- `--learning-rate`: 学习率，默认 1e-3
- `--lambda-reg`: L2 正则化系数，默认 0
- `--device`: 计算设备，选项：cpu, cuda

### 输出选项
- `--verbose`: 详细输出
- `--save-model`: 保存模型权重
- `--plot`: 绘制结果图表

## 输出文件

运行完成后，在 `--output-dir` 中生成：

- `u_network_model.pt`: 保存的模型（如果使用 --save-model）
- `training_log.json`: 训练日志
- `train_losses.npy`: 训练损失历史
- `test_losses.npy`: 测试损失历史
- `metrics.json`: 评估指标
- `predictions_vs_targets_*.png`: 预测对比图

## 数据格式说明

### .npy 格式
需要一个 shape=(n_samples, 3) 的数组，其中：
- 第 0 列：x 坐标
- 第 1 列：t 坐标
- 第 2 列：u 值

### .csv/.txt 格式
```
x1, u1, t1
x2, u2, t2
...
```

### .mat 格式
需要包含：
- `u` 或 `usol`: 时空场数据，shape=(nx, nt) 或 (nt, nx)
- `x` 或 `xx`: x 坐标
- `t` 或 `tt`: t 坐标

## 常见问题

### Q: 如何加快训练？
- 减少 `--epochs`
- 增加 `--batch-size`
- 减少 `--num-hidden-layers` 或 `--neurons-per-layer`
- 使用 GPU：`--device cuda`

### Q: 模型性能不好怎么办？
- 调整学习率（尝试 1e-4 到 1e-1）
- 增加隐藏层数或神经元数
- 尝试不同的激活函数（Rat 通常效果较好）
- 增加训练轮数
- 尝试正则化：`--lambda-reg 1e-4`

### Q: 如何计算导数？
```python
from test import compute_derivatives
u, atoms, ut = compute_derivatives(model, coords, num_spatial_dims=1, max_order=3)
# atoms 包含所有空间导数项
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

1. **数据标准化**: 模型会自动归一化坐标到 [-1, 1] 范围
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
