# Low-Rank Tensorized 项目创建总结

## 项目完成情况

✅ 已成功创建 **Low-Rank_Tensorized** 项目，完整提取了代理网络训练的所有核心代码。

## 目录结构

```
/Users/yuhanglan/Desktop/大学/博士/PINN/PII/Low-Rank_Tensorized/
├── __init__.py              # 包初始化文件，可作为 Python 包导入
├── main.py                  # 🎯 程序入口（命令行程序）
├── train.py                 # 训练函数集合
├── test.py                  # 测试和推理函数
├── loss.py                  # 损失函数定义
├── network.py               # 网络架构定义
├── measure.py               # 评估指标函数
├── config.py                # 配置文件和预设
├── examples.py              # 使用示例脚本
└── README.md                # 详细使用文档
```

## 各文件详细说明

### 1. **network.py** - 网络架构定义
提供了完整的神经网络定义：
- `U_Network`: 主网络类，用于近似 u(x,t)
- `Rational`: 可训练的有理激活函数
- `Sin`: 正弦激活函数
- `evaluate_u_derivatives()`: 通过自动微分计算多阶导数
- 导数生成函数（内部）

### 2. **train.py** - 训练模块 ⭐
核心训练函数和工具：
- `train_u_network()`: 基础训练函数，支持验证集
- `train_u_network_with_validation()`: 带早停机制的训练
- `_coord_normalization_from_bounds()`: 坐标自动归一化
- `normalize_coords_with_model()`: 使用模型保存的参数进行归一化
- `save_u_network()`: 保存模型（包含所有参数）
- `load_u_network()`: 加载模型

### 3. **test.py** - 测试和推理
提供了完整的评估工具：
- `evaluate_on_dataset()`: 在数据集上计算 MSE、MAE、RMSE、相对误差
- `predict()`: 快速预测接口
- `compute_derivatives()`: 计算自动微分导数
- 绘图函数：`plot_1d_predictions()`、`plot_predictions_vs_targets()`
- `test_on_regular_grid()`: 在规则网格上生成预测

### 4. **loss.py** - 损失函数
多种损失函数实现：
- `mse_loss()`: 均方误差
- `weighted_mse_loss()`: 加权 MSE
- `mae_loss()`: 平均绝对误差
- `huber_loss()`: Huber 损失（对异常值鲁棒）
- `regularized_loss()`: 带 L1/L2 正则化的损失

### 5. **measure.py** - 评估指标
完整的性能评估套件：
- 基本指标：MSE、RMSE、MAE、MAPE
- 统计指标：R²、Pearson/Spearman 相关系数
- 导数误差计算
- 模型复杂度指标（参数数、FLOPs）
- `compute_all_metrics()`: 一次性计算所有指标

### 6. **main.py** - 程序入口 🎯
完整的命令行程序：
- 支持多种数据格式：.npy、.csv、.txt、.mat
- 自动数据预处理和验证
- 完整的训练流程管理
- 自动评估和结果保存

### 7. **config.py** - 配置管理
灵活的配置系统：
- `TrainingConfig`: 数据类配置参数
- 4 个预设配置：fast_cpu, medium, high_accuracy, gpu
- `get_config()`: 灵活的配置获取函数

### 8. **examples.py** - 使用示例
5 个完整的示例脚本：
1. 基本训练流程
2. 自定义网络架构
3. 导数计算
4. 正则化训练
5. 模型保存/加载

### 9. **README.md** - 完整文档
包含：
- 功能特性说明
- 快速开始指南
- 详细的命令行参数说明
- 使用示例
- 常见问题解答

### 10. **__init__.py** - 包定义
使得整个目录可以作为 Python 包导入

## 快速使用方法

### 方法 1: 命令行使用（最简单）

```bash
# 基本用法
cd /Users/yuhanglan/Desktop/大学/博士/PINN/PII
python Low-Rank_Tensorized/main.py --input pde_discovery/data/burgers_sine.mat

# 完整用法
python Low-Rank_Tensorized/main.py \
    --input data.npy \
    --epochs 5000 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --num-hidden-layers 6 \
    --neurons-per-layer 64 \
    --output-dir ./results \
    --save-model \
    --plot
```

### 方法 2: Python 代码使用

```python
from Low_Rank_Tensorized.train import train_u_network
from Low_Rank_Tensorized.test import evaluate_on_dataset
from Low_Rank_Tensorized.measure import compute_all_metrics

# 加载数据
import numpy as np
coords = np.random.randn(1000, 2)  # [x, t]
values = np.sin(coords[:, 0]) * np.exp(-coords[:, 1])

# 训练
model, log = train_u_network(
    coords=coords,
    targets=values,
    input_dim=2,
    num_epochs=3000,
    device="cuda",
)

# 评估
eval_result = evaluate_on_dataset(model, coords, values)
metrics = compute_all_metrics(eval_result["predictions"], values)
print(f"RMSE: {metrics['rmse']:.6e}")
```

### 方法 3: 运行示例

```bash
cd Low-Rank_Tensorized
python examples.py
```

## 关键特性

✅ **完整独立** - 可以直接使用，无需依赖原项目
✅ **模块化设计** - 各模块相互独立，易于扩展
✅ **丰富的功能** - 训练、测试、评估、绘图一应俱全
✅ **灵活的配置** - 命令行参数或配置文件控制
✅ **自动微分** - PyTorch 自动求导计算各阶导数
✅ **多格式支持** - 支持 .npy、.csv、.txt、.mat
✅ **详细文档** - README + 使用示例 + 代码注释

## 与原项目的关系

从原项目 (pde_discovery) 中提取了：
- ✅ U_Network.py → network.py（网络定义）
- ✅ surrogate_autodiff.py → train.py（训练函数）
- ✅ 评估逻辑 → test.py + measure.py
- ✅ 新增：main.py（命令行程序）、loss.py、config.py

**完全独立**，可以直接替换或与原项目配合使用。

## 使用场景

1. **独立训练代理网络** - 不需要 PDE 发现，只训练神经网络近似
2. **参数优化** - 实验不同网络架构和训练参数
3. **导数计算** - 通过自动微分获得精确导数
4. **模型集成** - 将代理网络集成到其他系统中
5. **研究和教学** - 完整的代码示例和文档

## 下一步建议

1. **运行命令行程序**
   ```bash
   python main.py --input data.npy --output-dir ./my_results
   ```

2. **查看结果**
   - 检查 `./my_results/metrics.json` 查看评估指标
   - 查看 `training_log.json` 查看训练过程

3. **尝试示例脚本**
   ```bash
   python examples.py
   ```

4. **在自己的代码中导入使用**
   ```python
   from Low_Rank_Tensorized import train_u_network, compute_all_metrics
   ```

## 注意事项

- 建议使用 GPU 加速：`--device cuda`
- 对于大数据集，调整 `--batch-size` 提高效率
- 数据会自动归一化，坐标范围自动转换到 [-1, 1]
- 模型权重保存在 `.pt` 文件中，可跨平台加载

---

**项目创建完成！🎉**
所有文件已保存在：`/Users/yuhanglan/Desktop/大学/博士/PINN/PII/Low-Rank_Tensorized/`

可以立即开始使用！
