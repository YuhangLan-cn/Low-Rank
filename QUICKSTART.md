# 🚀 快速入门指南

## 项目已创建！

你要求的 **Low-Rank Tensorized** 项目已成功创建，包含 10 个文件，所有代理网络训练代码都已提取并整理好。

## 📁 完整文件列表

```
Low-Rank_Tensorized/
├── 📄 README.md                 ← 完整文档（必读！）
├── 📄 INSTALLATION.md           ← 安装和创建说明
├── 🎯 main.py                   ← 命令行程序入口（推荐先用这个）
├── 🔧 train.py                  ← 训练代码
├── ✅ test.py                   ← 测试和推理代码
├── 📊 measure.py                ← 评估指标
├── 🧠 network.py                ← 网络定义
├── 💔 loss.py                   ← 损失函数
├── ⚙️  config.py                ← 配置管理
├── 📚 examples.py               ← 5 个完整使用示例
└── 📦 __init__.py               ← 包初始化（可作为 Python 包导入）
```

## ⚡ 最快上手（3 种方式）

### 方式 1️⃣：命令行使用（最简单）

```bash
cd /Users/yuhanglan/Desktop/大学/博士/PINN/PII

# 最简单的用法（自动读取数据、训练、评估）
python Low-Rank_Tensorized/main.py --input pde_discovery/data/burgers_sine.mat

# 完整的用法（指定所有参数）
python Low-Rank_Tensorized/main.py \
    --input pde_discovery/data/burgers_sine.mat \
    --epochs 3000 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --num-hidden-layers 6 \
    --neurons-per-layer 64 \
    --output-dir ./results \
    --save-model \
    --plot
```

### 方式 2️⃣：运行示例脚本

```bash
cd /Users/yuhanglan/Desktop/大学/博士/PINN/PII/Low-Rank_Tensorized
python examples.py  # 运行所有示例（演示生成合成数据、训练、评估）
```

### 方式 3️⃣：在 Python 代码中使用

```python
# 导入
from Low_Rank_Tensorized.train import train_u_network
from Low_Rank_Tensorized.test import evaluate_on_dataset
from Low_Rank_Tensorized.measure import compute_all_metrics
import numpy as np

# 加载数据
coords = np.random.randn(1000, 2)  # [x, t]
values = np.sin(coords[:, 0]) * np.exp(-coords[:, 1])

# 训练网络
model, log = train_u_network(
    coords=coords,
    targets=values,
    input_dim=2,
    num_epochs=5000,
    batch_size=256,
    learning_rate=1e-3,
    device="cuda",  # 使用 GPU 加速
    verbose=True,
)

# 评估
eval_result = evaluate_on_dataset(model, coords, values)
metrics = compute_all_metrics(eval_result["predictions"], values)
print(f"RMSE: {metrics['rmse']:.6e}, R²: {metrics['r_squared']:.4f}")
```

## 📋 命令行参数速查

### 数据参数
```bash
--input data.npy              # 输入文件（必需）
--mat-key usol                # MATLAB 文件中的变量名
--output-dir ./results        # 输出目录
--test-split 0.2              # 测试集比例（0-1）
--noise-std 0.01              # 添加的噪声标准差
```

### 网络架构
```bash
--input-dim 2                 # 输入维度（x + t）
--num-hidden-layers 5         # 隐藏层数
--neurons-per-layer 50        # 每层神经元数
--activation Rat              # 激活函数（Rat/Tanh/Sin）
```

### 训练参数
```bash
--epochs 5000                 # 训练轮数
--batch-size 256              # 批大小
--learning-rate 1e-3          # 学习率
--lambda-reg 0.0              # L2 正则化系数
--device cpu                  # 计算设备（cpu/cuda）
```

### 输出选项
```bash
--save-model                  # 保存模型
--plot                        # 绘制结果图表
--verbose                     # 详细输出
```

## 🎯 典型工作流

### 1. 训练一个代理网络
```bash
python main.py --input data.npy --epochs 5000 --output-dir ./my_results
```

### 2. 查看结果
```bash
cat my_results/metrics.json         # 查看评估指标
cat my_results/training_log.json    # 查看训练过程
```

### 3. 加载已训练的模型
```python
from Low_Rank_Tensorized.train import load_u_network

model = load_u_network("my_results/u_network_model.pt")
```

### 4. 计算导数（用于 PDE 发现）
```python
from Low_Rank_Tensorized.test import compute_derivatives

u, atoms, ut = compute_derivatives(
    model,
    coords,
    num_spatial_dims=1,
    max_order=3,
)
# atoms 包含：u, u², u_x, u_xx, u_xxx
```

## 📚 各模块功能

| 模块 | 功能 | 关键函数 |
|------|------|---------|
| **network.py** | 网络架构 | `U_Network`, `evaluate_u_derivatives` |
| **train.py** | 训练流程 | `train_u_network`, `save_u_network`, `load_u_network` |
| **test.py** | 测试推理 | `evaluate_on_dataset`, `compute_derivatives`, `predict` |
| **loss.py** | 损失函数 | `mse_loss`, `regularized_loss`, `huber_loss` |
| **measure.py** | 评估指标 | `compute_all_metrics`, `r_squared`, `mean_squared_error` |
| **config.py** | 配置管理 | `TrainingConfig`, 预设配置 |
| **main.py** | 命令行程序 | 完整流程自动化 |
| **examples.py** | 示例代码 | 5 个完整示例 |

## 💡 常见问题速答

**Q: 如何加快训练？**
- 用 GPU：`--device cuda`
- 减少轮数：`--epochs 1000`
- 增加批大小：`--batch-size 512`

**Q: 模型性能不好？**
- 调整学习率（试试 1e-4 到 1e-1）
- 增加隐藏层：`--num-hidden-layers 8`
- 增加训练轮数

**Q: 如何计算导数？**
```python
u, atoms, ut = compute_derivatives(model, coords)
# atoms[:,2] 是 u_x, atoms[:,3] 是 u_xx 等
```

**Q: 支持什么数据格式？**
- .npy（NumPy）✅
- .csv/.txt（文本）✅
- .mat（MATLAB）✅

**Q: 如何在自己的项目中使用？**
```python
from Low_Rank_Tensorized import train_u_network, compute_all_metrics
```

## 🚀 下一步

1. **立即试用**
   ```bash
   python main.py --input pde_discovery/data/burgers_sine.mat --output-dir ./test_results
   ```

2. **阅读完整文档**
   ```bash
   cat README.md
   ```

3. **运行示例**
   ```bash
   python examples.py
   ```

4. **查看代码**
   - 打开 `main.py` 了解完整流程
   - 打开 `train.py` 了解训练细节
   - 打开 `examples.py` 了解使用方法

## 📝 项目特点

✅ **完全独立** - 不依赖 pde_discovery 的其他部分
✅ **即插即用** - 可直接运行，无需额外配置
✅ **文档齐全** - README + INSTALLATION + 代码注释
✅ **示例丰富** - 5 个完整使用示例
✅ **功能完整** - 训练 + 测试 + 评估 + 绘图
✅ **灵活配置** - 命令行参数 + Python API

---

**项目已完全准备好！🎉**

开始使用：
```bash
cd /Users/yuhanglan/Desktop/大学/博士/PINN/PII
python Low-Rank_Tensorized/main.py --help
```
