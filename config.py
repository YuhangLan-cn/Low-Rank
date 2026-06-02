"""
配置文件示例和默认参数。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置参数。"""
    # 数据参数
    input_file: str = "data.npy"  # 只支持 .npy 文件
    output_dir: str = "./output"
    noise_std: float = 0.0
    test_split: float = 0.2
    
    # 网络架构（input_dim 将从数据自动检测）
    num_hidden_layers: int = 5
    neurons_per_layer: int = 50
    activation_function: str = "Rat"  # "Rat", "Tanh", "Sin"
    
    # 训练参数
    num_epochs: int = 5000
    batch_size: int = 256
    learning_rate: float = 1e-3
    lambda_reg: float = 0.0
    device: str = "cpu"  # "cpu" or "cuda"
    
    # 输出参数
    save_model: bool = True


# 预设配置
PRESETS = {
    "fast_cpu": TrainingConfig(
        num_epochs=1000,
        batch_size=512,
        learning_rate=1e-2,
        num_hidden_layers=3,
        neurons_per_layer=32,
    ),
    "medium": TrainingConfig(
        num_epochs=5000,
        batch_size=256,
        learning_rate=1e-3,
        num_hidden_layers=5,
        neurons_per_layer=50,
    ),
    "high_accuracy": TrainingConfig(
        num_epochs=10000,
        batch_size=128,
        learning_rate=5e-4,
        num_hidden_layers=6,
        neurons_per_layer=64,
        lambda_reg=1e-5,
    ),
    "gpu": TrainingConfig(
        num_epochs=5000,
        batch_size=1024,
        learning_rate=1e-3,
        num_hidden_layers=5,
        neurons_per_layer=50,
        device="cuda",
    ),
}


def get_config(preset: Optional[str] = None, **kwargs) -> TrainingConfig:
    """
    获取配置。
    
    Args:
        preset: 预设名称（"fast_cpu", "medium", "high_accuracy", "gpu"）
        **kwargs: 要覆盖的参数
        
    Returns:
        TrainingConfig 对象
    """
    if preset is not None and preset in PRESETS:
        config = PRESETS[preset]
    else:
        config = TrainingConfig()
    
    # 覆盖参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown parameter: {key}")
    
    return config


def print_config(config: TrainingConfig) -> None:
    """打印配置信息。"""
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    
    print("\nData Parameters:")
    print(f"  Input file: {config.input_file}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Noise std: {config.noise_std}")
    print(f"  Test split: {config.test_split}")
    
    print("\nNetwork Architecture:")
    print("  Input dimension: detected from data")
    print(f"  Hidden layers: {config.num_hidden_layers}")
    print(f"  Neurons per layer: {config.neurons_per_layer}")
    print(f"  Activation: {config.activation_function}")
    
    print("\nTraining Parameters:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Lambda (L2 reg): {config.lambda_reg}")
    print(f"  Device: {config.device}")
    
    print("\nOutput:")
    print(f"  Save model: {config.save_model}")
    print("="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    # 方式 1: 使用预设
    config1 = get_config("medium")
    print_config(config1)
    
    # 方式 2: 自定义参数
    config2 = get_config(
        input_file="data.npy",
        num_epochs=3000,
        learning_rate=1e-2,
        device="cuda"
    )
    print_config(config2)
    
    # 方式 3: 基于预设修改
    config3 = get_config("high_accuracy", batch_size=512)
    print_config(config3)
