# PCB Decision Transformer Core

这是为PCB布局优化专门定制的Decision Transformer (DT)算法核心实现。状态表示从向量改为三通道掩码图像，能够处理空间布局约束和布线长度优化。

## 最终文件结构

```
DT/
├── __init__.py              # 主包初始化
├── README.md                # 项目说明
├── example.py               # 使用示例
└── models/                  # 模型模块
    ├── __init__.py
    ├── base_model.py        # 基础轨迹模型
    ├── dt_model.py          # PCB Decision Transformer 模型
    ├── gpt2_model.py        # 修改的 GPT-2 模型
    └── pcb_state_encoder.py # PCB状态编码器 (CNN/ViT)
```

## 核心组件

- `DecisionTransformer`: 主要的 PCB DT 模型类
- `PCBStateEncoder`: CNN-based 状态编码器
- `PCBStateEncoderViT`: ViT-based 状态编码器
- `TrajectoryModel`: 基础轨迹模型抽象类
- `GPT2Model`: 修改的 GPT-2 模型（移除了位置嵌入）

## PCB状态表示

状态使用四通道掩码图像表示：
- **Channel 0**: View Mask (已被占据的网格)
- **Channel 1**: Position Mask (合法放置位置 - 基础)
- **Channel 2**: Wire Mask (HPWL增量热力图)
- **Channel 3**: Position Mask (专门通道 - 动态上下文相关)

其中Channel 3的Position Mask是高度动态的，由环境根据当前待放置元件的具体类型、尺寸和候选旋转角度实时计算。## PCB动作空间

**结构化离散动作空间**: `(x, y, rotation)`
- **X坐标**: 0到N-1的离散网格坐标
- **Y坐标**: 0到N-1的离散网格坐标  
- **旋转角度**: 4个离散类别 (0°、90°、180°、270°)

**预测头**: 三个独立的分类器
- `predict_x`: 输出N个logits预测x坐标
- `predict_y`: 输出N个logits预测y坐标
- `predict_rotation`: 输出4个logits预测旋转角度

**训练损失**: 交叉熵损失 (Cross-Entropy Loss) 而非MSE

## 使用方法

```python
import torch
from models import DecisionTransformer

# 初始化PCB DT模型 - 动作空间现在是固定的 (x, y, rotation)
model = DecisionTransformer(
    grid_size=16,      # PCB网格大小 (N x N)
    hidden_size=128,   # 隐藏层大小
    max_length=20,     # 最大序列长度 K
    max_ep_len=1000,   # 最大episode长度
    n_layer=3,         # Transformer层数
    n_head=1,          # 注意力头数
    n_inner=4*128,     # 前馈网络维度
    use_vit=False,     # False=CNN编码器, True=ViT编码器
)

# 前向传播 - 使用四通道图像状态
states = torch.randn(1, 20, 4, 16, 16)  # (batch_size, seq_len, 4, N, N)
# 离散动作: [x, y, rotation] - x,y为0-N-1整数, rotation为0-3整数
actions = torch.randint(0, 16, (1, 20, 2))  # x, y坐标
rotations = torch.randint(0, 4, (1, 20, 1))  # 旋转角度
actions = torch.cat([actions, rotations], dim=-1).float()
rewards = torch.randn(1, 20, 1)
returns_to_go = torch.randn(1, 20, 1)
timesteps = torch.randint(0, 1000, (1, 20))

state_preds, action_preds, return_preds = model(
    states, actions, rewards, returns_to_go, timesteps
)
# action_preds: (x_logits, y_logits, rot_logits) - 三个logits元组

# 获取动作 (用于推理) - 返回离散动作元组
single_states = states.squeeze(0)  # (seq_len, 4, N, N)
single_actions = actions.squeeze(0)  # (seq_len, 3)
single_rewards = rewards.squeeze(0)
single_returns_to_go = returns_to_go.squeeze(0)
single_timesteps = timesteps.squeeze(0)

action = model.get_action(single_states, single_actions, single_rewards,
                         single_returns_to_go, single_timesteps)
# 返回: (x, y, rotation) 离散动作元组
```

## 架构特点

- **空间状态编码**: 使用CNN或ViT将三通道掩码图像编码为特征向量
- **序列建模**: 将轨迹建模为 `(return-to-go, state, action)` 序列
- **自回归预测**: 使用 GPT-2 架构进行自回归动作预测
- **位置预测**: 预测下一个组件的(x,y)坐标而不是完整状态
- **条件生成**: 通过 return-to-go 控制生成行为
- **多模态嵌入**: 分别嵌入状态、动作、回报和时间步

## 依赖

- PyTorch
- Transformers (Hugging Face)

## 测试

运行 `python example.py` 来测试CNN和ViT编码器版本。

## 应用场景

- PCB组件自动布局优化
- 考虑布线长度约束的布局规划
- 硬约束(位置限制)和软约束(HPWL最小化)的联合优化

## 参考

论文: [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)