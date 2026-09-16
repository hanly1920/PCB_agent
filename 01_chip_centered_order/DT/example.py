"""
Decision Transformer PCB布局优化示例
使用三通道掩码图像作为状态表示
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import DecisionTransformer

def main():
    # 初始化PCB DT模型参数
    grid_size = 16   # PCB网格大小 (16x16)
    hidden_size = 128
    max_length = 20  # 序列长度 K

    # 创建PCB DT模型 - 动作空间现在是固定的 (x, y, rotation)
    model = DecisionTransformer(
        grid_size=grid_size,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=1000,
        use_vit=False,  # 使用CNN编码器
        n_layer=3,
        n_head=1,
        n_inner=4*hidden_size,
    )

    print("PCB Decision Transformer 模型创建成功!")
    print(f"模型参数: grid_size={grid_size}, 动作空间=离散(x,y,rotation)")
    print(f"X坐标类别数: {grid_size}, Y坐标类别数: {grid_size}, 旋转类别数: 4")

    # 创建示例输入 - PCB状态为四通道掩码图像
    batch_size = 1
    seq_len = 10

    # states: (batch_size, seq_len, 4, N, N) - 四通道掩码图像
    # Channel 0: View Mask (已被占据的网格)
    # Channel 1: Position Mask (合法放置位置 - 基础)
    # Channel 2: Wire Mask (HPWL增量热力图)
    # Channel 3: Position Mask (专门通道 - 动态上下文相关)
    states = torch.randn(batch_size, seq_len, 4, grid_size, grid_size)

    # actions: (batch_size, seq_len, 3) - 离散动作 [x, y, rotation]
    # x, y: 0到grid_size-1的整数, rotation: 0-3的整数
    actions = torch.randint(0, grid_size, (batch_size, seq_len, 2))  # x, y坐标
    rotations = torch.randint(0, 4, (batch_size, seq_len, 1))       # 旋转角度
    actions = torch.cat([actions, rotations], dim=-1).float()       # 拼接为[x, y, rotation]

    # rewards和returns_to_go保持不变
    rewards = torch.randn(batch_size, seq_len, 1)
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"输入形状: states={states.shape}, actions={actions.shape}")

    # 前向传播
    with torch.no_grad():
        state_preds, action_preds, return_preds = model(
            states, actions, rewards, returns_to_go, timesteps
        )

    # 解包动作预测logits
    x_logits, y_logits, rot_logits = action_preds
    print(f"动作预测logits形状: x={x_logits.shape}, y={y_logits.shape}, rotation={rot_logits.shape}")
    print("state_preds现在预测下一个组件的(x,y)位置坐标")

    # 获取单个动作 (推理)
    # 注意: states需要是(seq_len, 4, N, N)格式, actions需要是(seq_len, 3)格式
    single_states = states.squeeze(0)  # 移除batch维度
    single_actions = actions.squeeze(0)
    single_rewards = rewards.squeeze(0)
    single_returns_to_go = returns_to_go.squeeze(0)
    single_timesteps = timesteps.squeeze(0)

    action = model.get_action(single_states, single_actions, single_rewards,
                             single_returns_to_go, single_timesteps)
    print(f"预测离散动作: {action} (x={action[0]}, y={action[1]}, rotation={action[2]})")

    print("PCB Decision Transformer 核心测试完成!")

def demonstrate_mask_enforcement():
    """演示Position Mask强制功能"""
    print("\n--- 演示Position Mask强制功能 ---")

    grid_size = 8   # 使用小网格便于演示
    model = DecisionTransformer(
        grid_size=grid_size,
        hidden_size=128,
        max_length=10,
        max_ep_len=1000,
        n_layer=2,
        n_head=4,
        n_inner=512,
    )

    batch_size = 1
    seq_len = 1

    # 创建状态 - 只允许左上角2x2区域
    states = torch.zeros(batch_size, seq_len, 4, grid_size, grid_size)

    # 设置Position Mask (Channel 3) - 只允许(0,0), (0,1), (1,0), (1,1)
    position_mask = torch.zeros(grid_size, grid_size)
    position_mask[:2, :2] = 1  # 左上角2x2区域
    states[:, :, 3] = position_mask

    # 创建任意动作（会被遮罩覆盖）
    actions = torch.zeros(batch_size, seq_len, 3, dtype=torch.long)
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.zeros(batch_size, seq_len, dtype=torch.long)

    print(f"网格大小: {grid_size}x{grid_size}")
    print("允许放置区域: (0,0) 到 (1,1)")

    with torch.no_grad():
        state_preds, action_preds, return_preds = model(
            states, actions, None, returns_to_go, timesteps
        )

    x_logits, y_logits, rot_logits = action_preds

    # 计算概率分布
    x_probs = torch.softmax(x_logits[0, 0], dim=0)
    y_probs = torch.softmax(y_logits[0, 0], dim=0)

    print("\nX坐标概率分布:")
    for i in range(grid_size):
        prob = x_probs[i].item()
        status = "✓" if i < 2 else "✗"
        print(".3f")

    print("\nY坐标概率分布:")
    for i in range(grid_size):
        prob = y_probs[i].item()
        status = "✓" if i < 2 else "✗"
        print(".3f")

    # 验证约束
    illegal_x_prob = x_probs[2:].sum().item()
    illegal_y_prob = y_probs[2:].sum().item()

    print("\n约束验证:")
    print(".6f")
    print(".6f")

    if illegal_x_prob < 1e-6 and illegal_y_prob < 1e-6:
        print("✓ 成功：模型只预测合法位置！")
    else:
        print("✗ 失败：存在非法位置预测")

    return illegal_x_prob < 1e-6 and illegal_y_prob < 1e-6

def test_vit_encoder():
    """测试ViT编码器版本"""
    print("\n--- 测试ViT编码器版本 ---")

    grid_size = 16
    hidden_size = 128
    max_length = 20

    # 创建使用ViT编码器的模型 - 动作空间现在是固定的
    model_vit = DecisionTransformer(
        grid_size=grid_size,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=1000,
        n_layer=3,
        n_head=1,
        n_inner=4*hidden_size,
        use_vit=True,  # 使用ViT编码器
    )

    print("PCB Decision Transformer (ViT) 模型创建成功!")

    # 测试前向传播
    batch_size = 1
    seq_len = 5
    states = torch.randn(batch_size, seq_len, 4, grid_size, grid_size)
    # 生成离散动作
    actions = torch.randint(0, grid_size, (batch_size, seq_len, 2))  # x, y
    rotations = torch.randint(0, 4, (batch_size, seq_len, 1))       # rotation
    actions = torch.cat([actions, rotations], dim=-1).float()
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        state_preds, action_preds, return_preds = model_vit(
            states, actions, None, returns_to_go, timesteps
        )

    # 解包ViT动作预测
    x_logits, y_logits, rot_logits = action_preds
    print(f"ViT动作预测logits形状: x={x_logits.shape}, y={y_logits.shape}, rotation={rot_logits.shape}")
    print("ViT编码器测试完成!")

if __name__ == "__main__":
    main()
    demonstrate_mask_enforcement()
    test_vit_encoder()