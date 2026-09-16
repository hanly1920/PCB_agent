import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('training_log.csv')

# 创建一个全局索引，用于x轴
df['global_idx'] = df['epoch'] * 8 + df['update_idx']

# 绘制loss, loss_x, loss_y, loss_rot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['global_idx'], df['loss'], label='loss', marker='o')
plt.title('Total Loss Over Time')
plt.xlabel('Global Update Index')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(df['global_idx'], df['loss_x'], label='loss_x', marker='o')
plt.title('Loss X Over Time')
plt.xlabel('Global Update Index')
plt.ylabel('Loss X')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(df['global_idx'], df['loss_y'], label='loss_y', marker='o')
plt.title('Loss Y Over Time')
plt.xlabel('Global Update Index')
plt.ylabel('Loss Y')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(df['global_idx'], df['loss_rot'], label='loss_rot', marker='o')
plt.title('Loss Rot Over Time')
plt.xlabel('Global Update Index')
plt.ylabel('Loss Rot')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_losses_detailed.png', dpi=150)
# plt.show()  # 移除以避免阻塞

# 检查周期性：比较每个epoch的loss模式
epochs = df['epoch'].unique()
for ep in epochs:
    ep_data = df[df['epoch'] == ep]
    plt.plot(ep_data['update_idx'], ep_data['loss'], label=f'Epoch {ep}', marker='o')

plt.title('Loss per Epoch (Checking Periodicity)')
plt.xlabel('Update Index within Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_per_epoch.png', dpi=150)
# plt.show()  # 移除