import torch
import torch.nn as nn
import torch.nn.functional as F


class PCBStateEncoder(nn.Module):
    """
    PCB状态编码器：将四通道掩码图像编码为特征向量

    输入: (batch_size, 4, N, N) - 四通道掩码图像
        Channel 0: View Mask (已被占据的网格) ∈ {0,1}^{N×N}
        Channel 1: Position Mask (合法放置位置 - 基础) ∈ {0,1}^{N×N}
        Channel 2: Wire Mask (HPWL增量热力图) ∈ ℝ^{N×N}
        Channel 3: Position Mask (专门通道 - 动态上下文相关) ∈ {0,1}^{N×N}

    输出: (batch_size, hidden_size) - 特征向量 (如512维)

    架构设计:
    - 多层CNN提取空间特征，特别关注Position Mask的几何约束
    - 自适应池化压缩到固定尺寸
    - 全连接层映射到隐空间
    """

    def __init__(self, grid_size=32, hidden_size=512):  # 增加默认hidden_size到512
        super().__init__()
        self.grid_size = grid_size
        self.hidden_size = hidden_size

        # 增强的CNN特征提取器 - 处理4通道输入
        self.conv_layers = nn.Sequential(
            # 输入: (4, N, N) - 现在是4通道
            nn.Conv2d(4, 64, kernel_size=3, padding=1),    # 64通道，适应4通道输入
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 128通道
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 空间尺寸减半

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256通道
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 保持256通道
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # 输出: (256, 8, 8) - 更大的特征图
        )

        # 特征映射到隐空间 - 两层MLP
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, hidden_size),  # 256*8*8 = 16384 -> hidden_size
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, state_masks):
        """
        Args:
            state_masks: (batch_size, 4, N, N) 四通道掩码图像

        Returns:
            features: (batch_size, hidden_size) 特征向量
        """
        # CNN特征提取
        conv_features = self.conv_layers(state_masks)  # (batch_size, 256, 8, 8)

        # 展平并映射到隐空间
        flattened = conv_features.view(conv_features.size(0), -1)  # (batch_size, 256*8*8)
        features = self.fc(flattened)  # (batch_size, hidden_size)

        return features


class PCBStateEncoderViT(nn.Module):
    """
    基于ViT的PCB状态编码器（可选替代方案）
    """

    def __init__(self, grid_size=32, hidden_size=128, patch_size=4, num_heads=8, num_layers=6):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        # 计算patch数量
        self.num_patches = (grid_size // patch_size) ** 2
        self.patch_dim = 4 * patch_size * patch_size  # 4 channels (更新为4通道)

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, hidden_size)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4*hidden_size,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, state_masks):
        """
        Args:
            state_masks: (batch_size, 4, N, N) 四通道掩码图像

        Returns:
            features: (batch_size, hidden_size) 特征向量
        """
        batch_size = state_masks.size(0)

        # 创建patches: (batch_size, num_patches, patch_dim)
        patches = self._create_patches(state_masks)

        # Patch embedding: (batch_size, num_patches, hidden_size)
        patch_embeddings = self.patch_embed(patches)

        # 添加位置编码
        patch_embeddings = patch_embeddings + self.pos_embed

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)

        # Transformer
        transformer_output = self.transformer(embeddings)

        # 使用CLS token作为特征
        features = transformer_output[:, 0]  # (batch_size, hidden_size)

        return features

    def _create_patches(self, x):
        """将图像分割为patches"""
        batch_size, channels, height, width = x.size()
        patch_size = self.patch_size

        # 确保可以整除
        assert height % patch_size == 0 and width % patch_size == 0

        # 重塑为patches
        x = x.view(batch_size, channels,
                  height // patch_size, patch_size,
                  width // patch_size, patch_size)

        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, -1, channels * patch_size * patch_size)

        return x