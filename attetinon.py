import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块

    通过全局平均池化和全局最大池化捕获通道间的依赖关系
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # ========== 修复：确保中间层通道数至少为1 ==========
        hidden_channels = max(1, in_channels // reduction_ratio)

        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, N, C) 对于ViT/Mamba架构
        # 转换为 (B, C, N) 以便使用池化
        x_permuted = x.permute(0, 2, 1)  # (B, C, N)

        # 平均池化和最大池化
        avg_out = self.avg_pool(x_permuted).squeeze(-1)  # (B, C)
        max_out = self.max_pool(x_permuted).squeeze(-1)  # (B, C)

        # 通过共享MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        # 融合并生成注意力权重
        out = self.sigmoid(avg_out + max_out)  # (B, C)

        # 扩展维度并应用到输入
        out = out.unsqueeze(1)  # (B, 1, C)
        return x * out  # 广播乘法


class SpatialAttention(nn.Module):
    """空间注意力模块

    通过沿通道维度的统计信息捕获空间依赖关系
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        # 1D卷积用于序列数据
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, N, C)
        # 转换为 (B, C, N)
        x_permuted = x.permute(0, 2, 1)  # (B, C, N)

        # 沿通道维度计算平均和最大
        avg_out = torch.mean(x_permuted, dim=1, keepdim=True)  # (B, 1, N)
        max_out, _ = torch.max(x_permuted, dim=1, keepdim=True)  # (B, 1, N)

        # 拼接
        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, N)

        # 卷积并生成注意力权重
        out = self.conv(out)  # (B, 1, N)
        out = self.sigmoid(out)

        # 转回 (B, N, 1) 并应用
        out = out.permute(0, 2, 1)  # (B, N, 1)
        return x * out  # 广播乘法


class HLCA(nn.Module):
    """CBAM完整模块：通道注意力 + 空间注意力

    Args:
        in_channels: 输入通道数（对于ViT/Mamba，即embedding维度）
        reduction_ratio: 通道注意力的降维比例
        kernel_size: 空间注意力的卷积核大小
        use_channel_attention: 是否使用通道注意力
        use_spatial_attention: 是否使用空间注意力
    """

    def __init__(self,
                 in_channels,
                 reduction_ratio=16,
                 kernel_size=7,
                 use_channel_attention=True,
                 use_spatial_attention=True):
        super(CBAM, self).__init__()

        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention

        if use_channel_attention:
            self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, N, C)
               B: batch size
               N: 序列长度（patch数量）
               C: 通道数（embedding维度）

        Returns:
            out: 注意力加权后的特征 (B, N, C)
        """
        # 通道注意力
        if self.use_channel_attention:
            x = self.channel_attention(x)

        # 空间注意力
        if self.use_spatial_attention:
            x = self.spatial_attention(x)

        return x


class HLCAFor2D(nn.Module):
    """用于2D特征图的CBAM（传统CNN用法）

    如果需要在重塑为2D特征图后使用，可以用这个版本
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMFor2D, self).__init__()
        self.channel_attention = ChannelAttention2D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention2D(kernel_size)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention2D(nn.Module):
    """2D版本的通道注意力"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ========== 修复：确保中间层通道数至少为1 ==========
        # 当in_channels很小时（如in_channels=1），避免除以reduction_ratio后变成0
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention2D(nn.Module):
    """2D版本的空间注意力"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention2D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out