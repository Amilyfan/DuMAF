import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden_channels = max(1, in_channels // reduction_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)  # (B, C, N)

        avg_out = self.avg_pool(x_permuted).squeeze(-1)  # (B, C)
        max_out = self.max_pool(x_permuted).squeeze(-1)  # (B, C)

        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        out = self.sigmoid(avg_out + max_out)  # (B, C)

        out = out.unsqueeze(1)  # (B, 1, C)
        return x * out


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, N, C)
        x_permuted = x.permute(0, 2, 1)  # (B, C, N)

        avg_out = torch.mean(x_permuted, dim=1, keepdim=True)  # (B, 1, N)
        max_out, _ = torch.max(x_permuted, dim=1, keepdim=True)  # (B, 1, N)

        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, N)

        out = self.conv(out)  # (B, 1, N)
        out = self.sigmoid(out)

        out = out.permute(0, 2, 1)  # (B, N, 1)
        return x * out


class CBAM(nn.Module):

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
        if self.use_channel_attention:
            x = self.channel_attention(x)

        if self.use_spatial_attention:
            x = self.spatial_attention(x)

        return x


class CBAMFor2D(nn.Module):

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

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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