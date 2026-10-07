import torch
import torch.nn as nn


# ==========================================================
# Channel Attention
# ==========================================================

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        reduced_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


# ==========================================================
# Spatial Attention
# ==========================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        attention = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.conv(attention)

        attention = self.sigmoid(attention)

        return x * attention


# ==========================================================
# CBAM
# ==========================================================

class CBAM(nn.Module):

    def __init__(self,
                 in_channels,
                 reduction=16,
                 kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels,
            reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size
        )

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x