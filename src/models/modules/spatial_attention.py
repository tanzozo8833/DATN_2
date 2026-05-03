import torch
import torch.nn as nn
import math

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, T, N = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, T, N).permute(0, 1, 3, 4, 2)
        k = k.view(B, self.num_heads, self.head_dim, T, N).permute(0, 1, 3, 4, 2)
        v = v.view(B, self.num_heads, self.head_dim, T, N).permute(0, 1, 3, 4, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        out = (attn @ v).permute(0, 1, 4, 2, 3).reshape(B, C, T, N)
        out = self.proj(out)
        out = self.bn(out)
        return out + x
