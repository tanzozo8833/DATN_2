import torch
import torch.nn as nn
import math
from src.models.modules.gqa_attention import GroupQueryAttention
from src.models.modules.ds_conv import TemporalDSConv

class SimpleAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, num_heads=8, num_groups=2):
        super().__init__()
        self.attention = GroupQueryAttention(in_channels, num_heads, num_groups)
        self.ln1 = nn.LayerNorm([num_points, in_channels])
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(in_channels, in_channels)
        )
        self.ln2 = nn.LayerNorm([num_points, in_channels])
        self.temporal_conv = TemporalDSConv(in_channels, out_channels, stride=(2,1))

    def forward(self, x, sgr=None):
        x = x.permute(0, 2, 3, 1)
        res = x
        x = self.attention(x, sgr=sgr)
        x = self.ln1(x + res)
        res = x
        x = self.ffn(x)
        x = self.ln2(x + res)
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        return x
