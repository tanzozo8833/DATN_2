import torch
import torch.nn as nn
import math
from src.models.modules.gqa_attention import GroupQueryAttention
from src.models.modules.ds_conv import TemporalDSConv

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).view(1, d_model, max_len, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        T = x.size(2)
        return x + self.pe[:, :, :T, :]

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln(x)
        return x

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_points):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(num_points, d_model)
        position = torch.arange(0, num_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        sin_vals = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)
        pe[:, 0::2] = sin_vals[:, :pe[:, 0::2].size(1)]
        pe[:, 1::2] = cos_vals[:, :pe[:, 1::2].size(1)]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe.transpose(0, 1).view(1, self.d_model, 1, x.size(3))
        return x + pe

class LightweightMSKAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, num_heads=8, num_groups=2):
        super().__init__()
        self.pe = SpatialPositionalEncoding(in_channels, num_points)
        self.attention = GroupQueryAttention(in_channels, num_heads, num_groups)
        self.ln1 = nn.LayerNorm([num_points, in_channels])
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(in_channels, in_channels)
        )
        self.ln2 = nn.LayerNorm([num_points, in_channels])
        self.temporal_attn = TemporalAttention(in_channels, num_heads=4)
        self.temporal_pe = TemporalPositionalEncoding(in_channels)
        self.temporal_conv = TemporalDSConv(in_channels, out_channels, stride=(2, 1))
    
    def forward(self, x, sgr=None):
        x = self.pe(x)
        x = x.permute(0, 2, 3, 1)
        res = x
        x = self.attention(x, sgr=sgr)
        x = self.ln1(x + res)
        res = x
        x = self.ffn(x)
        x = self.ln2(x + res)
        B, T, N, C = x.shape
        x_temp = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        x_temp = self.temporal_attn(x_temp)
        x = x_temp.reshape(B, N, T, C).permute(0, 2, 1, 3)
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_pe(x)
        x = self.temporal_conv(x)
        return x
