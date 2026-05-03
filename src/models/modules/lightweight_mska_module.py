import torch
import torch.nn as nn
import math
from src.models.modules.gqa_attention import GroupQueryAttention
from src.models.modules.ds_conv import TemporalDSConv

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_points):
        """
        Mã hóa vị trí không gian hỗ trợ cả số kênh lẻ (như 3 kênh tọa độ).
        """
        super(SpatialPositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(num_points, d_model)
        position = torch.arange(0, num_points, dtype=torch.float).unsqueeze(1)
        
        # Tính toán div_term cho số kênh chẵn gần nhất
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        sin_vals = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)
        
        # Gán giá trị cẩn thận để tránh lỗi lệch kích thước (mismatch)
        pe[:, 0::2] = sin_vals[:, :pe[:, 0::2].size(1)]
        pe[:, 1::2] = cos_vals[:, :pe[:, 1::2].size(1)]
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [Batch, C, T, N]
        """
        # Chuyển pe [N, C] -> [1, C, 1, N] để cộng vào tensor đầu vào
        pe = self.pe.transpose(0, 1).view(1, self.d_model, 1, x.size(3))
        return x + pe

class LightweightMSKAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, num_heads=8, num_groups=2):
        super(LightweightMSKAModule, self).__init__()
        
        self.pe = SpatialPositionalEncoding(in_channels, num_points)
        self.attention = GroupQueryAttention(in_channels, num_heads, num_groups)
        
        self.ln1 = nn.LayerNorm([num_points, in_channels])
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(in_channels, in_channels)
        )
        self.ln2 = nn.LayerNorm([num_points, in_channels])
        
        # DSConv để trích xuất đặc trưng thời gian và giảm chiều T[cite: 2]
        self.temporal_conv = TemporalDSConv(in_channels, out_channels, stride=(2, 1))

    def forward(self, x, sgr=None):
        """
        Đầu vào và đầu ra thống nhất dạng [Batch, C, T, N] để có thể xếp chồng modules.
        """
        # 1. Cộng Position Encoding[cite: 2]
        x = self.pe(x) 
        
        # 2. Chuyển sang [B, T, N, C] để xử lý Attention
        x = x.permute(0, 2, 3, 1)
        
        # Residual 1: Attention[cite: 2]
        res = x
        # x = self.embedding(x)
        x = self.attention(x, sgr=sgr)
        x = self.ln1(x + res)
        
        # Residual 2: Feed Forward[cite: 2]
        res = x
        x = self.ffn(x)
        x = self.ln2(x + res)
        
        # 3. Chuyển ngược lại [B, C, T, N] để chạy Conv thời gian[cite: 2]
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x) 
        
        return x