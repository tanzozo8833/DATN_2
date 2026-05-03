import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.ds_conv import DSConv2D

class LightweightHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3):
        """
        Mạng đầu đọc (Head Network) phiên bản nhẹ sử dụng DSConv.
        
        Args:
            in_channels (int): Số kênh đặc trưng đầu vào (thường là 256 hoặc 512).
            num_classes (int): Tổng số từ vựng (glosses) trong từ điển.
        """
        super(LightweightHead, self).__init__()
        
        # 1. Spatial Pooling: Giảm chiều không gian (N) về 1[cite: 2]
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1)) 
        
        # 2. Temporal Linear & Processing[cite: 2]
        self.temporal_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. Temporal Convolutional Block sử dụng DSConv[cite: 2]
        # kernel_size=3, stride=1 để giữ nguyên độ dài thời gian T
        self.ds_conv_block = nn.Sequential(
            DSConv2D(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
            DSConv2D(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        )
        
        # 4. Final Classifier: Chuyển đổi sang xác suất từ vựng[cite: 2]
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor đặc trưng [Batch, C, T, N]
        Returns:
            gloss_probabilities: [Batch, T, NumClasses]
        """
        # x: [B, C, T, N] -> Pooling không gian -> [B, C, T, 1]
        x = self.avg_pool(x).squeeze(-1) # [B, C, T]
        
        # Chuẩn bị cho Linear layer: [B, T, C]
        x = x.transpose(1, 2)
        b, t, c = x.shape
        
        # Áp dụng Temporal FC (cần flatten Batch và Time)
        x = x.reshape(-1, c)
        x = self.temporal_fc(x)
        x = x.view(b, t, c)
        
        # Áp dụng DSConv Block (cần dạng [B, C, T, 1])
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.ds_conv_block(x)
        
        # Đặc trưng cuối (Gloss Representation): [B, T, C]
        x = x.squeeze(-1).transpose(1, 2)
        
        # Phân loại: [B, T, NumClasses]
        logits = self.classifier(x)
        
        return logits