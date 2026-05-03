import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.ds_conv import DSConv2D

class LightweightHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.temporal_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=4,
                dim_feedforward=in_channels * 4,
                batch_first=True
            ),
            num_layers=2
        )
        self.ds_conv_block = nn.Sequential(
            DSConv2D(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
            DSConv2D(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        )
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x).squeeze(-1)
        x = x.transpose(1, 2)
        b, t, c = x.shape
        x = x.reshape(-1, c)
        x = self.temporal_fc(x)
        x = x.view(b, t, c)
        x = self.transformer(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.ds_conv_block(x)
        x = x.squeeze(-1).transpose(1, 2)
        logits = self.classifier(x)
        return logits
