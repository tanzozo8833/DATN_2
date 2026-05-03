import torch
import torch.nn as nn

class DSConv2D(nn.Module):
    """
    Depthwise Separable Convolution 2D.
    Thay thế cho Conv2D tiêu chuẩn để giảm tham số và FLOPs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DSConv2D, self).__init__()
        # 1. Depthwise Conv: Mỗi channel được học bởi một bộ lọc riêng biệt
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        # 2. Pointwise Conv: Kết hợp thông tin giữa các channels bằng kernel 1x1
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class TemporalDSConv(nn.Module):
    """
    Sử dụng DSConv để trích xuất đặc trưng dọc theo trục thời gian (T).
    Đây là thành phần cuối của mỗi Keypoint Attention Module[cite: 2].
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)):
        super(TemporalDSConv, self).__init__()
        # Trong MSKA, Conv thường áp dụng lên [Batch, C, T, N]
        # kernel_size=(3, 1) nghĩa là chỉ trượt dọc theo trục thời gian T
        self.conv = DSConv2D(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x shape: [Batch, C, T, N]
        return self.conv(x)