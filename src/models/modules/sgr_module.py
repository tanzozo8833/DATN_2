import torch
import torch.nn as nn

class SGRModule(nn.Module):
    def __init__(self, num_points):
        """
        Khởi tạo Ma trận Chỉnh tắc hóa không gian toàn cầu (Spatial Global Regularization).
        
        Args:
            num_points (int): Số lượng điểm then chốt (N). Trong project của bạn là 77.
        """
        super(SGRModule, self).__init__()
        # Khởi tạo ma trận N x N là một tham số có thể học được (learnable parameter)[cite: 2].
        # Chúng ta khởi tạo với giá trị ngẫu nhiên nhỏ để không làm nhiễu Attention Map ban đầu.
        self.sgr_matrix = nn.Parameter(torch.randn(num_points, num_points) * 0.01)

    def forward(self):
        """
        Trả về ma trận SGR để cộng trực tiếp vào ma trận chú ý không gian[cite: 2].
        """
        return self.sgr_matrix