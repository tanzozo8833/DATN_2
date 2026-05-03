import torch
import numpy as np
import random
import torch.nn.functional as F

class Augmentor:
    def __init__(self, rotation_range=0.2, temporal_range=(0.5, 1.5)):
        self.rotation_range = rotation_range
        self.temporal_range = temporal_range

    def normalize(self, x, width=256, height=256):
        normalized_x = x.clone()
        normalized_x[..., 0] = (x[..., 0] / width - 0.5) / 0.5
        normalized_x[..., 1] = ((height - x[..., 1]) / height - 0.5) / 0.5
        return normalized_x

    def random_rotate(self, x):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ], dtype=torch.float32).to(x.device)

        coords = x[..., :2]
        rotated_coords = torch.matmul(coords, rotation_matrix.t().to(coords.dtype))       
        x[..., :2] = rotated_coords
        return x

    def temporal_resample(self, x):
        """
        Xử lý tăng cường thời gian cho cả mẫu đơn và Batch[cite: 2].
        """
        scale = random.uniform(*self.temporal_range)
        
        if x.dim() == 3: # Một mẫu đơn lẻ
            t_orig, n, c = x.shape
            t_new = max(int(t_orig * scale), 5)
            x_reshaped = x.permute(1, 2, 0).reshape(1, -1, t_orig)
            x_resampled = F.interpolate(x_reshaped, size=t_new, mode='linear', align_corners=False)
            return x_resampled.reshape(n, c, t_new).permute(2, 0, 1), scale
        else: # Một Batch [B, T, N, C]
            b, t_orig, n, c = x.shape
            t_new = max(int(t_orig * scale), 5)
            x_reshaped = x.permute(0, 2, 3, 1).reshape(b, -1, t_orig)
            x_resampled = F.interpolate(x_reshaped, size=t_new, mode='linear', align_corners=False)
            return x_resampled.view(b, n, c, t_new).permute(0, 3, 1, 2), scale

    def __call__(self, x, apply_aug=True):
        x = self.normalize(x)
        scale = 1.0
        
        if apply_aug:
            x = self.random_rotate(x)
            x, scale = self.temporal_resample(x)
            
        return x, scale
    





