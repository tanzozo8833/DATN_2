import random
import torch
import torch.nn.functional as F


class Augmentor:
    """
    Augmentation trên keypoint tensor [T, 77, 3].

    Mỗi augmentation được áp dụng ngẫu nhiên theo xác suất tương ứng.
    """

    def __init__(
        self,
        rotation_range: float = 0.15,
        temporal_range: tuple = (0.7, 1.3),
        noise_std: float = 0.005,
        drop_prob: float = 0.05,
    ):
        self.rotation_range = rotation_range
        self.temporal_range = temporal_range
        self.noise_std = noise_std
        self.drop_prob = drop_prob

    def __call__(self, kpts: torch.Tensor) -> torch.Tensor:
        """kpts: Tensor[T, 77, 3]"""
        if random.random() < 0.5:
            kpts = self._temporal_resample(kpts)
        if random.random() < 0.5:
            kpts = self._add_noise(kpts)
        if random.random() < 0.3:
            kpts = self._random_rotation(kpts)
        if random.random() < 0.2:
            kpts = self._keypoint_dropout(kpts)
        return kpts

    def _temporal_resample(self, kpts: torch.Tensor) -> torch.Tensor:
        T, J, C = kpts.shape
        rate = random.uniform(*self.temporal_range)
        new_T = max(4, int(T * rate))
        # Interpolate: 1 × (J*C) × T → 1 × (J*C) × new_T
        x = kpts.reshape(T, J * C).T.unsqueeze(0)            # 1 × (J*C) × T
        x = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
        return x.squeeze(0).T.reshape(new_T, J, C)

    def _add_noise(self, kpts: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(kpts) * self.noise_std
        noise[:, :, 2] = 0.0    # không làm nhiễu kênh confidence
        return kpts + noise

    def _random_rotation(self, kpts: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        cos_a = torch.cos(torch.tensor(angle, dtype=kpts.dtype))
        sin_a = torch.sin(torch.tensor(angle, dtype=kpts.dtype))
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])    # 2×2
        xy = kpts[:, :, :2]                                     # T × J × 2
        T, J, _ = xy.shape
        rotated = (R @ xy.reshape(-1, 2).T).T.reshape(T, J, 2)
        kpts = kpts.clone()
        kpts[:, :, :2] = rotated
        return kpts

    def _keypoint_dropout(self, kpts: torch.Tensor) -> torch.Tensor:
        """Ngẫu nhiên zero-out một số keypoints (toàn bộ frame)."""
        T, J, C = kpts.shape
        mask = (torch.rand(J) > self.drop_prob).float()       # J
        kpts = kpts * mask.unsqueeze(0).unsqueeze(-1)
        return kpts
