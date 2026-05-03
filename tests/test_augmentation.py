import unittest
import torch
from src.utils.augmentation import Augmentor

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.augmentor = Augmentor(rotation_range=0.5, temporal_range=(0.5, 1.5))
        # Tạo dữ liệu giả lập [T=20, Points=77, Channels=3]
        # Giả sử tọa độ pixel nằm trong khoảng [0, 256]
        self.mock_kpts = torch.rand(20, 77, 3) * 200 

    def test_normalize(self):
        """Kiểm tra tọa độ sau chuẩn hóa có nằm trong khoảng [-1, 1]."""
        normalized = self.augmentor.normalize(self.mock_kpts)
        self.assertTrue(torch.all(normalized[..., :2] >= -1.1))
        self.assertTrue(torch.all(normalized[..., :2] <= 1.1))
        # Độ tin cậy (c) không được bị ảnh hưởng bởi normalize theo công thức này
        self.assertTrue(torch.equal(normalized[..., 2], self.mock_kpts[..., 2]))
        print("[v] Normalize chính xác.")

    def test_temporal_resample(self):
        """Kiểm tra thay đổi số lượng khung hình."""
        original_t = self.mock_kpts.shape[0]
        resampled = self.augmentor.temporal_resample(self.mock_kpts)
        new_t = resampled.shape[0]
        
        # T mới phải nằm trong khoảng [20*0.5, 20*1.5] -> [10, 30]
        self.assertTrue(10 <= new_t <= 30)
        self.assertEqual(resampled.shape[1], 77)
        self.assertEqual(resampled.shape[2], 3)
        print(f"[v] Temporal Resample chính xác (T: {original_t} -> {new_t}).")

    def test_random_rotate(self):
        """Kiểm tra xoay không làm thay đổi giá trị trung bình quá lớn và giữ nguyên độ tin cậy."""
        original_c = self.mock_kpts[..., 2].clone()
        rotated = self.augmentor.random_rotate(self.mock_kpts.clone())
        
        # Độ tin cậy phải giữ nguyên
        self.assertTrue(torch.allclose(rotated[..., 2], original_c))
        # Tọa độ X, Y phải thay đổi
        self.assertFalse(torch.equal(rotated[..., :2], self.mock_kpts[..., :2]))
        print("[v] Random Rotate chính xác.")

if __name__ == '__main__':
    unittest.main()