import unittest
import torch
from src.models.modules.sgr_module import SGRModule

class TestSGRModule(unittest.TestCase):
    def test_sgr_properties(self):
        num_points = 77
        model = SGRModule(num_points)
        
        # 1. Kiểm tra kích thước ma trận[cite: 2]
        matrix = model()
        self.assertEqual(matrix.shape, (num_points, num_points))
        
        # 2. Kiểm tra xem đây có phải là tham số có thể học được (cần tính gradient) không
        self.assertTrue(matrix.requires_grad)
        
        # 3. Thử nghiệm một bước lan truyền ngược giả lập
        loss = matrix.sum()
        loss.backward()
        self.assertIsNotNone(model.sgr_matrix.grad)
        
        print(f"[v] Ma trận SGR {num_points}x{num_points} khởi tạo chính xác và có thể cập nhật trọng số.")

if __name__ == '__main__':
    unittest.main()