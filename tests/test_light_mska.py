import unittest
import torch
import os
from src.models.light_mska import LightMSKA

class TestLightMSKA(unittest.TestCase):
    def setUp(self):
        # Đảm bảo đường dẫn config tồn tại cho test
        self.config_path = "configs/base_config.yaml"
        self.num_classes = 1000
        
        # Nếu chưa có file config thật, tạo một file giả lập cho test
        if not os.path.exists(self.config_path):
            os.makedirs("configs", exist_ok=True)
            with open(self.config_path, "w") as f:
                f.write("""
data:
  streams:
    body: {indices: [0,1,2,3,4,5,6,7,8]}
    left_hand: {indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
    right_hand: {indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
    mouth: {indices: [0,1,2,3,4,5,6,7,8,9,10,11]}
    face: {indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13]}
""")

    def test_model_output_structure(self):
        model = LightMSKA(self.config_path, self.num_classes)
        
        # Giả lập input: [Batch=2, T=32, N=77, C=3]
        x = torch.randn(2, 32, 77, 3)
        
        outputs = model(x)
        
        # 1. Kiểm tra số lượng đầu ra (5 luồng + 1 fuse = 6)[cite: 2]
        self.assertEqual(len(outputs), 6)
        self.assertIn('fuse', outputs)
        self.assertIn('body', outputs)
        
        # 2. Kiểm tra shape của fuse logits [B, T/8, NumClasses]
        # Stride=2 qua 3 lớp Encoder -> T: 32 -> 16 -> 8 -> 4
        expected_t = 32 // (2**3)
        self.assertEqual(outputs['fuse'].shape, (2, expected_t, self.num_classes))
        
        print(f"[v] LightMSKA hoạt động chính xác với 5 luồng. Output T={expected_t}.")

if __name__ == '__main__':
    unittest.main()