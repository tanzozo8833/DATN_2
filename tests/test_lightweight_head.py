import unittest
import torch
from src.models.lightweight_head import LightweightHead

class TestLightweightHead(unittest.TestCase):
    def test_head_output_shape(self):
        batch, c, t, n = 2, 256, 10, 77
        num_classes = 1000 # Giả lập 1000 từ vựng
        
        model = LightweightHead(in_channels=c, num_classes=num_classes)
        x = torch.randn(batch, c, t, n)
        
        logits = model(x)
        
        # Kết quả phải là [Batch, T, NumClasses][cite: 2]
        self.assertEqual(logits.shape, (batch, t, num_classes))
        print(f"[v] LightweightHead output logits shape chính xác: {logits.shape}")

    def test_backprop_compatibility(self):
        """Đảm bảo gradients có thể chảy ngược qua DSConv trong Head."""
        model = LightweightHead(in_channels=64, num_classes=10)
        x = torch.randn(2, 64, 5, 77, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        print("[v] Head hỗ trợ lan truyền ngược (backpropagation) hoàn chỉnh.")

if __name__ == '__main__':
    unittest.main()