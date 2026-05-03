import unittest
import torch
import torch.nn as nn
from src.models.modules.ds_conv import DSConv2D, TemporalDSConv

class TestDSConv(unittest.TestCase):
    def test_parameter_count(self):
        """Kiểm tra xem DSConv có thực sự ít tham số hơn Conv2D thường không."""
        in_c, out_c = 64, 128
        kernel = 3
        
        # Conv2D thường: in_c * out_c * k * k
        standard_conv = nn.Conv2d(in_c, out_c, kernel, bias=False)
        std_params = sum(p.numel() for p in standard_conv.parameters())
        
        # DSConv: (in_c * k * k) + (in_c * out_c * 1 * 1)
        ds_conv = DSConv2D(in_c, out_c, kernel, bias=False)
        ds_params = sum(p.numel() for p in ds_conv.parameters())
        
        print(f"\n[i] Standard Conv params: {std_params}")
        print(f"[i] DSConv params: {ds_params}")
        
        self.assertLess(ds_params, std_params)
        self.assertAlmostEqual(std_params / ds_params, 8.8, delta=1.0) # Tỷ lệ giảm xấp xỉ 9 lần
        print("[v] DSConv giúp giảm tham số đáng kể.")

    def test_temporal_ds_conv_shape(self):
        """Kiểm tra shape đầu ra khi giảm chiều thời gian (stride=2)."""
        # Giả lập input: [Batch=2, C=64, T=40, N=77]
        x = torch.randn(2, 64, 40, 77)
        
        # Stride=(2, 1) sẽ làm T: 40 -> 20
        model = TemporalDSConv(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        output = model(x)
        
        self.assertEqual(output.shape, (2, 128, 20, 77))
        print(f"[v] TemporalDSConv output shape chính xác: {output.shape}")

if __name__ == '__main__':
    unittest.main()