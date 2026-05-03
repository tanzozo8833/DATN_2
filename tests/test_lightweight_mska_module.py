import unittest
import torch
from src.models.modules.lightweight_mska_module import LightweightMSKAModule

class TestLightweightMSKAModule(unittest.TestCase):
    def test_forward_and_shape(self):
        batch, t, n, c_in = 2, 40, 77, 64
        c_out = 128
        
        model = LightweightMSKAModule(c_in, c_out, n)
        x = torch.randn(batch, t, n, c_in)
        sgr = torch.randn(n, n)
        
        output = model(x, sgr=sgr)
        
        # T/2 = 20 theo stride của TemporalDSConv
        self.assertEqual(output.shape, (batch, c_out, 20, n))
        print(f"[v] LightweightMSKAModule xử lý thành công. Output: {output.shape}")

if __name__ == '__main__':
    unittest.main()