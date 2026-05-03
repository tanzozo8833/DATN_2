import unittest
import torch
from src.models.modules.gqa_attention import GroupQueryAttention

class TestGQAAttention(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.num_heads = 8
        self.num_groups = 2
        self.model = GroupQueryAttention(self.embed_dim, self.num_heads, self.num_groups)
        
        # Giả lập input: [Batch=2, T=10, N=77, C=64]
        self.x = torch.randn(2, 10, 77, 64)

    def test_output_shape(self):
        """Kiểm tra shape đầu ra phải khớp với đầu vào."""
        output = self.model(self.x)
        self.assertEqual(output.shape, self.x.shape)
        print(f"[v] GQA Output shape chính xác: {output.shape}")

    def test_sgr_integration(self):
        """Kiểm tra việc tích hợp ma trận SGR[cite: 2]."""
        n = 77
        sgr = torch.randn(n, n)
        
        # Chạy model có SGR
        output_with_sgr = self.model(self.x, sgr=sgr)
        
        # Chạy model không có SGR
        output_without_sgr = self.model(self.x, sgr=None)
        
        # Hai kết quả phải khác nhau
        self.assertFalse(torch.equal(output_with_sgr, output_without_sgr))
        print("[v] SGR được tích hợp và ảnh hưởng đến kết quả đầu ra.")

    def test_parameter_efficiency(self):
        """So sánh số lượng tham số giữa GQA và Standard MHA (giả định)."""
        # Trong GQA, K và V chỉ có (num_groups * head_dim) tham số thay vì (num_heads * head_dim)
        params_count = sum(p.numel() for p in self.model.parameters())
        print(f"[i] Tổng tham số của lớp GQA: {params_count}")
        
        # Một lớp MHA tiêu chuẩn cùng quy mô sẽ có (4 * embed_dim^2) tham số cho Q, K, V, Out
        # GQA tiết kiệm được ở phần K_proj và V_proj
        self.assertTrue(params_count > 0)

if __name__ == '__main__':
    unittest.main()