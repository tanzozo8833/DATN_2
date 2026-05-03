import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_groups=2, dropout=0.1):
        """
        Triển khai Group Query Attention (GQA) tối ưu cho thiết bị di động.
        """
        super(GroupQueryAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim phải chia hết cho num_heads."
        assert num_heads % num_groups == 0, "num_heads phải chia hết cho num_groups."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.kv_heads = num_groups 
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, sgr=None):
        """
        Args:
            x: Tensor đầu vào [Batch, T, N, C]
            sgr: Ma trận Chỉnh tắc hóa không gian toàn cầu [N, N][cite: 2]
        """
        batch_size, t, n, c = x.shape
        
        # Project sang Q, K, V
        q = self.q_proj(x) # [B, T, N, C]
        k = self.k_proj(x) # [B, T, N, kv_heads * head_dim]
        v = self.v_proj(x) # [B, T, N, kv_heads * head_dim]
        
        # Reshape để đưa các Head ra chiều thứ 3: [Batch, T, Heads, N, HeadDim]
        q = q.view(batch_size, t, n, self.num_heads, self.head_dim).transpose(2, 3)
        k = k.view(batch_size, t, n, self.kv_heads, self.head_dim).transpose(2, 3)
        v = v.view(batch_size, t, n, self.kv_heads, self.head_dim).transpose(2, 3)
        
        # GQA Logic: Lặp lại K và V theo nhóm để khớp với số lượng đầu của Q
        num_q_per_kv = self.num_heads // self.num_groups
        if num_q_per_kv > 1:
            k = k.repeat_interleave(num_q_per_kv, dim=2)
            v = v.repeat_interleave(num_q_per_kv, dim=2)
        
        # Tính toán ma trận chú ý trên không gian (N x N)
        # q: [B, T, H, N, D] * k_T: [B, T, H, D, N] -> score: [B, T, H, N, N]
        attn_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Sử dụng hàm Tanh thay cho Softmax theo cấu trúc MSKA[cite: 2]
        attn_map = torch.tanh(attn_score)
        
        # Áp dụng SGR (Spatial Global Regularization)[cite: 2]
        if sgr is not None:
            # Broadcast sgr [N, N] cộng vào attn_map [B, T, H, N, N]
            attn_map = attn_map + sgr.view(1, 1, 1, n, n)
            
        attn_map = self.dropout(attn_map)
        
        # Kết hợp với giá trị V: [B, T, H, N, N] * [B, T, H, N, D] -> [B, T, H, N, D]
        out = torch.matmul(attn_map, v) 
        
        # Ghép các đầu chú ý lại và đưa về dạng [Batch, T, N, C]
        out = out.transpose(2, 3).contiguous().view(batch_size, t, n, c)
        return self.out_proj(out)