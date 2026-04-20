import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Tạo bảng mã hóa vị trí (sine và cosine)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Time, Dim)
        # Cộng bản đồ thời gian vào tensor dữ liệu
        x = x + self.pe[:, :x.size(1), :]
        return x

class CoordinateTransformer(nn.Module):
    def __init__(self, input_dim=6636, hidden_dim=512, nhead=8, num_layers=6, num_classes=1124, dropout=0.2):
        super(CoordinateTransformer, self).__init__()
        
        # 1. Conv1D Embedding: Học mối quan hệ giữa các bộ phận tay/mặt/người
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Lớp mã hóa thời gian: Giúp Transformer biết frame nào đứng trước/sau
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 3. Transformer Encoder: Học ngữ pháp thủ ngữ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # 4. Lớp dự đoán Gloss
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, input_lens):
        # A. Tạo Padding Mask (True tại vị trí là padding)
        batch_size, seq_len, _ = x.size()
        mask = torch.arange(seq_len).to(x.device).expand(batch_size, seq_len) >= input_lens.unsqueeze(1)
        mask = mask.to(torch.bool)
        # B. Conv1D Embedding (Cần transpose vì Conv1D nhận chiều Channel ở giữa)
        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        x = self.conv_embedding(x)
        x = x.transpose(1, 2) # (B, C, T) -> (B, T, C)
        
        # C. Thêm Positional Encoding (Quan trọng!)
        # Nhân căn bậc 2 hidden_dim để cân bằng trọng số (theo paper Transformer gốc)
        x = x * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)
        
        # D. Transformer với Padding Mask
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # E. Chuyển về logit
        logits = self.classifier(x)
        
        # Trả về kết quả (CTC Loss yêu cầu Time đứng đầu: T, B, C)
        return logits.transpose(0, 1)