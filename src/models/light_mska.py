import torch
import torch.nn as nn
import yaml
from src.models.modules.lightweight_mska_module import LightweightMSKAModule
from src.models.modules.sgr_module import SGRModule
from src.models.lightweight_head import LightweightHead

class LightMSKA(nn.Module):
    def __init__(self, config_path, num_classes, embed_dim=64):
        """
        Mô hình Light-MSKA hoàn chỉnh với 5 luồng dữ liệu.
        """
        super(LightMSKA, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.streams_cfg = self.config['data']['streams']
        self.num_classes = num_classes
        self.stream_names = ['body', 'left_hand', 'right_hand', 'mouth', 'face']
        
        self.input_projs = nn.ModuleDict() # Lớp chiếu đầu vào 3 -> 64
        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.sgrs = nn.ModuleDict()
        
        for name in self.stream_names:
            num_points = len(self.streams_cfg[name]['indices'])
            
            # 1. Bước nhúng đặc trưng thô (Feature Embedding): 3 -> 64
            # Giúp embed_dim chia hết cho num_heads=8
            self.input_projs[name] = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.LeakyReLU(0.1)
            )
            
            # 2. Các khối Encoder chuyên biệt (Giờ nhận vào 64 kênh)[cite: 2]
            self.encoders[name] = nn.Sequential(
                LightweightMSKAModule(64, 64, num_points),
                LightweightMSKAModule(64, 128, num_points),
                LightweightMSKAModule(128, 256, num_points)
            )
            
            self.sgrs[name] = SGRModule(num_points)
            self.heads[name] = LightweightHead(256, num_classes)

        self.fuse_head = LightweightHead(256, num_classes)

    def forward(self, x):
        batch_size, t, n_total, c = x.shape
        stream_logits = {}
        stream_features = []

        indices_map = {
            'body': (0, 9),
            'left_hand': (9, 30),
            'right_hand': (30, 51),
            'mouth': (51, 63),
            'face': (63, 77)
        }

        for name in self.stream_names:
            start, end = indices_map[name]
            x_stream = x[:, :, start:end, :] # [B, T, N_stream, 3]
            
            # Áp dụng Feature Embedding đầu tiên: [B, T, N, 3] -> [B, T, N, 64][cite: 2]
            x_stream = self.input_projs[name](x_stream)
            
            # Chuyển sang dạng [B, C, T, N] cho PE và Encoder
            x_stream = x_stream.permute(0, 3, 1, 2)
            
            sgr_mat = self.sgrs[name]()
            
            feat = x_stream
            for i, layer in enumerate(self.encoders[name]):
                # Truyền SGR vào khối chú ý[cite: 2]
                feat = layer(feat, sgr=sgr_mat if i == 0 else None)
            
            logits = self.heads[name](feat)
            stream_logits[name] = logits
            stream_features.append(feat)

        # Hợp nhất đặc trưng[cite: 2]
        fused_feat = torch.stack([torch.mean(f, dim=-1) for f in stream_features], dim=0).mean(dim=0).unsqueeze(-1)
        fuse_logits = self.fuse_head(fused_feat)
        stream_logits['fuse'] = fuse_logits

        return stream_logits