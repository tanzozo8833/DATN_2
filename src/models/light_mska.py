import torch
import torch.nn as nn
import yaml
from src.models.modules.lightweight_mska_module import LightweightMSKAModule
from src.models.modules.sgr_module import SGRModule
from src.models.lightweight_head import LightweightHead

class LightMSKA(nn.Module):
    def __init__(self, config_path, num_classes, embed_dim=128):
        super(LightMSKA, self).__init__()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.streams_cfg = self.config['data']['streams']
        self.num_classes = num_classes
        self.stream_names = ['body', 'left_hand', 'right_hand', 'mouth', 'face']

        self.input_projs = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.sgrs = nn.ModuleDict()
        self.context_encoders = nn.ModuleDict()

        for name in self.stream_names:
            num_points = len(self.streams_cfg[name]['indices'])

            self.input_projs[name] = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.LeakyReLU(0.1)
            )

            self.encoders[name] = nn.Sequential(
                LightweightMSKAModule(128, 128, num_points),
                LightweightMSKAModule(128, 256, num_points),
                LightweightMSKAModule(256, 256, num_points)
            )

            self.sgrs[name] = SGRModule(num_points)
            self.context_encoders[name] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    batch_first=True
                ),
                num_layers=2
            )
            self.heads[name] = LightweightHead(256, num_classes)

        self.fuse_weights = nn.Parameter(torch.ones(5, 1, 1, 1))
        self.fuse_head = LightweightHead(256, num_classes)

    def forward(self, x):
        batch_size, t, n_total, c = x.shape
        stream_logits = {}
        stream_features = []
        output_lengths = []

        indices_map = {
            'body': (0, 9),
            'left_hand': (9, 30),
            'right_hand': (30, 51),
            'mouth': (51, 63),
            'face': (63, 77)
        }

        for name in self.stream_names:
            start, end = indices_map[name]
            x_stream = x[:, :, start:end, :]

            x_stream = self.input_projs[name](x_stream)

            x_stream = x_stream.permute(0, 3, 1, 2)

            sgr_mat = self.sgrs[name]()

            feat = x_stream
            for i, layer in enumerate(self.encoders[name]):
                feat = layer(feat, sgr=sgr_mat if i == 0 else None)

            output_lengths.append(feat.size(2))

            B, C, T, N = feat.shape
            feat_reshaped = feat.permute(0, 2, 3, 1).reshape(B * T, N, C)
            feat_reshaped = self.context_encoders[name](feat_reshaped)
            feat = feat_reshaped.reshape(B, T, N, C).permute(0, 3, 1, 2)

            logits = self.heads[name](feat)
            stream_logits[name] = logits
            stream_features.append(feat.mean(dim=-1, keepdim=True))

        fused_feat = torch.cat(stream_features, dim=-1).mean(dim=-1)
        fuse_logits = self.fuse_head(fused_feat.unsqueeze(-1))
        stream_logits['fuse'] = fuse_logits
        stream_logits['output_lengths'] = output_lengths

        return stream_logits

    def get_output_length(self, input_length):
        T = input_length
        for _ in range(3):
            T = (T - 1) // 2 + 1
        return T
