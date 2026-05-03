import torch
import torch.nn as nn
import yaml
from src.models.lightweight_head import LightweightHead
from src.models.modules.spatial_attention import SpatialAttention

class StreamEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)
        self.attn1 = SpatialAttention(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU(0.1)
        self.attn2 = SpatialAttention(256)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.1)
        self.attn3 = SpatialAttention(256)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.attn1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.attn2(self.relu4(self.bn4(self.conv4(x))))
        x = self.attn3(self.relu5(self.bn5(self.conv5(x))))
        return x

class LightMSKA(nn.Module):
    def __init__(self, config_path, num_classes, embed_dim=128):
        super().__init__()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.streams_cfg = self.config['data']['streams']
        self.num_classes = num_classes
        self.stream_names = ['body', 'left_hand', 'right_hand', 'mouth', 'face']

        self.input_projs = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        for name in self.stream_names:
            num_points = len(self.streams_cfg[name]['indices'])

            self.input_projs[name] = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.LeakyReLU(0.1)
            )

            self.encoders[name] = StreamEncoder(embed_dim)

            self.heads[name] = LightweightHead(256, num_classes)

        self.fuse_head = LightweightHead(256 * 5, num_classes)

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

            feat = self.encoders[name](x_stream)

            output_lengths.append(feat.size(2))

            feat = feat.mean(dim=-1)

            logits = self.heads[name](feat)
            stream_logits[name] = logits
            stream_features.append(feat)

        fused_feat = torch.cat(stream_features, dim=1)
        output_lengths.append(fused_feat.size(2))
        fuse_logits = self.fuse_head(fused_feat)
        stream_logits['fuse'] = fuse_logits
        stream_logits['output_lengths'] = output_lengths

        return stream_logits

    def get_output_length(self, input_length):
        T = input_length
        for _ in range(3):
            T = (T - 1) // 2 + 1
        return T
