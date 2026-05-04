import torch
import torch.nn as nn
import math
import yaml
from src.models.visual_head import VisualHead
from src.models.modules.st_attention_block import STAttentionBlock


class DSTA(nn.Module):
    def __init__(self, num_frame=400, num_channel=3, cfg=None):
        super().__init__()
        self.cfg = cfg
        config = self.cfg['net']
        self.num_frame = num_frame

        param = {
            'num_subset': 2, 'glo_reg_s': True, 'att_s': True,
            'glo_reg_t': False, 'att_t': False, 'use_spatial_att': True,
            'use_temporal_att': False, 'use_pet': False, 'use_pes': True,
            'attentiondrop': 0.1,
        }

        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, config[0][0], 1),
            nn.BatchNorm2d(config[0][0]),
            nn.LeakyReLU(0.1),
        )
        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, config[0][0], 1),
            nn.BatchNorm2d(config[0][0]),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, config[0][0], 1),
            nn.BatchNorm2d(config[0][0]),
            nn.LeakyReLU(0.1),
        )
        self.mouth_input_map = nn.Sequential(
            nn.Conv2d(num_channel, config[0][0], 1),
            nn.BatchNorm2d(config[0][0]),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, config[0][0], 1),
            nn.BatchNorm2d(config[0][0]),
            nn.LeakyReLU(0.1),
        )

        num_node_map = {'body': 9, 'left_hand': 21, 'right_hand': 21, 'mouth': 12, 'face': 14}
        stream_map = {
            'body': 'body_graph_layers',
            'left_hand': 'left_graph_layers',
            'right_hand': 'right_graph_layers',
            'mouth': 'mouth_graph_layers',
            'face': 'face_graph_layers',
        }
        for stream_name, attr_name in stream_map.items():
            layers = nn.ModuleList()
            nf = num_frame
            num_node = num_node_map[stream_name]
            for in_channels, out_channels, inter_channels, t_kernel, stride in config:
                layers.append(STAttentionBlock(
                    in_channels, out_channels, inter_channels,
                    num_node=num_node, num_frame=nf,
                    t_kernel=t_kernel, stride=stride, **param
                ))
                nf = int(nf / stride + 0.5)
            setattr(self, attr_name, layers)

    def forward(self, x, stream_name):
        input_maps = {
            'body': self.body_input_map,
            'left_hand': self.left_input_map,
            'right_hand': self.right_input_map,
            'mouth': self.mouth_input_map,
            'face': self.face_input_map,
        }
        graph_layers = {
            'body': self.body_graph_layers,
            'left_hand': self.left_graph_layers,
            'right_hand': self.right_graph_layers,
            'mouth': self.mouth_graph_layers,
            'face': self.face_graph_layers,
        }

        feat = input_maps[stream_name](x)
        layers = graph_layers[stream_name]
        for layer in layers:
            feat = layer(feat)

        feat = feat.permute(0, 2, 1, 3).contiguous()
        feat = feat.mean(3)
        return feat


class LightMSKA(nn.Module):
    def __init__(self, config_path, num_classes):
        super().__init__()

        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.num_classes = num_classes
        self.stream_names = ['body', 'left_hand', 'right_hand', 'mouth', 'face']

        self.dsta = DSTA(num_frame=5000, num_channel=3, cfg=self.cfg['DSTA-Net'])

        head_cfg = {
            'body': 'body_visual_head',
            'left_hand': 'left_visual_head',
            'right_hand': 'right_visual_head',
        }
        self.heads = nn.ModuleDict()
        for name in self.stream_names:
            if name in head_cfg:
                cfg_key = head_cfg[name]
                self.heads[name] = VisualHead(cls_num=num_classes, **self.cfg['DSTA-Net'][cfg_key])
            else:
                self.heads[name] = VisualHead(
                    cls_num=num_classes,
                    input_size=256, hidden_size=512, ff_size=2048,
                    pe=True, ff_kernelsize=[3,3],
                )

        fuse_input_size = 256 * 5
        self.fuse_head = VisualHead(
            cls_num=num_classes,
            input_size=fuse_input_size, hidden_size=512, ff_size=2048,
            pe=True, ff_kernelsize=[3,3],
        )

    def forward(self, x):
        batch_size, t, n_total, c = x.shape
        stream_outputs = {}
        stream_features = []

        indices_map = {
            'body': (0, 9),
            'left_hand': (9, 30),
            'right_hand': (30, 51),
            'mouth': (51, 63),
            'face': (63, 77),
        }

        for name in self.stream_names:
            start, end = indices_map[name]
            x_stream = x[:, :, start:end, :]
            x_stream = x_stream.permute(0, 3, 1, 2)

            feat = self.dsta(x_stream, name)
            stream_features.append(feat)

            mask = torch.ones(batch_size, feat.size(1), device=feat.device)
            valid_len_in = torch.tensor([feat.size(1)] * batch_size, device=feat.device)
            head_output = self.heads[name](feat, mask, valid_len_in)
            stream_outputs[name] = head_output['gloss_logits']

        fused_feat = torch.cat(stream_features, dim=-1)
        mask = torch.ones(batch_size, fused_feat.size(1), device=fused_feat.device)
        valid_len_in = torch.tensor([fused_feat.size(1)] * batch_size, device=fused_feat.device)
        fuse_output = self.fuse_head(fused_feat, mask, valid_len_in)

        stream_outputs['fuse'] = fuse_output['gloss_logits']
        stream_outputs['output_lengths'] = [fused_feat.size(1)] * 6
        return stream_outputs

    def get_output_length(self, input_length):
        T = input_length
        for _ in range(3):
            T = (T - 1) // 2 + 1
        return max(T, 1)
