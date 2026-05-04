"""
5-Stream Lightweight CSLR Model — Phương án B (conservative)

Đây là kiến trúc v1 baseline + 1 cải tiến đã verify rủi ro thấp:
  - Velocity features (concat Δx, Δy → input mỗi joint = 5 channels)

KHÔNG có:
  - Per-stream/Global normalize  (đã chứng minh hại)
  - Multi-scale TCN              (chưa verify, giữ single kernel)
  - Aux CTC heads                (chưa verify, có thể nhiễu gradient)

Pipeline mỗi stream:
    Input → concat Velocity → Linear → DepthwiseSepConv × 2 → BiGRU
                                                                  ↓
                                                      Gated Fusion 5-stream
                                                                  ↓
                                                       BiGRU refine + CTC head

Stream layout trong 77-point array:
    pose       : [0 : 9]   (9 joints)
    left_hand  : [9 : 30]  (21 joints)
    right_hand : [30: 51]  (21 joints)
    mouth      : [51: 63]  (12 joints)
    face       : [63: 77]  (14 joints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


STREAMS = [
    ('pose',       slice(0,  9),  9),
    ('left_hand',  slice(9,  30), 21),
    ('right_hand', slice(30, 51), 21),
    ('mouth',      slice(51, 63), 12),
    ('face',       slice(63, 77), 14),
]


# ---------------------------------------------------------------------------
# Velocity feature  — chỉ là Δ giữa frame liên tiếp, không có learnable params
# ---------------------------------------------------------------------------

def compute_velocity(x: torch.Tensor) -> torch.Tensor:
    """
    x: B × T × J × 3   (raw keypoint, channels = x, y, conf)
    Trả về Δxy: B × T × J × 2   (frame đầu tiên = 0)
    """
    v = torch.zeros_like(x[..., :2])
    v[:, 1:] = x[:, 1:, :, :2] - x[:, :-1, :, :2]
    return v


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSepConv1d(nn.Module):
    """Depthwise separable Conv1d — nhẹ hơn full Conv1d (~9x ít params)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw   = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pw   = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):                                  # x: B × C × T
        return F.gelu(self.norm(self.pw(self.dw(x))))


class TCNBlock(nn.Module):
    """1 block TCN: DSConv + residual."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = DepthwiseSepConv1d(channels, kernel_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # B × C × T
        return self.drop(self.conv(x) + x)                  # residual


# ---------------------------------------------------------------------------
# Per-stream encoder
# ---------------------------------------------------------------------------

class StreamEncoder(nn.Module):
    """
    Pipeline: concat Velocity → Linear → TCN×N → BiGRU
    Input/Output: B × T × J × 3   →   B × T × (gru_hidden × 2)
    """

    def __init__(
        self,
        n_joints: int,
        embed_dim: int,
        tcn_channels: int,
        tcn_layers: int,
        gru_hidden: int,
        dropout: float,
        kernel_size: int = 3,
    ):
        super().__init__()

        # Channels per joint sau khi concat velocity: x, y, conf, vx, vy = 5
        input_dim = n_joints * 5

        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.adapt = nn.Conv1d(embed_dim, tcn_channels, 1) if embed_dim != tcn_channels else nn.Identity()

        self.tcn = nn.Sequential(*[
            TCNBlock(tcn_channels, kernel_size=kernel_size, dropout=dropout)
            for _ in range(tcn_layers)
        ])

        self.gru = nn.GRU(
            tcn_channels, gru_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.out_dim = gru_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B × T × J × 3
        v = compute_velocity(x)                             # B × T × J × 2
        x = torch.cat([x, v], dim=-1)                        # B × T × J × 5

        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)                           # B × T × (J*5)
        x = self.proj(x)                                     # B × T × embed_dim

        x = x.permute(0, 2, 1)                              # B × embed_dim × T
        x = self.adapt(x)                                    # B × tcn_channels × T
        x = self.tcn(x)                                      # B × tcn_channels × T
        x = x.permute(0, 2, 1)                              # B × T × tcn_channels

        x, _ = self.gru(x)                                  # B × T × (gru_hidden*2)
        return x


# ---------------------------------------------------------------------------
# Gated fusion (modality attention)
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    def __init__(self, n_streams: int, stream_dim: int, dropout: float):
        super().__init__()
        total = n_streams * stream_dim
        self.gate = nn.Sequential(
            nn.Linear(total, total // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total // 2, n_streams),
        )
        self.norm = nn.LayerNorm(stream_dim)

    def forward(self, streams):
        stacked = torch.stack(streams, dim=2)               # B × T × S × D
        B, T, S, D = stacked.shape

        concat = stacked.reshape(B, T, S * D)
        alpha  = F.softmax(self.gate(concat), dim=-1)       # B × T × S
        fused  = (alpha.unsqueeze(-1) * stacked).sum(dim=2)
        return self.norm(fused), alpha


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SLRModel(nn.Module):
    """
    5-Stream CSLR Model — Phương án B.

    Forward returns:
        log_probs:  B × T × num_classes   — main output cho CTC + decode
        attn_w:     B × T × 5             — fusion attention weights
        aux_logits: {}                    — empty (giữ tương thích với trainer)
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        tcn_channels: int = 128,
        tcn_layers: int = 2,
        kernel_size: int = 3,
        gru_hidden: int = 64,
        refine_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        stream_out_dim = gru_hidden * 2     # 128

        self.encoders = nn.ModuleDict({
            name: StreamEncoder(
                n_joints=n_joints,
                embed_dim=embed_dim,
                tcn_channels=tcn_channels,
                tcn_layers=tcn_layers,
                gru_hidden=gru_hidden,
                dropout=dropout,
                kernel_size=kernel_size,
            )
            for name, _, n_joints in STREAMS
        })

        self.fusion = GatedFusion(
            n_streams=len(STREAMS),
            stream_dim=stream_out_dim,
            dropout=dropout,
        )

        self.refine_gru = nn.GRU(
            stream_out_dim, refine_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.refine_norm = nn.LayerNorm(refine_hidden * 2)

        self.dropout  = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(refine_hidden * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, input_lengths=None, return_aux: bool = False):
        # 1. Encode từng stream
        stream_features = []
        for name, slc, _ in STREAMS:
            x_s = x[:, :, slc, :]
            h   = self.encoders[name](x_s)
            stream_features.append(h)

        # 2. Gated fusion
        fused, attn_w = self.fusion(stream_features)

        # 3. Temporal refinement
        refined, _ = self.refine_gru(fused)
        refined    = self.refine_norm(refined)

        # 4. CTC head
        logits    = self.ctc_head(self.dropout(refined))
        log_probs = F.log_softmax(logits, dim=-1)

        # Trả empty dict cho aux để giữ tương thích với trainer
        return log_probs, attn_w, {}

    def count_parameters(self, only_inference: bool = False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
