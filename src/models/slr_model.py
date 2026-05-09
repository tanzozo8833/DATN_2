"""
5-Stream Lightweight CSLR Model
Pipeline: Per-stream encoder → Gated Fusion → Temporal Refinement → CTC Head

Stream layout trong 77-point array (theo pre_processing.py):
    pose       : [0 : 9]   (9 joints)
    left_hand  : [9 : 30]  (21 joints)
    right_hand : [30: 51]  (21 joints)
    mouth      : [51: 63]  (12 joints)
    face       : [63: 77]  (14 joints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stream definitions (fixed, matches pre_processing.py ordering)
# ---------------------------------------------------------------------------
STREAMS = [
    ('pose',       slice(0,  9),  9),
    ('left_hand',  slice(9,  30), 21),
    ('right_hand', slice(30, 51), 21),
    ('mouth',      slice(51, 63), 12),
    ('face',       slice(63, 77), 14),
]


# ---------------------------------------------------------------------------
# Velocity feature — không có learnable params, chỉ là Δxy giữa các frame
# ---------------------------------------------------------------------------

def compute_velocity(x: torch.Tensor) -> torch.Tensor:
    """
    x  : B × T × J × 3   (channels = x, y, conf)
    out: B × T × J × 2   (Δx, Δy giữa frame t và t-1; frame 0 = 0)
    """
    v = torch.zeros_like(x[..., :2])
    v[:, 1:] = x[:, 1:, :, :2] - x[:, :-1, :, :2]
    return v


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _DepthwiseSepConv1d(nn.Module):
    """Depthwise separable Conv1d với dilation; nhẹ hơn full conv ~8-9x."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2     # giữ nguyên T
        self.dw = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, groups=channels,
        )
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):          # x: B x C x T
        return F.gelu(self.norm(self.pw(self.dw(x))))


class _TCNBlock(nn.Module):
    """
    Dilated Temporal Convolutional Network block (WaveNet-style).

    Stack N depthwise-separable convs với dilation tăng dần → receptive
    field tăng theo cấp số nhân: với kernel=3 và dilations=[1,2,4]
    thì receptive = 15.

    Residual: GELU( net(x) + skip(x) ).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilations: list,
        dropout: float,
    ):
        super().__init__()
        layers = []
        for i, d in enumerate(dilations):
            ch_in = in_ch if i == 0 else out_ch
            if ch_in != out_ch:
                layers.append(nn.Conv1d(ch_in, out_ch, 1))  # adapt channels
            layers.append(_DepthwiseSepConv1d(out_ch, kernel_size, dilation=d))
            if i < len(dilations) - 1:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):          # x: B x C x T
        return F.gelu(self.net(x) + self.skip(x))


class StreamEncoder(nn.Module):
    """
    Encoder cho một stream keypoint.

    Pipeline:
        [concat velocity →] Flatten → Linear → LayerNorm → TCN → BiGRU

    Channel per joint sau concat velocity = 5  (x, y, conf, Δx, Δy);
    nếu use_velocity=False thì giữ nguyên 3 channel gốc.

    Output: B × T × (gru_hidden × 2)
    """

    def __init__(
        self,
        n_joints: int,
        embed_dim: int,
        tcn_channels: int,
        tcn_kernel: int,
        tcn_dilations: list,
        gru_hidden: int,
        dropout: float,
        use_velocity: bool = True,
    ):
        super().__init__()
        self.use_velocity = use_velocity
        ch_per_joint = 5 if use_velocity else 3
        input_dim = n_joints * ch_per_joint

        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tcn = _TCNBlock(
            embed_dim, tcn_channels, tcn_kernel,
            dilations=tcn_dilations, dropout=dropout,
        )

        self.gru = nn.GRU(
            tcn_channels,
            gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = gru_hidden * 2

    def forward(self, x):          # x: B × T × J × 3
        if self.use_velocity:
            v = compute_velocity(x)            # B × T × J × 2
            x = torch.cat([x, v], dim=-1)      # B × T × J × 5

        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)             # B × T × (J*C)
        x = self.proj(x)                       # B × T × embed_dim
        x = x.permute(0, 2, 1)                 # B × embed_dim × T  (Conv1d)
        x = self.tcn(x)                        # B × tcn_channels × T
        x = x.permute(0, 2, 1)                 # B × T × tcn_channels
        x, _ = self.gru(x)                     # B × T × (gru_hidden*2)
        return x


class GatedFusion(nn.Module):
    """
    Modality-Attention Fusion.

    Tại mỗi thời điểm t, tính trọng số α_t ∈ R^5 bằng softmax(MLP(concat))
    rồi weighted-sum các stream embeddings.

    Input : list of n_streams tensors, mỗi tensor B × T × D
    Output: fused B × T × D,  attention_weights B × T × n_streams
    """

    def __init__(self, n_streams: int, stream_dim: int, dropout: float):
        super().__init__()
        self.n_streams = n_streams
        total_dim = n_streams * stream_dim

        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, n_streams),
        )
        self.norm = nn.LayerNorm(stream_dim)

    def forward(self, streams):
        # streams: list[Tensor B×T×D]
        stacked = torch.stack(streams, dim=2)          # B × T × S × D
        B, T, S, D = stacked.shape

        concat = stacked.reshape(B, T, S * D)          # B × T × (S*D)
        alpha = F.softmax(self.gate(concat), dim=-1)   # B × T × S

        fused = (alpha.unsqueeze(-1) * stacked).sum(dim=2)  # B × T × D
        return self.norm(fused), alpha


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SLRModel(nn.Module):
    """
    Lightweight 5-Stream Continuous Sign Language Recognition Model.

    Architecture:
        1. Per-stream encoder  (Linear → TCN → BiGRU)  × 5
        2. Gated modality fusion
        3. Temporal refinement  (BiGRU)
        4. CTC classification head
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        tcn_channels: int = 128,
        tcn_kernel: int = 3,
        tcn_dilations: list = (1, 2, 4),
        gru_hidden: int = 64,
        refine_hidden: int = 128,
        dropout: float = 0.3,
        use_velocity: bool = True,
        use_aux: bool = True,
    ):
        super().__init__()

        self.use_aux = use_aux

        stream_out_dim = gru_hidden * 2   # 128
        refine_out_dim = refine_hidden * 2   # 256

        # ---------- 1. Per-stream encoders ----------
        self.encoders = nn.ModuleDict({
            name: StreamEncoder(
                n_joints=n_joints,
                embed_dim=embed_dim,
                tcn_channels=tcn_channels,
                tcn_kernel=tcn_kernel,
                tcn_dilations=list(tcn_dilations),
                gru_hidden=gru_hidden,
                dropout=dropout,
                use_velocity=use_velocity,
            )
            for name, _, n_joints in STREAMS
        })

        # ---------- 2. Gated fusion ----------
        self.fusion = GatedFusion(
            n_streams=len(STREAMS),
            stream_dim=stream_out_dim,
            dropout=dropout,
        )

        # ---------- 3. Temporal refinement ----------
        self.refine_gru = nn.GRU(
            stream_out_dim,
            refine_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.refine_norm = nn.LayerNorm(refine_out_dim)

        # ---------- 4. CTC head (shared giữa main và aux nếu use_aux=True) ----------
        self.dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(refine_out_dim, num_classes)

        # ---------- 5. VAC Visual Enhancement (Min et al. 2021) ----------
        # Project visual features (sau fusion) lên cùng dim với refined features,
        # rồi share classifier weights để force visual và sequential representations
        # nằm cùng không gian semantic.
        if use_aux:
            self.visual_proj = nn.Sequential(
                nn.Linear(stream_out_dim, refine_out_dim),
                nn.LayerNorm(refine_out_dim),
            )

        # ---------- 6. Per-stream aux heads cho 3 modality chính (LH > RH > mouth)
        # Mỗi stream tự học gloss prediction → deep supervision, ép encoder mạnh hơn
        # thay vì chỉ dựa vào fusion. Share ctc_head với main.
        self.aux_stream_keys = ['left_hand', 'right_hand', 'mouth']
        if use_aux:
            self.stream_proj = nn.ModuleDict({
                k: nn.Sequential(
                    nn.Linear(stream_out_dim, refine_out_dim),
                    nn.LayerNorm(refine_out_dim),
                )
                for k in self.aux_stream_keys
            })

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
        """
        Args:
            x:              Tensor[B, T, 77, 3]
            input_lengths:  Tensor[B]  (placeholder, chưa dùng)
            return_aux:     nếu True và use_aux=True, trả thêm aux log_probs cho VAC loss

        Returns:
            return_aux=False:  (log_probs, attn_w)
            return_aux=True :  (log_probs, attn_w, aux_dict)
        """
        # 1. Encode từng stream
        stream_features = []
        stream_dict = {}
        for name, slc, _ in STREAMS:
            x_s = x[:, :, slc, :]
            h = self.encoders[name](x_s)
            stream_features.append(h)
            stream_dict[name] = h

        # 2. Gated fusion
        fused, attn_w = self.fusion(stream_features)  # B × T × stream_out_dim

        # 3. Temporal refinement
        refined, _ = self.refine_gru(fused)           # B × T × refine_out_dim
        refined = self.refine_norm(refined)

        # 4. Main CTC head
        logits = self.ctc_head(self.dropout(refined))
        log_probs = F.log_softmax(logits, dim=-1)

        # 5. Aux heads — đều share classifier weights với main head
        if return_aux and self.use_aux:
            aux_dict = {}

            # 5a. VAC Visual Enhancement (fused → ctc_head)
            v_proj = self.visual_proj(fused)
            aux_dict['visual'] = F.log_softmax(
                self.ctc_head(self.dropout(v_proj)), dim=-1,
            )

            # 5b. Per-stream aux cho LH, RH, mouth
            for k in self.aux_stream_keys:
                s_proj = self.stream_proj[k](stream_dict[k])
                aux_dict[k] = F.log_softmax(
                    self.ctc_head(self.dropout(s_proj)), dim=-1,
                )

            return log_probs, attn_w, aux_dict

        return log_probs, attn_w

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
