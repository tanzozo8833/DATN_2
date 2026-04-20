import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConvBlock(nn.Module):
    """Spatial convolution block - trích xuất features từ tọa độ 3D"""

    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SpatialFeatureExtractor(nn.Module):
    """Trích xuất spatial features từ 553 landmarks (face, hands, pose)"""

    def __init__(self, input_dim=6636, hidden_dim=512, dropout=0.2):
        super().__init__()

        # Landmark indices trong MediaPipe:
        # Face: 0-467 (468 landmarks)
        # Left Hand: 468-488 (21 landmarks)
        # Right Hand: 489-509 (21 landmarks)
        # Pose: 510-542 (33 landmarks)

        self.face_dim = 468 * 3
        self.hand_dim = 42 * 3  # 21 landmarks * 3 coords (both hands)
        self.pose_dim = 33 * 3

        # Face branch
        self.face_conv = nn.Sequential(
            SpatialConvBlock(self.face_dim, 256, kernel_size=5, dropout=dropout),
            SpatialConvBlock(256, 128, kernel_size=3, dropout=dropout),
            SpatialConvBlock(128, 64, kernel_size=3, dropout=dropout),
        )

        # Hand branch (left + right combined)
        self.hand_conv = nn.Sequential(
            SpatialConvBlock(self.hand_dim, 256, kernel_size=3, dropout=dropout),
            SpatialConvBlock(256, 128, kernel_size=3, dropout=dropout),
            SpatialConvBlock(128, 64, kernel_size=3, dropout=dropout),
        )

        # Pose branch
        self.pose_conv = nn.Sequential(
            SpatialConvBlock(self.pose_dim, 128, kernel_size=3, dropout=dropout),
            SpatialConvBlock(128, 64, kernel_size=3, dropout=dropout),
        )

        # Fusion layer
        total_features = 64 + 64 + 64  # face + hand + pose
        self.fusion = nn.Sequential(
            nn.Conv1d(total_features, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, T, 6636) - [pos, vel, acc] concatenated
        Returns: (B, T, hidden_dim)
        """
        B, T, _ = x.shape

        # Extract different body parts
        # Face: first 468*3
        x_face = x[:, :, : self.face_dim]  # (B, T, 1404)
        # Hands: next 42*3
        x_hand = x[:, :, self.face_dim : self.face_dim + self.hand_dim]  # (B, T, 126)
        # Pose: remaining
        x_pose = x[:, :, self.face_dim + self.hand_dim :]  # (B, T, 99)

        # Reshape for Conv1d: (B, C, T)
        x_face = x_face.transpose(1, 2)
        x_hand = x_hand.transpose(1, 2)
        x_pose = x_pose.transpose(1, 2)

        # Apply convolutions
        f_face = self.face_conv(x_face)  # (B, 64, T)
        f_hand = self.hand_conv(x_hand)  # (B, 64, T)
        f_pose = self.pose_conv(x_pose)  # (B, 64, T)

        # Concatenate and fuse
        fused = torch.cat([f_face, f_hand, f_pose], dim=1)  # (B, 192, T)
        out = self.fusion(fused)  # (B, hidden_dim, T)

        return out.transpose(1, 2)  # (B, T, hidden_dim)


class TemporalDownsampling(nn.Module):
    """Giảm sequence length bằng strided convolution"""

    def __init__(self, channels, stride=2):
        super().__init__()
        self.pool = nn.Conv1d(
            channels, channels, kernel_size=5, stride=stride, padding=2
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, T//stride, C)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return x


class TemporalEncoder(nn.Module):
    """Bidirectional LSTM + Self-Attention"""

    def __init__(self, hidden_dim=512, num_layers=3, dropout=0.2):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)

        # Self-Attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim)
        x = self.ln(lstm_out)

        # Self-Attention with residual
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn_out  # Residual
        x = self.attn_ln(x)

        return x


class SpatialTemporalModel(nn.Module):
    """
    CNN + BiLSTM + Attention + CTC
    Optimized for sign language recognition from keypoint coordinates
    """

    def __init__(
        self,
        input_dim=6636,
        hidden_dim=512,
        num_classes=1124,
        num_lstm_layers=3,
        dropout=0.2,
        use_downsampling=True,
        downsample_stride=2,
    ):
        super().__init__()

        self.use_downsampling = use_downsampling

        # 1. Spatial Feature Extraction (CNN)
        self.spatial_extractor = SpatialFeatureExtractor(
            input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # 2. Temporal Downsampling
        if use_downsampling:
            self.temporal_down = TemporalDownsampling(
                hidden_dim, stride=downsample_stride
            )

        # 3. Temporal Encoder (BiLSTM + Attention)
        self.temporal_encoder = TemporalEncoder(
            hidden_dim=hidden_dim, num_layers=num_lstm_layers, dropout=dropout
        )

        # 4. Additional Transformer layer for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, input_lens):
        """
        x: (B, T, 6636) - batch of coordinate sequences
        input_lens: (B,) - actual lengths before padding
        Returns: (T, B, num_classes) - logits for CTC
        """
        B, T, _ = x.size()

        # 1. Spatial CNN
        x = self.spatial_extractor(x)  # (B, T, hidden_dim)

        # 2. Create padding mask for attention
        if self.use_downsampling:
            # Adjust lengths after downsampling
            new_lens = (input_lens + 1) // 2
        else:
            new_lens = input_lens

        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(
            B, max_len
        ) >= new_lens.unsqueeze(1)

        # 3. Temporal Downsampling
        if self.use_downsampling:
            x = self.temporal_down(x)  # (B, T//2, hidden_dim)

        # 4. Temporal Encoder (BiLSTM + Attention)
        x = self.temporal_encoder(x, key_padding_mask=mask)

        # 5. Transformer for global context
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 6. Classification
        logits = self.classifier(x)  # (B, T, num_classes)

        # CTC requires: (T, B, C)
        return logits.transpose(0, 1)


def count_parameters(model):
    """Đếm số parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = SpatialTemporalModel(
        input_dim=6636, hidden_dim=512, num_classes=1124, num_lstm_layers=3, dropout=0.2
    )

    # Dummy input
    x = torch.randn(4, 100, 6636)  # (B, T, features)
    lens = torch.tensor([100, 80, 60, 50])

    output = model(x, lens)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (T, B, num_classes)
