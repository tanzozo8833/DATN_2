import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CoordinateDataset(Dataset):
    def __init__(self, pkl_data, npy_root, tokenizer, is_train=False):
        self.data = pkl_data
        self.npy_root = npy_root
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def _augment(self, x):
        """
        Strong augmentation cho sign language coordinates
        Chỉ augment features, KHÔNG thay đổi semantic của signs
        """
        if not self.is_train:
            return x

        B, T, D = x.shape

        # 1. Scaling (nhẹ hơn, 5%)
        scale = random.uniform(0.95, 1.05)
        x = x * scale

        # 2. Gaussian noise (rất nhẹ)
        noise = torch.randn_like(x) * 0.0005
        x = x + noise

        # 3. Random temporal dropout (5% frames)
        if T > 20 and random.random() < 0.3:
            num_drop = max(1, int(T * 0.05))
            drop_indices = random.sample(range(T), num_drop)
            x[:, drop_indices, :] = 0

        # 4. Small shift (dịch chuyển tọa độ)
        shift = (torch.rand(1, 1, D) - 0.5) * 0.02
        x = x + shift

        # 5. Time warping (thay đổi tốc độ signing) - chỉ cho features
        if T > 30 and random.random() < 0.2:
            warp_factor = random.uniform(0.85, 1.15)
            new_T = int(T * warp_factor)
            if new_T > 10 and new_T < T * 2:
                indices = torch.linspace(0, T - 1, new_T).long().clamp(0, T - 1)
                x = x[:, indices, :]

        return x

    def _extract_motion_features(self, x):
        """
        Feature Engineering nâng cao:
        - Position (tọa độ gốc)
        - Velocity (vận tốc)
        - Acceleration (gia tốc)
        - Jerk (thay đổi gia tốc) - quan trọng cho sign language!
        - Motion magnitude (tốc độ chuyển động tổng)

        Input: (B, T, 2212) - raw coordinates
        Output: (B, T, 6636) - enhanced features (2212 * 3)
        """
        B, T, D = x.shape

        # Position (tọa độ gốc)
        pos = x

        # Velocity: v[t] = x[t] - x[t-1]
        vel = torch.zeros_like(x)
        vel[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        # Acceleration: a[t] = v[t] - v[t-1]
        acc = torch.zeros_like(x)
        acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]

        # Jerk (thay đổi gia tốc) - rất quan trọng cho sign language
        # Jerk = a[t] - a[t-1]
        jerk = torch.zeros_like(x)
        jerk[:, 1:, :] = acc[:, 1:, :] - acc[:, :-1, :]

        # Concatenate all
        features = torch.cat([pos, vel, acc, jerk], dim=-1)  # (B, T, D*4)

        # Chiều dài = 2212 * 4 = 8848
        return features

    def __getitem__(self, index):
        item = self.data[index]
        video_id = item["name"]
        video_npy_path = os.path.join(self.npy_root, video_id + ".npy")

        if os.path.exists(video_npy_path):
            video_data = np.load(video_npy_path)
            video_tensor = torch.from_numpy(video_data).float()
        else:
            # Fallback: create empty tensor
            video_tensor = torch.zeros((item.get("num_frames", 1), 2212))

        # Áp dụng Augmentation nếu là tập Train (TRƯỚC KHI trích xuất features)
        if self.is_train:
            video_tensor = self._augment(video_tensor)

        # Trích xuất motion features (SAU KHI augmentation)
        video_tensor = self._extract_motion_features(video_tensor)

        # Encode gloss labels
        gloss_tokens = self.tokenizer.encode(item["gloss"])

        return {
            "input_data": video_tensor,
            "label": torch.LongTensor(gloss_tokens),
            "input_len": video_tensor.size(0),
            "label_len": len(gloss_tokens),
            "name": video_id,
        }


def collate_fn(batch):
    """
    Collate function cho batch có độ dài khác nhau
    """
    inputs = [item["input_data"] for item in batch]
    labels = [item["label"] for item in batch]
    input_lens = torch.LongTensor([item["input_len"] for item in batch])
    label_lens = torch.LongTensor([item["label_len"] for item in batch])
    names = [item["name"] for item in batch]

    # Pad sequences
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=1)  # 1 = <pad>

    return {
        "input_data": inputs_padded,
        "label": labels_padded,
        "input_len": input_lens,
        "label_len": label_lens,
        "name": names,
    }
