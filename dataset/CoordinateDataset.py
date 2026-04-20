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
        """Data Augmentation cho tọa độ"""
        # 1. Thêm nhiễu trắng (Gaussian Noise)
        noise = torch.randn_like(x) * 0.002
        x = x + noise
        
        # 2. Scaling (Phóng to/thu nhỏ hệ tọa độ)
        scale = random.uniform(0.95, 1.05)
        x = x * scale
        
        # 3. Shift (Dịch chuyển nhẹ)
        shift = (torch.rand(x.size(-1)) - 0.5) * 0.05
        x = x + shift.to(x.device)
        return x

    def _extract_motion_features(self, x):
        """Feature Engineering: Thêm Vận tốc và Gia tốc"""
        # x shape: (Time, 2212)
        
        # Vận tốc: v_t = x_t - x_{t-1}
        v = torch.zeros_like(x)
        v[1:] = x[1:] - x[:-1]
        
        # Gia tốc: a_t = v_t - v_{t-1}
        a = torch.zeros_like(v)
        a[1:] = v[1:] - v[:-1]
        
        # Nối lại: (Time, 2212 * 3 = 6636)
        return torch.cat([x, v, a], dim=-1)

    def __getitem__(self, index):
        item = self.data[index]
        video_id = item['name']
        video_npy_path = os.path.join(self.npy_root, video_id + ".npy")
        
        if os.path.exists(video_npy_path):
            video_data = np.load(video_npy_path)
            video_tensor = torch.from_numpy(video_data).float()
        else:
            video_tensor = torch.zeros((item.get('num_frames', 1), 2212))

        # Áp dụng Augmentation nếu là tập Train
        if self.is_train:
            video_tensor = self._augment(video_tensor)

        # Trích xuất thêm đặc trưng chuyển động
        video_tensor = self._extract_motion_features(video_tensor)

        gloss_tokens = self.tokenizer.encode(item['gloss'])
        return {
            'input_data': video_tensor,
            'label': torch.LongTensor(gloss_tokens),
            'input_len': video_tensor.size(0),
            'label_len': len(gloss_tokens),
            'name': video_id
        }

def collate_fn(batch):
    """
    Hàm gom nhóm các video có độ dài khác nhau vào cùng 1 Batch
    """
    inputs = [item['input_data'] for item in batch]
    labels = [item['label'] for item in batch]
    input_lens = torch.LongTensor([item['input_len'] for item in batch])
    label_lens = torch.LongTensor([item['label_len'] for item in batch])
    names = [item['name'] for item in batch]
    
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=1)
    
    return {
        'input_data': inputs_padded,
        'label': labels_padded,
        'input_len': input_lens,
        'label_len': label_lens,
        'names': names
    }