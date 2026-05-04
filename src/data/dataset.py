import pickle
import torch
from torch.utils.data import Dataset


class SLRDataset(Dataset):
    """
    Load dữ liệu từ file .pkl đã được pre_processing.

    Mỗi sample trong pkl có dạng:
        {
            'keypoint': Tensor[T, 77, 3],  # T frames, 77 keypoints, (x, y, conf)
            'gloss': List[int],             # chuỗi gloss ID đã mã hoá
        }

    Thứ tự 77 keypoints (theo pre_processing.py):
        Body(0:9) → Left Hand(9:30) → Right Hand(30:51) → Mouth(51:63) → Face(63:77)
    """

    def __init__(self, data_path: str, augmentor=None, phase: str = 'train'):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        self.samples = list(raw.values())
        self.augmentor = augmentor
        self.phase = phase

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        keypoints = sample['keypoint'].float()   # T x 77 x 3
        label = torch.tensor(sample['gloss'], dtype=torch.long)

        if self.augmentor is not None and self.phase == 'train':
            keypoints = self.augmentor(keypoints)

        return keypoints, label


def slr_collate_fn(batch):
    """
    Pad keypoints và labels trong batch về cùng độ dài.

    Returns:
        padded_kpts:    Tensor[B, T_max, 77, 3]
        padded_labels:  Tensor[B, L_max]   (pad bằng 1 = <pad>)
        input_lengths:  Tensor[B]           độ dài thực của mỗi sequence
        target_lengths: Tensor[B]           độ dài thực của mỗi label
    """
    keypoints_list, labels_list = zip(*batch)

    B = len(keypoints_list)
    T_max = max(kp.size(0) for kp in keypoints_list)
    L_max = max(lab.size(0) for lab in labels_list)

    padded_kpts = torch.zeros(B, T_max, 77, 3)
    input_lengths = torch.zeros(B, dtype=torch.long)

    for i, kp in enumerate(keypoints_list):
        T = kp.size(0)
        padded_kpts[i, :T] = kp
        input_lengths[i] = T

    PAD_ID = 1
    padded_labels = torch.full((B, L_max), PAD_ID, dtype=torch.long)
    target_lengths = torch.zeros(B, dtype=torch.long)

    for i, lab in enumerate(labels_list):
        L = lab.size(0)
        padded_labels[i, :L] = lab
        target_lengths[i] = L

    return padded_kpts, padded_labels, input_lengths, target_lengths
