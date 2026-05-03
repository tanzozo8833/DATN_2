import os
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SLRDataset(Dataset):
    def __init__(self, data_path, dict_path, phase='train'):
        """
        Khởi tạo Dataset cho Nhận dạng Ngôn ngữ Ký hiệu (SLR).
        
        Args:
            data_path (str): Đường dẫn đến file data processed (VD: test_77.pkl)
            dict_path (str): Đường dẫn đến file gloss2ids.pkl
            phase (str): 'train', 'dev', hoặc 'test'. Dùng để quyết định có bật Augmentation không.
        """
        super().__init__()
        self.phase = phase

        # 1. Load từ điển Gloss -> ID
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Không tìm thấy file từ điển tại: {dict_path}")
        with open(dict_path, 'rb') as f:
            self.gloss2ids = pickle.load(f)
            
        # Xác định ID của token không xác định (Unknown token)
        self.unk_id = self.gloss2ids.get('<unk>', 3)

        # 2. Load dữ liệu Keypoints
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        # Lưu lại danh sách các key để truy xuất theo index
        self.keys = list(self.raw_data.keys())
        print(f"[*] Đã tải {len(self.keys)} mẫu dữ liệu cho tập '{self.phase}'.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        video_id = self.keys[index]
        data = self.raw_data[video_id]

        # 1. Lấy tọa độ (đã là tensor từ bước tiền xử lý)
        kpts = data['keypoint'] 
        
        # 2. Lấy nhãn (BÂY GIỜ ĐÃ LÀ LIST CÁC IDs)
        label_ids = data['gloss'] 
        
        # Bạn có thể xóa bỏ hoàn toàn đoạn .split() và đoạn map gloss2ids cũ ở đây
        # vì label_ids hiện tại đã là [9, 10, 11...] rồi.

        # Chuyển thành LongTensor để truyền vào hàm Loss
        label_tensor = torch.LongTensor(label_ids)
        
        # Trả về dữ liệu (đảm bảo khớp với các biến mà Trainer nhận vào)
        return kpts, label_tensor, len(kpts), len(label_ids)


def slr_collate_fn(batch):
    """
    Hàm gom nhóm (collate) các mẫu thành 1 batch.
    Vì độ dài khung hình (T) của các video khác nhau, ta cần pad (đệm) chúng.
    Đồng thời trả về input_lengths và target_lengths để phục vụ cho CTC Loss.
    """
    keypoints = [item[0] for item in batch]  # List các tensor [T_i, 77, 3]
    labels = [item[1] for item in batch]     # List các tensor [L_i]

    # Tính toán độ dài thực tế của từng sequence trong batch
    input_lengths = torch.tensor([k.size(0) for k in keypoints], dtype=torch.long)
    target_lengths = torch.tensor([l.size(0) for l in labels], dtype=torch.long)

    # Pad các tensor điểm then chốt về cùng độ dài lớn nhất trong batch (T_max)
    # Kết quả: [Batch, T_max, 77, 3]
    padded_keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0.0)

    # Pad nhãn (labels). Giá trị padding_value thường là ID của <pad> (bạn có <pad>: 1)
    # Kết quả: [Batch, L_max]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=1)

    return padded_keypoints, padded_labels, input_lengths, target_lengths