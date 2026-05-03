import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import yaml

class SignPreprocessor:
    def __init__(self, config_path, dict_path):
        # 1. Tải cấu hình và từ điển
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(dict_path, 'rb') as f:
            self.gloss_dict = pickle.load(f)
        
        # Lấy danh sách indices cho 5 luồng (tổng 77 điểm)
        self.streams = self.config['data']['streams']
        self.all_indices = []
        for s in ['body', 'left_hand', 'right_hand', 'mouth', 'face']:
            self.all_indices.extend(self.streams[s]['indices'])

    def process_gloss(self, gloss_text):
        # 1. Chuyển nhãn gốc về chữ thường (.lower()) và tách từ
        words = gloss_text.lower().strip().split()
        label_ids = []
        
        for word in words:
            # 2. Tra cứu trong từ điển đã có (hiện đang là chữ thường)
            if word in self.gloss_dict:
                idx = self.gloss_dict[word]
                # CHỈ lấy ID thực tế (từ 9 trở đi)
                if idx >= 9: 
                    label_ids.append(idx)
            else:
                # Nếu vẫn không thấy, in ra để kiểm tra xem có ký tự lạ không
                # print(f"Cảnh báo: Từ '{word}' không có trong từ điển.")
                pass
                    
        return label_ids

    def run(self, raw_data_path, save_path):
        """
        Đọc dữ liệu thô (ví dụ file .pkl chứa 133/543 điểm) và trích xuất 77 điểm.
        """
        if not os.path.exists(raw_data_path):
            print(f"[!] Không tìm thấy dữ liệu thô tại {raw_data_path}")
            return

        with open(raw_data_path, 'rb') as f:
            raw_data = pickle.load(f)

        processed_data = {}
        print(f"[*] Đang xử lý: {raw_data_path}")

        for video_id, info in tqdm(raw_data.items()):
            # A. Trích xuất Keypoints (giả định shape ban đầu là [T, N, 3])
            kpts = info['keypoint'] # Ví dụ [100, 543, 3]
            if isinstance(kpts, np.ndarray):
                kpts = torch.from_numpy(kpts)
            
            # Chỉ lấy 77 điểm theo đúng thứ tự trong config
            selected_kpts = kpts[:, self.all_indices, :] # [T, 77, 3]

            # B. Xử lý nhãn Gloss
            label_ids = self.process_gloss(info['gloss'])

            # C. Chỉ lưu nếu video có nhãn hợp lệ (không trống sau khi lọc)
            if len(label_ids) > 0:
                processed_data[video_id] = {
                    'keypoint': selected_kpts,
                    'gloss': label_ids
                }

        # D. Lưu file pkl mới
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"[v] Đã lưu {len(processed_data)} mẫu vào {save_path}")

if __name__ == "__main__":
    # Cấu hình đường dẫn
    CONFIG = "configs/base_config.yaml"
    DICT = "data/processed/gloss2ids.pkl"
    
    preprocessor = SignPreprocessor(CONFIG, DICT)
    
    # Chạy cho tập Train và Dev
    preprocessor.run("data/raw/Phoenix-2014T.train", "data/processed/train_77.pkl")
    preprocessor.run("data/raw/Phoenix-2014T.dev", "data/processed/dev_77.pkl")