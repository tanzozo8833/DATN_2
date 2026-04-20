import pickle
import gzip  # Thêm thư viện này
import torch
import os
import sys

# Đảm bảo Python tìm thấy folder dataset
sys.path.append(os.getcwd())

from dataset.CoordinateDataset import CoordinateDataset

# 1. Giả lập một Tokenizer đơn giản
class SimpleTokenizer:
    def encode(self, text): 
        # Giả lập trả về danh sách ID cho chuỗi text
        return [10, 20, 30]

# 2. Đường dẫn đúng trên máy của bạn
pkl_path = r"C:\Users\TanHD6\Documents\DATN\data\phoenix-2014t\phoenix-2014t_cleaned.train"
coords_root = r"C:\Users\TanHD6\Documents\DATN\phoenix-coords"

# 3. Load dữ liệu (Sử dụng gzip.open thay vì open thông thường)
print(f"Đang nạp file: {pkl_path}...")
try:
    with gzip.open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Nạp thành công {len(raw_data)} mẫu dữ liệu.")
except Exception as e:
    print(f"Lỗi khi nạp file pkl: {e}")
    raw_data = []

# 4. Khởi tạo Dataset và chạy thử
if raw_data:
    dataset = CoordinateDataset(raw_data, coords_root, SimpleTokenizer())

    try:
        # Thử lấy mẫu đầu tiên
        sample = dataset[0]
        print("\n--- KIỂM TRA THÀNH CÔNG ---")
        print(f"Video Name: {sample['name']}")
        print(f"Hình dạng Tensor: {sample['input_data'].shape}") # Kỳ vọng: (T, 2212)
        print(f"Độ dài video (frames): {sample['input_len']}")
        print(f"Nhãn Gloss (ID): {sample['label']}")
    except Exception as e:
        print(f"\n--- LỖI XỬ LÝ DỮ LIỆU: {str(e)} ---")
        print("Gợi ý: Kiểm tra xem folder video tương ứng có nằm trong 'phoenix-coords' chưa.")