import pickle
import gzip
import os
import torch
from dataset.CoordinateDataset import CoordinateDataset

# 1. Giả lập Tokenizer để test
class MockTokenizer:
    def encode(self, text): return [5, 10, 15]

# 2. Định nghĩa đường dẫn (Dựa trên thông tin bạn cung cấp)
pkl_path = r"C:\Users\TanHD6\Documents\DATN\data\phoenix-2014t\phoenix-2014t_cleaned.train"
coords_root = r"C:\Users\TanHD6\Documents\DATN\phoenix-coords"

print("--- BẮT ĐẦU KIỂM TRA BƯỚC 1 ---")

# 3. Mở file .train (dạng nén gzip)
try:
    with gzip.open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"1. Nạp file .train thành công: {len(raw_data)} mẫu.")
    
    # 4. Khởi tạo Dataset
    dataset = CoordinateDataset(raw_data, coords_root, MockTokenizer())
    
    # 5. Thử lấy mẫu đầu tiên
    sample = dataset[0]
    print(f"2. Truy cập mẫu đầu tiên thành công.")
    print(f"   - Tên video: {sample['name']}")
    print(f"   - Hình dạng Tensor: {sample['input_data'].shape}") # Kỳ vọng: (T, 2212)
    print(f"   - Số lượng frame: {sample['input_len']}")
    
    if sample['input_data'].shape[1] == 2212:
        print("\n>>> KẾT QUẢ: ĐÚNG! Dữ liệu đã được phẳng hóa thành 2212 chiều.")
    else:
        print("\n>>> KẾT QUẢ: SAI kích thước đầu vào.")

except Exception as e:
    print(f"LỖI TRONG QUÁ TRÌNH KIỂM TRA: {e}")