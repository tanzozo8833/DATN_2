import os
import pandas as pd
from tqdm import tqdm

# Cấu hình đường dẫn
coords_root = r"C:\Users\TanHD6\Documents\DATN\phoenix-coords"
expected_rows = 553
splits = ['train', 'dev', 'test']
error_log_path = "csv_error_log.txt"

all_problematic_files = []

print("--- ĐANG BẮT ĐẦU QUÉT DỮ LIỆU ---")

for split in splits:
    split_path = os.path.join(coords_root, split)
    if not os.path.exists(split_path):
        print(f"Bỏ qua {split}: Không tìm thấy thư mục.")
        continue

    # Lấy danh sách tất cả các folder mẫu (video) trong split
    video_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    
    # Thanh tiến trình cho từng split
    for video_dir in tqdm(video_folders, desc=f"{split:5}", unit="video"):
        video_full_path = os.path.join(split_path, video_dir)
        
        # Tìm các file csv trong folder video này
        csv_files = [f for f in os.listdir(video_full_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(video_full_path, csv_file)
            try:
                # Đọc file (sử dụng usecols để tăng tốc độ nếu chỉ cần đếm dòng)
                # Hoặc dùng len(pd.read_csv) như cũ
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                if row_count != expected_rows:
                    all_problematic_files.append({
                        'split': split,
                        'video': video_dir,
                        'file': csv_file,
                        'rows': row_count,
                        'issue': 'Dư dòng' if row_count > expected_rows else 'Thiếu dòng'
                    })
            except Exception as e:
                all_problematic_files.append({
                    'split': split,
                    'video': video_dir,
                    'file': csv_file,
                    'rows': 0,
                    'issue': f'Lỗi đọc: {str(e)}'
                })

# Ghi kết quả ra file
with open(error_log_path, "w", encoding="utf-8") as f:
    f.write(f"BÁO CÁO KIỂM TRA TỌA ĐỘ CSV\n")
    f.write(f"Tổng số file lỗi: {len(all_problematic_files)}\n")
    f.write("="*80 + "\n")
    f.write(f"{'Split':<10} | {'Video Folder':<50} | {'File':<20} | {'Rows':<6} | {'Issue'}\n")
    f.write("-" * 110 + "\n")
    
    for item in all_problematic_files:
        f.write(f"{item['split']:<10} | {item['video']:<50} | {item['file']:<20} | {item['rows']:<6} | {item['issue']}\n")

print("\n" + "="*50)
print(f">>> HOÀN TẤT! Phát hiện {len(all_problematic_files)} file csv có kích thước khác thường.")
print(f">>> Danh sách chi tiết lưu tại: {error_log_path}")