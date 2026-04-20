import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Cấu hình
CSV_ROOT = r"C:\Users\TanHD6\Documents\DATN\phoenix-coords_fixed_final"
NPY_ROOT = r"C:\Users\TanHD6\Documents\DATN\phoenix-npy_2"
EXPECTED_DIM = 2212
EXPECTED_ROWS = 553

print("--- BẮT ĐẦU CHUYỂN ĐỔI CSV -> NPY ---")

for split in ['train', 'dev', 'test']:
    src_split_path = os.path.join(CSV_ROOT, split)
    dst_split_path = os.path.join(NPY_ROOT, split)
    
    if not os.path.exists(src_split_path): continue
    
    # Lấy danh sách video folders
    video_folders = [d for d in os.listdir(src_split_path) if os.path.isdir(os.path.join(src_split_path, d))]
    
    for video_dir in tqdm(video_folders, desc=f"Converting {split}"):
        src_video_path = os.path.join(src_split_path, video_dir)
        dst_video_file = os.path.join(dst_split_path, video_dir + ".npy")
        
        os.makedirs(dst_split_path, exist_ok=True)
        
        # Nếu đã convert rồi thì bỏ qua
        if os.path.exists(dst_video_file): continue
        
        csv_files = sorted([f for f in os.listdir(src_video_path) if f.endswith('.csv')])
        video_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(src_video_path, csv_file)).iloc[:EXPECTED_ROWS]
                feat = df[['x', 'y', 'z', 'visibility']].fillna(0).values.flatten()
                
                # Ép về đúng 2212 chiều
                if feat.shape[0] != EXPECTED_DIM:
                    fixed = np.zeros(EXPECTED_DIM)
                    fixed[:min(feat.shape[0], EXPECTED_DIM)] = feat[:min(feat.shape[0], EXPECTED_DIM)]
                    feat = fixed
                video_data.append(feat)
            except:
                video_data.append(np.zeros(EXPECTED_DIM))
        
        # Lưu thành 1 file .npy duy nhất cho cả video
        np.save(dst_video_file, np.array(video_data).astype(np.float32))

print(f"\n>>> HOÀN TẤT! Dữ liệu nhị phân lưu tại: {NPY_ROOT}")