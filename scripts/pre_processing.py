import pickle

import os

import torch

import yaml

from tqdm import tqdm



def pre_process_data(config_path, input_pkl, output_pkl):

    with open(config_path, 'r') as f:

        config = yaml.safe_load(f)

    with open(input_pkl, 'rb') as f:

        data = pickle.load(f)

   

    # KHÔNG DÙNG SORTED. Giữ nguyên thứ tự Body -> LH -> RH -> Mouth -> Face

    ordered_indices = []

    for stream_name in ['body', 'left_hand', 'right_hand', 'mouth', 'face']:

        ordered_indices.extend(config['data']['streams'][stream_name]['indices'])



    processed_data = {}

    print(f"--- Đang lọc dữ liệu về {len(ordered_indices)} điểm theo thứ tự YAML ---")

    for key, sample in tqdm(data.items()):

        # Lọc và ép kiểu về float32 để tránh lỗi "double != float" sau này

        kpt = torch.tensor(sample['keypoint'], dtype=torch.float32)

        sample['keypoint'] = kpt[:, ordered_indices, :]

        processed_data[key] = sample



    with open(output_pkl, 'wb') as f:

        pickle.dump(processed_data, f)



if __name__ == "__main__":

    # Ví dụ chạy: python scripts/pre_process.py

    pre_process_data("configs/base_config.yaml", "data/raw/Phoenix-2014T.test", "data/processed/test_77.pkl")