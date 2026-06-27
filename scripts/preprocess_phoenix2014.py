"""
Tiền xử lý RWTH-PHOENIX-Weather 2014 (bản gốc, không có bản dịch).

Đầu vào : data/raw/Phoenix-2014/Phoenix-2014.{train,dev,test}
           data/raw/Phoenix-2014/gloss2ids.pkl
Đầu ra  : data/processed/Phoenix-2014/{train,dev,test}_77.pkl
           data/processed/Phoenix-2014/gloss2ids.pkl   (copy từ raw)

Khác biệt so với Phoenix-2014T:
  - Keypoint shape gốc: [T, 133, 3] (không phải 543)
    => indices trong base_config.yaml (max=132) dùng trực tiếp được.
  - Gloss text & dict keys: UPPERCASE (không cần .lower()).
  - Special tokens: '<s>'=0, '<pad>'=1, '</s>'=2, '<unk>'=3
    => Lọc idx >= 4 (thay vì >= 9 của 2014T).
"""

import os
import shutil
import pickle
import yaml
import torch
import numpy as np
from tqdm import tqdm


SPECIAL_THRESHOLD = 4   # bỏ các token đặc biệt (0-3)


def load_indices(config_path: str) -> list[int]:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    streams = cfg['data']['streams']
    indices = []
    for name in ['body', 'left_hand', 'right_hand', 'mouth', 'face']:
        indices.extend(streams[name]['indices'])
    return indices


def process_split(raw_path: str, save_path: str, gloss2ids: dict, indices: list[int]):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Không tìm thấy: {raw_path}")

    with open(raw_path, 'rb') as f:
        raw_data = pickle.load(f)

    processed = {}
    skipped = 0

    for video_id, info in tqdm(raw_data.items(), desc=os.path.basename(raw_path)):
        kpts = info['keypoint']
        if isinstance(kpts, np.ndarray):
            kpts = torch.from_numpy(kpts).float()
        else:
            kpts = kpts.float()

        # Chọn 77 điểm theo thứ tự Body→LH→RH→Mouth→Face
        kpts_77 = kpts[:, indices, :]   # [T, 77, 3]

        # Xử lý nhãn: gloss là chuỗi UPPERCASE, dict cũng UPPERCASE
        gloss_text = info['gloss']
        if isinstance(gloss_text, str):
            words = gloss_text.strip().split()
        else:
            words = list(gloss_text)

        label_ids = []
        for w in words:
            idx = gloss2ids.get(w)
            if idx is not None and idx >= SPECIAL_THRESHOLD:
                label_ids.append(idx)

        if len(label_ids) == 0:
            skipped += 1
            continue

        processed[video_id] = {
            'keypoint': kpts_77,
            'gloss': label_ids,
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(processed, f)

    print(f"  Saved {len(processed):,} samples  (skipped {skipped}) -> {save_path}")
    return len(processed)


def main():
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG    = os.path.join(BASE_DIR, 'configs', 'base_config.yaml')
    RAW_DIR   = os.path.join(BASE_DIR, 'data', 'raw',       'Phoenix-2014')
    OUT_DIR   = os.path.join(BASE_DIR, 'data', 'processed', 'Phoenix-2014')
    DICT_RAW  = os.path.join(RAW_DIR, 'gloss2ids.pkl')
    DICT_OUT  = os.path.join(OUT_DIR, 'gloss2ids.pkl')

    print("[*] Đọc cấu hình keypoint indices từ", CONFIG)
    indices = load_indices(CONFIG)
    print(f"    Tổng {len(indices)} điểm được chọn (77 điểm)")

    print("[*] Đọc từ điển:", DICT_RAW)
    with open(DICT_RAW, 'rb') as f:
        gloss2ids = pickle.load(f)
    print(f"    Vocab size: {len(gloss2ids)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copy2(DICT_RAW, DICT_OUT)
    print(f"    Copied gloss2ids -> {DICT_OUT}")

    splits = [
        (os.path.join(RAW_DIR, 'Phoenix-2014.train'), os.path.join(OUT_DIR, 'train_77.pkl')),
        (os.path.join(RAW_DIR, 'Phoenix-2014.dev'),   os.path.join(OUT_DIR, 'dev_77.pkl')),
        (os.path.join(RAW_DIR, 'Phoenix-2014.test'),  os.path.join(OUT_DIR, 'test_77.pkl')),
    ]

    total = 0
    for raw_path, out_path in splits:
        n = process_split(raw_path, out_path, gloss2ids, indices)
        total += n

    print(f"\n[v] Hoàn thành. Tổng {total:,} mẫu đã xử lý.")
    print(f"    Output: {OUT_DIR}")


if __name__ == '__main__':
    main()
