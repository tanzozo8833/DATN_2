import yaml
import gzip
import pickle
from dataset.CoordinateDataset import CoordinateDataset
from modelling.Tokenizer import GlossTokenizer
from torch.utils.data import DataLoader
from dataset.CoordinateDataset import collate_fn

# 1. Load Config
with open("configs/s2g_coords.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("--- KIỂM TRA BƯỚC 3 ---")

# 2. Init Tokenizer
tokenizer = GlossTokenizer(cfg['data']['gloss_dict'])
print(f"1. Vocab size: {tokenizer.vocab_size}")

# 3. Load Raw Data
with gzip.open(cfg['data']['train_pkl'], 'rb') as f:
    raw_data = pickle.load(f)

# 4. Init Dataset & Dataloader
dataset = CoordinateDataset(raw_data, cfg['data']['coords_root'], tokenizer)
# Batch size lấy từ config
loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], 
                    shuffle=True, collate_fn=collate_fn)

# 5. Thử lấy 1 Batch thực tế
try:
    batch = next(iter(loader))
    print(f"2. Load Batch thành công!")
    print(f"   - Hình dạng Input Batch (B, T, D): {batch['input_data'].shape}")
    print(f"   - Hình dạng Label Batch (B, L): {batch['label'].shape}")
    
    # Thử decode nhãn đầu tiên trong batch
    first_label_ids = batch['label'][0].tolist()
    print(f"   - Giải mã nhãn mẫu: {tokenizer.decode(first_label_ids)}")
    
    print("\n>>> KẾT QUẢ: HOÀN HẢO! Dữ liệu và Nhãn đã sẵn sàng để đưa vào máy train.")
except Exception as e:
    print(f"LỖI: {e}")