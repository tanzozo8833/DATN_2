import yaml
import torch
import pickle
import os
from torch.utils.data import DataLoader
from src.datasets.slr_dataset import SLRDataset, slr_collate_fn
from src.models.light_mska import LightMSKA
from src.utils.augmentation import Augmentor
from src.trainer import SLRTrainer
import wandb

def main():
    # 1. Cấu hình hệ thống
    # Lưu ý: Đảm bảo các đường dẫn này khớp với cấu trúc thư mục của bạn
    config_path = "configs/base_config.yaml"
    dict_path = "data/processed/gloss2ids.pkl"
    train_data_path = "data/processed/train_77.pkl"
    dev_data_path = "data/processed/dev_77.pkl"
    
    train_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'epochs': 150,
        'batch_size': 16,
        'project_name': 'Light-MSKA-SLR'
    }

    # 2. Tải từ điển và xác định số lượng lớp (Gloss classes)
    if not os.path.exists(dict_path):
        print(f"[!] Lỗi: Không tìm thấy từ điển tại {dict_path}")
        return

    with open(dict_path, 'rb') as f:
        gloss_dict = pickle.load(f)
    num_classes = len(gloss_dict)
    print(f"[*] Hệ thống nhận diện: {num_classes} từ ký hiệu.")

    # 3. Khởi tạo Dataset & DataLoader
    # Sử dụng collate_fn để xử lý đệm (padding) cho các video có độ dài khác nhau
    train_dataset = SLRDataset(train_data_path, dict_path, phase='train')
    dev_dataset = SLRDataset(dev_data_path, dict_path, phase='dev')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=True, 
        collate_fn=slr_collate_fn,
        num_workers=4,
        pin_memory=True if train_params['device'] == 'cuda' else False
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=False, 
        collate_fn=slr_collate_fn,
        num_workers=2
    )

    # 4. Khởi tạo Mô hình Light-MSKA (5 luồng chuyên biệt)[cite: 2]
    model = LightMSKA(config_path, num_classes=num_classes)
    
    # 5. Khởi tạo Augmentor (Chuẩn hóa và Tăng cường dữ liệu)[cite: 2]
    # Chỉ áp dụng Rotation và Temporal Resampling để tránh overfitting[cite: 2]
    augmentor = Augmentor(rotation_range=0.2, temporal_range=(0.5, 1.5))
    
    # 6. Khởi tạo Trainer (CTC Loss + Self-Distillation)[cite: 2]
    trainer = SLRTrainer(model, train_loader, dev_loader, augmentor, train_params, gloss_dict)

    print(f"[*] Bắt đầu huấn luyện trên: {train_params['device']}")
    best_val_loss = float('inf')

    for epoch in range(1, train_params['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate(epoch)
        
        print(f"--- Kết quả Epoch {epoch} ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, path="weights/light_mska_best.pth")
            print(f"[!] Checkpoint tốt nhất được lưu.")
    
    # Kết thúc wandb session
    wandb.finish()

if __name__ == "__main__":
    main()