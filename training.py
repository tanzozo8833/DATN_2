import torch
import torch_directml
import yaml
import gzip
import pickle
import os
import wandb
import jiwer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Import từ các module đã tách (Khoa học hơn)
from dataset.CoordinateDataset import CoordinateDataset, collate_fn
from modelling.CoordinateModel import CoordinateTransformer
from modelling.Tokenizer import GlossTokenizer
from utils.recognition import ctc_beam_search_decoder 

def validate(model, val_loader, device, tokenizer):
    """Hàm validate luôn dùng Beam Search để biểu đồ mượt mà"""
    model.eval()
    all_preds, all_gt = [], []
    val_loss = 0
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs = batch['input_data'].to(device)
            labels = batch['label'].to(device)
            input_lens = batch['input_len'].to(device)
            label_lens = batch['label_len'].to(device)

            # Phải truyền input_lens để tạo Padding Mask
            logits = model(inputs, input_lens) 
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, labels, input_lens, label_lens)
            val_loss += loss.item()

            # Luôn dùng Beam Search ở bước Val để lấy WER chuẩn nhất
            logits_transposed = logits.transpose(0, 1) 
            for i in range(logits_transposed.size(0)):
                pred_str = ctc_beam_search_decoder(logits_transposed[i], tokenizer, beam_size=5)
                gt_str = tokenizer.decode(batch['label'][i].numpy())
                all_preds.append(pred_str)
                all_gt.append(gt_str)

    wer = jiwer.wer(all_gt, all_preds)
    return val_loss / len(val_loader), wer

def train():
    # 1. Load Cấu hình
    with open("configs/s2g_coords.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project="SignLanguage_DATN", 
        name="Transformer_V3_MotionFeatures_Augment", 
        config=cfg
    )

    device = torch_directml.device()
    tokenizer = GlossTokenizer(cfg['data']['gloss_dict'])

    # 2. Dataset - Quan trọng: Bật is_train=True để kích hoạt Augmentation
    print("Nạp dữ liệu Train (có Augmentation)...")
    with gzip.open(cfg['data']['train_pkl'], 'rb') as f:
        train_data = pickle.load(f)
    train_dataset = CoordinateDataset(train_data, cfg['data']['coords_root'], tokenizer, is_train=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], 
        shuffle=True, collate_fn=collate_fn, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    print("Nạp dữ liệu Validation...")
    with gzip.open(cfg['data']['dev_pkl'], 'rb') as f:
        dev_data = pickle.load(f)
    val_dataset = CoordinateDataset(dev_data, cfg['data']['coords_root'], tokenizer, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, collate_fn=collate_fn)

    # 3. Model - Nhớ chỉnh input_dim trong YAML thành 6636 (2212 * 3)
    model = CoordinateTransformer(
        input_dim=cfg['model']['input_dim'], 
        num_classes=cfg['model']['num_classes'],
        num_layers=6,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # 4. Loop
    best_wer = float('inf')
    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            inputs = batch['input_data'].to(device)
            labels = batch['label'].to(device)
            input_lens = batch['input_len'].to(device)
            label_lens = batch['label_len'].to(device)

            optimizer.zero_grad()
            
            # Truyền input_lens để dùng Padding Mask trong Transformer
            logits = model(inputs, input_lens) 
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            loss = criterion(log_probs, labels, input_lens, label_lens)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDATION ---
        avg_train_loss = train_loss / len(train_loader)
        # Validate luôn dùng Beam Search
        avg_val_loss, current_wer = validate(model, val_loader, device, tokenizer)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n[Epoch {epoch+1}] WER: {current_wer:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "wer": current_wer,
            "lr": current_lr,
            "epoch": epoch + 1
        })

        if current_wer < best_wer:
            best_wer = current_wer
            torch.save(model.state_dict(), f"{cfg['training']['save_path']}/best_model_v3.pt")
            print(f"--- LƯU MODEL TỐT NHẤT (WER: {best_wer:.4f}) ---")

    wandb.finish()

if __name__ == "__main__":
    train()