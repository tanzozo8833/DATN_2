import torch
import torch_directml
import yaml
import gzip
import pickle
import os
import wandb
import jiwer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Import từ các module đã tách
from dataset.CoordinateDataset import CoordinateDataset, collate_fn
from modelling.SpatialTemporalModel import SpatialTemporalModel
from modelling.Tokenizer import GlossTokenizer
from utils.recognition import ctc_beam_search_decoder


def validate(model, val_loader, device, tokenizer):
    """Validation với prediction logging để debug"""
    model.eval()
    all_preds, all_gt = [], []
    val_loss = 0
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs = batch["input_data"].to(device)
            labels = batch["label"].to(device)
            input_lens = batch["input_len"].to(device)
            label_lens = batch["label_len"].to(device)

            logits = model(inputs, input_lens)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, labels, input_lens, label_lens)
            val_loss += loss.item()

            # Use greedy decoder for faster validation
            logits_transposed = logits.transpose(0, 1)
            for i in range(logits_transposed.size(0)):
                pred_str = ctc_beam_search_decoder(
                    logits_transposed[i], tokenizer, beam_size=10
                )
                gt_str = tokenizer.decode(batch["label"][i].numpy())
                all_preds.append(pred_str)
                all_gt.append(gt_str)

    # Log sample predictions cho debug
    print("\n=== Sample Predictions (Debug) ===")
    num_samples = min(5, len(all_preds))
    for i in range(num_samples):
        gt = all_gt[i][:80]
        pred = all_preds[i][:80]
        print(f"[{i}] GT: {gt}")
        print(f"[{i}] PD: {pred}")
        print("---")
    print("==================================\n")

    wer = jiwer.wer(all_gt, all_preds)
    return val_loss / len(val_loader), wer


def train():
    # 1. Load Cấu hình
    with open("configs/s2g_coords.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project="SignLanguage_DATN",
        name="SpatialTemporal_CNN_LSTM_BeamSearch",
        config=cfg,
    )

    device = torch_directml.device()
    tokenizer = GlossTokenizer(cfg["data"]["gloss_dict"])

    # 2. Dataset
    print("Nạp dữ liệu Train...")
    with gzip.open(cfg["data"]["train_pkl"], "rb") as f:
        train_data = pickle.load(f)
    train_dataset = CoordinateDataset(
        train_data, cfg["data"]["coords_root"], tokenizer, is_train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Nạp dữ liệu Validation...")
    with gzip.open(cfg["data"]["dev_pkl"], "rb") as f:
        dev_data = pickle.load(f)
    val_dataset = CoordinateDataset(
        dev_data, cfg["data"]["coords_root"], tokenizer, is_train=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 3. Model mới - SpatialTemporalModel
    model = SpatialTemporalModel(
        input_dim=cfg["model"]["input_dim"],  # 8848 (2212 * 4)
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
        num_lstm_layers=cfg["model"].get("num_lstm_layers", 3),
        dropout=cfg["model"].get("dropout", 0.2),
        use_downsampling=cfg["model"].get("use_downsampling", True),
        downsample_stride=cfg["model"].get("downsample_stride", 2),
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=0.01
    )

    # OneCycleLR - tốt hơn ReduceLROnPlateau
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["training"]["learning_rate"],
        epochs=cfg["training"]["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # 4. Training Loop
    best_wer = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            inputs = batch["input_data"].to(device)
            labels = batch["label"].to(device)
            input_lens = batch["input_len"].to(device)
            label_lens = batch["label_len"].to(device)

            optimizer.zero_grad()

            logits = model(inputs, input_lens)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            loss = criterion(log_probs, labels, input_lens, label_lens)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDATION ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, current_wer = validate(model, val_loader, device, tokenizer)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n[Epoch {epoch + 1}] WER: {current_wer:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}"
        )

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "wer": current_wer,
                "lr": current_lr,
                "epoch": epoch + 1,
            }
        )

        if current_wer < best_wer:
            best_wer = current_wer
            torch.save(
                model.state_dict(), f"{cfg['training']['save_path']}/best_model_st.pt"
            )
            print(f"--- LƯU MODEL TỐT NHẤT (WER: {best_wer:.4f}) ---")

    wandb.finish()
    print(f"\nHOÀN TẤT! Best WER: {best_wer:.4f}")


if __name__ == "__main__":
    train()
