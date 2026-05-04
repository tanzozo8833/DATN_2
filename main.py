import yaml
import torch
import pickle
import os
from torch.utils.data import DataLoader
from src.datasets.slr_dataset import SLRDataset, slr_collate_fn
from src.models.light_mska import LightMSKA
from src.utils.augmentation import Augmentor
from src.trainer import SLRrainer
import wandb


def main():
    config_path = "configs/mska_config.yaml"
    dict_path = "data/processed/gloss2ids.pkl"
    train_data_path = "data/processed/train_77.pkl"
    dev_data_path = "data/processed/dev_77.pkl"

    train_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'epochs': 150,
        'batch_size': 16,
    }

    if not os.path.exists(dict_path):
        print(f"[!] Error: dict not found at {dict_path}")
        return

    with open(dict_path, 'rb') as f:
        gloss_dict = pickle.load(f)
    num_classes = len(gloss_dict)
    print(f"[*] System loaded: {num_classes} classes.")

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

    model = LightMSKA(config_path, num_classes=num_classes)

    augmentor = Augmentor(rotation_range=0.2, temporal_range=(0.5, 1.5))

    trainer = SLRrainer(model, train_loader, dev_loader, augmentor, train_params, gloss_dict)

    print(f"[*] Start training on: {train_params['device']}")
    best_val_loss = float('inf')

    for epoch in range(1, train_params['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate(epoch)

        print(f"--- Epoch {epoch} ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, path="weights/light_mska_best.pth")
            print(f"[!] Best checkpoint saved.")

    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
