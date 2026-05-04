import yaml
import pickle
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SLRDataset, slr_collate_fn
from src.models.slr_model import SLRModel
from src.utils.augmentation import Augmentor
from src.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> str:
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device


def main():
    cfg = load_config('configs/config.yaml')

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    device = resolve_device(cfg['training']['device'])
    print(f'[*] Device: {device}')

    # ------------------------------------------------------------------ #
    # Vocabulary
    # ------------------------------------------------------------------ #
    dict_path = cfg['data']['dict_path']
    with open(dict_path, 'rb') as f:
        gloss2ids = pickle.load(f)
    ids2gloss = {v: k for k, v in gloss2ids.items()}
    num_classes = len(gloss2ids)
    print(f'[*] Vocab size: {num_classes}')

    # ------------------------------------------------------------------ #
    # Dataset & DataLoader  (KHÔNG sửa, giữ augmentor cũ)
    # ------------------------------------------------------------------ #
    augmentor = Augmentor(
        rotation_range=0.15,
        temporal_range=(0.7, 1.3),
        noise_std=0.005,
        drop_prob=0.05,
    )

    train_ds = SLRDataset(cfg['data']['train_path'], augmentor=augmentor, phase='train')
    dev_ds   = SLRDataset(cfg['data']['dev_path'],   augmentor=None,      phase='dev')

    num_workers = cfg['data'].get('num_workers', 4)
    bs = cfg['training']['batch_size']

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=slr_collate_fn,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=bs, shuffle=False,
        collate_fn=slr_collate_fn,
        num_workers=max(num_workers // 2, 1),
    )
    print(f'[*] Train: {len(train_ds)} samples | Dev: {len(dev_ds)} samples')

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    mcfg = cfg['model']
    model = SLRModel(
        num_classes=num_classes,
        embed_dim=mcfg['embed_dim'],
        tcn_channels=mcfg['tcn_channels'],
        tcn_layers=mcfg['tcn_layers'],
        kernel_size=mcfg.get('kernel_size', 3),
        gru_hidden=mcfg['gru_hidden'],
        refine_hidden=mcfg['refine_hidden'],
        dropout=mcfg['dropout'],
    )
    print(f'[*] Model params: {model.count_parameters():,}')

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    tcfg = cfg['training']
    train_config = {
        'device':           device,
        'lr':               tcfg['lr'],
        'epochs':           tcfg['epochs'],
        'weight_decay':     tcfg['weight_decay'],
        'ctc_blank_id':     tcfg['ctc_blank_id'],
        'clip_grad':        tcfg['clip_grad'],
        'aux_loss_weight':  tcfg['aux_loss_weight'],
        'use_beam_eval':    tcfg['use_beam_eval'],
        'beam_size':        tcfg['beam_size'],
        'beam_top_k':       tcfg['beam_top_k'],
        'lr_factor':        tcfg['lr_factor'],
        'lr_patience':      tcfg['lr_patience'],
        'min_lr':           tcfg['min_lr'],
    }

    trainer = Trainer(model, train_loader, dev_loader, train_config, ids2gloss)

    # ------------------------------------------------------------------ #
    # Training loop với early stopping
    # ------------------------------------------------------------------ #
    save_dir         = tcfg['save_dir']
    early_stop_pat   = tcfg['early_stop_patience']
    early_stop_min   = tcfg['early_stop_min_epochs']

    best_wer = float('inf')
    patience_counter = 0

    for epoch in range(1, tcfg['epochs'] + 1):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics   = trainer.validate(epoch)

        # Step LR scheduler theo val_wer
        trainer.step_scheduler(val_metrics['val_wer'])

        print(
            f'Epoch {epoch:3d}/{tcfg["epochs"]} | '
            f'lr {trainer.get_lr():.2e} | '
            f'main {train_metrics["train_main_loss"]:.4f} | '
            f'aux {train_metrics["train_aux_loss"]:.4f} | '
            f'val_loss {val_metrics["val_loss"]:.4f} | '
            f'WER {val_metrics["val_wer"]:.4f}'
        )

        # Track best & early stop
        if val_metrics['val_wer'] < best_wer:
            best_wer = val_metrics['val_wer']
            patience_counter = 0
            trainer.save_checkpoint(epoch, best_wer, path=f'{save_dir}best.pth')
            print(f'  => Best WER {best_wer:.4f} — checkpoint saved.')
        else:
            patience_counter += 1
            if epoch >= early_stop_min and patience_counter >= early_stop_pat:
                print(f'  => Early stopping (patience={early_stop_pat} reached). Best WER: {best_wer:.4f}')
                break

        if epoch % 20 == 0:
            trainer.save_checkpoint(epoch, val_metrics['val_wer'], path=f'{save_dir}epoch_{epoch:03d}.pth')

    print(f'\n[*] Training complete. Best WER: {best_wer:.4f}')


if __name__ == '__main__':
    main()
