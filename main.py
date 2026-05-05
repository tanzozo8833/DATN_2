import yaml
import pickle
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SLRDataset, slr_collate_fn
from src.models.slr_model import SLRModel
from src.utils.augmentation import Augmentor
from src.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
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
    # Dataset & DataLoader
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
        tcn_kernel=mcfg['tcn_kernel'],
        tcn_layers=mcfg['tcn_layers'],
        gru_hidden=mcfg['gru_hidden'],
        refine_hidden=mcfg['refine_hidden'],
        dropout=mcfg['dropout'],
    )
    n_params = model.count_parameters()
    print(f'[*] Model params: {n_params:,}')

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    tcfg = cfg['training']
    train_config = {
        'device':          device,
        'lr':              tcfg['lr'],
        'epochs':          tcfg['epochs'],
        'weight_decay':    tcfg['weight_decay'],
        'ctc_blank_id':    tcfg['ctc_blank_id'],
        'clip_grad':       tcfg['clip_grad'],
        'aux_loss_weight': tcfg.get('aux_loss_weight', 1.0),
    }

    trainer = Trainer(model, train_loader, dev_loader, train_config, ids2gloss)

    # Optional: resume from checkpoint
    # trainer.load_checkpoint('weights/best.pth')

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    save_dir = tcfg['save_dir']
    best_wer = float('inf')

    for epoch in range(1, tcfg['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss, wer = trainer.validate(epoch)

        main_l = getattr(trainer, 'last_train_main', train_loss)
        aux_l  = getattr(trainer, 'last_train_aux', 0.0)
        print(
            f'Epoch {epoch:3d}/{tcfg["epochs"]} | '
            f'main {main_l:.4f} | aux {aux_l:.4f} | '
            f'Val Loss: {val_loss:.4f} | WER: {wer:.4f}'
        )

        if wer < best_wer:
            best_wer = wer
            trainer.save_checkpoint(epoch, wer, path=f'{save_dir}best.pth')
            print(f'  => Best WER {wer:.4f} — checkpoint saved.')

        # Periodic checkpoint mỗi 10 epoch
        if epoch % 10 == 0:
            trainer.save_checkpoint(epoch, wer, path=f'{save_dir}epoch_{epoch:03d}.pth')

    print(f'\n[*] Training complete. Best WER: {best_wer:.4f}')


if __name__ == '__main__':
    main()
