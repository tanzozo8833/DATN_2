import yaml
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.dataset import slr_collate_fn
from src.models.slr_model import SLRModel
from src.utils.metrics import calculate_wer, SPECIAL_IDS
from src.utils.decode import ctc_greedy_decode, ctc_beam_search_decode


UNK_ID = 3   # khớp với SPECIAL_IDS trong metrics


class TestDataset(Dataset):
    """
    Dataset cho test/eval — chấp nhận gloss ở cả dạng:
      - List[int]            (đã encode)
      - List[str]            (chưa encode, dùng gloss2ids để map)
      - str  ('a b c')       (chưa encode, split rồi map)
    """

    def __init__(self, data_path: str, gloss2ids: dict):
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        self.samples = list(raw.values())
        self.gloss2ids = gloss2ids

    def __len__(self):
        return len(self.samples)

    def _encode(self, gloss):
        if isinstance(gloss, str):
            tokens = gloss.split()
        else:
            tokens = list(gloss)

        if len(tokens) == 0:
            return []
        if isinstance(tokens[0], int):
            return tokens

        # Phoenix gloss text trong pkl là UPPERCASE, vocab là lowercase
        return [self.gloss2ids.get(t.lower(), UNK_ID) for t in tokens]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        keypoints = sample['keypoint'].float()
        ids = self._encode(sample['gloss'])
        label = torch.tensor(ids, dtype=torch.long)
        return keypoints, label


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> str:
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device


def ids_to_glosses(ids: list[int], ids2gloss: dict) -> list[str]:
    return [ids2gloss.get(i, f'<UNK:{i}>') for i in ids if i not in SPECIAL_IDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--checkpoint', default='weights/best.pth')
    parser.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override batch_size từ config')
    parser.add_argument('--show_samples', type=int, default=10,
                        help='số mẫu in ra để xem prediction vs ref')
    parser.add_argument('--save_preds', type=str, default=None,
                        help='nếu set, ghi predictions ra file txt')
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam'],
                        help='greedy hoặc prefix beam search')
    parser.add_argument('--beam_size', type=int, default=10,
                        help='số beam (chỉ áp dụng khi --decode beam)')
    parser.add_argument('--beam_top_k', type=int, default=15,
                        help='top-k char/frame pruning trong beam search')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = resolve_device(cfg['training']['device'])
    print(f'[*] Device     : {device}')
    print(f'[*] Checkpoint : {args.checkpoint}')
    print(f'[*] Split      : {args.split}')
    print(f'[*] Decode     : {args.decode}'
          + (f' (beam={args.beam_size}, top_k={args.beam_top_k})'
             if args.decode == 'beam' else ''))

    # Vocabulary
    with open(cfg['data']['dict_path'], 'rb') as f:
        gloss2ids = pickle.load(f)
    ids2gloss = {v: k for k, v in gloss2ids.items()}
    num_classes = len(gloss2ids)
    print(f'[*] Vocab size: {num_classes}')

    # Dataset
    split_path = cfg['data'][f'{args.split}_path']
    ds = TestDataset(split_path, gloss2ids=gloss2ids)
    bs = args.batch_size or cfg['training']['batch_size']
    loader = DataLoader(
        ds, batch_size=bs, shuffle=False,
        collate_fn=slr_collate_fn,
        num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=(device == 'cuda'),
    )
    print(f'[*] {args.split.capitalize()}: {len(ds)} samples')

    # Model
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
        use_velocity=mcfg.get('use_velocity', True),
        use_aux=mcfg.get('use_aux', True),
    ).to(device)
    print(f'[*] Model params: {model.count_parameters():,}')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[*] Loaded ckpt epoch={ckpt.get("epoch", "?")} '
          f'train_wer={ckpt.get("wer", "N/A")}')

    # Evaluate
    blank_id = cfg['training'].get('ctc_blank_id', 0)
    ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

    model.eval()
    total_loss = 0.0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for kpts, labels, input_lens, target_lens in tqdm(loader, desc=f'Eval {args.split}'):
            kpts        = kpts.to(device)
            labels      = labels.to(device)
            input_lens  = input_lens.to(device)
            target_lens = target_lens.to(device)

            log_probs, _ = model(kpts, input_lens)
            loss = ctc_loss(
                log_probs.permute(1, 0, 2).contiguous(),
                labels, input_lens, target_lens,
            )
            total_loss += loss.item()

            if args.decode == 'beam':
                decoded = ctc_beam_search_decode(
                    log_probs,
                    beam_size=args.beam_size,
                    blank_id=blank_id,
                    top_k=args.beam_top_k,
                    input_lengths=input_lens,
                )
            else:
                decoded = ctc_greedy_decode(log_probs, blank_id=blank_id)
            for i in range(len(kpts)):
                ref_len = target_lens[i].item()
                all_preds.append(decoded[i])
                all_refs.append(labels[i, :ref_len].tolist())

    avg_loss = total_loss / len(loader)
    wer = calculate_wer(all_preds, all_refs)

    print('\n' + '=' * 60)
    print(f'  {args.split.upper():4s} loss : {avg_loss:.4f}')
    print(f'  {args.split.upper():4s} WER  : {wer:.4f}  ({wer*100:.2f}%)')
    print('=' * 60)

    # Show some samples
    if args.show_samples > 0:
        print(f'\n[*] {args.show_samples} sample predictions:')
        for i in range(min(args.show_samples, len(all_preds))):
            ref_str  = ' '.join(ids_to_glosses(all_refs[i],  ids2gloss))
            pred_str = ' '.join(ids_to_glosses(all_preds[i], ids2gloss))
            print(f'  [{i:3d}] REF : {ref_str}')
            print(f'        PRED: {pred_str}')

    # Save predictions
    if args.save_preds:
        with open(args.save_preds, 'w', encoding='utf-8') as f:
            f.write(f'# {args.split} | WER {wer:.4f} | loss {avg_loss:.4f}\n')
            for i, (pred, ref) in enumerate(zip(all_preds, all_refs)):
                ref_str  = ' '.join(ids_to_glosses(ref,  ids2gloss))
                pred_str = ' '.join(ids_to_glosses(pred, ids2gloss))
                f.write(f'[{i}]\nREF : {ref_str}\nPRED: {pred_str}\n\n')
        print(f'\n[*] Predictions saved to {args.save_preds}')


if __name__ == '__main__':
    main()
