import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.utils.metrics import calculate_wer


class Trainer:
    """
    Training loop cho CSLR model với CTC loss.

    - Loss: CTCLoss (zero_infinity=True để tránh nan khi sequence rất ngắn)
    - Optimizer: AdamW + OneCycleLR scheduler
    - Decode: greedy CTC (collapse repeats + remove blank)
    """

    def __init__(self, model, train_loader, dev_loader, config, ids2gloss):
        self.device = config['device']
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.ids2gloss = ids2gloss

        self.blank_id = config.get('ctc_blank_id', 0)
        self.clip_grad = config.get('clip_grad', 5.0)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['epochs'],
            pct_start=0.1,
            anneal_strategy='cos',
        )

        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction='mean',
            zero_infinity=True,
        )

        self.best_wer = float('inf')

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Train E{epoch}', leave=False)
        for kpts, labels, input_lens, target_lens in pbar:
            kpts        = kpts.to(self.device)
            labels      = labels.to(self.device)
            input_lens  = input_lens.to(self.device)
            target_lens = target_lens.to(self.device)

            self.optimizer.zero_grad()

            log_probs, _ = self.model(kpts, input_lens)   # B × T × C
            # CTCLoss yêu cầu T × B × C
            log_probs_ctc = log_probs.permute(1, 0, 2).contiguous()

            loss = self.ctc_loss(log_probs_ctc, labels, input_lens, target_lens)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        return total_loss / n_batches

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_refs  = []

        pbar = tqdm(self.dev_loader, desc=f'Val   E{epoch}', leave=False)
        for kpts, labels, input_lens, target_lens in pbar:
            kpts        = kpts.to(self.device)
            labels      = labels.to(self.device)
            input_lens  = input_lens.to(self.device)
            target_lens = target_lens.to(self.device)

            log_probs, _ = self.model(kpts, input_lens)
            log_probs_ctc = log_probs.permute(1, 0, 2).contiguous()

            loss = self.ctc_loss(log_probs_ctc, labels, input_lens, target_lens)
            total_loss += loss.item()

            # Greedy decode
            decoded = self._greedy_decode(log_probs)   # list[list[int]]

            for i in range(len(kpts)):
                ref_len = target_lens[i].item()
                ref_ids = labels[i, :ref_len].tolist()
                all_preds.append(decoded[i])
                all_refs.append(ref_ids)

        val_loss = total_loss / len(self.dev_loader)
        wer = calculate_wer(all_preds, all_refs)
        return val_loss, wer

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _greedy_decode(self, log_probs: torch.Tensor) -> list[list[int]]:
        """
        Greedy CTC decode: argmax per frame → collapse repeats → remove blank.

        Args:
            log_probs: B × T × C
        Returns:
            list of decoded ID sequences
        """
        pred_ids = log_probs.argmax(dim=-1).cpu().tolist()   # B × T
        results = []
        for seq in pred_ids:
            out, prev = [], None
            for tok in seq:
                if tok != self.blank_id and tok != prev:
                    out.append(tok)
                prev = tok
            results.append(out)
        return results

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, wer: float, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'wer': wer,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f'[*] Loaded checkpoint epoch={ckpt["epoch"]} WER={ckpt.get("wer", "N/A")}')
        return ckpt['epoch']
