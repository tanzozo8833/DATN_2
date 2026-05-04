import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.utils.metrics import calculate_wer
from src.utils.decode import ctc_greedy_decode, ctc_beam_search_decode


class Trainer:
    """
    Trainer v2 với:
      - Auxiliary CTC supervision per stream (deep supervision)
      - Beam search decode (validation/inference)
      - ReduceLROnPlateau scheduler (theo val WER)
      - Greedy decode khi training để theo dõi nhanh, beam search lúc evaluate
    """

    def __init__(self, model, train_loader, dev_loader, config, ids2gloss):
        self.device = config['device']
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.ids2gloss = ids2gloss

        self.blank_id   = config.get('ctc_blank_id', 0)
        self.clip_grad  = config.get('clip_grad', 5.0)
        self.aux_weight = config.get('aux_loss_weight', 0.2)
        self.beam_size  = config.get('beam_size', 5)
        self.beam_top_k = config.get('beam_top_k', 10)
        self.use_beam_eval = config.get('use_beam_eval', True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # ReduceLROnPlateau theo val WER (mode='min')
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 5),
            min_lr=config.get('min_lr', 1e-6),
        )

        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction='mean',
            zero_infinity=True,
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_main = 0.0
        total_aux  = 0.0
        n_batches  = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Train E{epoch}', leave=False)
        for kpts, labels, input_lens, target_lens in pbar:
            kpts        = kpts.to(self.device)
            labels      = labels.to(self.device)
            input_lens  = input_lens.to(self.device)
            target_lens = target_lens.to(self.device)

            self.optimizer.zero_grad()

            log_probs, _, aux_log_probs = self.model(kpts, input_lens, return_aux=True)

            # Main CTC loss
            main = self._ctc(log_probs, labels, input_lens, target_lens)

            # Auxiliary CTC loss (mean over streams)
            if aux_log_probs:
                aux_losses = [
                    self._ctc(lp, labels, input_lens, target_lens)
                    for lp in aux_log_probs.values()
                ]
                aux = torch.stack(aux_losses).mean()
            else:
                aux = torch.tensor(0.0, device=self.device)

            loss = main + self.aux_weight * aux
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            total_main += main.item()
            total_aux  += aux.item() if isinstance(aux, torch.Tensor) else aux
            pbar.set_postfix(main=f'{main.item():.3f}', aux=f'{aux.item():.3f}')

        return {
            'train_main_loss': total_main / n_batches,
            'train_aux_loss':  total_aux / n_batches,
        }

    def _ctc(self, log_probs, labels, input_lens, target_lens):
        """Helper: log_probs B×T×C → permute to T×B×C for CTCLoss."""
        return self.ctc_loss(
            log_probs.permute(1, 0, 2).contiguous(),
            labels, input_lens, target_lens,
        )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int, decode_method: str = None) -> dict:
        """
        decode_method: 'greedy' | 'beam' | None (auto: beam nếu use_beam_eval)
        """
        self.model.eval()
        if decode_method is None:
            decode_method = 'beam' if self.use_beam_eval else 'greedy'

        total_loss = 0.0
        all_preds, all_refs = [], []

        pbar = tqdm(self.dev_loader, desc=f'Val   E{epoch} [{decode_method}]', leave=False)
        for kpts, labels, input_lens, target_lens in pbar:
            kpts        = kpts.to(self.device)
            labels      = labels.to(self.device)
            input_lens  = input_lens.to(self.device)
            target_lens = target_lens.to(self.device)

            # Eval không cần aux để nhanh hơn
            log_probs, _, _ = self.model(kpts, input_lens, return_aux=False)
            total_loss += self._ctc(log_probs, labels, input_lens, target_lens).item()

            # Decode
            if decode_method == 'beam':
                decoded = ctc_beam_search_decode(
                    log_probs,
                    beam_size=self.beam_size,
                    blank_id=self.blank_id,
                    top_k=self.beam_top_k,
                    input_lengths=input_lens,
                )
            else:
                decoded = ctc_greedy_decode(log_probs, blank_id=self.blank_id)

            for i in range(len(kpts)):
                ref_len = target_lens[i].item()
                all_preds.append(decoded[i])
                all_refs.append(labels[i, :ref_len].tolist())

        val_loss = total_loss / len(self.dev_loader)
        wer = calculate_wer(all_preds, all_refs)
        return {'val_loss': val_loss, 'val_wer': wer}

    # ------------------------------------------------------------------
    # Scheduler step
    # ------------------------------------------------------------------

    def step_scheduler(self, val_metric: float):
        self.scheduler.step(val_metric)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

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
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f'[*] Loaded checkpoint epoch={ckpt.get("epoch")} WER={ckpt.get("wer")}')
        return ckpt.get('epoch', 0)
