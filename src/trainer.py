import os
import copy
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.utils.metrics import calculate_wer


class _EMA:
    """
    Polyak averaging — duy trì 1 bản model có weights = trung bình mượt
    qua nhiều mini-batch.
        ema_t = decay · ema_{t-1} + (1 - decay) · model_t
    BN running stats (buffers) copy thẳng từ model.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ep, p in zip(self.ema_model.parameters(), model.parameters()):
            ep.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for eb, b in zip(self.ema_model.buffers(), model.buffers()):
            eb.copy_(b)


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

        self.blank_id   = config.get('ctc_blank_id', 0)
        self.clip_grad  = config.get('clip_grad', 5.0)
        # Per-aux weights: dict {key → weight}. Aux key nào không có trong dict
        # thì coi weight = 0 (an toàn khi config thiếu hoặc model trả thừa key).
        self.aux_weights = config.get('aux_weights', {})
        # Label smoothing cho CTC theo Liu et al. NeurIPS 2018 — KL(uniform || p)
        # đẩy distribution mượt hơn, chống over-confident peaks.
        # 0.0 = tắt; 0.1 là giá trị thường dùng cho CTC.
        self.label_smoothing = config.get('label_smoothing', 0.0)

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

        # EMA — eval bằng EMA model thay vì model gốc (gain ~0.5-1% WER free).
        # Decay 0.0 = tắt EMA hoàn toàn.
        ema_decay = config.get('ema_decay', 0.0)
        self.use_ema = ema_decay > 0.0
        self.ema = _EMA(self.model, decay=ema_decay) if self.use_ema else None

        self.best_wer = float('inf')

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
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

            out = self.model(kpts, input_lens, return_aux=True)
            if len(out) == 3:
                log_probs, _, aux = out
            else:
                log_probs, _ = out
                aux = {}

            main_loss = self._ctc(log_probs, labels, input_lens, target_lens)
            # Label smoothing chỉ áp dụng trên main head (aux đã regularize sẵn).
            if self.label_smoothing > 0.0:
                ls = self._smooth_loss(log_probs)
                main_loss = (1.0 - self.label_smoothing) * main_loss + self.label_smoothing * ls

            if aux:
                aux_total = torch.tensor(0.0, device=self.device)
                for k, lp in aux.items():
                    w = self.aux_weights.get(k, 0.0)
                    if w > 0:
                        aux_total = aux_total + w * self._ctc(
                            lp, labels, input_lens, target_lens,
                        )
            else:
                aux_total = torch.tensor(0.0, device=self.device)

            loss = main_loss + aux_total
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.scheduler.step()
            if self.use_ema:
                self.ema.update(self.model)

            total_loss += loss.item()
            total_main += main_loss.item()
            total_aux  += aux_total.item()
            pbar.set_postfix(
                main=f'{main_loss.item():.3f}',
                aux=f'{aux_total.item():.3f}',
            )

        # Store breakdown để main.py log
        self.last_train_main = total_main / n_batches
        self.last_train_aux  = total_aux  / n_batches
        return total_loss / n_batches

    def _ctc(self, log_probs, labels, input_lens, target_lens):
        """log_probs B×T×C → permute T×B×C cho CTCLoss."""
        return self.ctc_loss(
            log_probs.permute(1, 0, 2).contiguous(),
            labels, input_lens, target_lens,
        )

    def _smooth_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization tương đương KL(uniform || p_θ) (bỏ const).
        Tối thiểu hóa = đẩy distribution về uniform → chống over-confident.
        Liu et al. NeurIPS 2018: 'CTC with Maximum Entropy Regularization'.
        """
        # log_probs: B × T × C  → mean của (-log p) trên all (B, T, C)
        return -log_probs.mean()

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int):
        # Eval bằng EMA model nếu bật, fallback model gốc
        eval_model = self.ema.ema_model if self.use_ema else self.model
        eval_model.eval()
        total_loss = 0.0
        all_preds = []
        all_refs  = []

        pbar = tqdm(self.dev_loader, desc=f'Val   E{epoch}', leave=False)
        for kpts, labels, input_lens, target_lens in pbar:
            kpts        = kpts.to(self.device)
            labels      = labels.to(self.device)
            input_lens  = input_lens.to(self.device)
            target_lens = target_lens.to(self.device)

            log_probs, _ = eval_model(kpts, input_lens)
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
        # Khi bật EMA, model_state_dict = EMA weights (dùng cho test.py).
        # student_state_dict = weights gốc để resume train nếu cần.
        if self.use_ema:
            torch.save({
                'epoch': epoch,
                'wer': wer,
                'model_state_dict': self.ema.ema_model.state_dict(),
                'student_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        else:
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
