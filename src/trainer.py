import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import wandb
from src.utils.metrics import calculate_wer

try:
    from pyctcdecode import build_ctc_decoder
    CTC_AVAILABLE = True
except ImportError:
    CTC_AVAILABLE = False
    build_ctc_decoder = None


class SLRTrainer:
    def __init__(self, model, train_loader, dev_loader, augmentor, config, gloss_dict):
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.augmentor = augmentor
        self.config = config
        self.device = config['device']
        self.gloss_dict = gloss_dict
        self.id2gloss = {v: k for k, v in self.gloss_dict.items()}

        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')
        if CTC_AVAILABLE:
            self.beam_decoder = build_ctc_decoder(
                list(self.gloss_dict.keys()),
                model_path=None,
                alpha=0, beta=0,
                cutoff_top_n=40,
                cutoff_prob=1.0,
                beam_width=10,
                num_processes=4,
                blank_id=0,
                log_probs_input=True
            )
        else:
            self.beam_decoder = None
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=1e-2
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            total_steps=config['epochs'] * len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        wandb.init(project="Light-MSKA-SLR", config=config)

    def decode_beam(self, logits):
        if self.beam_decoder is None:
            return self.decode_greedy(logits)
        log_probs = F.log_softmax(logits, dim=-1)
        beam_results, beam_scores, timesteps, out_lens = self.beam_decoder.decode(log_probs)
        batch_predictions = []
        for i in range(beam_results.size(0)):
            best_beam = beam_results[i][0][:out_lens[i][0]]
            batch_predictions.append(best_beam.tolist())
        return batch_predictions

    def compute_loss(self, outputs, targets, input_lengths, target_lengths, is_training=True):
        total_ctc_loss = 0
        all_logits = []
        stream_names = ['body', 'left_hand', 'right_hand', 'mouth', 'face', 'fuse']

        for idx, name in enumerate(stream_names):
            logits = outputs[name]
            logits = torch.clamp(logits, min=-100, max=100)
            all_logits.append(logits)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            curr_input_lengths = torch.tensor(
                [min(self.model.get_output_length(l.item()), logits.size(1)) for l in input_lengths],
                dtype=torch.long, device=log_probs.device
            )
            loss = self.ctc_loss(log_probs, targets, curr_input_lengths, target_lengths)
            total_ctc_loss += loss

        if not is_training:
            return total_ctc_loss

        with torch.no_grad():
            avg_probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits]).mean(dim=0)

        total_distill_loss = 0
        for logits in all_logits:
            log_probs = F.log_softmax(logits, dim=-1)
            total_distill_loss += self.distill_loss(log_probs, avg_probs)

        return total_ctc_loss + total_distill_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]")

        for batch in pbar:
            kpts, labels, in_lens, tgt_lens = batch
            kpts, scale = self.augmentor(kpts, apply_aug=True)
            in_lens = (in_lens.float() * scale).long()

            kpts = kpts.to(self.device).float()
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(kpts)
            loss = self.compute_loss(outputs, labels, in_lens, tgt_lens, is_training=True)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            wandb.log({"step_train_loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_loss, "lr": self.optimizer.param_groups[0]['lr']})
        return avg_loss

    def decode_greedy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        max_indices = torch.argmax(probs, dim=-1)

        batch_predictions = []
        for i in range(max_indices.size(0)):
            preds = max_indices[i].cpu().numpy()
            decoded = []
            prev_idx = -1
            for idx in preds:
                if idx != 0 and idx != prev_idx:
                    decoded.append(int(idx))
                prev_idx = idx
            batch_predictions.append(decoded)
        return batch_predictions

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with open("val_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} EPOCH {epoch} {'='*20}\n")

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dev_loader, desc="[Validating]")):
                    kpts, labels, in_lens, tgt_lens = batch
                    kpts, _ = self.augmentor(kpts, apply_aug=False)
                    kpts, labels = kpts.to(self.device).float(), labels.to(self.device)

                    outputs = self.model(kpts)
                    try:
                        val_loss += self.compute_loss(outputs, labels, in_lens, tgt_lens, is_training=False).item()
                    except Exception as e:
                        print(f"Val batch {batch_idx} failed: {e}")
                        continue

                    preds = self.decode_greedy(outputs['fuse'])

                    for i in range(labels.size(0)):
                        target = labels[i][:tgt_lens[i]].cpu().numpy().tolist()
                        all_preds.append(preds[i])
                        all_targets.append(target)

                        if batch_idx < 2 and i < 3:
                            t_words = [self.id2gloss.get(x, f"ID:{x}") for x in target if x > 8]
                            p_words = [self.id2gloss.get(x, f"ID:{x}") for x in preds[i] if x > 8]
                            f.write(f"Gốc: {' '.join(t_words)} | IDs: {target}\n")
                            f.write(f"Đoán: {' '.join(p_words)} | IDs: {preds[i]}\n")
                            f.write("-" * 10 + "\n")

        avg_val_loss = val_loss / max(len(self.dev_loader), 1)
        current_wer = calculate_wer(all_preds, all_targets)

        wandb.log({"val_loss": avg_val_loss, "val_wer": current_wer * 100})
        print(f" -> Validation WER: {current_wer*100:.2f}%")
        return avg_val_loss

    def save_checkpoint(self, epoch, path="weights/best_model.pth"):
        os.makedirs("weights", exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict()}, path)
