import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import wandb 
from src.utils.metrics import calculate_wer

class SLRTrainer:
    def __init__(self, model, train_loader, dev_loader, augmentor, config, gloss_dict): # Thêm gloss_dict vào đây
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.augmentor = augmentor
        self.config = config
        self.device = config['device']
        self.gloss_dict = gloss_dict # Lưu gloss_dict
        self.id2gloss = {v: k for k, v in self.gloss_dict.items()}

        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])

        wandb.init(project="Light-MSKA-SLR", config=config)

    def compute_loss(self, outputs, targets, input_lengths, target_lengths):
        total_ctc_loss = 0
        all_logits = []
        stream_names = ['body', 'left_hand', 'right_hand', 'mouth', 'face', 'fuse']

        for name in stream_names:
            logits = outputs[name]
            all_logits.append(logits)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            curr_input_lengths = input_lengths // 8 
            loss = self.ctc_loss(log_probs, targets, curr_input_lengths, target_lengths)
            total_ctc_loss += loss

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
            loss = self.compute_loss(outputs, labels, in_lens, tgt_lens)
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            wandb.log({"step_train_loss": loss.item()})
            
        self.scheduler.step()
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

    def validate(self, epoch): # Thêm epoch vào tham số
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        # Mở file ghi log dự đoán
        with open("val_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} EPOCH {epoch} {'='*20}\n")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dev_loader, desc="[Validating]")):
                    kpts, labels, in_lens, tgt_lens = batch
                    kpts, _ = self.augmentor(kpts, apply_aug=False)
                    kpts, labels = kpts.to(self.device).float(), labels.to(self.device)
                    
                    outputs = self.model(kpts)
                    val_loss += self.compute_loss(outputs, labels, in_lens, tgt_lens).item()

                    preds = self.decode_greedy(outputs['fuse'])
                    
                    for i in range(labels.size(0)):
                        target = labels[i][:tgt_lens[i]].cpu().numpy().tolist()
                        all_preds.append(preds[i])
                        all_targets.append(target)
                        
                        # Ghi 3 mẫu đầu tiên của mỗi batch vào file để soi lỗi
                        if batch_idx < 2 and i < 3:
                            t_words = [self.id2gloss.get(x, f"ID:{x}") for x in target if x > 8]
                            p_words = [self.id2gloss.get(x, f"ID:{x}") for x in preds[i] if x > 8]
                            f.write(f"Gốc: {' '.join(t_words)} | IDs: {target}\n")
                            f.write(f"Đoán: {' '.join(p_words)} | IDs: {preds[i]}\n")
                            f.write("-" * 10 + "\n")

        avg_val_loss = val_loss / len(self.dev_loader)
        current_wer = calculate_wer(all_preds, all_targets)
        
        wandb.log({"val_loss": avg_val_loss, "val_wer": current_wer * 100})
        print(f" -> Validation WER: {current_wer*100:.2f}%")
        return avg_val_loss

    def save_checkpoint(self, epoch, path="weights/best_model.pth"):
        os.makedirs("weights", exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict()}, path)