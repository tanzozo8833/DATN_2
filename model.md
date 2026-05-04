# Mô tả Model — 5-Stream Lightweight CSLR

Bài toán: **Continuous Sign Language Recognition (CSLR)** — nhận một video ngôn ngữ ký hiệu (đã trích keypoint) và xuất ra chuỗi gloss tương ứng.

Tổng số tham số: **~1.27M** (rất nhẹ).

---

## Tổng quan pipeline

```
Raw video frames (đã có sẵn)
        │
        ▼
[Pre-processing] ───── trích 77 keypoints theo thứ tự cố định
        │
        ▼
Tensor [T, 77, 3]   (T frames, 77 điểm, mỗi điểm có x, y, confidence)
        │
        ▼
[Augmentation] ──── chỉ áp dụng khi train
        │
        ▼
[Tách 5 luồng]    pose / left_hand / right_hand / mouth / face
        │
        ▼
[Per-stream Encoder × 5]   Linear → TCN → BiGRU
        │
        ▼
[Gated Fusion]    attention adaptive trên 5 stream
        │
        ▼
[Temporal Refinement]   BiGRU
        │
        ▼
[CTC Head]    Linear → log_softmax
        │
        ▼
[CTC Decode]    greedy: collapse repeats + remove blank
        │
        ▼
Chuỗi gloss IDs  ────  ['druck', 'tief', 'kommen']
```

---

## Bước 1 — Dữ liệu đầu vào

### Định dạng file pkl

File: `data/processed/{train,dev,test}_77.pkl`

Mỗi sample là một dict:
```python
{
    'keypoint': Tensor[T, 77, 3],   # T = số frame, 77 keypoints, 3 = (x, y, confidence)
    'gloss':    List[int],          # đã được mã hoá thành ID (ví dụ [62, 61, 63])
}
```

### Bộ từ vựng

File: `data/processed/gloss2ids.pkl` — ánh xạ gloss → ID.
- **Tổng cộng 1124 token** (`num_classes = 1124`)
- 5 token đặc biệt:
  - `<s>` = 0 → dùng làm **CTC blank**
  - `<pad>` = 1 → padding cho label
  - `</s>` = 2
  - `<unk>` = 3
  - `<mask>` = 4

### Layout 77 keypoints

Theo `pre_processing.py`, 77 điểm được sắp xếp theo thứ tự **Body → LH → RH → Mouth → Face** (KHÔNG sort theo chỉ số gốc):

| Stream | Slice trong array 77 | Số điểm |
|--------|----------------------|---------|
| pose (body) | `[0:9]` | 9 |
| left_hand | `[9:30]` | 21 |
| right_hand | `[30:51]` | 21 |
| mouth | `[51:63]` | 12 |
| face | `[63:77]` | 14 |
| **Tổng** | | **77** |

---

## Bước 2 — Dataset & Collate

File: [src/data/dataset.py](src/data/dataset.py)

### `SLRDataset.__getitem__`
- Đọc tensor keypoint, ép kiểu `float32`.
- Đọc danh sách ID gloss làm label.
- Nếu là `phase='train'` và có `augmentor` → áp dụng augmentation.

### `slr_collate_fn` (pad batch)
- Pad keypoints: tất cả các sequence trong batch về cùng độ dài `T_max` (pad 0).
- Pad labels: pad bằng ID `1` (`<pad>`) về độ dài `L_max`.
- Trả về thêm `input_lengths` và `target_lengths` — **bắt buộc cho CTCLoss**.

Output collate:
```
padded_kpts:    [B, T_max, 77, 3]
padded_labels:  [B, L_max]
input_lengths:  [B]
target_lengths: [B]
```

---

## Bước 3 — Augmentation (chỉ train)

File: [src/utils/augmentation.py](src/utils/augmentation.py)

Bốn phép biến đổi xác suất (random per sample):

| Augmentation | Xác suất | Mô tả |
|---|---|---|
| Temporal resample | 0.5 | Co/giãn thời gian, rate ∈ [0.7, 1.3] (nội suy linear) |
| Gaussian noise | 0.5 | Thêm nhiễu N(0, 0.005²) vào (x, y), không động vào confidence |
| Random rotation | 0.3 | Xoay (x, y) với góc ∈ [-0.15, 0.15] rad |
| Keypoint dropout | 0.2 | Zero-out ngẫu nhiên một số keypoint (toàn bộ frame) |

---

## Bước 4 — Tách 5 luồng

Trong `SLRModel.forward`, batch keypoint được chia theo 5 slice cố định (`STREAMS` trong [src/models/slr_model.py](src/models/slr_model.py)):

```python
x_pose       = x[:, :, 0:9, :]    # B × T × 9  × 3
x_left_hand  = x[:, :, 9:30, :]   # B × T × 21 × 3
x_right_hand = x[:, :, 30:51, :]  # B × T × 21 × 3
x_mouth      = x[:, :, 51:63, :]  # B × T × 12 × 3
x_face       = x[:, :, 63:77, :]  # B × T × 14 × 3
```

Mỗi luồng đi qua một **encoder riêng biệt** (5 encoder).

---

## Bước 5 — Per-stream Encoder

Class `StreamEncoder` trong [src/models/slr_model.py](src/models/slr_model.py).

Mỗi luồng đi qua các bước sau:

### 5.1. Flatten + Linear projection
```
[B, T, J, 3]  ──flatten──>  [B, T, J×3]  ──Linear──>  [B, T, 128]
                                          + LayerNorm + GELU + Dropout
```

Mục tiêu: ánh xạ tọa độ thô về không gian embedding `embed_dim=128`.

### 5.2. TCN (Temporal Convolutional Network)
```
[B, 128, T]  ──Depthwise Sep Conv1d × 2──>  [B, 128, T]
              kernel=3, residual
```

- Dùng **depthwise separable conv** (depthwise + pointwise) — nhẹ hơn full conv ~8–9 lần.
- BatchNorm + GELU + residual.
- Mục tiêu: học **chuyển động cục bộ theo thời gian** (motion patterns ngắn).

### 5.3. BiGRU
```
[B, T, 128]  ──BiGRU(hidden=64, bidirectional)──>  [B, T, 128]
```

- 1 layer BiGRU, hidden=64 mỗi chiều → output 128.
- Mục tiêu: học **phụ thuộc dài hơn theo thời gian** (long-range dependencies).

**Output của mỗi stream encoder: `[B, T, 128]`**

---

## Bước 6 — Gated Fusion (Modality Attention)

Class `GatedFusion` trong [src/models/slr_model.py](src/models/slr_model.py).

5 stream embeddings được hợp nhất bằng cách học trọng số adaptive theo từng frame:

### 6.1. Stack 5 stream
```
streams = [h_pose, h_left, h_right, h_mouth, h_face]   # 5 × [B, T, 128]
stacked = stack(dim=2)                                  # [B, T, 5, 128]
```

### 6.2. Tính attention weights α_t
```
concat = stacked.reshape(B, T, 5×128)                   # [B, T, 640]
α_t = softmax( MLP(concat) , dim=-1)                    # [B, T, 5]
```

MLP: `Linear(640 → 320) → GELU → Dropout → Linear(320 → 5)`.

### 6.3. Weighted sum
```
fused = Σ_i  α_t[i] × stacked[:, :, i, :]               # [B, T, 128]
fused = LayerNorm(fused)
```

**Ý nghĩa**:
- Khi sign chủ yếu ở tay → α_t cho `left/right_hand` cao.
- Khi mouthing quan trọng → α_t cho `mouth/face` cao.
- `pose` đóng vai trò context, giúp disambiguation.

α_t được trả về làm `attention_weights` — có thể dùng để **visualize**.

---

## Bước 7 — Temporal Refinement

```
[B, T, 128]  ──BiGRU(hidden=128, bidirectional)──>  [B, T, 256]
                                                     + LayerNorm
```

Mục tiêu: học ngữ cảnh **toàn câu** sau khi đã hợp nhất các luồng — boundary giữa các gloss, gloss nào đi trước/sau, độ dài kéo giãn.

---

## Bước 8 — CTC Head

```
[B, T, 256]  ──Dropout──>  ──Linear(256 → 1124)──>  [B, T, 1124]
                                                     ──log_softmax──>  log_probs
```

Output cuối:
- `log_probs`: `[B, T, 1124]` — log-probability cho mỗi gloss tại mỗi frame (kèm blank).
- `attention_weights`: `[B, T, 5]` — trọng số fusion (debug/visualize).

---

## Bước 9 — Loss: CTC

File: [src/trainer.py](src/trainer.py)

Tại sao dùng CTC?
- Trong CSLR, **không có alignment frame-to-gloss** sẵn có (chỉ biết chuỗi gloss tổng).
- CTC tự động marginalize trên mọi alignment hợp lệ.

```python
ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# log_probs: [B, T, C] → permute thành [T, B, C] cho CTCLoss
loss = ctc_loss(
    log_probs.permute(1, 0, 2),
    labels,                # [B, L]
    input_lengths,         # [B]   độ dài T thật của mỗi sequence
    target_lengths,        # [B]   độ dài L thật của mỗi label
)
```

`zero_infinity=True` giúp tránh `nan` khi sequence ngắn hơn label.

---

## Bước 10 — Optimization

| Hyperparameter | Giá trị |
|---|---|
| Optimizer | AdamW |
| Base LR | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | OneCycleLR (cosine, 10% warmup) |
| Gradient clipping | max_norm = 5.0 |
| Batch size | 16 |
| Epochs | 150 |

---

## Bước 11 — Inference (Greedy CTC Decode)

File: [src/trainer.py](src/trainer.py) — `Trainer._greedy_decode`

```python
pred_ids = log_probs.argmax(dim=-1)            # [B, T]

# Với mỗi sequence:
#   1. Bỏ token blank (id = 0)
#   2. Collapse repeats (ababbc → abc)
for tok in seq:
    if tok != blank_id and tok != prev:
        out.append(tok)
    prev = tok
```

Có thể nâng cấp lên **beam search** + **language model** sau này nếu cần.

---

## Bước 12 — Evaluation: WER

File: [src/utils/metrics.py](src/utils/metrics.py)

Word Error Rate trên chuỗi gloss:

```
WER = Σ levenshtein(pred, ref) / Σ len(ref)
```

Lọc bỏ special tokens (id < 5) trước khi tính. **WER càng thấp càng tốt** (best ~0.20 trên Phoenix-2014T).

---

## Tóm tắt số chiều dữ liệu xuyên suốt

```
Input batch          [B, T, 77, 3]
  ↓ tách 5 luồng
pose                 [B, T, 9,  3]
left_hand            [B, T, 21, 3]
right_hand           [B, T, 21, 3]
mouth                [B, T, 12, 3]
face                 [B, T, 14, 3]
  ↓ flatten + Linear
mỗi stream           [B, T, 128]
  ↓ TCN
mỗi stream           [B, T, 128]
  ↓ BiGRU
mỗi stream           [B, T, 128]
  ↓ Gated Fusion
fused                [B, T, 128]   + α [B, T, 5]
  ↓ Refine BiGRU
refined              [B, T, 256]
  ↓ CTC head
log_probs            [B, T, 1124]
  ↓ CTC decode
gloss_ids            [B, ≤T]
```

---

## Phân bổ tham số (~1.27M)

| Block | Số params (xấp xỉ) |
|---|---|
| 5 × StreamEncoder | ~750K |
| Gated Fusion | ~205K |
| Refine BiGRU | ~198K |
| CTC Head (Linear 256→1124) | ~288K |
| **Total** | **~1.27M** |

---

## File cấu hình

[configs/config.yaml](configs/config.yaml) — chứa toàn bộ hyperparameter (đường dẫn data, kiến trúc model, training params). Có thể tinh chỉnh mà không cần sửa code.
