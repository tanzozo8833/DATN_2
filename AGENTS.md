# AGENTS.md

## Project Overview
Vietnamese Sign Language recognition using CTC-based Transformer on PHOENIX-2014T coordinates.

## Key Commands

```bash
# Activate venv first (REQUIRED)
.\venv\Scripts\Activate.ps1

# Training
python training.py

# Preprocessing (CSV -> NPY)
python preprocess_coords.py

# Requires config at: configs/s2g_coords.yaml
```

## Important Patterns

### Config
- All paths in `configs/s2g_coords.yaml` are **absolute Windows paths**
- Edit config to change: data paths, model params, batch size, learning rate

### Data Pipeline
- **Input (CSV)**: 553 landmarks/frame, columns: x, y, z, visibility → 2212 dims
- ** gloss2ids.pkl**: Vocab size 1124, special tokens: `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, `<mask>`=4
- **Data splits**: train=7096, dev=519, test=642 (gzip compressed pickle)
- **NPY shape**: `(T, 2212)` per video

### Model Architecture
- **Input**: 6636-dim (2212 × 3: position, velocity, acceleration)
- **Model**: Transformer encoder with Conv1D embedding, 6 layers, 8 heads, 512 hidden
- **Output**: CTC loss, 1124 classes, blank token ID = 0

### Training Details
- **Must activate venv** before running any python command
- Use `is_train=True` in `CoordinateDataset` to enable augmentation
- **Always pass `input_lens` to model forward()** for proper padding mask
- Validation uses beam search (beam_size=5) for accurate WER
- Best model saved: `checkpoints_3/best_model_v3.pt`

### Data Preprocessing
- Source: `phoenix-coords_fixed_final/{split}/{video_id}/images*.csv`
- Output: `phoenix-npy_2/{split}/{video_id}.npy`

### Dependencies
- `torch_directml` (GPU on Windows, not CUDA)
- `wandb` (logging), `jiwer` (WER), `tqdm`