# Multimodal Deepfake Detection — Data Preprocessing Pipeline

## Quick Start (Lab Machine — Windows)

### Step 0: Setup Environment (one-time)
```
cd C:\Users\Komal-Sch\Desktop\deepfake_detection\DataPrep
setup.bat
```
This installs Miniconda (if needed), creates the `dfd` conda env, and installs all dependencies.

If conda is not found, the script will tell you to install Miniconda first:
- Download: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
- Install with default settings, CHECK "Add to PATH"
- Close and reopen VS Code terminal
- Run `setup.bat` again

### Step 1: Download Dataset
```
conda activate dfd
python download_dataset.py
```
Downloads LAV-DF (~25.6 GB) from HuggingFace. Requires `huggingface-cli login` if prompted.

### Step 2: Explore Dataset
```
python explore_dataset.py
```
Prints dataset stats, class distribution, and verifies file existence.

### Step 3: Sample Dataset (5000 videos)
```
python sample_dataset.py
```
Stratified sampling maintaining class/split proportions. Saves `sampled_metadata.json`.

### Step 4: Preprocess Audio (~15-30 min)
```
python preprocess_audio.py
```
Extracts audio → mel-spectrograms → float16 `.pt` tensors.

### Step 5: Preprocess Video (~1-3 hrs)
```
python preprocess_video.py
```
Extracts frames → MTCNN face crop → normalized float16 `.pt` tensors. This is the slowest step.

### Step 6: Build Metadata CSV
```
python build_metadata.py
```
Links video/audio tensors with labels into `metadata.csv`.

### Step 7: Verify
```
python dataset.py
```
Loads one batch from each split, prints shapes. If this passes, data is ready for training.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `setup.bat` | One-click env setup (conda, packages, folders) |
| `config.py` | All paths, hyperparameters, settings |
| `download_dataset.py` | Downloads LAV-DF from HuggingFace |
| `explore_dataset.py` | Dataset stats and verification |
| `sample_dataset.py` | Stratified sampling (5K from 136K) |
| `preprocess_audio.py` | Audio → mel-spectrogram tensors |
| `preprocess_video.py` | Video → face-cropped frame tensors |
| `build_metadata.py` | Creates final metadata.csv |
| `dataset.py` | PyTorch Dataset + DataLoader |

## Key Settings (in config.py)

- **SAMPLE_SIZE**: 5000 (change if needed)
- **SAVE_DTYPE**: float16 (halves disk usage)
- **NUM_FRAMES**: 16 per video
- **FRAME_SIZE**: 224×224
- **AUDIO_DURATION**: 3 seconds at 16 kHz

## Disk Space Estimates (5K videos, float16)

| Component | Size |
|-----------|------|
| Raw dataset (LAV-DF) | ~25.6 GB |
| Video tensors | ~22 GB |
| Audio tensors | ~0.25 GB |
| **Total needed** | **~48 GB** |

## Troubleshooting

- **"conda not found"**: Install Miniconda, check "Add to PATH", restart terminal
- **"CUDA not available"**: Check `nvidia-smi` works, reinstall PyTorch with correct CUDA version
- **MTCNN slow**: Runs faster on GPU. Check `torch.cuda.is_available()` returns True
- **Disk space**: Reduce SAMPLE_SIZE in config.py
- **Audio extraction fails**: Some .mp4 files may have no audio track; these are logged in `_failed_audio.txt`
- **Resume after interruption**: Re-run the same script; it skips already-processed files
