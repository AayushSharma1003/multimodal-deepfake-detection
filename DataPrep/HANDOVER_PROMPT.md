# HANDOVER PROMPT — Paste this entire message into a new Claude chat

---

You are helping me with a research project on multimodal deepfake detection. I am Komal Yadav, a PhD scholar at Bennett University. My associate research intern Aayush Sharma has written all the code and set up the pipeline. I need your help understanding the project, running the code on our lab machine, and troubleshooting any issues.

Below is everything you need to know about this project. Read it carefully and remember it throughout our conversation.

---

## PROJECT OVERVIEW

We are building a **Cross-Modal Transformer for Audio-Visual Deepfake Detection with Explainability**. This is a **conference paper** (short-form). The model takes a video as input, processes the visual (face) and audio (speech) separately using pretrained encoders, then fuses them using cross-modal attention to detect whether the video is real or a deepfake.

The key novelty is the **bidirectional cross-modal transformer** — visual features attend to audio features and vice versa — which captures audio-visual mismatches like lip-sync errors or speech timing mismatches that single-modality detectors would miss.

After training, we use **Grad-CAM++** on both branches for explainability — showing which face regions and which audio time-frequency regions the model focused on.

---

## ARCHITECTURE

```
VIDEO BRANCH:
  Video frames (16, 3, 224, 224)
  → EfficientNet-B0 (pretrained on ImageNet, partially fine-tuned)
  → Linear projection (1280 → 256) + positional encoding
  → Transformer encoder (2 layers, 4 heads)
  → Visual features (16, 256)

AUDIO BRANCH:
  Mel-spectrogram (1, 128, T)
  → VGGish (pretrained on AudioSet, partially fine-tuned)
  → Linear projection (128 → 256) + positional encoding
  → Transformer encoder (2 layers, 4 heads)
  → Audio features (T', 256)

FUSION:
  → Cross-modal Transformer (2-3 layers, bidirectional cross-attention)
  → Mean pool each branch + concatenate → 512-d vector
  → MLP classifier → Real/Fake (binary) or 4-class output

EXPLAINABILITY (inference-time only):
  → Grad-CAM++ on EfficientNet (face heatmaps)
  → Grad-CAM++ on VGGish (audio spectrogram heatmaps)
  → Cross-modal attention weight visualization
```

**Why these choices:**
- **EfficientNet-B0** over ResNet-50: 5.3M vs 25.6M params, more efficient
- **VGGish** over Wav2Vec2: simpler, aligns with paper description, pretrained on AudioSet
- **2 Transformer layers**: sequence is only 16 tokens, diminishing returns beyond 2 layers
- **d_model=256, 4 heads**: lightweight to fit on GPU alongside pretrained backbones
- **Bidirectional cross-attention**: V→A catches lip-sync, A→V catches speech-onset mismatches

---

## DATASET: LAV-DF (Localized Audio-Visual Deepfake)

- Downloaded from HuggingFace: `ControlNet/LAV-DF`
- ~136K total videos, but we are using a **stratified sample of 5,000 videos** (full dataset exceeded 400 GB during preprocessing)
- Audio is embedded inside .mp4 files — there are NO separate audio files

**Directory structure:**
```
LAV-DF/
├── metadata.json       ← all labels and metadata
├── train/              ← .mp4 video files
├── dev/                ← validation split (NOT called "val")
└── test/               ← test split
```

**Labels** are based on two fields in metadata.json:
- `modify_video=False, modify_audio=False` → **Real** (binary: 0, 4-class: 0)
- `modify_video=True, modify_audio=False` → **Fake-Video** (binary: 1, 4-class: 1)
- `modify_video=False, modify_audio=True` → **Fake-Audio** (binary: 1, 4-class: 2)
- `modify_video=True, modify_audio=True` → **Fake-Both** (binary: 1, 4-class: 3)

---

## LAB MACHINE DETAILS

- **Location:** Bennett University AR/VR Lab
- **OS:** Windows 10
- **User account:** `C:\Users\Komal-Sch`
- **GPU:** NVIDIA RTX A6000 (48 GB VRAM)
- **CUDA:** 13.1 (driver 591.59)
- **Remote access:** AnyDesk (use v6.x on the connecting machine to avoid keyboard issues with v9 mismatch)
- **VS Code:** Already installed

---

## CODE FILES — What Each One Does

All code files should be placed at:
```
C:\Users\Komal-Sch\Desktop\deepfake_detection\DataPrep\
```

Here is every file in execution order:

### 1. `setup.bat` — FIRST FILE TO RUN (double-click)
**What it does:** Fully automatic environment setup. Downloads and installs Miniconda (if not present), creates a conda environment named `dfd` with Python 3.11, installs all dependencies with pinned versions (PyTorch 2.3.1 + CUDA 12.1, numpy 1.26.4, facenet-pytorch 2.6.0, opencv, pandas, etc.), creates a Jupyter kernel, and creates project folders.

**How to run:** Just double-click the file. A command prompt window will open. Do not close it. It takes about 15-25 minutes.

**First run behavior:** If Miniconda is not installed, it will download and install it automatically (~80 MB download), then ask you to close the window and double-click setup.bat again. The second run will create the environment and install everything.

**Output:** 
- Conda environment `dfd` ready to use
- Folders created: `Desktop\deepfake_detection\DataPrep\` and `Desktop\LAV-DF-preprocessed\video_tensors\` and `Desktop\LAV-DF-preprocessed\audio_tensors\`
- At the end, it prints a verification showing all package versions and whether the GPU is detected

**What success looks like:**
```
  PyTorch:       2.3.1
  CUDA:          True
  GPU:           NVIDIA RTX A6000
  torchaudio:    2.3.1
  torchvision:   0.18.1
  numpy:         1.26.4
  opencv:        4.10.0
  MTCNN:         OK

  SETUP COMPLETE!
```

### 2. `config.py` — Configuration (do NOT run directly)
**What it does:** Contains all paths, hyperparameters, and settings. Every other script imports from this file. If you need to change the dataset location or sample size, edit this file.

**Key settings:**
- `SAMPLE_SIZE = 5000` — how many videos to sample
- `SAVE_DTYPE = "float16"` — saves disk space
- `DATASET_ROOT` — where LAV-DF is stored
- `OUTPUT_DIR` — where preprocessed tensors go

**You can run it to check paths:** `python config.py` will print all paths and whether they exist.

### 3. `download_dataset.py` — Download LAV-DF
**What it does:** Downloads the LAV-DF dataset (~25.6 GB) from HuggingFace to `Desktop\LAV-DF\LAV-DF\`.

**How to run:**
```
conda activate dfd
cd C:\Users\Komal-Sch\Desktop\deepfake_detection\DataPrep
python download_dataset.py
```

**Notes:** 
- Takes a long time depending on internet speed
- If it asks for HuggingFace login, run `huggingface-cli login` first and paste your token
- If download is interrupted, re-run and it will resume from where it stopped
- If the dataset is already downloaded (from a previous session), skip this step

### 4. `explore_dataset.py` — Explore Dataset
**What it does:** Prints dataset statistics — total videos, class distribution per split (train/dev/test), sample entries, and checks if video files actually exist on disk.

**How to run:**
```
python explore_dataset.py
```

**Why run this:** To verify the dataset downloaded correctly and to see the class distribution before sampling.

### 5. `sample_dataset.py` — Stratified Sampling
**What it does:** Randomly samples 5,000 videos from the full 136K dataset, maintaining the original class proportions across all splits (train/dev/test) and all 4 classes (Real, Fake-Video, Fake-Audio, Fake-Both). Saves `sampled_metadata.json` to the output directory.

**How to run:**
```
python sample_dataset.py
```

**Output:** `Desktop\LAV-DF-preprocessed\sampled_metadata.json`

### 6. `preprocess_audio.py` — Audio Preprocessing
**What it does:** For each of the 5,000 sampled videos:
1. Extracts audio from the .mp4 file
2. Converts to mono, resamples to 16 kHz
3. Pads or crops to 3 seconds
4. Computes a log mel-spectrogram (128 mel bins)
5. Saves as a float16 `.pt` tensor file

**How to run:**
```
python preprocess_audio.py
```

**Time:** ~15-30 minutes for 5K videos
**Disk:** ~250 MB total
**Output:** `.pt` files in `Desktop\LAV-DF-preprocessed\audio_tensors\`

**Resume capability:** If interrupted, re-run and it will skip already-processed files.

### 7. `preprocess_video.py` — Video Preprocessing (SLOWEST STEP)
**What it does:** For each of the 5,000 sampled videos:
1. Loads the video with OpenCV
2. Uniformly samples 16 frames
3. Detects and crops faces using MTCNN (neural network face detector)
4. If face detection fails on some frames, uses the nearest successful detection
5. If face detection fails on ALL frames, falls back to center-cropping
6. Resizes to 224×224, normalizes with ImageNet statistics
7. Saves as a float16 `.pt` tensor file

**How to run:**
```
python preprocess_video.py
```

**Time:** ~1-3 hours for 5K videos (faster with GPU, which we have)
**Disk:** ~22 GB total
**Output:** `.pt` files in `Desktop\LAV-DF-preprocessed\video_tensors\`

**Resume capability:** If interrupted, re-run and it will skip already-processed files.

### 8. `build_metadata.py` — Build Metadata CSV
**What it does:** Creates a CSV file that links each video's preprocessed tensor paths with its labels. Only includes entries where BOTH video and audio tensors exist. Prints class distribution stats.

**How to run:**
```
python build_metadata.py
```

**Output:** `Desktop\LAV-DF-preprocessed\metadata.csv`

### 9. `dataset.py` — Verify Everything Works
**What it does:** Loads the PyTorch Dataset, creates DataLoaders, loads one batch from each split (train/dev/test), and prints tensor shapes. This verifies the entire pipeline is working.

**How to run:**
```
python dataset.py
```

**What success looks like:**
```
  Video shape:  torch.Size([16, 3, 224, 224])
  Audio shape:  torch.Size([1, 128, 94])
  Label:        0

  ALL CHECKS PASSED!
  Dataset is ready for training.
```

---

## COMPLETE EXECUTION ORDER (copy-paste ready)

After placing all files in `C:\Users\Komal-Sch\Desktop\deepfake_detection\DataPrep\`:

```
Step 1: Double-click setup.bat (wait 15-25 min)
        If it says "close and re-open" → close window, double-click setup.bat again

Step 2: Open VS Code, open terminal (Ctrl+`), then:

conda activate dfd
cd C:\Users\Komal-Sch\Desktop\deepfake_detection\DataPrep

python download_dataset.py
python explore_dataset.py
python sample_dataset.py
python preprocess_audio.py
python preprocess_video.py
python build_metadata.py
python dataset.py
```

If every script completes without errors, the data is ready for model training (model.py and train.py, which Aayush will write next).

---

## COMMON ISSUES AND FIXES

### "conda is not recognized"
Miniconda was installed but the PATH wasn't updated. Close all terminals/cmd windows and open a new one. If still not working, the PATH needs to be added manually:
```
set PATH=%USERPROFILE%\Miniconda3;%USERPROFILE%\Miniconda3\Scripts;%USERPROFILE%\Miniconda3\condabin;%PATH%
```

### "CUDA is not available" / "GPU not detected"
PyTorch is installed for CUDA 12.1 but the lab machine has CUDA 13.1 — this should still work (backward compatible). If it doesn't:
```
conda activate dfd
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### "No module named X"
The conda environment might not be activated:
```
conda activate dfd
```
You should see `(dfd)` at the start of your terminal prompt.

### "metadata.json not found"
The dataset path in config.py doesn't match where the dataset was downloaded. Check:
```
python config.py
```
This will print the expected path. If it's wrong, edit `DATASET_ROOT` in config.py.

### Audio/video preprocessing fails on some files
This is normal — some .mp4 files may be corrupted or have no audio track. Failed files are logged in `_failed_audio.txt` or `_failed_video.txt`. A few failures out of 5,000 are fine.

### "Out of disk space"
Check disk space with `dir C:\ /s` or just look at This PC. The preprocessing needs about 48 GB free. If tight, reduce `SAMPLE_SIZE` in config.py to 3000 or even 2000.

### Preprocessing interrupted / machine restarted
Just re-run the same script. All preprocessing scripts skip files that already exist, so they resume from where they stopped.

### "torch.load() error" or "weights_only" warning
Not critical — this is a PyTorch 2.x safety warning. The code already handles it with `weights_only=True`.

---

## WHAT'S DONE AND WHAT'S NEXT

**Done:**
- Architecture designed and approved
- All preprocessing code written (9 files)
- Lab machine accessed, GPU confirmed (RTX A6000, 48 GB VRAM)
- setup.bat ready for one-click installation

**Next (preprocessing — you can do this):**
1. Transfer DataPrep folder to lab machine
2. Run setup.bat
3. Run the scripts in order (download → explore → sample → audio → video → build → verify)

**After preprocessing (Aayush will handle):**
- Write model.py (the full neural network architecture)
- Write train.py (training loop with mixed precision, discriminative learning rates)
- Write evaluate.py (metrics, Grad-CAM++ visualization, attention maps)
- Run training experiments and ablations
- Write the paper

---

## IMPORTANT NOTES

- The validation split in LAV-DF is called `dev`, NOT `val` — the code handles this correctly
- Audio must be extracted FROM .mp4 files — there are NO separate .wav files
- All tensors are saved as float16 to save disk space — they are automatically converted to float32 during training
- The `dfd` conda environment must always be activated before running any Python script
- If any script fails, share the full error message with Aayush or paste it into this Claude chat for help

---

This is the complete project context. Please confirm you understand the project, the architecture, the code pipeline, and the lab machine setup. When I ask for help, use this context to give me accurate, specific answers.
