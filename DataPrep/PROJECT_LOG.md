# Multimodal Deepfake Detection — Project Log

**Project:** Cross-Modal Transformer for Audio-Visual Deepfake Detection with Explainability  
**Authors:** Komal Yadav (PhD Scholar, Lead), Aayush Sharma (Associate Research Intern)  
**Institution:** Bennett University  
**Target:** Conference paper (short-form, code + results)  
**Started:** May 2026  
**Last Updated:** 02 June 2026

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Design Decisions & Rationale](#3-design-decisions--rationale)
4. [Dataset](#4-dataset)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Training Plan](#6-training-plan)
7. [Experiment Plan & Ablations](#7-experiment-plan--ablations)
8. [Environment & Infrastructure](#8-environment--infrastructure)
9. [Progress Tracker](#9-progress-tracker)
10. [Key Findings](#10-key-findings)
11. [Issues & Solutions Log](#11-issues--solutions-log)
12. [Meeting Notes](#12-meeting-notes)
13. [References](#13-references)

---

## 1. Project Overview

Komal ma'am had an existing paper on explainable multimodal deepfake detection. A similar paper was found to be already published. Her senior approved continuing as a conference paper since those are shorter. We are implementing the code from scratch, running experiments, and producing results.

**Core Idea:** Detect deepfakes by jointly analyzing video (face) and audio (speech) using separate encoders, then fusing them via cross-modal attention. The cross-modal transformer captures audio-visual mismatches (e.g., lip-sync errors, speech-onset vs lip-movement timing) that single-modality detectors miss.

**Explainability:** Post-training Grad-CAM++ on both branches + cross-modal attention weight visualization to show which audio-video pairs the model linked.

---

## 2. Architecture

### 2.1 Model Pipeline

```
VIDEO BRANCH:
  Video frames (16, 3, 224, 224)
  → EfficientNet-B0 (pretrained ImageNet, partially fine-tuned, 1280-d output)
  → Linear projection (1280 → 256) + positional encoding
  → Transformer encoder (2 layers, 4 heads, from scratch)
  → Visual feature sequence (16, 256)

AUDIO BRANCH:
  Mel-spectrogram (1, 128, T)
  → VGGish (pretrained AudioSet, partially fine-tuned, 128-d output)
  → Linear projection (128 → 256) + positional encoding
  → Transformer encoder (2 layers, 4 heads, from scratch)
  → Audio feature sequence (T', 256)

FUSION:
  → Cross-modal Transformer (2-3 layers, bidirectional cross-attention)
    - Visual queries attend to Audio keys/values
    - Audio queries attend to Visual keys/values
  → Mean pool each branch + concatenate → 512-d vector
  → MLP classifier → Real/Fake (binary) or 4-class

EXPLAINABILITY (post-training, inference-time only):
  → Grad-CAM++ on EfficientNet-B0 (visual saliency heatmaps on face regions)
  → Grad-CAM++ on VGGish (audio saliency on mel-spectrogram time-frequency regions)
  → Cross-modal attention weight visualization (which audio-video pairs the model linked)
```

### 2.2 Tensor Shapes Through the Pipeline

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw video input | (B, 16, 3, 224, 224) | 16 frames, ImageNet normalized |
| After EfficientNet-B0 | (B, 16, 1280) | Per-frame features |
| After video projection | (B, 16, 256) | + positional encoding |
| After video transformer | (B, 16, 256) | Self-attended visual features |
| Raw audio input | (B, 1, 128, 94) | Log mel-spectrogram (T=94 confirmed) |
| After VGGish | (B, T', 128) | Per-segment features |
| After audio projection | (B, T', 256) | + positional encoding |
| After audio transformer | (B, T', 256) | Self-attended audio features |
| After cross-modal fusion | (B, 512) | Mean-pooled + concatenated |
| After classifier | (B, num_classes) | Binary or 4-class logits |

### 2.3 Parameter Count Estimates

| Component | Params | Trainable? |
|-----------|--------|------------|
| EfficientNet-B0 | ~5.3M | Partial (last 2 blocks + head) |
| VGGish | ~62M | Partial (last layers) |
| Video projection + PE | ~330K | Yes |
| Audio projection + PE | ~33K | Yes |
| Video transformer encoder (2L) | ~1.1M | Yes (from scratch) |
| Audio transformer encoder (2L) | ~1.1M | Yes (from scratch) |
| Cross-modal transformer (2L) | ~1.1M | Yes (from scratch) |
| MLP classifier | ~130K | Yes |
| **Total** | **~71M** | **~4M trainable** |

---

## 3. Design Decisions & Rationale

### Backbone Choices

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Video encoder | EfficientNet-B0 | ResNet-50 | 5.3M vs 25.6M params; better efficiency for GPU training |
| Audio encoder | VGGish | Wav2Vec2 | Faithful to paper's "1D CNN" description; simpler; pretrained on AudioSet |
| Face detection | MTCNN | RetinaFace | Standard, fast, good alignment; works well with facenet-pytorch |

### Architecture Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| d_model | 256 | Lightweight, fits alongside pretrained backbones on GPU |
| n_heads | 4 | 256/4 = 64-d per head, sufficient for short sequences |
| dim_feedforward | 512 | 2x d_model, standard ratio |
| Encoder layers | 2 | Sequence is only 16 tokens; 1 layer = full connectivity, 2 = second-order interactions, diminishing returns beyond |
| Cross-modal layers | 2-3 | Where the novelty lives; more capacity here |
| Fusion output | Mean pool + concat | Simple, effective for conference paper |
| Cross-attention | Bidirectional | V→A catches lip-sync; A→V catches speech-onset mismatches |

### Preprocessing Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frames per video | 16 | Standard (TimeSformer, ViViT); O(n²) attention is cheap at 16 |
| Frame size | 224×224 | EfficientNet-B0 pretrained input size |
| Audio sample rate | 16 kHz | Human speech < 8 kHz (Nyquist); VGGish pretrained rate |
| Audio duration | 3 seconds | Enough for sentence-level prosody |
| Mel bins | 128 | Standard, matches VGGish |
| n_fft / hop_length | 1024 / 512 | Standard config matching VGGish pretraining |
| Storage dtype | float16 | Halves disk usage; cast to float32 at training time |

### Sampling Decision

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sample size | 5,000 / 136K | Full dataset preprocessing exceeded 400 GB disk; 5K is sufficient for conference paper with pretrained encoders; stratified to maintain class/split proportions |

---

## 4. Dataset

### 4.1 LAV-DF (Localized Audio-Visual Deepfake)

- Purpose-built for audio-visual deepfake detection
- Videos with manipulated audio, video, or both
- ~25.6 GB total (as .tar), available on HuggingFace (ControlNet/LAV-DF) — gated dataset, requires HuggingFace login and access approval
- NO separate audio files — audio embedded in .mp4, extracted with ffmpeg subprocess

### 4.2 Directory Structure

```
LAV-DF/LAV-DF/LAV-DF/          ← NOTE: triple nesting after HuggingFace download + tar extraction
├── metadata.json               ← ALL labels and metadata (131.2 MB)
├── metadata.min.json           ← minified version (32.3 MB)
├── README.md
├── train/                      ← 78,703 .mp4 files
├── dev/                        ← 31,501 .mp4 files (validation split, NOT called "val")
└── test/                       ← 26,100 .mp4 files
```

### 4.3 metadata.json Fields

```json
{
  "file": "test/000001.mp4",
  "n_fakes": 0,
  "fake_periods": [],
  "duration": 4.224,
  "original": null,
  "modify_video": false,
  "modify_audio": false,
  "split": "test",
  "video_frames": 103,
  "audio_channels": 1,
  "audio_frames": 65536
}
```

### 4.4 Label Mapping

| modify_video | modify_audio | 4-class Label | Binary Label |
|:---:|:---:|:---:|:---:|
| False | False | 0 (Real) | 0 (Real) |
| True | False | 1 (Fake-Video) | 1 (Fake) |
| False | True | 2 (Fake-Audio) | 1 (Fake) |
| True | True | 3 (Fake-Both) | 1 (Fake) |

### 4.5 Class Distribution (Full Dataset — 136,304 total)

| Split | Real | Fake-Video | Fake-Audio | Fake-Both | Total |
|-------|------|------------|------------|-----------|-------|
| Train | 21,254 (27.0%) | 19,271 (24.5%) | 19,088 (24.3%) | 19,090 (24.3%) | 78,703 |
| Dev | 8,271 (26.3%) | 7,820 (24.8%) | 7,709 (24.5%) | 7,701 (24.4%) | 31,501 |
| Test | 6,906 (26.5%) | 6,452 (24.7%) | 6,373 (24.4%) | 6,369 (24.4%) | 26,100 |
| **Total** | **36,431 (26.7%)** | **33,543 (24.6%)** | **33,170 (24.3%)** | **33,160 (24.3%)** | **136,304** |

**Observation:** Near-perfectly balanced 4-class dataset. Binary split is 26.7% Real / 73.3% Fake (expected: 3 fake types vs 1 real type). Will use weighted CrossEntropyLoss or balanced accuracy to handle binary imbalance.

### 4.6 Sampled Dataset (5,000 total, stratified)

| Split | Real | Fake-Video | Fake-Audio | Fake-Both | Total |
|-------|------|------------|------------|-----------|-------|
| Train | 780 (27.0%) | 707 (24.5%) | 700 (24.2%) | 700 (24.2%) | 2,887 |
| Dev | 303 (26.2%) | 287 (24.8%) | 283 (24.5%) | 282 (24.4%) | 1,155 |
| Test | 253 (26.4%) | 237 (24.7%) | 234 (24.4%) | 234 (24.4%) | 958 |
| **Total** | **1,336 (26.7%)** | **1,231 (24.6%)** | **1,217 (24.3%)** | **1,216 (24.3%)** | **5,000** |

Average duration: 8.60 seconds. Total duration of full dataset: 325.5 hours.

---

## 5. Preprocessing Pipeline

### 5.1 Video Preprocessing

```
For each sampled .mp4:
  1. Load video with OpenCV
  2. Uniformly sample 16 frames
  3. MTCNN face detection + crop + align (GPU-accelerated)
     - Fallback 1: nearest good bounding box if frame fails
     - Fallback 2: center-crop if ALL frames fail
  4. Resize to 224×224
  5. Normalize with ImageNet mean/std
  6. Save as float16 .pt → shape (16, 3, 224, 224)
```

**Actual results (02 June 2026):**
- Processed: 5,000 / 5,000 (0 failures)
- Time: 12.4 min on RTX A6000 (6.7 videos/sec)
- Tensor shape: (16, 3, 224, 224) confirmed
- File size per tensor: 4.59 MB
- Total disk: 22.4 GB

### 5.2 Audio Preprocessing

```
For each sampled .mp4:
  1. Extract audio from .mp4 using ffmpeg subprocess (NOT torchaudio.load — no backends on Windows)
  2. Convert to mono, resample to 16 kHz (done by ffmpeg)
  3. Load wav with soundfile
  4. Pad or center-crop to 3 seconds (48000 samples)
  5. Compute mel-spectrogram (n_mels=128, n_fft=1024, hop=512)
  6. Log scaling: log(mel + 1e-9)
  7. Save as float16 .pt → shape (1, 128, 94)
```

**Actual results (02 June 2026):**
- Processed: 5,000 / 5,000 (0 failures)
- Time: 15.6 min (5.35 videos/sec)
- Tensor shape: (1, 128, 94) confirmed — T=94 time steps for 3-sec audio
- File size per tensor: 24.7 KB
- Total disk: ~120 MB

### 5.3 Actual Disk Usage (5K videos, float16)

| Component | Estimated | Actual |
|-----------|-----------|--------|
| Video tensors | ~22 GB | 22.4 GB |
| Audio tensors | ~0.25 GB | ~0.12 GB |
| Raw dataset (.tar + extracted) | ~25.6 GB | ~51 GB (tar not deleted yet) |
| **Total** | **~48 GB** | **~74 GB (can reclaim ~25 GB by deleting .tar)** |

---

## 6. Training Plan

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| LR (pretrained backbones) | 1e-5 |
| LR (new layers) | 1e-4 |
| Scheduler | Cosine annealing |
| Weight decay | 1e-2 |
| Mixed precision | torch.cuda.amp |
| Batch size | 8–16 (can go to 32 on A6000) |
| Epochs | 20–30 |
| Checkpoint criterion | Best validation AUC |
| Dropout | 0.1 |
| Binary class weights | [3.0, 1.0] (upweight Real to handle 27/73 imbalance) |

### Evaluation Metrics
- Balanced accuracy (primary — handles class imbalance)
- Precision / Recall / F1 (per class)
- AUC-ROC
- Confusion matrix

---

## 7. Experiment Plan & Ablations

### 7.1 Core Experiments

| # | Experiment | Status |
|---|-----------|--------|
| E1 | Binary classification (Real vs Fake) | ❌ Pending |
| E2 | 4-class classification | ❌ Pending |
| E3 | Cross-dataset generalization (train LAV-DF, test FakeAVCeleb) | ❌ Pending |

### 7.2 Planned Ablations

| # | Ablation | Purpose |
|---|---------|---------|
| A1 | Transformer depth: 1, 2, 3, 4 layers | Find optimal depth for short sequences |
| A2 | Cross-modal vs simple concatenation fusion | Justify cross-attention novelty |
| A3 | Bidirectional vs unidirectional cross-attention | Justify both directions |
| A4 | Video-only vs Audio-only vs Multimodal | Show multimodal benefit |
| A5 | With vs without Grad-CAM++ explainability | Qualitative comparison |

### 7.3 Results Table Template

| Model Variant | Accuracy | Bal. Acc | AUC | Precision | Recall | F1 |
|--------------|----------|----------|-----|-----------|--------|-----|
| Full model (proposed) | | | | | | |
| Video-only | | | | | | |
| Audio-only | | | | | | |
| Concat fusion (no cross-attn) | | | | | | |
| Unidirectional cross-attn | | | | | | |
| 1 encoder layer | | | | | | |
| 3 encoder layers | | | | | | |
| 4 encoder layers | | | | | | |

---

## 8. Environment & Infrastructure

### 8.1 Lab Machine (Primary — Preprocessing + Training)
- **Location:** Bennett University AR/VR Lab (remote via AnyDesk)
- **OS:** Windows 10
- **GPU:** NVIDIA RTX A6000 (48 GB VRAM)
- **CUDA:** 13.1 (driver 591.59)
- **User:** Komal-Sch
- **Access:** AnyDesk v6.x from Mac (keyboard issues with v9 mismatch — use on-screen keyboard workaround)
- **Conda env:** `dfd` (Python 3.11.9)
- **GitHub repo cloned:** Yes, at `C:\Users\Komal-Sch\Desktop\deepfake_detection\`

### 8.2 MacBook (Secondary — Code writing, planning)
- **Machine:** MacBook M2 Pro
- **OS:** macOS
- **Role:** Code development, remote access to lab machine
- **Conda env:** `dfd` (Python 3.11)

### 8.3 Kaggle (Backup training)
- **GPU:** T4 (16 GB VRAM)
- **Use:** If lab machine is unavailable; upload preprocessed tensors as Kaggle Dataset

### 8.4 Dependencies (Pinned)

```
torch==2.3.1 (CUDA 12.1)
torchvision==0.18.1
torchaudio==2.3.1
numpy==1.26.4
facenet-pytorch==2.6.0 (--no-deps)
Pillow==10.4.0
opencv-python==4.10.0.84
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.9.2
tqdm==4.66.5
ipykernel==6.29.5
huggingface_hub==0.24.6
soundfile (added 02 June — needed for audio loading on Windows)
ffmpeg (conda-forge — needed for audio extraction from mp4)
```

---

## 9. Progress Tracker

### Phase 1: Setup & Preprocessing

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Architecture designed | ✅ Done | May 2026 | Approved by Komal ma'am |
| Preprocessing pipeline designed | ✅ Done | May 2026 | |
| LAV-DF dataset downloaded (MacBook) | ✅ Done | May 2026 | ~25.6 GB from HuggingFace |
| Conda env setup (MacBook) | ✅ Done | May 2026 | |
| Code files written (DataPrep) | ✅ Done | 31 May 2026 | 9 files + setup.bat |
| Lab machine access (AnyDesk) | ✅ Done | 31 May 2026 | RTX A6000 confirmed |
| Lab machine env setup | ✅ Done | 02 June 2026 | setup.bat + manual Miniconda install |
| LAV-DF download on lab machine | ✅ Done | 02 June 2026 | 48 min download, required tar extraction |
| Dataset exploration | ✅ Done | 02 June 2026 | 136,304 videos, near-balanced 4-class |
| Stratified sampling (5K) | ✅ Done | 02 June 2026 | 5,000 videos, proportions maintained |
| Audio preprocessing | ✅ Done | 02 June 2026 | 5,000/5,000, 0 failures, 15.6 min |
| Video preprocessing | ✅ Done | 02 June 2026 | 5,000/5,000, 0 failures, 12.4 min |
| Metadata CSV build | ✅ Done | 02 June 2026 | 5,000 entries, all matched |
| Dataset verification | ✅ Done | 02 June 2026 | ALL CHECKS PASSED |

### Phase 2: Model & Training

| Task | Status | Date | Notes |
|------|--------|------|-------|
| model.py (full architecture) | ❌ Pending | | Next up |
| train.py (training loop) | ❌ Pending | | |
| evaluate.py (metrics + viz) | ❌ Pending | | |
| Binary classification training | ❌ Pending | | |
| 4-class classification training | ❌ Pending | | |
| Ablation experiments | ❌ Pending | | |
| Grad-CAM++ visualization | ❌ Pending | | |

### Phase 3: Paper

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Results tables | ❌ Pending | | |
| Figures (architecture, Grad-CAM) | ❌ Pending | | |
| Paper draft | ❌ Pending | | |
| Venue selection | ❌ Pending | | |

---

## 10. Key Findings

### 10.1 Dataset Observations
- LAV-DF is near-perfectly balanced for 4-class: each class ~24-27%
- Binary split is 26.7% Real / 73.3% Fake — natural consequence of 3 fake types, not a dataset flaw
- Average video duration: 8.60 seconds, total: 325.5 hours
- All 136,304 videos verified present on disk

### 10.2 Preprocessing Observations
- Full dataset preprocessing (136K videos) exceeded 400 GB on MacBook — switched to 5K stratified sample
- Lab machine RTX A6000 (48 GB VRAM) processed 5K videos in 12.4 min — A6000 is dramatically faster than estimated 1-3 hours
- MTCNN face detection ran at 6.7 videos/sec on A6000 with zero failures
- Audio preprocessing: T=94 time steps for 3-second audio at hop_length=512
- torchaudio had zero audio backends on Windows — fixed by switching to ffmpeg subprocess + soundfile
- float16 storage confirmed working: tensors auto-cast to float32 at DataLoader time
- Audio file size is negligible (24.7 KB each) vs video (4.59 MB each)

### 10.3 Training Results
- *Pending*

### 10.4 Ablation Results
- *Pending*

---

## 11. Issues & Solutions Log

| # | Date | Issue | Solution | Status |
|---|------|-------|----------|--------|
| 1 | May 2026 | Similar paper already published | Senior approved: publish as conference paper | ✅ Resolved |
| 2 | May 2026 | facenet-pytorch version pin conflicts | Install with --no-deps | ✅ Resolved |
| 3 | May 2026 | numpy 2.x incompatibility | Pin numpy==1.26.4 | ✅ Resolved |
| 4 | 31 May | Full preprocessing fills 400 GB disk | Stratified sample 5K + float16 storage | ✅ Resolved |
| 5 | 31 May | AnyDesk v9 keyboard broken (sends Win key) | Downgraded to AnyDesk v6.6.0 on Mac; use on-screen keyboard, toggle Win key off | ✅ Resolved |
| 6 | 31 May | AnyDesk v6↔v9 version mismatch causes stuck Win key | Toggle Win key on Windows on-screen keyboard to release it | ✅ Resolved |
| 7 | 31 May | Lab machine has no Python/Conda | setup.bat auto-installs; manual Miniconda download when auto-download failed | ✅ Resolved |
| 8 | 31 May | TeamViewer LAN mode blocked remote access | Didn't work; sticking with AnyDesk | ❌ Abandoned |
| 9 | 02 June | HuggingFace 401 — LAV-DF is gated dataset | huggingface-cli login with access token | ✅ Resolved |
| 10 | 02 June | Dataset downloaded as .tar, not extracted | python tarfile.extractall() — created triple-nested LAV-DF/LAV-DF/LAV-DF path | ✅ Resolved |
| 11 | 02 June | DATASET_ROOT path wrong after tar extraction | Updated config.py to add extra "LAV-DF" nesting | ✅ Resolved |
| 12 | 02 June | torchaudio.list_audio_backends() returns [] on Windows | Replaced torchaudio.load() with ffmpeg subprocess + soundfile; added soundfile to deps | ✅ Resolved |

---

## 12. Meeting Notes

### 31 May 2026 — Lab Machine Setup
- Accessed AR/VR lab machine via AnyDesk
- Confirmed GPU: RTX A6000 (48 GB VRAM), CUDA 13.1
- VS Code already installed on lab machine
- Python/Conda NOT installed — setup.bat will handle
- All DataPrep code files written and ready to transfer
- Decided to use 5K stratified sample instead of full 136K dataset (storage constraint)
- TeamViewer attempted but failed (LAN mode); sticking with AnyDesk

### 02 June 2026 — Full Preprocessing Run
- Ran entire preprocessing pipeline on lab machine via AnyDesk
- setup.bat: Miniconda auto-download failed (network), installed manually. Env creation and package install succeeded. All pinned versions verified.
- download_dataset.py: 25.6 GB downloaded in 48 min. Required tar extraction and config.py path fix (triple-nested folder).
- explore_dataset.py: 136,304 videos confirmed, near-balanced 4-class distribution.
- sample_dataset.py: 5,000 videos sampled, stratified across splits and classes.
- preprocess_audio.py: Initially failed — torchaudio had zero backends on Windows. Fixed by switching to ffmpeg subprocess + soundfile. After fix: 5,000/5,000 processed, 0 failures, 15.6 min.
- preprocess_video.py: 5,000/5,000 processed, 0 failures, 12.4 min on A6000 (6.7 videos/sec). Way faster than estimated 1-3 hours.
- build_metadata.py: 5,000 entries, all video+audio tensor pairs matched.
- dataset.py: ALL CHECKS PASSED. DataLoader verified for both binary and 4-class labels.
- **Phase 1 is 100% complete. Ready for model.py and train.py.**

---

## 13. References

- LAV-DF Dataset: https://huggingface.co/datasets/ControlNet/LAV-DF
- EfficientNet: Tan & Le, ICML 2019
- VGGish: Hershey et al., ICASSP 2017
- Grad-CAM++: Chattopadhyay et al., WACV 2018
- Cross-Modal Transformers: Tsai et al. (MulT), ACL 2019
