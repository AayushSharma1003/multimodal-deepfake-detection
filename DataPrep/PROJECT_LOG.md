# Multimodal Deepfake Detection — Project Log

**Project:** Cross-Modal Transformer for Audio-Visual Deepfake Detection with Explainability  
**Authors:** Komal Yadav (PhD Scholar, Lead), Aayush Sharma (Associate Research Intern)  
**Institution:** Bennett University  
**Target:** Conference paper (short-form, code + results)  
**Started:** May 2026  
**Last Updated:** 03 June 2026

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
  → GradientSHAP (per-pixel attribution with Shapley-value guarantees, via captum)
  → Modality occlusion study (zero out one branch, measure performance drop)
```

### 2.2 Tensor Shapes Through the Pipeline

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw video input | (B, 16, 3, 224, 224) | 16 frames, ImageNet normalized |
| After EfficientNet-B0 | (B, 16, 1280) | Per-frame features |
| After video projection | (B, 16, 256) | + positional encoding |
| After video transformer | (B, 16, 256) | Self-attended visual features |
| Raw audio input | (B, 1, 128, 101) | Log mel-spectrogram (T=101 with center=True padding) |
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
| Sample size | 20,000 / 136K | Originally 5K; scaled to 20K on 03 June for research rigour (~958 test samples per class). Balanced 4-class (25% each). Fits within 500 GB disk budget. |

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

### 4.6 Sampled Dataset — v1 (5,000 total, DEPRECATED)

| Split | Real | Fake-Video | Fake-Audio | Fake-Both | Total |
|-------|------|------------|------------|-----------|-------|
| Train | 780 (27.0%) | 707 (24.5%) | 700 (24.2%) | 700 (24.2%) | 2,887 |
| Dev | 303 (26.2%) | 287 (24.8%) | 283 (24.5%) | 282 (24.4%) | 1,155 |
| Test | 253 (26.4%) | 237 (24.7%) | 234 (24.4%) | 234 (24.4%) | 958 |
| **Total** | **1,336 (26.7%)** | **1,231 (24.6%)** | **1,217 (24.3%)** | **1,216 (24.3%)** | **5,000** |

**Status:** Deleted on 03 June 2026. Replaced by 20K sample for research rigour.

### 4.7 Sampled Dataset — v2 (20,000 total, ACTIVE)

Resampled for research rigour: 20K gives ~958 test samples per class (vs ~240 in 5K), enabling robust per-class metrics and bootstrap significance testing.

| Split | Real | Fake-Video | Fake-Audio | Fake-Both | Total |
|-------|------|------------|------------|-----------|-------|
| Train | 2,887 | 2,887 | 2,887 | 2,887 | 11,548 |
| Dev | 1,156 | 1,156 | 1,155 | 1,155 | 4,622 |
| Test | 958 | 958 | 957 | 957 | 3,830 |
| **Total** | **5,001 (25.0%)** | **5,001 (25.0%)** | **4,999 (25.0%)** | **4,999 (25.0%)** | **20,000** |

**Key differences from v1:** Perfectly balanced 4-class distribution (25% each, not proportional to original). This is intentional — balanced classes eliminate the need for class weighting and simplify evaluation.

Average duration: 8.60 seconds. Total duration of full dataset: 325.5 hours.

---

## 5. Preprocessing Pipeline

### 5.1 Video Preprocessing

```
For each sampled .mp4:
  1. Load video with OpenCV
  2. Uniformly sample 16 frames
  3. MTCNN face detection + crop + align (GPU-accelerated)
     - Fallback: center-crop if face detection fails
  4. Resize to 224×224
  5. Normalize with ImageNet mean/std
  6. Save as float16 .pt → shape (16, 3, 224, 224)
```

**5K run (02 June 2026):** 5,000/5,000, 0 failures, 12.4 min (6.7 vids/sec)
**20K run (03 June 2026):** 20,000/20,000, 0 failures, 4h37m (1.2 vids/sec — includes audio processing in same loop)

### 5.2 Audio Preprocessing

```
For each sampled .mp4:
  1. Extract audio from .mp4 using ffmpeg subprocess
  2. Convert to mono, resample to 16 kHz
  3. Load wav with soundfile
  4. Take first 1 second (16,000 samples), zero-pad if shorter
  5. Compute mel-spectrogram (n_mels=128, n_fft=1024, hop=160, center=True)
  6. Log scaling: log(mel + 1e-9)
  7. Save as float16 .pt → shape (1, 128, 101)
```

**5K run (02 June 2026):** shape (1, 128, 94) — used hop=512, 3-sec audio
**20K run (03 June 2026):** shape (1, 128, 101) — used hop=160, 1-sec audio, center=True padding gives 101 frames

### 5.3 Actual Disk Usage

| Component | 5K (v1, deleted) | 20K (v2, active) |
|-----------|------------------|-------------------|
| Video tensors | 22.4 GB | ~92 GB |
| Audio tensors | ~0.12 GB | ~0.5 GB |
| Metadata + JSON | <1 MB | <1 MB |
| **Total preprocessed** | **~22.5 GB** | **~92.5 GB** |
| Raw dataset (extracted, tar deleted) | ~47.6 GB | ~47.6 GB |
| **Total on Desktop** | — | **~140 GB / 500 GB budget** |

---

## 6. Training Plan

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Optimizer | AdamW | |
| LR (pretrained backbones) | 2e-5 | Scaled for batch_size=16 (linear scaling rule) |
| LR (new layers) | 2e-4 | Scaled for batch_size=16 |
| Scheduler | Linear warmup (5%) + cosine annealing | Warmup prevents early transformer instability |
| Weight decay | 1e-2 | |
| Mixed precision | torch.cuda.amp (GradScaler) | |
| Batch size | 16 | A6000 48 GB VRAM |
| Epochs | 25 (max) | Early stopping patience = 7 |
| Checkpoint criterion | Best validation AUC | |
| Dropout | 0.1 | |
| Gradient clipping | max_norm = 1.0 | |
| Seeds | 7, 42, 123 | 3 seeds for statistical validity |
| 4-class weights | [1.0, 1.0, 1.0, 1.0] | Balanced dataset, uniform weights |

### Evaluation Metrics
- Balanced accuracy (primary — handles any residual class imbalance)
- Per-class Precision / Recall / F1
- Macro-averaged AUC-ROC (one-vs-rest)
- Confusion matrix (4×4)
- Expected Calibration Error (ECE)
- Computational cost (training time, inference throughput)

### Statistical Methodology
- All experiments run with 3 seeds → report mean ± std
- Paired bootstrap test (N=10,000) for significance claims between configurations
- 95% confidence intervals

---

## 7. Experiment Plan & Ablations

### 7.1 Core Experiments

| # | Experiment | Runs | Status |
|---|-----------|------|--------|
| E1 | 4-class classification on LAV-DF (20K) | 3 seeds | ❌ Pending |
| E2 | Cross-dataset generalization: train LAV-DF → test DFDC (binary) | 3 seeds | ❌ Future |

**Note:** Binary LAV-DF classification dropped — 4-class is more informative and subsumes binary (collapse 3 fake classes for binary metrics). DFDC evaluation will collapse 4-class → binary at inference time (no retraining).

### 7.2 Ablation Study 1: Transformer Depth (12 runs)

Vary per-branch transformer encoder layers AND cross-modal layers together.

| Config | Encoder layers | Cross-modal layers | Seeds | Purpose |
|--------|---------------|-------------------|-------|---------|
| D1 | 1 | 1 | 7, 42, 123 | Minimum viable depth |
| D2 | 2 | 2 | 7, 42, 123 | Default/moderate |
| D3 | 3 | 3 | 7, 42, 123 | Deeper |
| D4 | 4 | 4 | 7, 42, 123 | Diminishing returns? |

### 7.3 Ablation Study 2: Fusion Strategy (12 runs)

Uses best depth from Study 1. Tests whether cross-modal attention is necessary.

| Config | Mode | Description | Seeds |
|--------|------|-------------|-------|
| F1 | video_only | Video branch only, no audio | 7, 42, 123 |
| F2 | audio_only | Audio branch only, no video | 7, 42, 123 |
| F3 | concat | Both branches, concatenate (no cross-attention) | 7, 42, 123 |
| F4 | cross_modal | Both branches + bidirectional cross-attention (ours) | 7, 42, 123 |

### 7.4 Modality Occlusion Study (no training, evaluation only)

Run on best model from 7.3. Same weights, different inputs:

| Condition | Video input | Audio input | Purpose |
|-----------|-----------|-------------|---------|
| Full model | Real | Real | Baseline |
| Audio only | Zeros | Real | Quantify audio contribution |
| Video only | Real | Zeros | Quantify video contribution |

### 7.5 Explainability Analysis (no training, evaluation only)

| Method | What it shows | Output |
|--------|--------------|--------|
| Grad-CAM++ (video) | Where on face the model attends | Frame heatmaps |
| Grad-CAM++ (audio) | Where in spectrogram the model attends | Spectrogram heatmaps |
| Cross-modal attention weights | Which video-audio pairs are linked | Attention matrices |
| GradientSHAP | Per-pixel attribution with Shapley guarantees | Attribution maps |

### 7.6 Results Table Template

| Model Variant | Bal. Acc (mean±std) | AUC (mean±std) | Per-class F1 | ECE |
|--------------|---------------------|----------------|-------------|-----|
| **Depth ablation** | | | | |
| 1 layer | | | | |
| 2 layers | | | | |
| 3 layers | | | | |
| 4 layers | | | | |
| **Fusion ablation** | | | | |
| Video-only | | | | |
| Audio-only | | | | |
| Concat fusion | | | | |
| Cross-modal (ours) | | | | |
| **Modality occlusion** | | | | |
| Full model | | | | |
| Video zeroed | | | | |
| Audio zeroed | | | | |

### 7.7 Paper Figures Plan

| # | Figure | Type | Source |
|---|--------|------|--------|
| F1 | Architecture diagram | Schematic | Manual/tikz |
| F2 | Train/val loss curves (depth ablation) | 4 subplots, mean±std | plot_results.py |
| F3 | Depth ablation comparison | Bar chart (bacc + AUC) with error bars | plot_results.py |
| F4 | Fusion ablation comparison | Bar chart (bacc + AUC) with error bars | plot_results.py |
| F5 | Confusion matrix (best model) | 4×4 heatmap | plot_results.py |
| F6 | Per-class F1 comparison | Grouped bar chart | plot_results.py |
| F7 | Modality occlusion results | Bar chart | plot_results.py |
| F8 | Grad-CAM++ video examples | Face heatmaps (3–4 samples) | run_experiments.py |
| F9 | Grad-CAM++ audio examples | Spectrogram heatmaps (3–4 samples) | run_experiments.py |
| F10 | Cross-modal attention maps | Attention matrices | run_experiments.py |
| F11 | GradientSHAP attributions | Attribution overlays | run_experiments.py |

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
captum (needed for GradientSHAP explainability — install before evaluation)
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
| Audio preprocessing (5K) | ✅ Done | 02 June 2026 | 5,000/5,000, 0 failures, 15.6 min |
| Video preprocessing (5K) | ✅ Done | 02 June 2026 | 5,000/5,000, 0 failures, 12.4 min |
| Metadata CSV build (5K) | ✅ Done | 02 June 2026 | 5,000 entries, all matched |
| Dataset verification (5K) | ✅ Done | 02 June 2026 | ALL CHECKS PASSED |
| **Resampling to 20K** | ✅ Done | 03 June 2026 | Balanced 4-class (25% each), 3 seeds planned |
| **20K preprocessing** | ✅ Done | 03 June 2026 | 20,000/20,000, 0 failures, 4h37m |
| **20K verification** | ✅ Done | 03 June 2026 | All tensors verified, metadata CSV clean |
| **Old 5K data deleted** | ✅ Done | 03 June 2026 | Freed ~22.5 GB |
| **Tar archive deleted** | ✅ Done | 03 June 2026 | Freed ~24 GB |

### Phase 2: Model Training & Experiments

| Task | Status | Date | Notes |
|------|--------|------|-------|
| config.py updated (20K, 4-class) | 🔄 In progress | 03 June 2026 | |
| dataset.py updated (20K paths) | ❌ Pending | | |
| model.py (configurable architecture) | ❌ Pending | | Configurable depth + fusion mode |
| run_experiments.py (train + eval + ablations) | ❌ Pending | | Single file: training, testing, evaluation |
| Depth ablation (4 configs × 3 seeds) | ❌ Pending | | 12 runs, ~42 hours estimated |
| Fusion ablation (4 configs × 3 seeds) | ❌ Pending | | 12 runs, ~42 hours estimated |
| Modality occlusion study | ❌ Pending | | Evaluation only, no training |
| Grad-CAM++ + GradientSHAP | ❌ Pending | | Qualitative explainability |
| analysis.py (plots + bootstrap) | ❌ Pending | | All paper figures + significance tests |

### Phase 3: Paper

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Results tables | ❌ Pending | | |
| Figures (11 planned) | ❌ Pending | | See Section 7.7 |
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
- Full dataset preprocessing (136K videos) exceeded 400 GB on MacBook — switched to stratified sample
- 5K sample: Lab machine RTX A6000 processed 5K videos in 12.4 min (6.7 videos/sec)
- 20K sample: 20,000/20,000 in 4h37m (1.2 it/s combined video+audio), zero failures
- Audio tensor shape changed between runs: (1,128,94) at hop=512/3sec → (1,128,101) at hop=160/1sec with center=True
- MTCNN face detection ran at 6.7 videos/sec on A6000 with zero failures across both 5K and 20K
- torchaudio had zero audio backends on Windows — fixed by switching to ffmpeg subprocess + soundfile
- torchaudio.transforms.MelSpectrogram works fine (no backend needed for transforms, only for I/O)
- float16 storage confirmed working: tensors auto-cast to float32 at DataLoader time
- Disk usage scales linearly: 5K=22.5GB, 20K=92.5GB (as expected)

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
| 13 | 03 June | 5K test set too small for per-class statistical claims (~240/class) | Resampled to 20K (958/class in test set) | ✅ Resolved |
| 14 | 03 June | Tar archive at wrong nesting level (LAV-DF.tar at LAV-DF/LAV-DF/) | Found with recursive search, deleted to free 24 GB | ✅ Resolved |
| 15 | 03 June | Audio tensor shape (1,128,101) differs from 5K run (1,128,94) | Different params: 20K uses hop=160/1sec/center=True vs 5K hop=512/3sec. Model handles dynamically. | ✅ Noted |
| 16 | 03 June | Git push rejected from Mac (remote ahead) | Uploaded via GitHub web UI instead | ✅ Resolved |

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

### 02 June 2026 — Full Preprocessing Run (5K)
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

### 03 June 2026 — 20K Resampling + Research Design

**Decisions made:**
- Scaled from 5K → 20K samples for research rigour (958 test samples/class vs 240)
- Switched to 4-class prediction on LAV-DF (binary on DFDC planned for future)
- Balanced 4-class sampling (25% each) instead of proportional
- Added 3-seed training (7, 42, 123) for statistical validity
- Added paired bootstrap significance testing (replacing Wilcoxon — underpowered at 3 seeds)
- Added warmup scheduler (5% linear warmup → cosine decay)
- Scaled LR with batch size (1e-5→2e-5 pretrained, 1e-4→2e-4 new) per linear scaling rule
- Added explainability: GradientSHAP (captum) + modality occlusion study alongside Grad-CAM++
- Added Expected Calibration Error (ECE) as evaluation metric

**Preprocessing run:**
- Deleted tar archive (freed ~24 GB)
- resample_and_preprocess.py: 20K sampled (balanced 4-class) + preprocessed
- 20,000/20,000 processed, 0 failures, 4h37m on A6000
- Audio shape: (1, 128, 101) — different from 5K due to parameter changes
- Old 5K preprocessed data deleted (freed ~22.5 GB)
- Verified: metadata CSV clean, all tensor files exist

**Code file plan:**
- `config.py` — central configuration (updated)
- `dataset.py` — data loading (update paths)
- `model.py` — architecture with configurable depth + fusion mode
- `run_experiments.py` — training + ablation orchestration + test evaluation + explainability
- `analysis.py` — all paper figures + bootstrap significance tests

---

## 13. References

- LAV-DF Dataset: https://huggingface.co/datasets/ControlNet/LAV-DF
- EfficientNet: Tan & Le, ICML 2019
- VGGish: Hershey et al., ICASSP 2017
- Grad-CAM++: Chattopadhyay et al., WACV 2018
- Cross-Modal Transformers: Tsai et al. (MulT), ACL 2019
- GradientSHAP: Erion et al., NeurIPS 2021 (via captum library)
- Linear LR Scaling: Goyal et al., 2017
