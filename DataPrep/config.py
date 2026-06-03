"""
config.py — Central configuration for multimodal deepfake detection
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────

DATASET_ROOT = r"C:\Users\Komal-Sch\Desktop\LAV-DF\LAV-DF\LAV-DF"
PREPROCESSED_ROOT = r"C:\Users\Komal-Sch\Desktop\LAV-DF-preprocessed-20k"
METADATA_CSV = os.path.join(PREPROCESSED_ROOT, "metadata.csv")
CHECKPOINT_DIR = os.path.join(PREPROCESSED_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PREPROCESSED_ROOT, "results")
PLOTS_DIR = os.path.join(PREPROCESSED_ROOT, "plots")

# ─── Dataset ──────────────────────────────────────────────────────────

NUM_CLASSES = 4
LABEL_TYPE = "4class"          # "4class" or "binary"
CLASS_NAMES = ["Real", "Fake-Video", "Fake-Audio", "Fake-Both"]

# Class weights (balanced dataset ≈ 25% each, so uniform)
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# ─── Model architecture ──────────────────────────────────────────────

D_MODEL = 256
N_HEADS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# These are DEFAULTS; ablation runner overrides them
NUM_ENCODER_LAYERS = 2
NUM_CROSS_MODAL_LAYERS = 2

# ─── Training ─────────────────────────────────────────────────────────

BATCH_SIZE = 16
NUM_EPOCHS = 25
LR_PRETRAINED = 2e-5           # scaled for batch_size=16 (was 1e-5 at bs=8)
LR_NEW_LAYERS = 2e-4           # scaled for batch_size=16 (was 1e-4 at bs=8)
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.05            # 5% of total steps for linear warmup
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 7

# ─── Ablation study ──────────────────────────────────────────────────

SEEDS = [7, 42, 123]

# Depth ablation: vary transformer encoder layers (per-branch + cross-modal)
DEPTH_ABLATION = [1, 2, 3, 4]

# Fusion ablation (uses best depth from depth ablation)
FUSION_MODES = [
    "video_only",       # only video branch, no audio
    "audio_only",       # only audio branch, no video
    "concat",           # both branches, simple concatenation (no cross-attention)
    "cross_modal",      # both branches + bidirectional cross-attention (ours)
]

# ─── Evaluation ───────────────────────────────────────────────────────

GRADCAM_SAMPLES = 8            # number of samples for Grad-CAM++ visualisation
BOOTSTRAP_N = 10_000           # bootstrap resamples for significance test
<<<<<<< Updated upstream
BOOTSTRAP_CI = 0.95            # confidence interval
=======
BOOTSTRAP_CI = 0.95            # confidence interval
>>>>>>> Stashed changes
