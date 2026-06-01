"""
config.py — Single source of truth for the deepfake detection project.
All paths, hyperparameters, and settings live here.

IMPORTANT: Adjust DATASET_ROOT and OUTPUT_DIR if your dataset is on a different drive.
"""

import os
import platform

# =============================================================================
# AUTO-DETECT OS & SET BASE PATHS
# =============================================================================
IS_WINDOWS = platform.system() == "Windows"
HOME = os.path.expanduser("~")

if IS_WINDOWS:
    # Lab machine (Windows) — adjust drive letter if dataset is on D:/ etc.
    BASE_DIR = os.path.join(HOME, "Desktop", "deepfake_detection")
    DATASET_ROOT = os.path.join(HOME, "Desktop", "LAV-DF", "LAV-DF")
    OUTPUT_DIR = os.path.join(HOME, "Desktop", "LAV-DF-preprocessed")
else:
    # MacBook (macOS)
    BASE_DIR = os.path.join(HOME, "Desktop", "deepfake_detection")
    DATASET_ROOT = os.path.join(HOME, "Desktop", "LAV-DF", "LAV-DF")
    OUTPUT_DIR = os.path.join(HOME, "Desktop", "LAV-DF-preprocessed")

# =============================================================================
# DATASET PATHS
# =============================================================================
METADATA_JSON = os.path.join(DATASET_ROOT, "metadata.json")
SAMPLED_METADATA_JSON = os.path.join(OUTPUT_DIR, "sampled_metadata.json")

# Output directories for preprocessed tensors
VIDEO_TENSOR_DIR = os.path.join(OUTPUT_DIR, "video_tensors")
AUDIO_TENSOR_DIR = os.path.join(OUTPUT_DIR, "audio_tensors")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

# =============================================================================
# SAMPLING CONFIG
# =============================================================================
SAMPLE_SIZE = 5000          # Total videos to sample (stratified across 4 classes)
RANDOM_SEED = 42            # For reproducibility
SAVE_DTYPE = "float16"      # float16 to halve disk usage (~47 GB → ~24 GB)

# =============================================================================
# VIDEO PREPROCESSING
# =============================================================================
NUM_FRAMES = 16             # Uniformly sampled frames per video
FRAME_SIZE = 224            # EfficientNet-B0 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MTCNN_MIN_FACE_SIZE = 40   # Minimum face size for MTCNN detection
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]  # Detection thresholds

# =============================================================================
# AUDIO PREPROCESSING
# =============================================================================
AUDIO_SAMPLE_RATE = 16000   # 16 kHz (VGGish pretrained)
AUDIO_DURATION_SEC = 3.0    # seconds to keep
AUDIO_NUM_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC)  # 48000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# =============================================================================
# LABEL MAPPING
# =============================================================================
# 4-class labels
LABEL_MAP_4CLASS = {
    (False, False): 0,  # Real-Real
    (True, False): 1,   # Fake-Video
    (False, True): 2,   # Fake-Audio
    (True, True): 3,    # Fake-Both
}
LABEL_NAMES_4CLASS = ["Real", "Fake-Video", "Fake-Audio", "Fake-Both"]

# Binary labels
LABEL_MAP_BINARY = {
    (False, False): 0,  # Real
    (True, False): 1,   # Fake
    (False, True): 1,   # Fake
    (True, True): 1,    # Fake
}

# =============================================================================
# TRAINING HYPERPARAMETERS (for later use in train.py)
# =============================================================================
BATCH_SIZE = 8
NUM_EPOCHS = 25
LR_PRETRAINED = 1e-5        # EfficientNet, VGGish
LR_NEW_LAYERS = 1e-4        # Transformers, classifier
WEIGHT_DECAY = 1e-2
D_MODEL = 256
N_HEADS = 4
DIM_FEEDFORWARD = 512
NUM_ENCODER_LAYERS = 2
NUM_CROSS_MODAL_LAYERS = 2
DROPOUT = 0.1

# =============================================================================
# DATALOADER
# =============================================================================
NUM_WORKERS = 4 if IS_WINDOWS else 4
PIN_MEMORY = True


def create_output_dirs():
    """Create all necessary output directories."""
    dirs = [OUTPUT_DIR, VIDEO_TENSOR_DIR, AUDIO_TENSOR_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}")


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPFAKE DETECTION — CONFIG")
    print("=" * 60)
    print(f"OS:              {platform.system()}")
    print(f"Dataset root:    {DATASET_ROOT}")
    print(f"Output dir:      {OUTPUT_DIR}")
    print(f"Sample size:     {SAMPLE_SIZE}")
    print(f"Save dtype:      {SAVE_DTYPE}")
    print(f"Metadata JSON:   {METADATA_JSON}")
    print(f"JSON exists:     {os.path.exists(METADATA_JSON)}")
    print()
    print("Creating output directories...")
    create_output_dirs()
    print("\nDone!")
