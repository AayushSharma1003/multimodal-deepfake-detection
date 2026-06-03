"""
resample_and_preprocess.py — 20K LAV-DF Resampling + Preprocessing
====================================================================
Usage:
    python resample_and_preprocess.py --phase sample     # Step 1: Sample 20K, print stats
    python resample_and_preprocess.py --phase preprocess  # Step 2: Preprocess sampled videos
    python resample_and_preprocess.py --phase both        # Both steps sequentially

Resumable: skips already-processed tensors on restart.
"""

import os
import json
import argparse
import subprocess
import warnings
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── CONFIGURATION (edit these if paths differ) ──────────────────────

# Original dataset
DATASET_ROOT = r"C:\Users\Komal-Sch\Desktop\LAV-DF\LAV-DF\LAV-DF"
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.min.json")  # smaller, faster
METADATA_FALLBACK = os.path.join(DATASET_ROOT, "metadata.json")

# Output directory (new, separate from old 5K)
OUTPUT_ROOT = r"C:\Users\Komal-Sch\Desktop\LAV-DF-preprocessed-20k"
VIDEO_TENSOR_DIR = os.path.join(OUTPUT_ROOT, "video_tensors")
AUDIO_TENSOR_DIR = os.path.join(OUTPUT_ROOT, "audio_tensors")
METADATA_CSV = os.path.join(OUTPUT_ROOT, "metadata.csv")
SAMPLE_JSON = os.path.join(OUTPUT_ROOT, "sampled_entries.json")

# Sampling
TOTAL_SAMPLES = 20_000
RANDOM_SEED = 42

# Video preprocessing
NUM_FRAMES = 16
FRAME_SIZE = 224
MTCNN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio preprocessing
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 1.0  # take first 1 second
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160
# Expected output: (1, 128, 94)

# ─── 4-CLASS LABEL MAPPING ───────────────────────────────────────────

LABEL_MAP = {
    (False, False): {"label_4class": 0, "label_name": "Real"},
    (True,  False): {"label_4class": 1, "label_name": "Fake-Video"},
    (False, True):  {"label_4class": 2, "label_name": "Fake-Audio"},
    (True,  True):  {"label_4class": 3, "label_name": "Fake-Both"},
}


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD METADATA + STRATIFIED SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def load_metadata():
    """Load the full LAV-DF metadata JSON."""
    path = METADATA_FILE if os.path.exists(METADATA_FILE) else METADATA_FALLBACK
    print(f"[sample] Loading metadata from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[sample] Total entries in dataset: {len(data):,}")
    return data


def assign_labels(entries):
    """Add 4-class label and binary label to each entry."""
    for e in entries:
        mv = e["modify_video"]
        ma = e["modify_audio"]
        info = LABEL_MAP[(mv, ma)]
        e["label_4class"] = info["label_4class"]
        e["label_name"] = info["label_name"]
        e["label_binary"] = 0 if (not mv and not ma) else 1
    return entries


def stratified_sample(entries, n_total, seed):
    """
    Stratified sampling: maintain original split proportions AND
    class balance within each split.
    """
    rng = np.random.RandomState(seed)

    # Group by (split, label_4class)
    groups = {}
    for e in entries:
        key = (e["split"], e["label_4class"])
        groups.setdefault(key, []).append(e)

    # Count per split to compute split proportions
    split_counts = Counter(e["split"] for e in entries)
    total = sum(split_counts.values())
    split_ratios = {s: c / total for s, c in split_counts.items()}

    print(f"\n[sample] Original split distribution:")
    for s in ["train", "dev", "test"]:
        if s in split_counts:
            print(f"  {s}: {split_counts[s]:,} ({split_ratios[s]:.1%})")

    # Allocate samples per split
    split_targets = {}
    for s in ["train", "dev", "test"]:
        split_targets[s] = int(round(n_total * split_ratios.get(s, 0)))
    # Fix rounding
    diff = n_total - sum(split_targets.values())
    split_targets["train"] += diff

    # Within each split, allocate equally across 4 classes (balanced)
    sampled = []
    for split_name in ["train", "dev", "test"]:
        target = split_targets[split_name]
        per_class = target // 4
        remainder = target % 4

        for cls_idx in range(4):
            key = (split_name, cls_idx)
            pool = groups.get(key, [])
            n_pick = per_class + (1 if cls_idx < remainder else 0)
            n_pick = min(n_pick, len(pool))

            if n_pick < len(pool):
                indices = rng.choice(len(pool), size=n_pick, replace=False)
                sampled.extend([pool[i] for i in indices])
            else:
                sampled.extend(pool)
                if n_pick > len(pool):
                    print(f"  [warn] {split_name}/class_{cls_idx}: wanted {n_pick}, "
                          f"only {len(pool)} available")

    return sampled


def run_sampling():
    """Execute the full sampling pipeline."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    entries = load_metadata()
    entries = assign_labels(entries)

    sampled = stratified_sample(entries, TOTAL_SAMPLES, RANDOM_SEED)

    # Print statistics
    print(f"\n[sample] Sampled {len(sampled):,} videos")
    print(f"\n[sample] Split distribution:")
    split_counter = Counter(e["split"] for e in sampled)
    for s in ["train", "dev", "test"]:
        print(f"  {s}: {split_counter.get(s, 0):,}")

    print(f"\n[sample] 4-class distribution:")
    class_counter = Counter(e["label_name"] for e in sampled)
    for name in ["Real", "Fake-Video", "Fake-Audio", "Fake-Both"]:
        print(f"  {name}: {class_counter.get(name, 0):,} "
              f"({class_counter.get(name, 0)/len(sampled):.1%})")

    print(f"\n[sample] Per-split class breakdown:")
    for s in ["train", "dev", "test"]:
        split_entries = [e for e in sampled if e["split"] == s]
        cls_counts = Counter(e["label_name"] for e in split_entries)
        print(f"  {s}:")
        for name in ["Real", "Fake-Video", "Fake-Audio", "Fake-Both"]:
            print(f"    {name}: {cls_counts.get(name, 0)}")

    # Save sampled entries for preprocessing step
    with open(SAMPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(sampled, f)
    print(f"\n[sample] Saved sampled entries → {SAMPLE_JSON}")

    return sampled


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def init_mtcnn():
    """Initialise MTCNN face detector."""
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(
        image_size=FRAME_SIZE,
        margin=40,
        keep_all=False,
        post_process=False,  # return 0-255 tensor, we normalise ourselves
        device=MTCNN_DEVICE,
    )
    print(f"[preprocess] MTCNN on {MTCNN_DEVICE}")
    return mtcnn


def init_mel_transform():
    """Initialise mel spectrogram transform (no backend needed)."""
    import torchaudio.transforms as T
    mel = T.MelSpectrogram(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    print(f"[preprocess] MelSpectrogram: sr={AUDIO_SAMPLE_RATE}, "
          f"n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}")
    return mel


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract num_frames uniformly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(None)
    cap.release()
    return frames


def process_video(video_path, mtcnn):
    """
    Extract faces from 16 frames → (16, 3, 224, 224) float16 tensor.
    Falls back to center crop if MTCNN fails on a frame.
    """
    frames = extract_frames(video_path)
    if frames is None:
        return None

    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    processed = []
    for frame in frames:
        if frame is None:
            # Blank frame fallback
            processed.append(torch.zeros(3, FRAME_SIZE, FRAME_SIZE))
            continue

        # Try MTCNN face detection
        face = mtcnn(frame)  # returns (3, 224, 224) or None

        if face is not None:
            face = face / 255.0  # MTCNN with post_process=False gives 0-255
            face = (face - mean) / std
            processed.append(face)
        else:
            # Center crop fallback
            from PIL import Image
            from torchvision import transforms
            pil_img = Image.fromarray(frame)
            transform = transforms.Compose([
                transforms.CenterCrop(min(pil_img.size)),
                transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            processed.append(transform(pil_img))

    tensor = torch.stack(processed)  # (16, 3, 224, 224)
    return tensor.half()  # float16


def process_audio(video_path, mel_transform, temp_dir):
    """
    Extract audio from video → mel spectrogram → (1, 128, 94) float16 tensor.
    Uses ffmpeg subprocess + soundfile (no torchaudio backend needed).
    """
    import soundfile as sf

    video_id = Path(video_path).stem
    temp_wav = os.path.join(temp_dir, f"{video_id}.wav")

    # Extract audio with ffmpeg
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                      # no video
        "-acodec", "pcm_s16le",     # 16-bit PCM
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",                 # mono
        temp_wav,
        "-y",                       # overwrite
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(temp_wav):
        return None

    try:
        audio, sr = sf.read(temp_wav)
    except Exception:
        return None
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    # Take first AUDIO_DURATION_SEC seconds
    n_samples = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC)
    if len(audio) < n_samples:
        # Pad with zeros if too short
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]

    # Compute mel spectrogram
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, n_samples)
    mel = mel_transform(waveform)       # (1, n_mels, time)
    mel = torch.log(mel + 1e-9)         # log-mel

    return mel.half()  # float16, shape (1, 128, 94)


def run_preprocessing():
    """Execute preprocessing on all sampled videos."""
    # Load sampled entries
    if not os.path.exists(SAMPLE_JSON):
        print("[preprocess] No sampled entries found. Run --phase sample first.")
        return

    with open(SAMPLE_JSON, "r") as f:
        sampled = json.load(f)
    print(f"[preprocess] Loaded {len(sampled):,} sampled entries")

    # Create output directories
    os.makedirs(VIDEO_TENSOR_DIR, exist_ok=True)
    os.makedirs(AUDIO_TENSOR_DIR, exist_ok=True)
    temp_dir = os.path.join(OUTPUT_ROOT, "_temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    # Init models
    mtcnn = init_mtcnn()
    mel_transform = init_mel_transform()

    # Process each video
    success = 0
    failures = []
    metadata_rows = []

    for entry in tqdm(sampled, desc="[preprocess]"):
        video_file = entry["file"]                              # e.g. "train/136719.mp4"
        video_id = Path(video_file).stem                        # e.g. "136719"
        video_path = os.path.join(DATASET_ROOT, video_file)     # full path

        # Output paths
        vt_path = os.path.join(VIDEO_TENSOR_DIR, f"{video_id}.pt")
        at_path = os.path.join(AUDIO_TENSOR_DIR, f"{video_id}.pt")

        # Skip if already processed (resumability)
        if os.path.exists(vt_path) and os.path.exists(at_path):
            # Still add to metadata
            metadata_rows.append({
                "video_id": video_id,
                "original_file": video_file,
                "split": entry["split"],
                "video_tensor": vt_path,
                "audio_tensor": at_path,
                "label_binary": entry["label_binary"],
                "label_4class": entry["label_4class"],
                "label_name": entry["label_name"],
                "modify_video": entry["modify_video"],
                "modify_audio": entry["modify_audio"],
                "duration": entry.get("duration", 0),
            })
            success += 1
            continue

        # Check video exists
        if not os.path.exists(video_path):
            failures.append((video_id, "file not found"))
            continue

        # Process video
        video_tensor = process_video(video_path, mtcnn)
        if video_tensor is None:
            failures.append((video_id, "video processing failed"))
            continue

        # Process audio
        audio_tensor = process_audio(video_path, mel_transform, temp_dir)
        if audio_tensor is None:
            failures.append((video_id, "audio processing failed"))
            continue

        # Save tensors
        torch.save(video_tensor, vt_path)
        torch.save(audio_tensor, at_path)

        metadata_rows.append({
            "video_id": video_id,
            "original_file": video_file,
            "split": entry["split"],
            "video_tensor": vt_path,
            "audio_tensor": at_path,
            "label_binary": entry["label_binary"],
            "label_4class": entry["label_4class"],
            "label_name": entry["label_name"],
            "modify_video": entry["modify_video"],
            "modify_audio": entry["modify_audio"],
            "duration": entry.get("duration", 0),
        })
        success += 1

    # Cleanup temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    # Save metadata CSV
    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_CSV, index=False)

    # Report
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Success: {success:,}")
    print(f"  Failures: {len(failures)}")
    if failures:
        print(f"\n  First 10 failures:")
        for vid, reason in failures[:10]:
            print(f"    {vid}: {reason}")
    print(f"\n  Video tensors: {VIDEO_TENSOR_DIR}")
    print(f"  Audio tensors: {AUDIO_TENSOR_DIR}")
    print(f"  Metadata CSV:  {METADATA_CSV}")

    # Verify shapes on a random sample
    sample_vt = torch.load(
        os.path.join(VIDEO_TENSOR_DIR, os.listdir(VIDEO_TENSOR_DIR)[0]),
        weights_only=True,
    )
    sample_at = torch.load(
        os.path.join(AUDIO_TENSOR_DIR, os.listdir(AUDIO_TENSOR_DIR)[0]),
        weights_only=True,
    )
    print(f"\n  Video tensor shape: {sample_vt.shape} ({sample_vt.dtype})")
    print(f"  Audio tensor shape: {sample_at.shape} ({sample_at.dtype})")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="20K LAV-DF Resampling + Preprocessing")
    parser.add_argument(
        "--phase",
        choices=["sample", "preprocess", "both"],
        default="both",
        help="Which phase to run (default: both)",
    )
    args = parser.parse_args()

    if args.phase in ("sample", "both"):
        run_sampling()

    if args.phase in ("preprocess", "both"):
        run_preprocessing()
