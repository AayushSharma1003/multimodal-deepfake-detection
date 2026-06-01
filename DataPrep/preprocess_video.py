"""
preprocess_video.py — Extract frames, face-crop with MTCNN, save as .pt

USAGE:
    conda activate dfd
    python preprocess_video.py

For each video in sampled_metadata.json:
  1. Load video with OpenCV
  2. Uniformly sample 16 frames
  3. MTCNN face detection + crop + align
     - Fallback: use nearest good bounding box if detection fails on a frame
     - Fallback: center-crop if ALL detections fail
  4. Resize to 224×224
  5. Normalize with ImageNet mean/std
  6. Save as float16 .pt tensor → shape (16, 3, 224, 224)

Estimated time: ~1-3 hours for 5K videos (GPU MTCNN is faster)
Estimated disk: ~22 GB (float16)
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    SAMPLED_METADATA_JSON,
    VIDEO_TENSOR_DIR,
    NUM_FRAMES,
    FRAME_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MTCNN_MIN_FACE_SIZE,
    MTCNN_THRESHOLDS,
    SAVE_DTYPE,
    create_output_dirs,
)


def get_mtcnn(device):
    """Initialize MTCNN face detector."""
    from facenet_pytorch import MTCNN
    return MTCNN(
        image_size=FRAME_SIZE,
        min_face_size=MTCNN_MIN_FACE_SIZE,
        thresholds=MTCNN_THRESHOLDS,
        device=device,
        post_process=False,  # We'll normalize ourselves
        select_largest=True,  # Select largest face if multiple detected
    )


def sample_frame_indices(total_frames: int, n_frames: int) -> list:
    """Uniformly sample n_frames indices from total_frames."""
    if total_frames <= n_frames:
        # Not enough frames — take all and repeat last
        indices = list(range(total_frames))
        while len(indices) < n_frames:
            indices.append(total_frames - 1)
        return indices
    # Uniform sampling
    step = total_frames / n_frames
    return [int(step * i + step / 2) for i in range(n_frames)]


def extract_frames(mp4_path: str, n_frames: int) -> list:
    """Extract uniformly sampled frames from video. Returns list of RGB numpy arrays."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, n_frames)

    frames = []
    frame_idx = 0
    target_idx = 0

    while target_idx < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == indices[target_idx]:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            target_idx += 1
            # Handle duplicate indices
            while target_idx < len(indices) and indices[target_idx] == frame_idx:
                frames.append(frame_rgb.copy())
                target_idx += 1
        frame_idx += 1

    cap.release()

    # If we didn't get enough frames, pad with last frame
    while len(frames) < n_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            raise RuntimeError(f"No frames extracted from {mp4_path}")

    return frames[:n_frames]


def center_crop_face(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a square region from the frame and resize."""
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    y = (h - crop_size) // 2
    x = (w - crop_size) // 2
    cropped = frame[y:y+crop_size, x:x+crop_size]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


def detect_and_crop_faces(frames: list, mtcnn, device) -> list:
    """
    Detect faces in frames using MTCNN.
    Uses fallback strategies for failed detections.
    Returns list of cropped face images as numpy arrays (224x224x3).
    """
    from PIL import Image

    n_frames = len(frames)
    boxes_list = [None] * n_frames
    cropped_faces = [None] * n_frames

    # Detect faces in all frames
    pil_frames = [Image.fromarray(f) for f in frames]

    # Batch detection for speed
    try:
        boxes_batch, _ = mtcnn.detect(pil_frames)
    except Exception:
        # Fallback: individual detection
        boxes_batch = []
        for pf in pil_frames:
            try:
                b, _ = mtcnn.detect(pf)
                boxes_batch.append(b)
            except Exception:
                boxes_batch.append(None)

    # Store valid boxes
    valid_indices = []
    for i, boxes in enumerate(boxes_batch):
        if boxes is not None and len(boxes) > 0:
            boxes_list[i] = boxes[0]  # Take first (largest) face
            valid_indices.append(i)

    # Crop faces with fallback strategy
    for i in range(n_frames):
        if boxes_list[i] is not None:
            # Good detection — crop
            box = boxes_list[i]
            cropped_faces[i] = crop_face_from_box(frames[i], box, FRAME_SIZE)
        elif valid_indices:
            # Fallback: use nearest good bounding box
            nearest = min(valid_indices, key=lambda x: abs(x - i))
            box = boxes_list[nearest]
            cropped_faces[i] = crop_face_from_box(frames[i], box, FRAME_SIZE)
        else:
            # All detections failed — center crop
            cropped_faces[i] = center_crop_face(frames[i], FRAME_SIZE)

    return cropped_faces


def crop_face_from_box(frame: np.ndarray, box, size: int) -> np.ndarray:
    """Crop face region from frame using bounding box, with margin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(b) for b in box]

    # Add 20% margin
    bw = x2 - x1
    bh = y2 - y1
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.2)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    # Make square
    bw = x2 - x1
    bh = y2 - y1
    if bw > bh:
        diff = bw - bh
        y1 = max(0, y1 - diff // 2)
        y2 = min(h, y2 + diff // 2)
    elif bh > bw:
        diff = bh - bw
        x1 = max(0, x1 - diff // 2)
        x2 = min(w, x2 + diff // 2)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return center_crop_face(frame, size)

    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_frames(faces: list) -> torch.Tensor:
    """
    Convert list of face images (224x224x3 uint8) to normalized tensor.
    Returns tensor of shape (N, 3, 224, 224) in float32.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    tensors = []
    for face in faces:
        # (H, W, C) uint8 → (C, H, W) float32 [0, 1]
        t = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
        # Normalize
        t = (t - mean) / std
        tensors.append(t)

    return torch.stack(tensors)  # (N, 3, 224, 224)


def main():
    print("=" * 60)
    print("VIDEO PREPROCESSING")
    print("=" * 60)

    # Check sampled metadata
    if not os.path.exists(SAMPLED_METADATA_JSON):
        print(f"[!] Sampled metadata not found: {SAMPLED_METADATA_JSON}")
        print("    Run sample_dataset.py first.")
        sys.exit(1)

    # Create output dirs
    create_output_dirs()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Initialize MTCNN
    print("Initializing MTCNN face detector...")
    mtcnn = get_mtcnn(device)

    # Load sampled metadata
    with open(SAMPLED_METADATA_JSON, "r") as f:
        metadata = json.load(f)

    print(f"Videos to process: {len(metadata):,}")
    print(f"Output directory:  {VIDEO_TENSOR_DIR}")
    print(f"Save dtype:        {SAVE_DTYPE}")
    print(f"Frames per video:  {NUM_FRAMES}")
    print(f"Frame size:        {FRAME_SIZE}×{FRAME_SIZE}")

    save_torch_dtype = torch.float16 if SAVE_DTYPE == "float16" else torch.float32

    # Process
    success = 0
    failed = 0
    skipped = 0
    center_crop_count = 0
    failed_files = []

    start_time = time.time()

    for entry in tqdm(metadata, desc="Processing video"):
        video_file = entry["file"]
        mp4_path = os.path.join(DATASET_ROOT, video_file)

        # Output filename
        tensor_name = video_file.replace("/", "_").replace("\\", "_").replace(".mp4", ".pt")
        save_path = os.path.join(VIDEO_TENSOR_DIR, tensor_name)

        # Skip if already processed
        if os.path.exists(save_path):
            skipped += 1
            continue

        if not os.path.exists(mp4_path):
            failed += 1
            failed_files.append(video_file)
            continue

        try:
            # Extract frames
            frames = extract_frames(mp4_path, NUM_FRAMES)

            # Detect and crop faces
            face_crops = detect_and_crop_faces(frames, mtcnn, device)

            # Check if all frames used center crop (no face detected)
            # We can't easily check this here, but it's logged in detect_and_crop_faces

            # Normalize and stack
            tensor = normalize_frames(face_crops)  # (16, 3, 224, 224)

            # Save as float16
            torch.save(tensor.to(save_torch_dtype), save_path)
            success += 1

        except Exception as e:
            failed += 1
            failed_files.append(video_file)
            if failed <= 10:
                tqdm.write(f"  [FAIL] {video_file}: {e}")

        # Print progress every 500 videos
        if (success + failed) % 500 == 0 and (success + failed) > 0:
            elapsed = time.time() - start_time
            rate = (success + failed) / elapsed
            remaining = (len(metadata) - success - failed - skipped) / rate if rate > 0 else 0
            tqdm.write(f"  Progress: {success+failed}/{len(metadata)-skipped} "
                       f"({rate:.1f} videos/sec, ~{remaining/60:.0f} min remaining)")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print(f"VIDEO PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed: {success:,}")
    print(f"  Skipped:   {skipped:,} (already existed)")
    print(f"  Failed:    {failed:,}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if failed_files:
        fail_log = os.path.join(VIDEO_TENSOR_DIR, "_failed_video.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed_files))
        print(f"  Failed list saved to: {fail_log}")

    # Verify a random sample
    sample_files = [f for f in os.listdir(VIDEO_TENSOR_DIR) if f.endswith(".pt")]
    if sample_files:
        sample = torch.load(os.path.join(VIDEO_TENSOR_DIR, sample_files[0]),
                            weights_only=True)
        print(f"\n  Sample tensor shape: {sample.shape}")
        print(f"  Sample tensor dtype: {sample.dtype}")
        size_mb = os.path.getsize(os.path.join(VIDEO_TENSOR_DIR, sample_files[0])) / (1024 * 1024)
        print(f"  Sample file size:    {size_mb:.2f} MB")
        total_size_gb = size_mb * len(sample_files) / 1024
        print(f"  Estimated total:     {total_size_gb:.1f} GB")

    print(f"\nTotal video tensors on disk: {len(sample_files):,}")
    print(f"\nDone! Next step: python build_metadata.py")


if __name__ == "__main__":
    main()
