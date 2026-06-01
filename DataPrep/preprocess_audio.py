"""
preprocess_audio.py — Extract audio from .mp4, compute mel-spectrograms, save as .pt

USAGE:
    conda activate dfd
    python preprocess_audio.py

For each video in sampled_metadata.json:
  1. Extract audio from .mp4 using torchaudio
  2. Convert to mono, resample to 16 kHz
  3. Pad or center-crop to 3 seconds (48000 samples)
  4. Compute log mel-spectrogram (128 mel bins)
  5. Save as float16 .pt tensor → shape (1, 128, T)

Estimated time: ~15-30 min for 5K videos
Estimated disk: ~250 MB (float16)
"""

import json
import os
import sys
import time

import torch
import torchaudio
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    SAMPLED_METADATA_JSON,
    AUDIO_TENSOR_DIR,
    AUDIO_SAMPLE_RATE,
    AUDIO_NUM_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    SAVE_DTYPE,
    create_output_dirs,
)


def process_audio(mp4_path: str, mel_transform) -> torch.Tensor:
    """
    Extract audio from .mp4, preprocess, and return mel-spectrogram tensor.
    Returns tensor of shape (1, 128, T) in float32.
    """
    # Load audio from mp4
    waveform, sr = torchaudio.load(mp4_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target sample rate
    if sr != AUDIO_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_SAMPLE_RATE)
        waveform = resampler(waveform)

    # Pad or center-crop to fixed length
    n_samples = waveform.shape[1]
    if n_samples < AUDIO_NUM_SAMPLES:
        # Pad with zeros
        pad_amount = AUDIO_NUM_SAMPLES - n_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif n_samples > AUDIO_NUM_SAMPLES:
        # Center crop
        start = (n_samples - AUDIO_NUM_SAMPLES) // 2
        waveform = waveform[:, start:start + AUDIO_NUM_SAMPLES]

    # Compute mel-spectrogram
    mel_spec = mel_transform(waveform)  # (1, n_mels, T)

    # Log scaling
    mel_spec = torch.log(mel_spec + 1e-9)

    return mel_spec


def main():
    print("=" * 60)
    print("AUDIO PREPROCESSING")
    print("=" * 60)

    # Check sampled metadata
    if not os.path.exists(SAMPLED_METADATA_JSON):
        print(f"[!] Sampled metadata not found: {SAMPLED_METADATA_JSON}")
        print("    Run sample_dataset.py first.")
        sys.exit(1)

    # Create output dirs
    create_output_dirs()

    # Load sampled metadata
    with open(SAMPLED_METADATA_JSON, "r") as f:
        metadata = json.load(f)

    print(f"Videos to process: {len(metadata):,}")
    print(f"Output directory:  {AUDIO_TENSOR_DIR}")
    print(f"Save dtype:        {SAVE_DTYPE}")

    # Create mel-spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    save_torch_dtype = torch.float16 if SAVE_DTYPE == "float16" else torch.float32

    # Process
    success = 0
    failed = 0
    skipped = 0
    failed_files = []

    start_time = time.time()

    for entry in tqdm(metadata, desc="Processing audio"):
        video_file = entry["file"]  # e.g., "train/000001.mp4"
        mp4_path = os.path.join(DATASET_ROOT, video_file)

        # Output filename: use flat name to avoid nested dirs
        # "train/000001.mp4" → "train_000001.pt"
        tensor_name = video_file.replace("/", "_").replace("\\", "_").replace(".mp4", ".pt")
        save_path = os.path.join(AUDIO_TENSOR_DIR, tensor_name)

        # Skip if already processed
        if os.path.exists(save_path):
            skipped += 1
            continue

        if not os.path.exists(mp4_path):
            failed += 1
            failed_files.append(video_file)
            continue

        try:
            mel_tensor = process_audio(mp4_path, mel_transform)
            # Save as float16
            torch.save(mel_tensor.to(save_torch_dtype), save_path)
            success += 1
        except Exception as e:
            failed += 1
            failed_files.append(video_file)
            if failed <= 5:  # Print first 5 errors
                tqdm.write(f"  [FAIL] {video_file}: {e}")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print(f"AUDIO PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed: {success:,}")
    print(f"  Skipped:   {skipped:,} (already existed)")
    print(f"  Failed:    {failed:,}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if failed_files:
        fail_log = os.path.join(AUDIO_TENSOR_DIR, "_failed_audio.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed_files))
        print(f"  Failed list saved to: {fail_log}")

    # Verify a random sample
    sample_files = [f for f in os.listdir(AUDIO_TENSOR_DIR) if f.endswith(".pt")]
    if sample_files:
        sample = torch.load(os.path.join(AUDIO_TENSOR_DIR, sample_files[0]),
                            weights_only=True)
        print(f"\n  Sample tensor shape: {sample.shape}")
        print(f"  Sample tensor dtype: {sample.dtype}")
        print(f"  Sample file size:    {os.path.getsize(os.path.join(AUDIO_TENSOR_DIR, sample_files[0])) / 1024:.1f} KB")

    print(f"\nTotal audio tensors on disk: {len(sample_files):,}")
    print(f"\nDone! Next step: python preprocess_video.py")


if __name__ == "__main__":
    main()
