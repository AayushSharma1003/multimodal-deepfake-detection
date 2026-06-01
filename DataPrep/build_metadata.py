"""
build_metadata.py — Build metadata.csv linking preprocessed tensors with labels.

USAGE:
    conda activate dfd
    python build_metadata.py

Creates a CSV with columns:
  video_id, split, video_tensor_path, audio_tensor_path, 
  label_binary, label_4class, modify_video, modify_audio, duration
  
Only includes entries where BOTH video and audio tensors exist.
"""

import json
import os
import sys

import pandas as pd
from config import (
    SAMPLED_METADATA_JSON,
    VIDEO_TENSOR_DIR,
    AUDIO_TENSOR_DIR,
    METADATA_CSV,
    OUTPUT_DIR,
    LABEL_MAP_4CLASS,
    LABEL_MAP_BINARY,
    LABEL_NAMES_4CLASS,
)


def main():
    print("=" * 60)
    print("BUILD METADATA CSV")
    print("=" * 60)

    if not os.path.exists(SAMPLED_METADATA_JSON):
        print(f"[!] Sampled metadata not found: {SAMPLED_METADATA_JSON}")
        print("    Run sample_dataset.py first.")
        sys.exit(1)

    with open(SAMPLED_METADATA_JSON, "r") as f:
        metadata = json.load(f)

    print(f"Sampled entries: {len(metadata):,}")

    rows = []
    missing_video = 0
    missing_audio = 0
    missing_both = 0

    for entry in metadata:
        video_file = entry["file"]  # e.g., "train/000001.mp4"
        tensor_name = video_file.replace("/", "_").replace("\\", "_").replace(".mp4", ".pt")

        video_tensor_path = os.path.join(VIDEO_TENSOR_DIR, tensor_name)
        audio_tensor_path = os.path.join(AUDIO_TENSOR_DIR, tensor_name)

        has_video = os.path.exists(video_tensor_path)
        has_audio = os.path.exists(audio_tensor_path)

        if not has_video and not has_audio:
            missing_both += 1
            continue
        if not has_video:
            missing_video += 1
            continue
        if not has_audio:
            missing_audio += 1
            continue

        mv = entry["modify_video"]
        ma = entry["modify_audio"]

        # Use basename without extension as video_id
        video_id = os.path.splitext(os.path.basename(video_file))[0]

        rows.append({
            "video_id": video_id,
            "original_file": video_file,
            "split": entry["split"],
            "video_tensor": video_tensor_path,
            "audio_tensor": audio_tensor_path,
            "label_binary": LABEL_MAP_BINARY[(mv, ma)],
            "label_4class": LABEL_MAP_4CLASS[(mv, ma)],
            "label_name": LABEL_NAMES_4CLASS[LABEL_MAP_4CLASS[(mv, ma)]],
            "modify_video": int(mv),
            "modify_audio": int(ma),
            "duration": entry.get("duration", 0),
        })

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nSaved to: {METADATA_CSV}")
    print(f"Total usable entries: {len(df):,}")

    # Missing report
    if missing_video + missing_audio + missing_both > 0:
        print(f"\nMissing tensors:")
        print(f"  Video only missing: {missing_video}")
        print(f"  Audio only missing: {missing_audio}")
        print(f"  Both missing:       {missing_both}")

    # Stats
    print(f"\n{'=' * 50}")
    print("SPLIT DISTRIBUTION")
    print(f"{'=' * 50}")
    for split in ["train", "dev", "test"]:
        subset = df[df["split"] == split]
        print(f"\n  {split.upper()} ({len(subset):,} samples):")
        for label_name in LABEL_NAMES_4CLASS:
            count = len(subset[subset["label_name"] == label_name])
            pct = count / len(subset) * 100 if len(subset) > 0 else 0
            print(f"    {label_name:<15} {count:>6,}  ({pct:.1f}%)")

    print(f"\n{'=' * 50}")
    print("BINARY DISTRIBUTION")
    print(f"{'=' * 50}")
    real = len(df[df["label_binary"] == 0])
    fake = len(df[df["label_binary"] == 1])
    print(f"  Real: {real:,} ({real/len(df)*100:.1f}%)")
    print(f"  Fake: {fake:,} ({fake/len(df)*100:.1f}%)")

    print(f"\nAverage duration: {df['duration'].mean():.2f}s")
    print(f"\nDone! Next step: python dataset.py")


if __name__ == "__main__":
    main()
