"""
explore_dataset.py — Explore LAV-DF dataset structure and statistics.

USAGE:
    conda activate dfd
    python explore_dataset.py

Run this FIRST to verify the dataset is correctly downloaded and
understand the class distribution before sampling.
"""

import json
import os
import sys
from collections import Counter

from config import (
    DATASET_ROOT,
    METADATA_JSON,
    LABEL_MAP_4CLASS,
    LABEL_NAMES_4CLASS,
)


def main():
    print("=" * 60)
    print("LAV-DF DATASET EXPLORER")
    print("=" * 60)

    # --- Check paths ---
    print(f"\nDataset root: {DATASET_ROOT}")
    print(f"Metadata JSON: {METADATA_JSON}")

    if not os.path.exists(DATASET_ROOT):
        print(f"\n[!] Dataset root not found: {DATASET_ROOT}")
        print("    Run download_dataset.py first.")
        sys.exit(1)

    if not os.path.exists(METADATA_JSON):
        print(f"\n[!] metadata.json not found: {METADATA_JSON}")
        sys.exit(1)

    # --- List top-level contents ---
    print(f"\nContents of {DATASET_ROOT}:")
    for item in sorted(os.listdir(DATASET_ROOT)):
        full = os.path.join(DATASET_ROOT, item)
        if os.path.isdir(full):
            count = len(os.listdir(full))
            print(f"  📁 {item}/ ({count} files)")
        else:
            size_mb = os.path.getsize(full) / (1024 * 1024)
            print(f"  📄 {item} ({size_mb:.1f} MB)")

    # --- Load metadata ---
    print("\nLoading metadata.json...")
    with open(METADATA_JSON, "r") as f:
        metadata = json.load(f)

    total = len(metadata)
    print(f"Total entries: {total}")

    # --- Split distribution ---
    split_counts = Counter()
    class_counts = Counter()           # overall
    class_per_split = {}               # per split
    duration_total = 0.0

    for entry in metadata:
        split = entry["split"]
        mv = entry["modify_video"]
        ma = entry["modify_audio"]
        label_4c = LABEL_MAP_4CLASS[(mv, ma)]
        label_name = LABEL_NAMES_4CLASS[label_4c]

        split_counts[split] += 1
        class_counts[label_name] += 1

        if split not in class_per_split:
            class_per_split[split] = Counter()
        class_per_split[split][label_name] += 1

        duration_total += entry.get("duration", 0)

    # --- Print split stats ---
    print(f"\n{'Split':<10} {'Count':>8} {'Percentage':>12}")
    print("-" * 32)
    for split in ["train", "dev", "test"]:
        c = split_counts.get(split, 0)
        pct = c / total * 100
        print(f"{split:<10} {c:>8,} {pct:>11.1f}%")
    print(f"{'TOTAL':<10} {total:>8,}")

    # --- Print class distribution (overall) ---
    print(f"\n{'Class':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 37)
    for name in LABEL_NAMES_4CLASS:
        c = class_counts.get(name, 0)
        pct = c / total * 100
        print(f"{name:<15} {c:>8,} {pct:>11.1f}%")

    # --- Print class distribution per split ---
    for split in ["train", "dev", "test"]:
        if split not in class_per_split:
            continue
        split_total = split_counts[split]
        print(f"\n  {split.upper()} split ({split_total:,} total):")
        for name in LABEL_NAMES_4CLASS:
            c = class_per_split[split].get(name, 0)
            pct = c / split_total * 100 if split_total > 0 else 0
            print(f"    {name:<15} {c:>6,}  ({pct:.1f}%)")

    # --- Duration stats ---
    avg_dur = duration_total / total if total > 0 else 0
    print(f"\nTotal duration:   {duration_total / 3600:.1f} hours")
    print(f"Average duration: {avg_dur:.2f} seconds")

    # --- Sample entries ---
    print("\n" + "=" * 60)
    print("SAMPLE ENTRIES (first 3)")
    print("=" * 60)
    for i, entry in enumerate(metadata[:3]):
        print(f"\n--- Entry {i} ---")
        for key in ["file", "split", "modify_video", "modify_audio",
                     "duration", "n_fakes", "original"]:
            print(f"  {key}: {entry.get(key)}")
        label_4c = LABEL_MAP_4CLASS[(entry["modify_video"], entry["modify_audio"])]
        print(f"  → 4-class label: {label_4c} ({LABEL_NAMES_4CLASS[label_4c]})")

    # --- Check if video files exist ---
    print("\n" + "=" * 60)
    print("FILE EXISTENCE CHECK")
    print("=" * 60)
    missing = 0
    checked = 0
    for entry in metadata[:100]:  # check first 100
        fpath = os.path.join(DATASET_ROOT, entry["file"])
        if not os.path.exists(fpath):
            missing += 1
            if missing <= 3:
                print(f"  [MISSING] {fpath}")
        checked += 1

    if missing == 0:
        print(f"  [OK] All {checked} checked files exist.")
    else:
        print(f"  [!] {missing}/{checked} files missing in first {checked} entries.")

    print("\nDone! Dataset looks good." if missing == 0 else
          "\n[!] Some files are missing. Check your download.")


if __name__ == "__main__":
    main()
