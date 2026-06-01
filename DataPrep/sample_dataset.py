"""
sample_dataset.py — Stratified sampling of LAV-DF to create a manageable subset.

USAGE:
    conda activate dfd
    python sample_dataset.py

Samples SAMPLE_SIZE videos from metadata.json with stratified sampling:
- Maintains class proportions (Real, Fake-Video, Fake-Audio, Fake-Both)
- Maintains split proportions (train/dev/test)
- Saves sampled_metadata.json to OUTPUT_DIR
"""

import json
import os
import random
import sys
from collections import defaultdict

from config import (
    METADATA_JSON,
    SAMPLED_METADATA_JSON,
    OUTPUT_DIR,
    SAMPLE_SIZE,
    RANDOM_SEED,
    LABEL_MAP_4CLASS,
    LABEL_NAMES_4CLASS,
)


def main():
    print("=" * 60)
    print("STRATIFIED DATASET SAMPLING")
    print("=" * 60)

    if not os.path.exists(METADATA_JSON):
        print(f"[!] metadata.json not found: {METADATA_JSON}")
        print("    Run download_dataset.py and explore_dataset.py first.")
        sys.exit(1)

    # Load full metadata
    with open(METADATA_JSON, "r") as f:
        metadata = json.load(f)

    total = len(metadata)
    print(f"Total entries in dataset: {total:,}")
    print(f"Target sample size:       {SAMPLE_SIZE:,}")
    print(f"Random seed:              {RANDOM_SEED}")

    # Group entries by (split, class)
    groups = defaultdict(list)
    for entry in metadata:
        split = entry["split"]
        mv = entry["modify_video"]
        ma = entry["modify_audio"]
        label = LABEL_MAP_4CLASS[(mv, ma)]
        key = (split, label)
        groups[key].append(entry)

    # Print group sizes
    print(f"\nOriginal distribution:")
    print(f"{'Split':<8} {'Class':<15} {'Count':>8}")
    print("-" * 35)
    for split in ["train", "dev", "test"]:
        for label_idx, label_name in enumerate(LABEL_NAMES_4CLASS):
            key = (split, label_idx)
            count = len(groups.get(key, []))
            print(f"{split:<8} {label_name:<15} {count:>8,}")

    # Calculate proportional sample sizes
    random.seed(RANDOM_SEED)
    sampled = []
    sample_ratio = SAMPLE_SIZE / total

    print(f"\nSampling at ratio: {sample_ratio:.4f}")
    print(f"\nSampled distribution:")
    print(f"{'Split':<8} {'Class':<15} {'Original':>10} {'Sampled':>10}")
    print("-" * 47)

    total_sampled = 0
    for split in ["train", "dev", "test"]:
        for label_idx, label_name in enumerate(LABEL_NAMES_4CLASS):
            key = (split, label_idx)
            group = groups.get(key, [])
            n_sample = max(1, round(len(group) * sample_ratio))  # at least 1

            # Don't sample more than available
            n_sample = min(n_sample, len(group))

            selected = random.sample(group, n_sample)
            sampled.extend(selected)
            total_sampled += n_sample

            print(f"{split:<8} {label_name:<15} {len(group):>10,} {n_sample:>10,}")

    print(f"\n{'TOTAL':<25} {total:>10,} {total_sampled:>10,}")

    # Save sampled metadata
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SAMPLED_METADATA_JSON, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"\nSaved sampled metadata to: {SAMPLED_METADATA_JSON}")
    print(f"Actual sample size: {len(sampled):,}")

    # Quick verification
    from collections import Counter
    split_check = Counter(e["split"] for e in sampled)
    print(f"\nVerification — samples per split:")
    for split in ["train", "dev", "test"]:
        print(f"  {split}: {split_check.get(split, 0):,}")

    print("\nDone! Next step: python preprocess_audio.py")


if __name__ == "__main__":
    main()
