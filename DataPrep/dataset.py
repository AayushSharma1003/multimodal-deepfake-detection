"""
dataset.py — PyTorch Dataset + DataLoader for deepfake detection.

USAGE:
    conda activate dfd
    python dataset.py          # Run standalone to verify loading works

Used by train.py during training. Loads preprocessed video and audio tensors
from disk and returns them with labels.
"""

import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    METADATA_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    LABEL_NAMES_4CLASS,
)


class DeepfakeDataset(Dataset):
    """
    Dataset for multimodal deepfake detection.
    
    Loads preprocessed video tensors (16, 3, 224, 224) and 
    audio tensors (1, 128, T) from disk.
    
    Args:
        metadata_csv: Path to metadata.csv
        split: One of "train", "dev", "test"
        label_type: "binary" or "4class"
    """

    def __init__(self, metadata_csv: str, split: str, label_type: str = "binary"):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.label_type = label_type
        self.label_col = f"label_{label_type}"

        if self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found. "
                             f"Available: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load tensors (saved as float16, convert to float32 for training)
        video = torch.load(row["video_tensor"], weights_only=True).float()
        audio = torch.load(row["audio_tensor"], weights_only=True).float()

        label = torch.tensor(row[self.label_col], dtype=torch.long)

        return {
            "video": video,       # (16, 3, 224, 224)
            "audio": audio,       # (1, 128, T)
            "label": label,       # scalar
            "video_id": row["video_id"],
        }


def get_dataloaders(metadata_csv: str, label_type: str = "binary",
                    batch_size: int = BATCH_SIZE) -> dict:
    """
    Create DataLoaders for train, dev, and test splits.
    
    Returns dict: {"train": DataLoader, "dev": DataLoader, "test": DataLoader}
    """
    loaders = {}
    for split in ["train", "dev", "test"]:
        dataset = DeepfakeDataset(metadata_csv, split, label_type)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=(split == "train"),
        )
    return loaders


# ===== STANDALONE VERIFICATION =====
def main():
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    if not os.path.exists(METADATA_CSV):
        print(f"[!] metadata.csv not found: {METADATA_CSV}")
        print("    Run build_metadata.py first.")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    print(f"Total entries in CSV: {len(df):,}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    # Test loading each split
    for split in ["train", "dev", "test"]:
        print(f"\n--- Testing {split.upper()} split ---")
        dataset = DeepfakeDataset(METADATA_CSV, split, label_type="binary")
        print(f"  Samples: {len(dataset):,}")

        if len(dataset) == 0:
            print(f"  [!] No samples in {split} split!")
            continue

        # Load first sample
        sample = dataset[0]
        print(f"  Video shape:  {sample['video'].shape}")
        print(f"  Video dtype:  {sample['video'].dtype}")
        print(f"  Audio shape:  {sample['audio'].shape}")
        print(f"  Audio dtype:  {sample['audio'].dtype}")
        print(f"  Label:        {sample['label'].item()}")
        print(f"  Video ID:     {sample['video_id']}")

    # Test DataLoader
    print(f"\n--- Testing DataLoader (batch_size={BATCH_SIZE}) ---")
    loaders = get_dataloaders(METADATA_CSV, label_type="binary", batch_size=BATCH_SIZE)

    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"\n  {split.upper()} batch:")
        print(f"    Video: {batch['video'].shape}")
        print(f"    Audio: {batch['audio'].shape}")
        print(f"    Labels: {batch['label'].tolist()}")

    # Also test 4-class
    print(f"\n--- Testing 4-class labels ---")
    dataset_4c = DeepfakeDataset(METADATA_CSV, "train", label_type="4class")
    sample_4c = dataset_4c[0]
    print(f"  4-class label: {sample_4c['label'].item()} "
          f"({LABEL_NAMES_4CLASS[sample_4c['label'].item()]})")

    print(f"\n{'=' * 60}")
    print("ALL CHECKS PASSED!")
    print(f"{'=' * 60}")
    print(f"\nDataset is ready for training.")
    print(f"Next step: Write and run model.py + train.py")


if __name__ == "__main__":
    main()
