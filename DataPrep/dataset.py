"""
dataset.py — Dataset and DataLoader for preprocessed LAV-DF tensors
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import METADATA_CSV, BATCH_SIZE


class DeepfakeDataset(Dataset):
    """
    Loads preprocessed video/audio tensors from disk.
    Tensors are stored as float16 and cast to float32 at load time.
    """

    def __init__(self, metadata_csv, split, label_type="4class"):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_col = "label_4class" if label_type == "4class" else "label_binary"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video = torch.load(row["video_tensor"], map_location="cpu", weights_only=True).float()
        audio = torch.load(row["audio_tensor"], map_location="cpu", weights_only=True).float()
        label = int(row[self.label_col])
        return {
            "video": video,         # (16, 3, 224, 224)
            "audio": audio,         # (1, 128, T)
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": row["video_id"],
        }


def get_dataloaders(metadata_csv=METADATA_CSV, label_type="4class",
                    batch_size=BATCH_SIZE, num_workers=4):
    """
    Returns dict of DataLoaders: {"train": ..., "dev": ..., "test": ...}
    """
    loaders = {}
    for split in ["train", "dev", "test"]:
        ds = DeepfakeDataset(metadata_csv, split, label_type)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders


if __name__ == "__main__":
    loaders = get_dataloaders()
    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"{split}: {len(loader.dataset)} samples | "
              f"video={batch['video'].shape} audio={batch['audio'].shape} "
<<<<<<< Updated upstream
              f"label={batch['label'].shape}")
=======
              f"label={batch['label'].shape}")
>>>>>>> Stashed changes
