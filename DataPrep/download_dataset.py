"""
download_dataset.py — Downloads LAV-DF dataset from HuggingFace.

USAGE:
    conda activate dfd
    python download_dataset.py

This downloads ~25.6 GB. Make sure you have enough disk space.
If download is interrupted, re-run and it will resume.
"""

import os
import sys

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    from config import DATASET_ROOT

    # Target directory: one level up since snapshot_download creates a subfolder
    download_dir = os.path.dirname(DATASET_ROOT)
    os.makedirs(download_dir, exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING LAV-DF DATASET")
    print("=" * 60)
    print(f"Repository:  ControlNet/LAV-DF")
    print(f"Target dir:  {download_dir}")
    print(f"Size:        ~25.6 GB")
    print()
    print("This will take a while depending on your internet speed.")
    print("If interrupted, re-run this script to resume.\n")

    confirm = input("Proceed with download? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    try:
        path = snapshot_download(
            repo_id="ControlNet/LAV-DF",
            repo_type="dataset",
            local_dir=DATASET_ROOT,
            resume_download=True,
        )
        print(f"\n[OK] Dataset downloaded to: {path}")
    except Exception as e:
        print(f"\n[!] Download error: {e}")
        print("\nIf you get an auth error, run:")
        print("  huggingface-cli login")
        print("Then re-run this script.")
        sys.exit(1)

    # Verify
    metadata_path = os.path.join(DATASET_ROOT, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"[OK] metadata.json found at {metadata_path}")
    else:
        print(f"[!] metadata.json NOT found. Check if dataset extracted correctly.")
        print(f"    Expected at: {metadata_path}")
        print(f"    Contents of {DATASET_ROOT}:")
        if os.path.exists(DATASET_ROOT):
            for item in os.listdir(DATASET_ROOT):
                print(f"      {item}")


if __name__ == "__main__":
    main()
