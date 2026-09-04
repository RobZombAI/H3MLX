#!/usr/bin/env python3
"""
📥 H3MLX Automated Model Downloader
Downloads official MiniMax-H3 / PDD-8Step weights with multi-part resumption.
"""

import os
import sys
import argparse
from pathlib import Path

DEFAULT_TARGET_DIR = Path.home() / "h3-models" / "MiniMax-H3-PDD-8Step"
REPO_ID = "MiniMax/MiniMax-H3"

def main():
    parser = argparse.ArgumentParser(description="Download MiniMax H3 weights for Apple Silicon inference")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_TARGET_DIR), help="Target local directory")
    parser.add_argument("--model", type=str, default=REPO_ID, help="Hugging Face Repository ID")
    parser.add_argument("--pdd-only", action="store_true", default=True, help="Download optimized PDD 8-Step weights (~24 GB)")
    args = parser.parse_args()

    target_path = Path(args.dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("👑 H3MLX AUTOMATED MODEL DOWNLOADER")
    print(f"📦 Target Repository: {args.model}")
    print(f"💾 Local Destination: {target_path}")
    print("=" * 70)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing 'huggingface_hub'...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    print("\nStarting download with resume support...")
    try:
        snapshot_download(
            repo_id=args.model,
            local_dir=str(target_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\nModel weights downloaded successfully to: {target_path}")
    except Exception as e:
        print(f"\nError during HuggingFace download:\n{e}", file=sys.stderr)
        print("\nAlternatively, you can manually place model weights in:")
        print(f"   {target_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
