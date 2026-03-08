#!/usr/bin/env python
"""Push the gym-env Space to HuggingFace Hub.

Usage:
    python scripts/push-hf-space.py --user YOUR_HF_USERNAME
    python scripts/push-hf-space.py --user warrenlow --space wrl-dragon-gym
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push gym-env Space to HF Hub")
    parser.add_argument("--user", required=True, help="HuggingFace username")
    parser.add_argument("--space", default="wrl-dragon-gym", help="Space name")
    args = parser.parse_args()

    repo_id = f"{args.user}/{args.space}"
    api = HfApi()

    # Create or get the Space
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"Space repo: {repo_id}")

    # Upload files
    space_dir = Path(__file__).resolve().parent.parent / "spaces" / "gym-env"
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
    )

    space_url = f"https://{args.user}-{args.space}.hf.space"
    print(f"\nPushed to: https://huggingface.co/spaces/{repo_id}")
    print(f"API URL:   {space_url}")
    print(f"\nUse this URL in training:")
    print(f"  modal run src/meta/modal_train.py --gym-space-url {space_url}")


if __name__ == "__main__":
    main()
