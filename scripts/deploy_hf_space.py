"""Automated 1-click deploy script to Hugging Face Spaces."""

import argparse
import os
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file


def deploy_space(
    space_name: str,
    hf_token: str,
    checkpoint_path: str = "checkpoints/coral_best.pth",
    app_dir: str = "app",
) -> str:
    """Create and deploy interactive Gradio app to Hugging Face Spaces.

    Args:
        space_name: Full repository identifier (e.g., 'username/dr-severity-grading').
        hf_token: HuggingFace user access token with write permissions.
        checkpoint_path: Path to best model checkpoint to include in the Space.
        app_dir: Directory containing Gradio application files.

    Returns:
        URL of the deployed Hugging Face Space.
    """
    api = HfApi(token=hf_token)

    print(f"🚀 Creating / verifying Hugging Face Space: {space_name}...")
    create_repo(
        repo_id=space_name,
        repo_type="space",
        space_sdk="gradio",
        token=hf_token,
        exist_ok=True,
    )

    # Prepare staging directory
    staging_dir = "outputs/hf_space_staging"
    os.makedirs(staging_dir, exist_ok=True)

    # Copy app files
    for item in os.listdir(app_dir):
        src_path = os.path.join(app_dir, item)
        dst_path = os.path.join(staging_dir, item)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # Copy checkpoint into staging checkpoints/
    staging_ckpt_dir = os.path.join(staging_dir, "checkpoints")
    os.makedirs(staging_ckpt_dir, exist_ok=True)
    if os.path.exists(checkpoint_path):
        dst_ckpt = os.path.join(staging_ckpt_dir, os.path.basename(checkpoint_path))
        print(f"📦 Packaging model weights: {checkpoint_path} -> {dst_ckpt}")
        shutil.copy2(checkpoint_path, dst_ckpt)
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")

    # Copy src/ package into staging so imports work cleanly inside the Space
    staging_src_dir = os.path.join(staging_dir, "src")
    if os.path.exists(staging_src_dir):
        shutil.rmtree(staging_src_dir)
    shutil.copytree("src", staging_src_dir)

    print(f"Uploading files to Hugging Face Spaces...")
    upload_folder(
        folder_path=staging_dir,
        repo_id=space_name,
        repo_type="space",
        token=hf_token,
    )

    space_url = f"https://huggingface.co/spaces/{space_name}"
    print(f"\n🎉 Successfully deployed! Your live permanent app is accessible at:")
    print(f"👉 {space_url}\n")
    return space_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy app to Hugging Face Spaces.")
    parser.add_argument("--space", type=str, required=True, help="HF Space repo name, e.g. username/dr-severity-grading")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face User Access Token (write access)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/coral_best.pth", help="Path to checkpoint")
    args = parser.parse_args()

    deploy_space(
        space_name=args.space,
        hf_token=args.token,
        checkpoint_path=args.checkpoint,
    )
