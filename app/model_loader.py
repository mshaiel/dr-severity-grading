"""Model loader for HuggingFace Hub and local checkpoint cache."""

import os
from typing import Optional
import torch

from src.models.model_factory import DRSeverityModel, build_model
from src.utils.checkpoint import load_checkpoint


def load_app_model(
    repo_id: Optional[str] = None,
    filename: str = "coral_best.pth",
    local_checkpoint_path: Optional[str] = "checkpoints/coral_best.pth",
    device: torch.device = torch.device("cpu"),
) -> DRSeverityModel:
    """Load model from Hugging Face Hub or local path with CPU optimization.

    Args:
        repo_id: HuggingFace repository ID (e.g. 'username/dr-severity-grading').
        filename: Checkpoint filename in HF repository.
        local_checkpoint_path: Fallback local path if repo_id is None.
        device: Torch device (defaults to CPU for HF Spaces free tier).

    Returns:
        Loaded DRSeverityModel in eval mode.
    """
    checkpoint_file = None

    if repo_id:
        try:
            from huggingface_hub import hf_hub_download

            checkpoint_file = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            print(f"HF Hub download failed: {e}. Checking local cache...")

    if not checkpoint_file or not os.path.exists(checkpoint_file):
        if local_checkpoint_path and os.path.exists(local_checkpoint_path):
            checkpoint_file = local_checkpoint_path

    class Config:
        backbone = "efficientnet_b0"
        pretrained = False
        num_classes = 5
        drop_rate = 0.2
        drop_path_rate = 0.1

    model = build_model(Config(), loss_name="coral")

    if checkpoint_file and os.path.exists(checkpoint_file):
        ckpt = load_checkpoint(checkpoint_file, device=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded weights from {checkpoint_file}")
    else:
        print("Warning: No checkpoint loaded. Model initialized with random weights.")

    model.to(device)
    model.eval()
    return model
