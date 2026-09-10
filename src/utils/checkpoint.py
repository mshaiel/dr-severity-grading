import os
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
) -> None:
    """Save model checkpoint and optionally copy/save to best filepath.

    Args:
        state: Dictionary containing state_dict, optimizer, epoch, metrics, etc.
        filepath: Destination path for current checkpoint.
        is_best: Whether this checkpoint achieved the best validation score.
        best_filepath: Destination path for best checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        os.makedirs(os.path.dirname(best_filepath), exist_ok=True)
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load model checkpoint safely on target device.

    Args:
        filepath: Path to the .pth checkpoint file.
        device: Torch device to map loaded tensors.

    Returns:
        Loaded state dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    return torch.load(filepath, map_location=device, weights_only=False)
