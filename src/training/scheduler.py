"""Learning rate scheduler factory."""

from typing import Any
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine_warm_restarts",
    t_0: int = 10,
    t_mult: int = 1,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    """Build learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_name: Identifier for scheduler type.
        t_0: Number of iterations/epochs for the first restart.
        t_mult: Factor by which T increases after each restart.
        min_lr: Minimum learning rate floor.

    Returns:
        Configured PyTorch learning rate scheduler.
    """
    if scheduler_name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=min_lr,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
