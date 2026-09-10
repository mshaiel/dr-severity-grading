"""Focal Loss for Multi-Class Imbalanced Classification.

Reference:
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2980-2988.

Mathematical Formulation:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground truth class.
    Modulating factor (1 - p_t)^gamma dynamically scales down easy examples
    and focuses gradient updates on hard, ambiguous cases.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-Class Focal Loss with support for class weighting and focusing parameter gamma."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            gamma: Focusing parameter gamma >= 0. When gamma=0, reduces to standard CE.
            alpha: Optional class weights tensor of shape (num_classes,).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between logits and target class labels.

        Args:
            logits: Predicted unnormalized scores of shape (N, C).
            targets: Ground truth class indices of shape (N,).

        Returns:
            Computed scalar or per-sample loss.
        """
        # Cross entropy loss without reduction: -log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)

        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = modulating_factor * ce_loss

        # Apply class weights alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
