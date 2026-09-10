"""Standard Cross-Entropy Loss with class-weighting and label smoothing."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Wrapper around PyTorch CrossEntropyLoss supporting inverse class weights."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(targets.device) if self.weight is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
