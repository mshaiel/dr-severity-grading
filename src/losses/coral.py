"""Consistent Rank Logits (CORAL) Ordinal Regression Loss.

Reference:
    Cao, W., Mirjalili, V., & Raschka, S. (2020).
    Rank-consistent ordinal regression for neural networks with applications
    to age estimation. Pattern Recognition Letters, 140, 325-331.

Mathematical Formulation:
    For a K-class ordinal classification task (e.g., DR severity grades 0 to K-1),
    the ordinal scale is decomposed into K-1 binary classification tasks:
        Task k: Is true_grade >= k? for k in {1, 2, ..., K-1}

    Binary targets:
        y_k = 1 if true_grade >= k else 0

    Predicted threshold probability:
        p_k = sigmoid(W^T x + b_k)

    The weight vector W is shared across all K-1 binary classifiers, while each
    classifier has its own bias term b_k.

    Loss function:
        L_CORAL = - sum_{k=1}^{K-1} lambda_k [ y_k * log(p_k) + (1 - y_k) * log(1 - p_k) ]

    Predicted rank / grade:
        grade = sum_{k=1}^{K-1} 1[p_k > 0.5]
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_ordinal(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert integer grade labels into K-1 binary ordinal targets.

    Args:
        labels: Tensor of integer class grades with shape (N,) or (N, 1),
                where each label is in {0, 1, ..., num_classes - 1}.
        num_classes: Total number of ordinal classes (5 for DR grading).

    Returns:
        Binary target tensor of shape (N, num_classes - 1) with dtype float32.
        For sample with grade g:
            levels[:, k-1] = 1.0 if g >= k else 0.0, for k in 1..num_classes-1.
    """
    labels = labels.view(-1, 1)
    # Thresholds: 1, 2, ..., num_classes - 1
    thresholds = torch.arange(1, num_classes, device=labels.device).view(1, -1)
    return (labels >= thresholds).float()


def coral_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    importance_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute CORAL binary cross entropy loss across K-1 threshold tasks.

    Args:
        logits: Raw linear outputs from CoralHead with shape (N, K - 1).
        targets: Target integer labels (N,) or binary ordinal targets (N, K - 1).
        importance_weights: Optional per-threshold importance weights (K - 1,).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar or per-sample CORAL loss tensor.
    """
    num_thresholds = logits.size(1)

    if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
        targets = label_to_ordinal(targets, num_classes=num_thresholds + 1)

    # Compute binary cross entropy with logits for numerical stability
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    if importance_weights is not None:
        importance_weights = importance_weights.to(logits.device).view(1, -1)
        bce = bce * importance_weights

    # Sum across the K-1 binary classification tasks per sample
    sample_loss = torch.sum(bce, dim=1)

    if reduction == "mean":
        return torch.mean(sample_loss)
    elif reduction == "sum":
        return torch.sum(sample_loss)
    elif reduction == "none":
        return sample_loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Decode raw CORAL logits into predicted integer severity grades.

    Grade = sum_{k=1}^{K-1} 1[sigmoid(logit_k) > threshold]

    Args:
        logits: Raw logits tensor of shape (N, K - 1).
        threshold: Decision boundary probability threshold (default: 0.5).

    Returns:
        Predicted integer grades tensor of shape (N,) with values in {0, ..., K-1}.
    """
    probs = torch.sigmoid(logits)
    binary_predictions = (probs > threshold).long()
    return torch.sum(binary_predictions, dim=1)


def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 binary threshold logits into a normalized K-class probability distribution.

    Uses adjacent categories formulation:
        P(grade >= 0) = 1.0
        P(grade >= k) = sigmoid(logit_{k-1}) for k in 1..K-1
        P(grade >= K) = 0.0

        P(grade = c) = max(0, P(grade >= c) - P(grade >= c + 1))
        Normalized to ensure sum = 1.0.

    Args:
        logits: Tensor of shape (N, K - 1).

    Returns:
        Categorical probability tensor of shape (N, K).
    """
    batch_size, num_thresholds = logits.shape
    num_classes = num_thresholds + 1
    device = logits.device

    # Threshold probabilities P(grade >= k) for k = 1..K-1
    cum_probs = torch.sigmoid(logits)

    # Pad with P(grade >= 0) = 1.0 and P(grade >= K) = 0.0
    ones = torch.ones(batch_size, 1, device=device)
    zeros = torch.zeros(batch_size, 1, device=device)
    extended_probs = torch.cat([ones, cum_probs, zeros], dim=1)  # shape (N, K + 1)

    # P(grade = c) = P(grade >= c) - P(grade >= c + 1)
    class_probs = extended_probs[:, :-1] - extended_probs[:, 1:]

    # Clip negative values due to any non-monotonicity and renormalize
    class_probs = torch.clamp(class_probs, min=1e-7)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
    return class_probs


class CoralLoss(nn.Module):
    """PyTorch nn.Module wrapper for CORAL ordinal loss."""

    def __init__(
        self,
        num_classes: int = 5,
        importance_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        if importance_weights is not None:
            self.register_buffer("importance_weights", importance_weights)
        else:
            self.importance_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return coral_loss(
            logits=logits,
            targets=targets,
            importance_weights=self.importance_weights,
            reduction=self.reduction,
        )
