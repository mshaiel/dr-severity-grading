"""Evaluation metrics including Quadratic Weighted Kappa (QWK), F1, ECE, and TTA inference."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.losses.coral import coral_predict, coral_probabilities


def compute_qwk(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Quadratic Weighted Kappa (QWK).

    QWK is the primary clinical metric for Diabetic Retinopathy grading. It
    penalizes classification errors quadratically proportional to distance:
        w_{i,j} = (i - j)^2 / (K - 1)^2

    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        QWK score as a float in [-1, 1].
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.ravel().astype(np.int64)
    y_pred = y_pred.ravel().astype(np.int64)

    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_ece(
    probs: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    num_bins: int = 10,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Compute Expected Calibration Error (ECE) and bin statistics.

    ECE measures the difference between expected accuracy and confidence:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        probs: Predicted class probability distribution of shape (N, num_classes).
        y_true: Ground truth class labels of shape (N,).
        num_bins: Number of confidence bins (typically 10 or 15).

    Returns:
        Tuple of (ece_score, bin_details_dict).
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    y_true = y_true.ravel().astype(np.int64)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    total_samples = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include right boundary on the last bin
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        bin_counts.append(bin_size)

        if bin_size > 0:
            avg_acc = np.mean(accuracies[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            ece += (bin_size / total_samples) * np.abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    bin_details = {
        "bin_boundaries": bin_boundaries,
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
    }
    return float(ece), bin_details


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive suite of evaluation metrics."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.ravel().astype(np.int64)
    y_pred = y_pred.ravel().astype(np.int64)

    metrics = {
        "qwk": compute_qwk(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    for c, score in enumerate(per_class_f1):
        metrics[f"f1_class_{c}"] = float(score)

    if probs is not None:
        ece, _ = compute_ece(probs, y_true)
        metrics["ece"] = ece

    return metrics


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    is_ordinal: bool = True,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform Test-Time Augmentation (TTA) with horizontal flip.

    From Addendum Q1:
    Averages predictions from the original image and its horizontal flip (2 passes).
    For CORAL head: Averages sigmoid probabilities before rank-decoding.
    For CE head: Averages softmax probabilities before argmax.

    Args:
        model: PyTorch DRSeverityModel.
        image_tensor: Input tensor of shape (B, C, H, W).
        is_ordinal: True if CORAL loss / head is used.
        threshold: Decision threshold for CORAL rank decoding.

    Returns:
        Tuple of (predicted_grades, averaged_probabilities).
    """
    model.eval()

    # Pass 1: Original image
    logits_orig = model(image_tensor).flatten(1)

    # Pass 2: Horizontal flip (along width dimension)
    flipped_tensor = torch.flip(image_tensor, dims=[-1])
    logits_flip = model(flipped_tensor).flatten(1)

    if is_ordinal:
        # Average sigmoid probabilities across thresholds
        sig_orig = torch.sigmoid(logits_orig)
        sig_flip = torch.sigmoid(logits_flip)
        avg_sig = (sig_orig + sig_flip) / 2.0

        # Decode grade from averaged sigmoids
        grades = torch.sum((avg_sig > threshold).long(), dim=1)

        # Convert averaged sigmoids to class probability distribution
        batch_size, num_thresholds = avg_sig.shape
        device = avg_sig.device
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)
        extended = torch.cat([ones, avg_sig, zeros], dim=1)
        probs = extended[:, :-1] - extended[:, 1:]
        probs = torch.clamp(probs, min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
    else:
        probs_orig = F.softmax(logits_orig, dim=1)
        probs_flip = F.softmax(logits_flip, dim=1)
        probs = (probs_orig + probs_flip) / 2.0
        grades = torch.argmax(probs, dim=1)

    return grades, probs
