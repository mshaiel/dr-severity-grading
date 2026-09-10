"""Temperature scaling and reliability diagram generation for model calibration."""

from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.evaluation.metrics import compute_ece
from src.losses.coral import coral_probabilities


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling calibration on validation logits.

    Reference:
        Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.

    Finds optimal scalar parameter T > 0 such that softmax(z / T)
    minimizes cross-entropy on a held-out validation set.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize temperature to 1.0 (log(T) = 0.0)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature parameter."""
        temp = self.temperature.clamp(min=0.01)
        return logits / temp

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        is_ordinal: bool = False,
    ) -> float:
        """Optimize temperature parameter on validation set using L-BFGS.

        Args:
            logits: Validation logits of shape (N, C) or (N, K - 1).
            labels: Ground truth integer labels of shape (N,).
            lr: Learning rate for optimizer.
            max_iter: Maximum optimization steps.
            is_ordinal: Whether logits correspond to CORAL thresholds.

        Returns:
            Optimal temperature value as float.
        """
        device = logits.device
        self.to(device)
        labels = labels.to(device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            if is_ordinal:
                # Convert scaled threshold logits to class probability distribution
                probs = coral_probabilities(scaled_logits)
                loss = F.nll_loss(torch.log(probs + 1e-8), labels)
            else:
                loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())


def plot_reliability_diagram(
    probs: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    num_bins: int = 10,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """Plot a publication-grade reliability diagram with ECE annotation.

    Args:
        probs: Predicted probability distribution (N, num_classes).
        y_true: Ground truth labels (N,).
        num_bins: Number of confidence bins.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    ece, details = compute_ece(probs, y_true, num_bins=num_bins)

    bin_centers = (details["bin_boundaries"][:-1] + details["bin_boundaries"][1:]) / 2.0
    bin_accs = details["bin_accuracies"]
    bin_confs = details["bin_confidences"]
    bin_width = 1.0 / num_bins

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)

    # Actual model calibration bars
    ax.bar(
        bin_centers,
        bin_accs,
        width=bin_width * 0.9,
        color="#2563EB",
        alpha=0.8,
        edgecolor="black",
        label="Observed Accuracy",
    )

    # Shaded gap showing calibration error
    gap = np.abs(bin_accs - bin_confs)
    ax.bar(
        bin_centers,
        gap,
        bottom=np.minimum(bin_accs, bin_confs),
        width=bin_width * 0.9,
        color="#EF4444",
        alpha=0.3,
        hatch="//",
        label="Calibration Gap",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    return fig
