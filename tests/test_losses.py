"""Unit tests for CORAL ordinal loss, Focal loss, and Weighted CE."""

import pytest
import torch
from src.losses.coral import (
    CoralLoss,
    coral_loss,
    coral_predict,
    coral_probabilities,
    label_to_ordinal,
)
from src.losses.cross_entropy import WeightedCrossEntropyLoss
from src.losses.focal import FocalLoss


def test_label_to_ordinal():
    """Verify conversion of integer labels to K-1 binary ordinal levels."""
    labels = torch.tensor([0, 1, 2, 3, 4])
    # For num_classes=5, K-1=4 thresholds: [>=1, >=2, >=3, >=4]
    ord_targets = label_to_ordinal(labels, num_classes=5)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],  # Grade 0: none are >= 1
            [1.0, 0.0, 0.0, 0.0],  # Grade 1: >= 1
            [1.0, 1.0, 0.0, 0.0],  # Grade 2: >= 1 and >= 2
            [1.0, 1.0, 1.0, 0.0],  # Grade 3: >= 1, >= 2, >= 3
            [1.0, 1.0, 1.0, 1.0],  # Grade 4: all 4 are 1.0
        ]
    )
    assert torch.allclose(ord_targets, expected)


def test_coral_loss_differentiable():
    """Verify CORAL loss computes gradients correctly."""
    logits = torch.randn(8, 4, requires_grad=True)
    labels = torch.randint(0, 5, (8,))

    loss = coral_loss(logits, labels)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_coral_predict_monotonicity():
    """Verify CORAL grade decoding produces exact expected integer ranks."""
    # High confidence for all thresholds -> Grade 4
    logits_grade4 = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    assert coral_predict(logits_grade4).item() == 4

    # Confidence only for threshold >= 1 -> Grade 1
    logits_grade1 = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    assert coral_predict(logits_grade1).item() == 1

    # Low confidence for all thresholds -> Grade 0
    logits_grade0 = torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
    assert coral_predict(logits_grade0).item() == 0


def test_coral_probabilities_sum_to_one():
    """Verify CORAL probability conversion produces valid categorical distributions."""
    logits = torch.randn(10, 4)
    probs = coral_probabilities(logits)

    assert probs.shape == (10, 5)
    assert torch.all(probs >= 0.0)
    # Each row must sum to 1.0 within numerical tolerance
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_focal_loss():
    """Verify Focal Loss computation and gradient flow."""
    logits = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([0, 2, 4, 1])

    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, labels)
    assert loss.item() >= 0.0

    loss.backward()
    assert logits.grad is not None


def test_weighted_cross_entropy():
    """Verify WeightedCrossEntropyLoss."""
    weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5])
    criterion = WeightedCrossEntropyLoss(weight=weights)

    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    loss = criterion(logits, labels)
    assert loss.item() >= 0.0
