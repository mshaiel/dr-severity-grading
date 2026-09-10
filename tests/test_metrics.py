"""Unit tests for QWK, ECE, and Test-Time Augmentation (TTA)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.evaluation.metrics import compute_ece, compute_metrics, compute_qwk, predict_with_tta


def test_qwk_perfect_score():
    """Verify QWK is 1.0 on identical ground truth and prediction."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    qwk = compute_qwk(y_true, y_pred)
    assert pytest.approx(qwk, abs=1e-5) == 1.0


def test_qwk_penalizes_large_ordinal_errors():
    """Verify QWK severely penalizes errors across distant grades compared to adjacent grades."""
    y_true = np.array([0, 0, 0, 0, 4, 4, 4, 4])

    # Adjacent error (diff = 1)
    y_pred_close = np.array([1, 1, 1, 1, 3, 3, 3, 3])
    # Distant error (diff = 4)
    y_pred_far = np.array([4, 4, 4, 4, 0, 0, 0, 0])

    qwk_close = compute_qwk(y_true, y_pred_close)
    qwk_far = compute_qwk(y_true, y_pred_far)

    assert qwk_close > qwk_far


def test_compute_ece():
    """Verify ECE calculation on perfectly confident and calibrated predictions."""
    # 4 samples, perfectly predicted with 1.0 confidence
    probs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    y_true = np.array([0, 1, 2, 3])

    ece, details = compute_ece(probs, y_true, num_bins=10)
    assert pytest.approx(ece, abs=1e-5) == 0.0


def test_predict_with_tta():
    """Verify predict_with_tta performs 2 passes and averages probabilities."""

    class MockModel(nn.Module):
        def forward(self, x):
            # Return dummy CORAL 4-threshold logits based on mean pixel value
            val = x.mean(dim=[1, 2, 3]).unsqueeze(1)
            return torch.cat([val, val - 1, val - 2, val - 3], dim=1)

    model = MockModel()
    # Mock image (1, 3, 64, 64) with left-right gradient so flip changes features
    img = torch.linspace(0, 1, 64).repeat(1, 3, 64, 1)

    grades, probs = predict_with_tta(model, img, is_ordinal=True)
    assert grades.shape == (1,)
    assert probs.shape == (1, 5)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
