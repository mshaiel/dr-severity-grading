"""Unit tests for Grad-CAM explainability and overlay generation."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.explainability.gradcam import GradCAM, overlay_gradcam


class SimpleConvNet(nn.Module):
    """Small CNN with Conv2d layers for testing GradCAM hooks."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 4)  # 4 thresholds for CORAL

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        features = self.pool(x).flatten(1)
        return self.fc(features)


def test_gradcam_generation():
    """Verify GradCAM generates 2D normalized heatmap of input spatial dimensions."""
    model = SimpleConvNet()
    gradcam = GradCAM(model)

    dummy_input = torch.randn(1, 3, 64, 64)
    heatmap = gradcam.generate_heatmap(dummy_input, is_ordinal=True)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6

    gradcam.remove_hooks()


def test_overlay_gradcam():
    """Verify overlay blending returns uint8 RGB image matching dimensions."""
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap = np.ones((100, 100), dtype=np.float32) * 0.8

    overlay = overlay_gradcam(base, heatmap, alpha=0.5)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
