"""Unit tests for Ben Graham preprocessing pipeline."""

import numpy as np
from PIL import Image
from src.data.preprocessing import (
    apply_clahe,
    ben_graham_pipeline,
    crop_circular_roi,
    mask_circle,
    subtract_local_mean,
)


def test_crop_circular_roi(sample_fundus_image):
    """Ensure circular ROI crop removes outside black boundaries."""
    cropped = crop_circular_roi(sample_fundus_image, tolerance=7)
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    # Cropped height and width should be less than or equal to original
    assert cropped.shape[0] <= sample_fundus_image.shape[0]
    assert cropped.shape[1] <= sample_fundus_image.shape[1]


def test_apply_clahe(sample_fundus_image):
    """Validate CLAHE preserves dimensions and enhances contrast range."""
    enhanced = apply_clahe(sample_fundus_image)
    assert enhanced.shape == sample_fundus_image.shape
    assert enhanced.dtype == np.uint8
    assert enhanced.min() >= 0
    assert enhanced.max() <= 255


def test_subtract_local_mean(sample_fundus_image):
    """Validate Ben Graham local mean subtraction."""
    normalized = subtract_local_mean(sample_fundus_image, sigma=10.0)
    assert normalized.shape == sample_fundus_image.shape
    assert normalized.dtype == np.uint8


def test_ben_graham_pipeline_standard(sample_fundus_image):
    """Validate end-to-end Ben Graham preprocessing."""
    target_size = 380
    result = ben_graham_pipeline(sample_fundus_image, image_size=target_size)
    assert isinstance(result, np.ndarray)
    assert result.shape == (target_size, target_size, 3)
    assert result.dtype == np.uint8


def test_ben_graham_pipeline_intermediates(sample_fundus_image):
    """Validate 3-stage intermediate extraction for Gradio UI."""
    target_size = 380
    result = ben_graham_pipeline(sample_fundus_image, image_size=target_size, return_intermediates=True)
    assert isinstance(result, dict)
    assert "raw" in result
    assert "cropped" in result
    assert "clahe_normalized" in result

    assert result["cropped"].shape == (target_size, target_size, 3)
    assert result["clahe_normalized"].shape == (target_size, target_size, 3)


def test_ben_graham_with_pil_input(sample_fundus_image):
    """Verify pipeline accepts PIL Image inputs seamlessly."""
    pil_img = Image.fromarray(sample_fundus_image)
    result = ben_graham_pipeline(pil_img, image_size=256)
    assert result.shape == (256, 256, 3)
