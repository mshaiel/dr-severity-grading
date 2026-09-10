"""Inference pipeline for Gradio application with 3-stage preprocessing visualizer."""

from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.preprocessing import (
    crop_circular_roi,
    apply_clahe,
    subtract_local_mean,
    mask_circle,
)
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.metrics import predict_with_tta
from src.explainability.gradcam import GradCAM, overlay_gradcam
from src.losses.coral import coral_probabilities

GRADE_LABELS = {
    0: "Grade 0: No DR",
    1: "Grade 1: Mild DR",
    2: "Grade 2: Moderate DR",
    3: "Grade 3: Severe DR",
    4: "Grade 4: Proliferative DR",
}


def run_app_inference(
    pil_image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    use_tta: bool = True,
    is_ordinal: bool = True,
    image_size: int = 380,
) -> Tuple[Image.Image, Image.Image, Image.Image, str, Dict[str, float], Image.Image]:
    """Execute full clinical inference returning intermediate visual stages.

    From Addendum Q3:
    Returns three intermediate images:
      1. Raw input PIL image
      2. Circle cropped + resized image
      3. CLAHE normalized image (what the model sees)
    Together with:
      - Predicted grade label
      - Confidence dictionary per class
      - Grad-CAM overlay cast on the CLAHE image

    Returns:
        (raw_img, cropped_img, clahe_img, grade_str, conf_dict, gradcam_overlay)
    """
    raw_rgb = np.array(pil_image.convert("RGB"))

    # Stage 1: Crop circular ROI
    cropped = crop_circular_roi(raw_rgb)
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        cropped = raw_rgb
    cropped_resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Stage 2: CLAHE + Local Mean Subtraction (Ben Graham)
    clahe_enhanced = apply_clahe(cropped_resized)
    local_mean_sub = subtract_local_mean(clahe_enhanced, sigma=image_size / 30.0)
    clahe_final = mask_circle(local_mean_sub)

    # Convert intermediate images to PIL for Gradio display
    raw_pil = pil_image
    cropped_pil = Image.fromarray(cropped_resized)
    clahe_pil = Image.fromarray(clahe_final)

    # Prepare input tensor for model
    img_float = clahe_final.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    norm_img = (img_float - mean) / std

    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Model inference (with horizontal-flip TTA if enabled)
    if use_tta:
        grades, probs = predict_with_tta(model, tensor, is_ordinal=is_ordinal)
    else:
        with torch.no_grad():
            grades, probs = model.predict(tensor)

    pred_grade = grades.item()
    class_probs = probs[0].cpu().numpy()

    grade_str = GRADE_LABELS[pred_grade]
    conf_dict = {GRADE_LABELS[i]: float(class_probs[i]) for i in range(5)}

    # Generate Grad-CAM explainability overlay (on the unaugmented image)
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_heatmap(tensor, target_index=pred_grade, is_ordinal=is_ordinal)
    gradcam.remove_hooks()

    # Overlay Grad-CAM onto the CLAHE normalized image (panel 3)
    overlay_rgb = overlay_gradcam(clahe_final, heatmap, alpha=0.45)
    overlay_pil = Image.fromarray(overlay_rgb)

    return raw_pil, cropped_pil, clahe_pil, grade_str, conf_dict, overlay_pil
