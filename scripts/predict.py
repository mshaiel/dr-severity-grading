"""Single-image clinical inference and explainability script."""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch

from src.data.preprocessing import ben_graham_pipeline
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.metrics import predict_with_tta
from src.explainability.gradcam import GradCAM, overlay_gradcam
from src.losses.coral import coral_predict, coral_probabilities
from src.models.model_factory import build_model
from src.utils.checkpoint import load_checkpoint

GRADE_DESCRIPTIONS = {
    0: "No Diabetic Retinopathy (Healthy)",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy (PDR - Urgent)",
}


def preprocess_for_inference(image: np.ndarray, image_size: int = 380) -> torch.Tensor:
    """Preprocess image with Ben Graham and normalize to PyTorch tensor."""
    bg_img = ben_graham_pipeline(image, image_size=image_size)
    img_float = bg_img.astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    norm_img = (img_float - mean) / std

    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, bg_img


def main():
    parser = argparse.ArgumentParser(description="Predict DR severity grade for a single fundus image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input fundus image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Model backbone name")
    parser.add_argument("--loss", type=str, default="coral", choices=["coral", "cross_entropy", "focal"])
    parser.add_argument("--use-tta", action="store_true", help="Apply Horizontal-Flip Test-Time Augmentation")
    parser.add_argument("--output", type=str, default="results/prediction_overlay.png", help="Path to save overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at {args.image}")

    raw_image = cv2.imread(args.image)
    if raw_image is None:
        raise ValueError(f"Failed to read image at {args.image}")
    raw_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    class ModelConfig:
        backbone = args.backbone
        pretrained = False
        num_classes = 5
        drop_rate = 0.2
        drop_path_rate = 0.1

    model = build_model(ModelConfig(), loss_name=args.loss)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    is_ordinal = (args.loss == "coral")

    # Preprocess
    input_tensor, bg_img = preprocess_for_inference(raw_rgb, image_size=380)
    input_tensor = input_tensor.to(device)

    # Inference
    if args.use_tta:
        grade_tensor, prob_tensor = predict_with_tta(model, input_tensor, is_ordinal=is_ordinal)
    else:
        with torch.no_grad():
            grade_tensor, prob_tensor = model.predict(input_tensor)

    pred_grade = grade_tensor.item()
    probs = prob_tensor[0].cpu().numpy()

    print("\n" + "=" * 50)
    print("🔍 CLINICAL DIABETIC RETINOPATHY PREDICTION")
    print("=" * 50)
    print(f"Predicted Grade: {pred_grade} — {GRADE_DESCRIPTIONS[pred_grade]}")
    print("\nClass Probability Distribution:")
    for c, p in enumerate(probs):
        bar = "█" * int(p * 30)
        print(f"  Grade {c} ({GRADE_DESCRIPTIONS[c][:15]:<15}): {p * 100:5.2f}% | {bar}")
    print("=" * 50)

    # Generate Grad-CAM explainability overlay on top of CLAHE-normalized image
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_heatmap(input_tensor, is_ordinal=is_ordinal)
    gradcam.remove_hooks()

    overlay = overlay_gradcam(bg_img, heatmap)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, overlay_bgr)
    print(f"Saved Grad-CAM explainability overlay to: {args.output}\n")


if __name__ == "__main__":
    main()
