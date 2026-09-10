"""Gradio Web Application for Diabetic Retinopathy Severity Grading & Trustworthiness."""

import os
import gradio as gr
from PIL import Image
import torch

try:
    from app.inference import run_app_inference
    from app.model_loader import load_app_model
except ImportError:
    from inference import run_app_inference
    from model_loader import load_app_model

# Load model on CPU for Hugging Face Spaces compatibility
device = torch.device("cpu")
model = load_app_model(
    repo_id=os.environ.get("HF_REPO_ID", None),
    filename="coral_best.pth",
    local_checkpoint_path="checkpoints/coral_best.pth",
    device=device,
)


def predict(image: Image.Image, use_tta: bool):
    """Run app inference wrapper for Gradio interface."""
    if image is None:
        return None, None, None, "No Image Provided", {}, None

    return run_app_inference(
        pil_image=image,
        model=model,
        device=device,
        use_tta=use_tta,
        is_ordinal=True,
    )


# Find any sample images for interactive clicking
sample_images_dir = "app/assets/sample_images"
example_images = []
if os.path.exists(sample_images_dir):
    for f in os.listdir(sample_images_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            example_images.append([os.path.join(sample_images_dir, f), True])

# Modern theme with clinical blue/indigo palette
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="Diabetic Retinopathy Clinical Grading") as demo:
    gr.HTML(
        """
        <div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%); border-radius: 14px; color: white; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px;">👁️ Diabetic Retinopathy Severity Grading</h1>
            <p style="margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.9; font-weight: 400;">
                Clinical Decision Support powered by <strong>EfficientNet-B0</strong> · <strong>CORAL Ordinal Regression</strong> · <strong>Ben Graham CLAHE</strong> · <strong>Grad-CAM</strong>
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Retinal Fundus Image", type="pil")
            use_tta_cb = gr.Checkbox(
                label="Apply Horizontal-Flip Test-Time Augmentation (TTA)",
                value=True,
                info="Averages threshold sigmoids across horizontal flips before ordinal rank decoding.",
            )
            submit_btn = gr.Button("🔬 Run Diagnostic Assessment", variant="primary", size="lg")

            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=[input_img, use_tta_cb],
                    label="Click to Test Sample Fundus Images",
                )

            gr.Markdown(
                """
                > 💡 **Production Architecture Note:** Deployed on **EfficientNet-B0** for sub-second CPU inference on free cloud tiers. 
                > The headline benchmark model (**EfficientNet-B3**) is preserved for maximum Quadratic Weighted Kappa (QWK) analysis in the repository.
                """
            )

        with gr.Column(scale=2):
            gr.Markdown("### 🔬 3-Stage Ben Graham Preprocessing Pipeline")
            with gr.Row():
                raw_panel = gr.Image(label="① Raw Input", type="pil")
                cropped_panel = gr.Image(label="② Circle Crop + Resize", type="pil")
                clahe_panel = gr.Image(label="③ CLAHE Normalized (Model Input)", type="pil")

            gr.Markdown("### 🩺 Diagnostic Prediction & Explainability")
            with gr.Row():
                with gr.Column():
                    grade_label = gr.Label(label="Predicted Severity Grade", num_top_classes=1)
                    confidence_plot = gr.Label(label="Class Probability Distribution", num_top_classes=5)
                with gr.Column():
                    gradcam_panel = gr.Image(label="Grad-CAM Lesion Heatmap Overlay", type="pil")

            with gr.Accordion("📖 Clinical Severity Scale & Explainability Guide", open=False):
                gr.Markdown(
                    """
                    * **Grade 0 (No DR):** No apparent microaneurysms or retinal lesions.
                    * **Grade 1 (Mild NPDR):** Microaneurysms only.
                    * **Grade 2 (Moderate NPDR):** More than microaneurysms, but less than severe NPDR.
                    * **Grade 3 (Severe NPDR):** >20 intraretinal hemorrhages in each of 4 quadrants, venous beading, or IRMA.
                    * **Grade 4 (Proliferative DR):** Neovascularization or vitreous/preretinal hemorrhage (Urgent referral).
                    * **Grad-CAM Interpretation:** Red/yellow hot zones highlight spatial features (e.g. vascular abnormalities and exudates) that most heavily influenced the ordinal boundary score.
                    """
                )

    submit_btn.click(
        fn=predict,
        inputs=[input_img, use_tta_cb],
        outputs=[
            raw_panel,
            cropped_panel,
            clahe_panel,
            grade_label,
            confidence_plot,
            gradcam_panel,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
