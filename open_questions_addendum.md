# Open Questions Addendum

> Companion document to `implementation_plan.md`.  
> Addresses the four deferred design decisions. Does not modify the base plan.

---

## Q1 — Test-Time Augmentation (TTA)

### Decision
Use **horizontal flip only**. Average predictions from the original image and its horizontal flip (2 forward passes total).

### Why This Specific Scope
- EfficientNet-B0 inference on CPU takes ~50–100ms per pass. Two passes = ~100–200ms. Unnoticeable in a Gradio demo.
- Horizontal flip is clinically safe for fundus images — the optic disc / fovea anatomy is not meaningfully distorted.
- Vertical flip or rotation TTA would require verifying that your Ben Graham circle-crop doesn't produce asymmetric padding artifacts — not worth the debugging complexity here.
- Using 5+ augmentations (full TTA) would complicate the Grad-CAM overlay logic (you'd need to average heatmaps too, or only run Grad-CAM on the original). Keep it simple.

### Where It Lives
- **`src/evaluation/metrics.py`** — add a `predict_with_tta(model, image_tensor)` utility function.
- **`app/inference.py`** — call this function instead of a raw forward pass.
- **`scripts/evaluate.py`** — add a `--use-tta` flag (Hydra override: `inference.tta=true`).

### Implementation Notes
```
# Pseudocode — actual implementation goes in Phase 7

def predict_with_tta(model, image_tensor):
    # image_tensor shape: (1, C, H, W)
    flipped = torch.flip(image_tensor, dims=[-1])   # horizontal flip
    
    logits_orig   = model(image_tensor)
    logits_flip   = model(flipped)
    
    # For CE head: average softmax probabilities
    # For CORAL head: average sigmoid probabilities, then rank-decode
    probs = (sigmoid_or_softmax(logits_orig) + sigmoid_or_softmax(logits_flip)) / 2
    return probs
```

**Key nuance for CORAL:** Average the sigmoid probabilities *before* decoding the rank — not the decoded integer grades. Averaging grades directly (e.g., (2 + 3) / 2 = 2.5) loses information and breaks the ordinal decode logic.

### Grad-CAM Interaction
In the Gradio app, run Grad-CAM on the **original image only** (no flip). The TTA prediction produces the final grade/confidence. Grad-CAM is for visual explanation — running it on the average of two images would produce a confusing heatmap. Explain this separation in the app's UI text.

### Impact on Results Table
Add a TTA column to the final results table in the README:

| Model | Loss | Val QWK | Val QWK + TTA | Δ TTA |
|---|---|---|---|---|
| B0 | CORAL | ~0.XX | ~0.XX | +0.0X |

---

## Q2 — W&B Micro-Sweep Strategy

### Decision
Run a **3–5 run learning rate sweep** using W&B Sweeps on the CORAL model only. Fixed: all other hyperparameters. Log everything, publish the dashboard link in the README.

### Why This Scope
- A sweep proves you know MLOps practices beyond "I ran one training script."
- Kaggle/Colab GPU quota is finite. 5 runs × 30 epochs at ~20 min/run = ~1.7 GPU hours on Kaggle. Well within the free weekly quota.
- Sweeping *only* LR avoids confounding variables. It also produces a clean, readable W&B parallel coordinates plot — much better for README screenshots than a chaotic 50-run grid.

### Sweep Config (YAML)
This file goes in `configs/sweep/coral_lr_sweep.yaml`:

```yaml
program: scripts/train.py
method: bayes          # Bayesian optimization — smarter than grid for 5 runs
metric:
  name: val/qwk
  goal: maximize
parameters:
  training.lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  # Everything else is pinned:
  loss: coral           # fixed
  model: efficientnet_b0  # fixed
  training.epochs: 30   # fixed
  training.batch_size: 32  # fixed
```

### How to Launch (Colab cell)
```bash
wandb sweep configs/sweep/coral_lr_sweep.yaml   # returns SWEEP_ID
wandb agent <SWEEP_ID> --count 5                # runs 5 agents sequentially
```

### What to Screenshot for README
1. **Parallel coordinates plot** — shows LR vs. Val QWK. Shows you understand hyperparameter sensitivity.
2. **Run comparison table** — W&B auto-generates this. Embed it as a PNG in the README.
3. **Best run config** — W&B highlights the winning run. Mention the winning LR in your model card.

### Timing in the Phases
Run the sweep **after Phase 3** (CORAL baseline run), before Phase 4 (calibration). The sweep winner's checkpoint becomes your `coral_best.pth` used for all downstream analysis.

---

## Q3 — Gradio Preprocessing Side-by-Side Panel

### Decision
Show a **three-panel comparison** inside the Gradio app:
1. Raw uploaded image (as-is from the user)
2. After circle-crop + resize
3. After CLAHE contrast normalization (the final model input)

### Why Three Panels, Not Two
The Ben Graham pipeline has two visually distinct stages:
- Circle-crop removes the black border/vignetting → dramatic visual change.
- CLAHE adjusts the contrast within the circle → subtle but clinically meaningful.

Splitting these into two visible intermediate steps shows you understand **each** operation's role, not just the combined effect. Recruiters who hover on this will be impressed.

### Where It Lives
- **`app/inference.py`** — the preprocessing pipeline must be written as discrete, inspectable steps that return intermediate images, not just the final tensor.
- **`app/app.py`** — Gradio layout with `gr.Row()` containing three `gr.Image()` outputs.

### Gradio Layout Sketch
```python
# Pseudocode for app/app.py

with gr.Blocks() as demo:
    gr.Markdown("## Diabetic Retinopathy Severity Grading")
    
    with gr.Row():
        input_img = gr.Image(label="Upload Retinal Fundus Image", type="pil")
    
    submit_btn = gr.Button("Analyze", variant="primary")
    
    gr.Markdown("### Preprocessing Pipeline")
    with gr.Row():
        raw_panel    = gr.Image(label="① Raw Input")
        cropped_panel = gr.Image(label="② Circle Crop + Resize")
        clahe_panel  = gr.Image(label="③ CLAHE Normalized (Model Input)")
    
    gr.Markdown("### Prediction")
    with gr.Row():
        grade_label  = gr.Label(label="Predicted Grade")
        confidence_plot = gr.BarPlot(label="Confidence per Class")
    
    gr.Markdown("### Explainability")
    gradcam_panel = gr.Image(label="Grad-CAM Overlay")
    
    submit_btn.click(
        fn=run_inference,
        inputs=[input_img],
        outputs=[raw_panel, cropped_panel, clahe_panel,
                 grade_label, confidence_plot, gradcam_panel]
    )
```

### Implementation Notes for `app/inference.py`
The function `run_inference(pil_image)` must return intermediate images in PIL/numpy format (not tensors) so Gradio can display them:

```
1. raw_img       = pil_image (pass through directly)
2. cropped_img   = apply_circle_crop_and_resize(pil_image) → return as PIL
3. clahe_img     = apply_clahe(cropped_img) → return as PIL
4. model_tensor  = apply_normalization(clahe_img) → tensor for model
5. Run model(model_tensor) → get grade + probabilities
6. Run Grad-CAM on model_tensor using clahe_img as the overlay base
```

**Key:** The Grad-CAM overlay goes on top of the CLAHE-normalized image (panel ③), not the raw image. The heatmap corresponds to the features the model actually saw.

### CPU-Friendliness Check
Ben Graham with OpenCV (CLAHE, GaussianBlur, circle-mask) runs in ~5–10ms on CPU for a 380×380 image. No issue.

---

## Q4 — EfficientNet-B3 Stretch Goal

### Decision
Run B3 as a **single final experiment** after the full B0 pipeline is validated. Use the best configuration found from B0 experiments (CORAL loss + winning LR from sweep). Do not re-run the full ablation study on B3 — that would waste GPU quota.

### Scope
| What to do with B3 | What NOT to do with B3 |
|---|---|
| One CORAL training run (best config) | CE baseline on B3 |
| Eval on APTOS val + Messidor-2 | Focal loss ablation on B3 |
| Grad-CAM gallery | Another W&B sweep |
| Add to results table | Full calibration analysis |

The README story becomes: *"We validated the full pipeline on B0 for speed. B3 gave us our best QWK."* Simple, credible, impressive.

### Architectural Differences to Account For

| Property | EfficientNet-B0 | EfficientNet-B3 |
|---|---|---|
| Input resolution | 224×224 (or 380×380) | 300×300 (or 416×416) |
| Parameters | ~5.3M | ~12M |
| timm model name | `efficientnet_b0` | `efficientnet_b3` |
| Recommended batch size (T4) | 32 | 16–24 |
| Training time / epoch | ~4–6 min | ~8–12 min |

**Resolution note:** The `timm` default for B3 is 300×300. However, since APTOS images are retinal fundus images with significant detail, you can push to 380×380 for B3 too (same as B0) — just drop batch size to 16 to stay within T4 VRAM. Add this as a Hydra config override: `data.image_size=380`.

### Config File to Add
`configs/model/efficientnet_b3.yaml`:

```yaml
backbone: efficientnet_b3
pretrained: true
image_size: 380
drop_rate: 0.3        # B3 has more parameters, use slightly higher dropout
drop_path_rate: 0.2
```

### Where It Lives in the Phases
- **Phase 0:** Add `efficientnet_b3.yaml` config. The `model_factory.py` already reads from config — no code changes needed if you build the factory correctly.
- **Phase 3 (after B0 CORAL is done):** Run `python scripts/train.py model=efficientnet_b3 loss=coral training.lr=<winning_lr> training.batch_size=16`.
- **Phase 8 (polish):** Add B3 row to the README results table.

### README Comparison Table Target
| Model | Loss | Val QWK | Messidor QWK | Params | Notes |
|---|---|---|---|---|---|
| EfficientNet-B0 | Cross-Entropy | ~0.XX | ~0.XX | 5.3M | Baseline |
| EfficientNet-B0 | Focal (γ=2.0) | ~0.XX | ~0.XX | 5.3M | Imbalance ablation |
| EfficientNet-B0 | CORAL | ~0.XX | ~0.XX | 5.3M | **Main contribution** |
| EfficientNet-B0 | CORAL + TTA | ~0.XX | ~0.XX | 5.3M | Inference enhancement |
| **EfficientNet-B3** | **CORAL** | **~0.XX** | **~0.XX** | **12M** | **Best result ⭐** |

This table is the single most impressive thing a recruiter will see in the first 30 seconds. Bold the B3 CORAL row.

### Deployment Decision
Deploy **B0** to HuggingFace Spaces, not B3. Reason: B3 + TTA on CPU will be noticeably slower in a free HF Space (no GPU), and the app will feel sluggish to reviewers clicking through your portfolio. B0 + TTA gives a responsive demo. B3 is for the headline QWK number in the README. Mention this tradeoff explicitly in the app's description — it shows production-awareness.

---

## Summary of Addendum Changes to Phases

| Phase | Change |
|---|---|
| Phase 0 | Add `configs/model/efficientnet_b3.yaml`. Add `predict_with_tta()` stub to `src/evaluation/metrics.py`. |
| Phase 3 | After B0 CORAL run: run W&B sweep (Q2). Winner becomes `coral_best.pth`. After sweep: run B3 CORAL (Q4). |
| Phase 5 | Grad-CAM gallery uses original image + CLAHE panel as overlay base (Q3). |
| Phase 6 | Domain shift eval uses TTA (`--use-tta` flag) for final Messidor-2 numbers (Q1). |
| Phase 7 | Gradio app shows 3-panel preprocessing comparison (Q3). TTA enabled in `inference.py` (Q1). Deploy B0 only (Q4). |
| Phase 8 | README results table has 5 rows including B3 (Q4). W&B sweep screenshot embedded (Q2). |
