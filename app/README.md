---
title: Diabetic Retinopathy Severity Grading
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# 👁️ Diabetic Retinopathy Clinical Grading & Trustworthiness

This Hugging Face Space hosts the interactive clinical demo for **Diabetic Retinopathy Severity Grading**:

* **Backbone:** EfficientNet-B0
* **Loss Formulation:** Consistent Rank Logits (CORAL) Ordinal Regression
* **Preprocessing:** Ben Graham Circular ROI + Contrast Limited Adaptive Histogram Equalization (CLAHE)
* **Explainability:** Grad-CAM lesion localization heatmaps
* **Test-Time Augmentation (TTA):** Horizontal flip sigmoid probability averaging before ordinal decoding

### Usage
1. Upload any retinal fundus photograph (or click one of the pre-loaded clinical samples).
2. Toggle Test-Time Augmentation (TTA) if desired.
3. Click **Run Diagnostic Assessment**.
4. Inspect the 3 intermediate preprocessing stages, the predicted grade, class probabilities, and the Grad-CAM lesion overlay!
