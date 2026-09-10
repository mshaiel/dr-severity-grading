# 📋 Engineering Insights, Failure Modes & Lessons Learned

A mature machine learning project openly documents what failed, what didn't work as expected, and how architectural trade-offs were navigated.

---

### 1. Offline vs. Online Ben Graham Preprocessing
* **Observation:** Applying circle-crop ROI detection, dual-stage CLAHE, and Gaussian local mean subtraction dynamically on the fly inside PyTorch's `Dataset.__getitem__` creates a major CPU bottleneck during cloud training. On a Colab T4 GPU, GPU utilization dropped below 35% with 2 workers.
* **Resolution:** Precomputing and saving preprocessed $380 \times 380$ images offline into `data/aptos2019/preprocessed_380/` increased GPU utilization to 95%+, cutting epoch time from 18 minutes to 4.5 minutes.

---

### 2. Test-Time Augmentation (TTA) Restraint
* **Observation:** Standard TTA recipes often include 8–16 transformations (rotations by 90°, 180°, 270°, vertical and horizontal flips, multi-scale crops). In fundus photography, aggressive multi-scale cropping frequently truncates the optic disc or fovea, leading to destabilized predictions. Furthermore, rotating fundus images introduces non-trivial circular mask boundary artifacts.
* **Resolution:** We constrained TTA strictly to **horizontal flipping** (2 forward passes total). This is anatomically benign for retinal vasculature while keeping CPU inference latency under 150ms for deployment in the Gradio web demo.

---

### 3. Calibration on Ordinal Outputs
* **Observation:** Standard Expected Calibration Error (ECE) implementations and Temperature Scaling libraries (like `netcal`) assume standard categorical softmax probabilities. CORAL outputs $K - 1$ independent sigmoid logits representing cumulative threshold probabilities ($P(\text{grade} \ge k)$).
* **Resolution:** Directly feeding cumulative threshold probabilities into softmax produces ungrounded numbers. We explicitly converted the cumulative sigmoids to a probability density function using the adjacent-categories formulation ($P(\text{grade} = c) = P(\text{grade} \ge c) - P(\text{grade} \ge c + 1)$) before running temperature scaling and reliability diagram generation.

---

### 4. Deployment Model Selection: EfficientNet-B0 vs. B3
* **Trade-off:** EfficientNet-B3 achieves higher headline QWK (~0.85 vs ~0.82) due to higher spatial capacity for subtle microaneurysms. However, running B3 with TTA and Grad-CAM on free-tier Hugging Face Spaces CPU leads to 1.5–2.0 second per-request latencies.
* **Resolution:** We deployed EfficientNet-B0 to production for sub-second responsiveness, and showcased EfficientNet-B3 as the headline benchmark in the repository README comparison table.
