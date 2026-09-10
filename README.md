# 👁️ Diabetic Retinopathy Severity Grading: Ordinal Regression & Model Calibration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![W&B Tracked](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/muhammadshaiel-comsats-university-islamabad/dr-severity-grading)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-dr--severity--grading--coral-blue)](https://huggingface.co/mshaiel2004/dr-severity-grading-coral)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live_Space-yellow)](https://huggingface.co/spaces/mshaiel2004/dr-severity-grading)
[![CI Tests](https://img.shields.io/badge/CI%20Tests-22%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **"Given a retinal fundus image, how severe is the patient's diabetic retinopathy — and can we trust the model's answer?"**  
> An end-to-end machine learning study implementing **Consistent Rank Logits (CORAL) Ordinal Regression**, **Ben Graham Preprocessing**, **Temperature-Scaled Calibration**, and **Grad-CAM Explainability** evaluated on the APTOS 2019 Blindness Detection benchmark.

> ⚠️ **Disclaimer:** *This is a machine learning research portfolio project evaluating ordinal loss functions and calibration techniques. It is not a diagnostic tool and has not been clinically validated.*

---

## 📊 Benchmark Results

| Model Architecture | Loss Formulation | Val QWK | Full Eval QWK | Val ECE | Optimal Temp ($T$) | Checkpoint Artifact |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | Cross-Entropy + Class Weights | **0.8980** | **0.9279** | 0.0428 (0.0095) | 1.000 | `checkpoints/ce_baseline_best.pth` |
| EfficientNet-B0 | Focal Loss ($\gamma=2.0$) | **0.8964** | **0.9150** | **0.0348** | 1.050 | `checkpoints/focal_best.pth` |
| EfficientNet-B0 | **CORAL Ordinal Regression** | **0.7383** | **0.7510** | 0.2641 | 1.269 | `checkpoints/coral_best.pth` |
| EfficientNet-B0 | **CORAL + Temperature Scaling** | **0.7420** | **0.7680** | **0.2501** ($\Delta -0.014$) | **1.2694** | `checkpoints/coral_best.pth` |

> 📌 **Engineering Analysis on CORAL Performance vs. Baseline:**  
> While CORAL mathematically guarantees rank-consistent boundaries by design, it achieved a lower validation QWK (0.7383) than the weighted Cross-Entropy baseline (0.8980). In APTOS 2019, class frequencies are heavily skewed (Grade 0 accounts for ~50% of the data, while Grade 3 accounts for only 5.3%). In standard Cross-Entropy, inverse-frequency weights ($W_c = N / (K \cdot N_c)$) directly counter this skew at the gradient level. In contrast, standard CORAL decomposes the task into $K-1=4$ unweighted binary classifiers evaluated with a default decision threshold ($\tau = 0.5$). Without per-threshold loss reweighting ($\lambda_k$) or threshold optimization, the shared weight vector drifts toward the dominant healthy class. Documenting this trade-off highlights that structural ordinal constraints require dedicated threshold calibration when deployed under extreme class imbalance.

---

## 🔬 Key Engineering Contributions & Methodology

1. **Baseline Model:** EfficientNet-B0 trained with Categorical Cross-Entropy and inverse-frequency class weights ($W_c = \frac{N}{K \cdot N_c}$).
2. **Class Imbalance Ablation:** Focal Loss ($\gamma=2.0$) dynamically scaling hard example gradients: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$.
3. **Core Contribution — CORAL Ordinal Loss:** Reformulation of 5-class grading into $K-1=4$ binary thresholds with structural parameter sharing ($\mathbf{w} \in \mathbb{R}^{D \times 1}$ with independent biases $b_k$) to enforce rank consistency ($P(\text{grade} \ge 3) \le P(\text{grade} \ge 2)$).
4. **Model Calibration & Reliability:** Expected Calibration Error (ECE) reduction via post-hoc Temperature Scaling optimized with L-BFGS, visualized with reliability diagrams.
5. **Grad-CAM Lesion Localization:** Gradient-weighted activation mapping targeting EfficientNet's deepest convolutional block, overlaid onto CLAHE-normalized fundus inputs.
6. **Production Deployment:** Interactive Gradio web application with a 3-stage preprocessing visualizer and Horizontal-Flip Test-Time Augmentation (TTA).

---

## 🖼️ Explainability & Calibration Visualizations

### Grad-CAM Lesion Localization Heatmap
![Grad-CAM Overlay](results/figures/sample_gradcam_overlay.png)
*Grad-CAM localizes relevant retinal pathology without being influenced by camera border artifacts.*

### Calibrated Reliability Diagram
![Reliability Diagram](results/figures/calibration_efficientnet_b0_coral.png)
*Post-hoc Temperature Scaling reliability diagram comparing observed empirical accuracy against confidence bins.*

---

## 🛠️ Repository Architecture

```
dr-severity-grading/
├── configs/                     # Hydra hierarchical configuration
│   ├── config.yaml              # Master configuration
│   ├── model/                   # Model configs (efficientnet_b0, efficientnet_b3)
│   ├── loss/                    # Losses (cross_entropy, focal, coral)
│   ├── data/                    # Datasets (aptos, messidor2)
│   ├── training/                # Training pipelines (baseline, ordinal)
│   └── sweep/                   # W&B Bayesian hyperparameter sweeps
├── src/                         # Core modular package
│   ├── data/                    # Preprocessing (Ben Graham), datasets, transforms
│   ├── models/                  # EfficientNet backbone, ClassificationHead, CoralHead
│   ├── losses/                  # Pure PyTorch CORAL, Focal, Weighted CE
│   ├── training/                # Trainer with W&B logging, CosineAnnealingWarmRestarts
│   ├── evaluation/              # QWK, ECE, TemperatureScaler, TTA
│   ├── explainability/          # Grad-CAM hooks and overlay generators
│   └── utils/                   # Seed, logging, and checkpointing
├── scripts/                     # Hydra-driven CLI commands (train, eval, calibrate, predict)
├── app/                         # Gradio deployment app with 3-panel preprocessing view
├── notebooks/                   # EDA, calibration, Grad-CAM, and master Colab pipeline
└── tests/                       # Complete 22-item PyTest unit test suite
```

---

## 🌐 Interactive Web Application Deployment

The application features:
1. **3-Stage Preprocessing Visualizer:** `[① Raw Input]` $\to$ `[② Circle Crop & Resize]` $\to$ `[③ CLAHE Normalized]`.
2. **Severity Grading:** Categorical confidence distribution.
3. **Grad-CAM Heatmap:** Lesion localization over the normalized retinal view.
4. **Horizontal-Flip TTA:** Averages threshold probabilities before ordinal rank decoding.

### Local Execution:
```bash
python app/app.py
```

### Hugging Face Spaces Deployment:
The application is pre-configured for Hugging Face Spaces:
```bash
cd app
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/dr-severity-grading
git add .
git commit -m "Deploy clinical DR grading demo"
git push -u origin main
```

---

## 🧪 Reproducing Experiments & Unit Tests

### Run Unit Test Suite:
```bash
pytest tests/ -v
```

### Reproduce Training Runs:
```bash
# Phase 1: Cross-Entropy Baseline
python scripts/train.py loss=cross_entropy model=efficientnet_b0 training=baseline

# Phase 2: Focal Loss Ablation
python scripts/train.py loss=focal model=efficientnet_b0 loss.gamma=2.0

# Phase 3: CORAL Ordinal Regression
python scripts/train.py loss=coral model=efficientnet_b0 training=ordinal
```

---

## 📜 Documentation & Deep Dives
* [CORAL Loss Mathematical Explanation](docs/coral_loss_explained.md)
* [Engineering Lessons Learned & Architectural Trade-Offs](docs/lessons_learned.md)

---

## ⚖️ License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
