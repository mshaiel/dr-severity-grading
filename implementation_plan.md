# Diabetic Retinopathy Severity Grading — Full Implementation Plan

> **Role:** Staff ML Engineer mentoring an AI graduate  
> **Target Audience:** Master's admissions committees + industry internship recruiters  
> **Stack:** PyTorch · EfficientNet · CORAL Ordinal Loss · Grad-CAM · W&B · Hydra · Gradio · HuggingFace Hub

---

## 1. Project Overview & Narrative

The project answers a clinically meaningful question: *"Given a retinal fundus image, how severe is the patient's diabetic retinopathy — and can we trust the model's answer?"* 

The narrative arc is deliberate and tells a story of rigorous ML engineering:

1. **Baseline** — EfficientNet-B0 + Cross-Entropy (the strawman)
2. **Contribution** — Switch to CORAL ordinal loss (the improvement)
3. **Rigor** — Calibration, Grad-CAM XAI, QWK delta (proving it works *and* is trustworthy)
4. **Generalization** — Zero-shot eval on Messidor-2 (proving it isn't overfit)
5. **Deployment** — Gradio app on HF Spaces (proving it's production-aware)

---

## 2. Repository Structure

```
dr-severity-grading/
│
├── README.md                    # Project overview, badges (W&B, HF Space), results table
├── LICENSE
├── .gitignore                   # Excludes data/, runs/, *.pth > 100MB
├── requirements.txt             # Local dev dependencies
├── requirements-colab.txt       # Colab-specific overrides (e.g., pinned torch version)
├── setup.py                     # Makes src/ importable as a package
│
├── configs/                     # Hydra config files (YAML)
│   ├── config.yaml              # Master config (imports from sub-groups)
│   ├── model/
│   │   ├── efficientnet_b0.yaml
│   │   └── efficientnet_b3.yaml  # (optional future upgrade)
│   ├── loss/
│   │   ├── cross_entropy.yaml
│   │   ├── focal.yaml
│   │   └── coral.yaml
│   ├── data/
│   │   ├── aptos.yaml
│   │   └── messidor2.yaml
│   └── training/
│       ├── baseline.yaml
│       └── ordinal.yaml
│
├── data/                        # NOT committed to git (in .gitignore)
│   ├── aptos2019/
│   │   ├── train_images/
│   │   └── train.csv
│   └── messidor2/
│       ├── images/
│       └── messidor_data.csv
│
├── notebooks/                   # Exploration & reporting (committed to git)
│   ├── 00_eda.ipynb             # Class distribution, image quality analysis
│   ├── 01_preprocessing_debug.ipynb   # Validate Ben Graham pipeline on 10 samples
│   ├── 02_calibration_analysis.ipynb  # Reliability diagrams post-training
│   └── 03_gradcam_gallery.ipynb       # Visual XAI gallery for README
│
├── src/                         # Core importable package
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Ben Graham circle-crop + CLAHE pipeline
│   │   ├── aptos_dataset.py     # APTOS PyTorch Dataset class
│   │   ├── messidor_dataset.py  # Messidor-2 PyTorch Dataset class
│   │   ├── transforms.py        # Albumentations augmentation pipelines (train/val/test)
│   │   └── samplers.py          # Weighted random sampler for class imbalance
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py          # EfficientNet-B0 with swappable head
│   │   ├── heads.py             # CE head (softmax) vs. CORAL head (sigmoid chains)
│   │   └── model_factory.py     # Builds model from Hydra config
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── cross_entropy.py     # Standard CE with optional class weights
│   │   ├── focal.py             # Focal Loss (from scratch, well-commented)
│   │   └── coral.py             # CORAL ordinal loss (from scratch, well-commented)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Main training loop with W&B logging
│   │   ├── scheduler.py         # LR scheduler factory (CosineAnnealingWarmRestarts)
│   │   └── early_stopping.py    # Patience-based early stopping on val QWK
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # QWK, Cohen's Kappa, per-class F1, ECE
│   │   ├── calibration.py       # Temperature scaling + reliability diagram generation
│   │   └── domain_shift.py      # Evaluation harness for Messidor-2 (OOD eval)
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py           # Grad-CAM implementation (hooks on EfficientNet last conv)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py              # Global reproducibility seed
│       ├── logging_utils.py     # Structured logger setup
│       └── checkpoint.py        # Save/load model checkpoints + W&B artifact upload
│
├── scripts/                     # CLI entry points (run from repo root)
│   ├── train.py                 # Main training script (Hydra-driven)
│   ├── evaluate.py              # Offline evaluation on val/test set
│   ├── calibrate.py             # Post-hoc temperature scaling
│   └── predict.py               # Single-image inference script
│
├── app/                         # Gradio deployment app
│   ├── app.py                   # Main Gradio app entry point
│   ├── model_loader.py          # Loads model from HF Hub
│   ├── inference.py             # Preprocessing + prediction + Grad-CAM pipeline
│   ├── requirements.txt         # App-only dependencies (lighter than training)
│   └── assets/
│       └── sample_images/       # A few demo images (gradable, annotated)
│
├── tests/                       # Unit & integration tests
│   ├── test_preprocessing.py    # Validate Ben Graham pipeline output shape/range
│   ├── test_datasets.py         # Dataset __len__ and __getitem__ smoke tests
│   ├── test_losses.py           # CORAL loss numerical correctness tests
│   ├── test_metrics.py          # QWK known-value sanity tests
│   └── test_gradcam.py          # Grad-CAM output shape tests
│
└── results/                     # Committed lightweight artifacts
    ├── figures/                 # Reliability diagrams, confusion matrices, QWK tables
    └── tables/                  # CSV files with experiment comparison tables
```

---

## 3. Phased Implementation Plan

### Phase 0 — Local Scaffolding (Laptop / GTX 1050) · ~2 days

**Goal:** Get the full pipeline running on 50 images. No model training yet.

| Task | Location | Details |
|---|---|---|
| Init repo + git | Root | Create `.gitignore`, `setup.py`, `README.md` stub |
| Install local deps | `requirements.txt` | PyTorch CPU or tiny CUDA, albumentations, hydra-core, pytest |
| Write preprocessing | `src/data/preprocessing.py` | Ben Graham pipeline: resize → circle-mask → CLAHE → normalize |
| Write APTOS dataset | `src/data/aptos_dataset.py` | `__getitem__` returns `(image_tensor, label, image_path)` |
| Write Messidor dataset | `src/data/messidor_dataset.py` | Filters to `adjudicated_gradable == 1` only |
| Write augmentations | `src/data/transforms.py` | Train: RandomHorizontalFlip, RandomRotate90, ColorJitter, CoarseDropout. Val/Test: Resize + Normalize only |
| Write CORAL loss | `src/losses/coral.py` | Implement from scratch with full docstring |
| Write Focal Loss | `src/losses/focal.py` | Implement from scratch with full docstring |
| Write model | `src/models/backbone.py` + `heads.py` | EfficientNet-B0 from `timm`, two interchangeable heads |
| Write metrics | `src/evaluation/metrics.py` | QWK using `sklearn.metrics.cohen_kappa_score(weights='quadratic')` |
| Smoke test | `tests/` | Run all unit tests. Dataset returns correct shapes. Loss is differentiable |
| Notebook EDA | `notebooks/00_eda.ipynb` | Plot class distribution bar chart, show 5 images per class |
| Preprocessing debug | `notebooks/01_preprocessing_debug.ipynb` | Visualize before/after Ben Graham on 10 images |

**Definition of Done:** `pytest tests/` passes on CPU. A dataloader iterates without error.

---

### Phase 1 — Cloud Training Run 1: CE Baseline (Colab T4) · ~1 day

**Goal:** Establish a reproducible baseline score with W&B logging.

**Config:** `configs/loss/cross_entropy.yaml` + `configs/training/baseline.yaml`

| Task | Details |
|---|---|
| Upload data to Google Drive | Mount Drive in Colab. Store `aptos2019/` there |
| Install Colab deps | `!pip install -r requirements-colab.txt` then `!pip install -e .` |
| Configure W&B | `wandb.init(project="dr-severity-grading", name="run-ce-baseline")` |
| Run training | `python scripts/train.py loss=cross_entropy model=efficientnet_b0` |
| Log to W&B | Log: train/val loss, val QWK, val accuracy, per-epoch LR, GPU memory |
| Save checkpoint | Best model by val QWK → `checkpoints/ce_baseline_best.pth` + W&B artifact |
| Record results | Val QWK, per-class F1, confusion matrix → `results/tables/run1_ce_baseline.csv` |

**Training config:**
- Epochs: 30 (with early stopping, patience=7)
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingWarmRestarts (T_0=10)
- Batch size: 32 (Colab T4 should handle this at 380×380)
- Image size: 380×380 (matches EfficientNet-B0 input)
- Pretrained: ImageNet weights via `timm`
- Class weighting: **Yes** (inverse frequency, as a baseline imbalance strategy)

---

### Phase 2 — Cloud Training Run 2: Focal Loss Ablation (Colab T4) · ~0.5 days

**Goal:** Measure the effect of Focal Loss vs. class weighting on imbalanced classes.

**Config:** `configs/loss/focal.yaml`

| Task | Details |
|---|---|
| Run training | `python scripts/train.py loss=focal model=efficientnet_b0` |
| Sweep gamma | Log runs for γ ∈ {0.5, 1.0, 2.0} as a W&B sweep |
| Compare | Add results row to experiment comparison table |

---

### Phase 3 — Cloud Training Run 3: CORAL Ordinal Loss (Colab T4) · ~1 day

**Goal:** The main contribution. Prove ordinal loss improves QWK over CE.

**Config:** `configs/loss/coral.yaml` + `configs/training/ordinal.yaml`

**Key architectural difference:** The model head now outputs `K-1 = 4` sigmoid logits (one per rank threshold), not 5 softmax logits. Grade prediction = `sum(sigmoid(logits) > 0.5)`.

| Task | Details |
|---|---|
| Switch head | Use `CoralHead` from `src/models/heads.py` |
| Run training | `python scripts/train.py loss=coral model=efficientnet_b0` |
| Log to W&B | Same metrics + CORAL-specific: predicted rank distribution |
| Record QWK delta | Δ QWK = CORAL_QWK − CE_QWK. This is the headline result |
| Save checkpoint | Best CORAL model → `checkpoints/coral_best.pth` + W&B artifact |

---

### Phase 4 — Calibration Analysis (Colab / Local) · ~0.5 days

**Goal:** Check if confidence scores are trustworthy. Generate reliability diagrams for the README.

| Task | Location | Details |
|---|---|---|
| Compute ECE | `src/evaluation/calibration.py` | Expected Calibration Error before temperature scaling |
| Temperature scaling | `src/evaluation/calibration.py` | Post-hoc calibration on val set (single scalar T parameter) |
| Reliability diagrams | `notebooks/02_calibration_analysis.ipynb` | Plot for CE model, Focal model, CORAL model |
| Record ECE | Pre/post calibration ECE → results table |

**Note on CORAL calibration:** CORAL outputs sigmoid probabilities per threshold, not a standard softmax distribution. The calibration analysis should convert CORAL outputs to a grade probability distribution using the adjacent-categories parameterization before computing ECE. Document this nuance explicitly in comments — it shows depth.

---

### Phase 5 — Grad-CAM XAI (Colab / Local) · ~0.5 days

**Goal:** Prove the model localizes clinically relevant lesions, not image artifacts.

| Task | Location | Details |
|---|---|---|
| Implement Grad-CAM | `src/explainability/gradcam.py` | Register forward/backward hooks on `model.features[-1]` (last EfficientNet conv block) |
| Generate overlays | `notebooks/03_gradcam_gallery.ipynb` | For 5 images per class: original → Grad-CAM heatmap → overlay |
| Qualitative analysis | Notebook | Do activations align with microaneurysms/hemorrhages? Note any failures |
| Save gallery | `results/figures/gradcam_gallery.png` | Commit this to git for README embedding |

**Implementation detail:** For CORAL (sigmoid head), Grad-CAM must target a specific class boundary (e.g., "probability of grade ≥ 2"). Use the sum of the relevant sigmoid outputs as the target scalar. Document this in the code.

---

### Phase 6 — Cross-Dataset Domain Shift Evaluation (Colab) · ~0.5 days

**Goal:** Measure how well the APTOS-trained model generalizes to Messidor-2 (zero-shot).

| Task | Location | Details |
|---|---|---|
| Load Messidor-2 | `src/data/messidor_dataset.py` | Filter: `adjudicated_gradable == 1` only (~1,748 images, but filter may reduce) |
| Run evaluation | `scripts/evaluate.py --dataset messidor2 --checkpoint coral_best` | |
| Report metrics | `src/evaluation/domain_shift.py` | QWK, per-class F1, confusion matrix on Messidor-2 |
| Compare to APTOS val | The gap is your "domain shift penalty." Discuss in README | |

**Expected insight:** There will be a performance drop. That's OK and expected — the key is to *quantify* it and discuss *why* (label definition differences, image acquisition protocol, etc.).

---

### Phase 7 — Gradio App + HuggingFace Deployment · ~1 day

**Goal:** Production-ready demo that runs on HF Spaces CPU (no GPU).

**Architecture Decision:** The training repo lives on GitHub. The model weights live on HuggingFace Hub (uploaded as a model artifact). The HF Space is a **separate** repo that downloads the model from HF Hub at startup.

| Task | Location | Details |
|---|---|---|
| Upload weights to HF Hub | HF website | `from huggingface_hub import upload_file`. Upload `coral_best.pth` + `config.yaml` |
| Write model loader | `app/model_loader.py` | `hf_hub_download()` to pull weights. CPU-friendly inference mode |
| Write inference pipeline | `app/inference.py` | Ben Graham preprocessing → model forward → grade + confidence + Grad-CAM overlay |
| Write Gradio app | `app/app.py` | Inputs: image upload. Outputs: grade label, confidence bar chart, Grad-CAM overlay image |
| Test locally | `python app/app.py` | |
| Create HF Space | HF website | Create a new Space, push `app/` contents to it |
| Link from README | `README.md` | Add HF Space badge + link |

**Gradio UI Layout:**
```
[Image Upload] → [Ben Graham Preview] → [Grade: Moderate (Grade 2)]
                                      → [Confidence Bar Chart]
                                      → [Grad-CAM Overlay]
```

---

### Phase 8 — Polish & Portfolio Packaging · ~1 day

**Goal:** Make the GitHub repo impressive to someone spending 60 seconds on it.

| Task | Details |
|---|---|
| README.md | Hero image, W&B badge, HF Space badge, results table (CE vs Focal vs CORAL), Grad-CAM gallery, dataset credits |
| Results table | Markdown table: Model × {Val QWK, Val ECE, Messidor QWK} |
| CORAL loss explainer | Add `docs/coral_loss_explained.md` with math + diagram |
| Code quality | Run `black` + `isort` + `flake8` on all `src/` files |
| `pytest` CI | Add GitHub Actions `.github/workflows/tests.yml` to run `pytest` on every push |
| Model card | HF Hub model card with training details, evaluation results, intended use, limitations |

---

## 4. Technical Library Decisions

### 4.1 Model Backbone
| Choice | Library | Notes |
|---|---|---|
| EfficientNet-B0 | `timm` (`pip install timm`) | `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)` strips the head |

### 4.2 Ordinal Loss — CORAL (from scratch)
CORAL (Consistent Rank Logits) reformulates K-class ordinal regression as K-1 binary classification tasks sharing all parameters except the final bias terms.

**Loss formula:**
$$\mathcal{L}_{CORAL} = -\sum_{k=1}^{K-1} \left[ \lambda_k \left( y_k \log \hat{p}_k + (1 - y_k) \log(1 - \hat{p}_k) \right) \right]$$

where $y_k = \mathbb{1}[\text{true\_grade} \geq k]$ (binary labels per threshold) and $\hat{p}_k = \sigma(W^T x + b_k)$ (shared weights, per-threshold biases).

**Predicted grade:** $\hat{y} = \sum_{k=1}^{K-1} \mathbb{1}[\hat{p}_k > 0.5]$

**Key advantage over standard CE:** The loss *enforces rank consistency* — the model cannot predict P(grade≥3) > P(grade≥2), which is a logical guarantee CE does not provide.

**Implementation:** `src/losses/coral.py` — write from scratch with the math documented in the docstring. This is your interview talking point.

### 4.3 QWK Calculation
```python
from sklearn.metrics import cohen_kappa_score
qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
```
**No extra library needed.** The key is to log this as the primary validation metric (not accuracy), since QWK penalizes large ordinal errors more heavily.

### 4.4 Grad-CAM
| Option | Notes |
|---|---|
| `pytorch-grad-cam` (`pip install grad-cam`) | **Recommended.** By Jacob Gildenblat. Supports `EigenCAM`, `GradCAM`, `GradCAM++`. Handles EfficientNet natively. Use `EigenCAM` for CORAL's multi-output head since it doesn't require a single scalar target. |
| Custom hook implementation | Write if you want to show depth in a specific notebook, but use the library in production |

**For the app:** Use `pytorch-grad-cam` directly. It's battle-tested.  
**For the interview:** Be able to explain Grad-CAM from scratch (grad of class score w.r.t. feature maps → global average pool → ReLU).

### 4.5 Calibration
| Tool | Notes |
|---|---|
| `netcal` (`pip install netcal`) | ECE computation + reliability diagram plotting |
| Manual temperature scaling | Implement in `src/evaluation/calibration.py` (single parameter, optimized on val set with NLL) |

### 4.6 Augmentation
```
albumentations==1.4.x
```
**Train pipeline:**
1. `RandomResizedCrop(380, 380, scale=(0.8, 1.0))`
2. `HorizontalFlip(p=0.5)`
3. `VerticalFlip(p=0.5)`
4. `RandomRotate90(p=0.5)`
5. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5)`
6. `CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)` (regularization)
7. `Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`

**Val/Test pipeline:**
1. `Resize(380, 380)`
2. `Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`

### 4.7 Full Dependency List

**`requirements.txt` (local dev):**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.4.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.16.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
grad-cam>=1.4.8
netcal>=1.3.5
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
huggingface_hub>=0.19.0
```

**`requirements-colab.txt`:**
```
# Colab already has torch, torchvision, numpy, pandas, PIL, sklearn, matplotlib
timm>=0.9.0
albumentations>=1.4.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.16.0
grad-cam>=1.4.8
netcal>=1.3.5
huggingface_hub>=0.19.0
```

---

## 5. Data Pipeline Design

### 5.1 Ben Graham Preprocessing (APTOS standard)
This is the canonical preprocessing for APTOS. Every serious submission uses it.

```
1. Load image (BGR)
2. Detect circular ROI: find the largest contour or use radius = 90% of min(H, W)/2
3. Crop to circle bounding box
4. Resize to 380×380
5. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel of LAB colorspace
6. Subtract local mean (Gaussian blur subtracted with addWeighted) → reduces vignetting
7. Convert back to RGB tensor
```

This must be applied **offline** (precompute and save processed images) on Colab to avoid CPU bottleneck during training.

### 5.2 Class Imbalance Strategy

**APTOS class distribution (approximate):**
| Grade | Label | Count | % |
|---|---|---|---|
| No DR | 0 | 1,805 | 49.3% |
| Mild | 1 | 370 | 10.1% |
| Moderate | 2 | 999 | 27.3% |
| Severe | 3 | 193 | 5.3% |
| PDR | 4 | 295 | 8.1% |

**Strategy (use both, then ablate):**
- **Class Weighting:** `weight_i = total_samples / (num_classes × count_i)` passed to loss function
- **Weighted Random Sampler:** `torch.utils.data.WeightedRandomSampler` to over-sample minority classes at the batch level
- **Focal Loss ablation:** γ sweep via W&B hyperparameter sweep

### 5.3 Train/Val Split
- APTOS has no official validation set. Use **stratified 80/20 split** on the training CSV.
- Stratify by label to maintain class distribution in both splits.
- Fix random seed (42) for reproducibility.
- **Messidor-2 is strictly held out** — never used for any training or hyperparameter selection.

---

## 6. Evaluation Framework

### 6.1 Primary Metric: Quadratic Weighted Kappa (QWK)
- Range: [-1, 1]. Target: > 0.80 for Moderate/Good performance on APTOS.
- Report on: APTOS val, APTOS test (if exists), Messidor-2.

### 6.2 Experiment Comparison Table (target output)
| Model | Loss | Val QWK | Val ECE | Messidor QWK |
|---|---|---|---|---|
| EfficientNet-B0 | Cross-Entropy + Class Weights | ~0.XX | ~0.XX | ~0.XX |
| EfficientNet-B0 | Focal (γ=2.0) + Class Weights | ~0.XX | ~0.XX | ~0.XX |
| EfficientNet-B0 | CORAL Ordinal | ~0.XX | ~0.XX | ~0.XX |
| EfficientNet-B0 | CORAL + Temperature Scaling | ~0.XX | ~0.XX | ~0.XX |

### 6.3 Per-Class Breakdown
Always report per-class F1 alongside QWK. The rare classes (Severe, PDR) are where your model will struggle most — being transparent about this is a signal of maturity.

---

## 7. Key Technical Talking Points (for interviews & SOP)

1. **"Why ordinal regression instead of classification?"** — DR grades are not independent categories. A model confusing Grade 0 with Grade 4 should be penalized more than confusing Grade 2 with Grade 3. CORAL encodes this prior structurally, not just via the loss weighting.

2. **"How does CORAL handle the output layer differently?"** — Instead of a softmax over K classes, CORAL uses K-1 sigmoid outputs with *shared weights but independent biases*. This weight-sharing constraint is what enforces rank consistency.

3. **"What does Grad-CAM tell you that accuracy doesn't?"** — A model can achieve high accuracy by exploiting spurious correlations (e.g., image borders, scanner artifacts). Grad-CAM shows *where* the model looks. If it localizes microaneurysms in Grade 1 images, that's clinical evidence the model learned the right features.

4. **"What is calibration and why does it matter clinically?"** — A well-calibrated model that says "70% confident Grade 2" should be correct ~70% of the time when making that prediction. In clinical decision support, overconfident models (ECE >> 0) are dangerous. Temperature scaling is a simple, effective post-hoc fix.

5. **"Why does performance drop on Messidor-2?"** — Domain shift from different acquisition equipment, preprocessing conventions, and labeling protocols. This is expected and honest. The key is to quantify the gap and discuss mitigation strategies (e.g., test-time augmentation, domain adaptation).

---

## 8. GitHub Repository Best Practices (Portfolio-Specific)

- **README hero:** Start with a Grad-CAM gallery image and the results table. Don't bury the lede.
- **Badges:** W&B run badge, HF Spaces badge, GitHub Actions CI badge, Python version badge.
- **Reproducibility section:** Exact commands to reproduce each training run from a fresh clone.
- **Model card:** Both in the HF Hub model card AND in the README.
- **Commit hygiene:** Meaningful commit messages. No giant "added everything" commits.
- **Tag releases:** `v1.0-baseline`, `v2.0-coral`. Admissions committees can see your iteration history.
- **Discussion of failures:** A `docs/lessons_learned.md` or a section in the README discussing what didn't work is highly regarded by both academia and industry.

---

## 9. Estimated Timeline

| Phase | Description | Duration |
|---|---|---|
| 0 | Local scaffolding + unit tests | 2 days |
| 1 | CE baseline training + W&B logging | 1 day |
| 2 | Focal loss ablation sweep | 0.5 days |
| 3 | CORAL ordinal training | 1 day |
| 4 | Calibration analysis | 0.5 days |
| 5 | Grad-CAM XAI gallery | 0.5 days |
| 6 | Messidor-2 domain shift eval | 0.5 days |
| 7 | Gradio app + HF Space deployment | 1 day |
| 8 | Polish, README, CI, model card | 1 day |
| **Total** | | **~8 days of focused work** |

---

## 10. Open Questions / Decisions Log

- [ ] Will you use **test-time augmentation (TTA)** during inference? (Improves QWK ~0.01–0.02, adds complexity to the app)
- [ ] Do you want a **W&B hyperparameter sweep** for learning rate / batch size, or keep configs fixed to keep compute costs low?
- [ ] For the Gradio app, do you want a **before/after preprocessing comparison** panel (good for demos) or keep it minimal?
- [ ] Do you want to experiment with **EfficientNet-B3** as a stretch goal if B0 results are disappointing?
