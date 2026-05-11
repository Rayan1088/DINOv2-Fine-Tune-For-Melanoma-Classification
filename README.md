# DINOv2-Fine-Tune-For-Melanoma-Classification
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/DINOv2-4285F4?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
</p>

<h1 align="center">🔬 Melanoma Classification with DINOv2</h1>

<p align="center">
  <b>Fine-tuning DINOv2 (ViT-S/14) on dermoscopic images for binary melanoma detection</b><br>
  <i>Image-only pipeline · Focal Loss · Layer-wise LR Decay · 3 Architecture Comparisons</i>
</p>

<p align="center">
  <a href="#-dataset--preprocessing">Dataset</a> •
  <a href="#-model-architecture">Architecture</a> •
  <a href="#-loss-function">Loss</a> •
  <a href="#-optimisation">Optimisation</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-reproducibility">Reproducibility</a> •
  <a href="#-experimental-setup">Setup</a>
</p>

---

## 📋 Overview

This project fine-tunes **DINOv2-Small (ViT-S/14)** — a self-supervised Vision Transformer with 22M parameters — for binary classification of dermoscopic skin lesion images into **benign** and **melanoma** classes. The pipeline addresses extreme class imbalance (~1.76% melanoma) through a multi-layered strategy combining weighted sampling, focal loss, and a custom melanoma-prioritising evaluation metric.

---

## 📂 Dataset & Preprocessing

**Source:** [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) (Kaggle)

| Split | Samples | Melanoma (Class 1) | Ratio |
|-------|---------|---------------------|-------|
| Train | 80% (4 folds) | ~1.76% | ~57:1 |
| Validation | 20% (1 fold) | ~1.76% | ~57:1 |

Only image data and target labels are used — all metadata columns (patient ID, sex, age, anatomical site, diagnosis) are excluded to build a purely image-based classification framework.

**Data Splitting:** Stratified 5-Fold Cross-Validation (`StratifiedKFold`, seed=42) preserves the class distribution across all folds.

### Data Augmentation

Training images pass through an extensive augmentation pipeline designed to simulate real-world dermoscopic variability:

| Category | Transformations |
|----------|----------------|
| **Geometric** | Random resized crop (scale 0.75–1.0), horizontal/vertical flip, 90° rotation, affine (translate ±5%, scale 0.9–1.1×, rotate ±45°) |
| **Skin Tone & Lighting** | Colour jitter (brightness, contrast, saturation, hue), HSV shifts, random gamma (γ ∈ [80, 120]) |
| **Dermoscope Artefacts** | Random brightness-contrast, CLAHE (clip=2.0, 8×8 grid) |
| **Noise & Blur** | Gaussian noise, Gaussian blur, motion blur, median blur |
| **Occlusion Simulation** | CoarseDropout (4–8 holes, 8–24 px) to simulate hair/ruler markings |

Validation images receive **only** resizing and normalisation. Both splits are normalised using the official DINOv2 ImageNet statistics:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Handling Class Imbalance at the Data Level

A `WeightedRandomSampler` oversamples the minority class during training. Melanoma samples receive a weight equal to the negative-to-positive ratio, while benign samples receive a weight of 1.0 — ensuring roughly balanced batches despite the 57:1 imbalance.

---

## 🧠 Model Architecture

### Backbone: DINOv2-Small (ViT-S/14)

```
Input (224×224 RGB) → 14×14 Patches → [CLS] + 4 Register + 196 Patch Tokens → 12 Encoder Layers
                                         ↓
                                   384-dim features
```

| Property | Value |
|----------|-------|
| Parameters | 22M |
| Patch Size | 14×14 |
| Hidden Dim | 384 |
| Encoder Layers | 12 |
| Register Tokens | 4 |
| Pretraining | Self-distillation (no labels) |

### Feature Extraction Strategy

The **"cls_only"** strategy is used: only the 384-dimensional `[CLS]` token representation is passed to the classification head.

> Two alternative strategies are defined but unused: `cls_patch_mean` (768-dim) and `cls_patch_mean_register` (1152-dim).

### Classification Head

```
[CLS] Token (384)
    │
    ▼
LayerNorm(384)
    │
    ▼
Linear(384 → 512) → GELU → Dropout(0.3)
    │
    ▼
Linear(512 → 256) → GELU → Dropout(0.3)
    │
    ▼
Linear(256 → 1) → Sigmoid
    │
    ▼
Binary Prediction (Benign / Melanoma)
```

### Fine-Tuning Architectures

Three layer-freezing strategies are compared to determine the optimal depth of backbone adaptation:

| Architecture | Frozen Layers | Trainable Layers | Trainable Params |
|-------------|---------------|-------------------|-----------------|
| **Arch 01** | Patch embeddings only | Encoder 0–11, LayerNorm, Head | ~96.6% |
| **Arch 02** | Patch embeddings + Encoder 0–4 | Encoder 5–11, LayerNorm, Head | ~57.0% |
| **Arch 03** | Patch embeddings + Encoder 8–11 | Encoder 0–7, LayerNorm, Head | ~64.9% |

---

## ⚖️ Loss Function

**Focal Loss** addresses class imbalance at the loss level:

```
FL(pₜ) = −αₜ · (1 − pₜ)ᵞ · log(pₜ)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `α` | 0.9 | Melanoma receives 90% class-balancing weight, benign 10% |
| `γ` | 2.0 | Down-weights easy examples, focuses on hard/ambiguous cases |
| `reduction` | mean | Average loss across the batch |

> An alternative weighted BCE loss (`BCEWithLogitsLoss` with `pos_weight` from class weights `[1.0, 10.0]`) is implemented but Focal Loss is used for all experiments due to better performance under extreme imbalance.

---

## ⚙️ Optimisation

### Optimiser: AdamW with LLRD

**Layer-wise Learning Rate Decay (LLRD)** assigns different learning rates at different network depths:

```
Classification Head     → 1×10⁻⁵  (highest)
Final LayerNorm         → 1×10⁻⁵
Encoder Layer 11        → 1×10⁻⁵
Encoder Layer 10        → 1×10⁻⁵ × 0.9¹
Encoder Layer 9         → 1×10⁻⁵ × 0.9²
        ⋮                      ⋮
Encoder Layer 0         → 1×10⁻⁵ × 0.9¹¹ (lowest)
```

| Parameter | Value |
|-----------|-------|
| Weight Decay | 0.01 |
| Head LR | 1×10⁻⁵ |
| Backbone LR | 1×10⁻⁵ |
| LLRD Decay Factor | 0.9 |

### Learning Rate Schedule

A two-phase schedule via `SequentialLR` with **per-step** (not per-epoch) updates:

| Phase | Epochs | Strategy |
|-------|--------|----------|
| **Warmup** | 1–2 | Linear warmup from 10% → 100% of base LR |
| **Cosine Annealing** | 3–30 | Cosine decay to min LR of 7×10⁻⁶ |

### Mixed Precision Training

Automatic Mixed Precision (AMP) via `torch.cuda.amp` with `GradScaler` casts forward passes to float16 where safe. Gradient clipping (max norm = 1.0) is applied after unscaling to prevent training instability.

---

## 📊 Evaluation

### Metrics

Models are evaluated on the validation set after each epoch:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Macro F1 / Precision / Recall | Class-balanced performance |
| Per-class Recall | Separate recall for benign (class 0) and melanoma (class 1) |
| Recall Gap | Class 1 recall − Class 0 recall |
| ROC-AUC | Discrimination ability (probability-based) |
| PR-AUC | Precision-recall trade-off (probability-based) |
| Confusion Matrix | TP, TN, FP, FN distribution |

Decision threshold: **0.5** (applied to sigmoid probabilities).

### Selection Score (Model Monitoring)

A custom metric prioritises melanoma detection while penalising excessive recall imbalance:

```
If Recall_class1 ≤ Recall_class0  →  Score = -1.0  (failure)
Otherwise                         →  Score = Recall_class1 − 0.5 × (Recall_class1 − Recall_class0)
```

This ensures selected models achieve high melanoma sensitivity without completely sacrificing benign classification.

### Early Stopping & Checkpointing

| Component | Configuration |
|-----------|--------------|
| **Early Stopping** | Patience = 20 epochs, min Δ = 1×10⁻⁴, mode = max |
| **Checkpointing** | Saves model state, optimiser state, scheduler state, epoch, best score |
| **Checkpoint Paths** | Independent per architecture |

---

## 🔁 Reproducibility

All random seeds are fixed to **42** across every source of randomness:

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

A dedicated `torch.Generator` controls the `WeightedRandomSampler` index order, and a worker init function seeds each `DataLoader` worker process.

---

## 🖥️ Experimental Setup

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Max Epochs | 30 |
| Batch Size | 32 |
| DataLoader Workers | 4 |
| Architectures Compared | 3 |

All three architectures are trained independently with **identical** hyperparameters, data splits, and augmentation pipelines — differing only in their layer-freezing strategy. Results are compared via per-architecture training curves and cross-architecture comparison plots.

---

## 📁 Project Structure

```
├── fine-tune-dinov-melanoma-classification-img-only.ipynb   # Main notebook
├── arch01_checkpoint/
│   └── best_dinov2s_melanoma.pth                            # Best model — Arch 01
├── arch02_checkpoint/
│   └── best_dinov2s_melanoma.pth                            # Best model — Arch 02
├── arch03_checkpoint/
│   └── best_dinov2s_melanoma.pth                            # Best model — Arch 03
└── README.md
```

---

## 🛠️ Requirements

```
torch
torchvision
transformers
albumentations
huggingface_hub
opencv-python
scikit-learn
numpy
pandas
seaborn
matplotlib
tqdm
```

---

<p align="center">
  <i>Built for the SIIM-ISIC Melanoma Classification Challenge</i>
</p>
