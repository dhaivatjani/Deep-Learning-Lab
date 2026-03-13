# IT549: Deep Learning – Lab 3
## Image-Based AQI Classification using CNN and MobileNetV2

**Name:** Dhaivat Jani <br>
**Student ID:** 202511053 <br>
**Course:** IT549 – Deep Learning <br>

---

## Objective

Predict **Air Quality Index (AQI) class** from location photographs using two deep learning approaches and compare their performance:

1. **Basic CNN** – trained from scratch
2. **MobileNetV2** – transfer learning from ImageNet pretrained weights

---

## Dataset

- `data.csv` – maps image filenames to AQI class labels
- `sampled_images/` – folder containing all location images
- Only two columns are used: `Filename` (image path) and `AQI_Class` (target label)

**6 AQI Classes:**

| Label | Class |
|---|---|
| a_Good | Good |
| b_Moderate | Moderate |
| c_Unhealthy_for_Sensitive_Groups | Unhealthy for Sensitive Groups |
| d_Unhealthy | Unhealthy |
| e_Very_Unhealthy | Very Unhealthy |
| f_Severe | Severe |

---

## Project Structure

```
Lab03/
├── Lab03.ipynb          # Main notebook (all tasks)
├── best_basic_cnn.pth   # Best CNN checkpoint (saved during training)
├── best_mobilenet.pth   # Best MobileNetV2 checkpoint (saved during training)
└── README.md
```
---

## Tasks Overview

### Task 1 – Data Preparation
- Loaded `data.csv` and kept only `Filename` and `AQI_Class`
- Computed dataset mean and std from the training set for normalization:
  - Mean: `[0.543, 0.578, 0.601]`
  - Std: `[0.198, 0.197, 0.210]`
- Applied augmentation on training set: random horizontal flip, rotation (±10°), colour jitter
- Stratified 70 / 15 / 15 train / val / test split using `SEED=42`

### Task 2 – Basic CNN (From Scratch)
3-block CNN with a single Conv2d per block:

```
Input (3×224×224)
→ Conv Block 1: 32 filters  → MaxPool → 112×112
→ Conv Block 2: 64 filters  → MaxPool → 56×56
→ Conv Block 3: 128 filters → MaxPool → 28×28
→ AdaptiveAvgPool(4×4)
→ Linear(2048 → 256) → ReLU → Dropout(0.5) → Linear(256 → 6)
```

Each conv block: `Conv2d (bias=False) → BatchNorm2d → ReLU → MaxPool2d → Dropout2d(0.25)`

Trained with Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau for 20 epochs.

### Task 3 – MobileNetV2 (Transfer Learning)
- Loaded MobileNetV2 pretrained on ImageNet (`IMAGENET1K_V1`)
- Replaced classifier head: `Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→6)`
- **Phase 1** – backbone frozen, only the new head trained (20 epochs, lr=1e-3)
- **Phase 2** – last 3 backbone blocks unfrozen and fine-tuned alongside the head (15 epochs, lr=1e-4)

### Task 4 – Evaluation
Evaluated both models on the held-out test set using accuracy, precision, recall, F1-score (weighted), and per-class confusion matrices.

### Task 5 – Training Curves
Plotted epoch vs. train/val loss and accuracy for both models, plus a side-by-side validation accuracy comparison.

**Why transfer learning helps:** MobileNetV2's backbone already encodes rich low-level features (edges, textures, shapes) from ImageNet. Rather than learning everything from random weights, only the head needs to map existing features to AQI classes — converging faster with less risk of overfitting on a moderate-sized dataset.

### Task 6 – Misclassification Analysis
Collected 10 misclassified test images per model and visualised them with actual vs predicted labels.

**Common reasons for misclassification:**
- Adjacent AQI classes (e.g. Moderate vs Unhealthy for Sensitive Groups) differ only in subtle haze density
- Lighting conditions at dawn/dusk mimic atmospheric effects unrelated to actual pollution
- Foreground objects (buildings, trees) can dominate the frame and hide the sky signal
- Camera settings (contrast, saturation) vary across devices without reflecting true AQI

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Basic CNN (Scratch) | 0.679 | 0.674 | 0.679 | 0.669 |
| MobileNetV2 (Transfer) | 0.951 | 0.951 | 0.951 | 0.951 |

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| CNN epochs | 20 |
| MobileNet Phase 1 epochs | 20 |
| MobileNet Phase 2 epochs | 15 |
| Learning rate | 1e-3 |
| Phase 2 LR | 1e-4 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Random seed | 42 |

## Conclusion

This project demonstrates the effectiveness of transfer learning for image classification tasks with limited data.

**Key findings:**
- CNN trained from scratch achieved moderate performance.
- MobileNetV2 significantly improved classification accuracy.
- Transfer learning allows models to leverage pretrained visual features, resulting in faster convergence and higher accuracy.
