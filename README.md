# Rotten Fruit Classification with Transfer Learning 🍎🍌🍊

This project applies **transfer learning with a pretrained VGG16 model** to classify fruit images as **fresh or rotten** across multiple categories. The goal is to demonstrate how a strong pretrained vision backbone can achieve high accuracy on a relatively small, real-world dataset through careful fine-tuning and data augmentation.

<p align="center">
  <img src="image/fruits.png" alt="Fresh vs Rotten Fruit Examples" width="700"/>
</p>

---

## Overview
This project implements a **multi-class image classification pipeline** to distinguish between fresh and rotten fruits using **transfer learning** with a pretrained **VGG16** convolutional neural network.

The goal is to demonstrate how pretrained ImageNet models can be adapted to small, domain-specific datasets using:
- Feature extraction
- Data augmentation
- Fine-tuning
- GPU-accelerated training (CUDA)

The final model achieves **>95% validation accuracy**, exceeding the target performance threshold.

---

## Problem Statement
Accurately identifying food freshness is a practical computer vision problem with applications in:
- Food quality control
- Supply chain automation
- Waste reduction systems

This project focuses on classifying images into six categories:
- Fresh apples
- Fresh bananas
- Fresh oranges
- Rotten apples
- Rotten bananas
- Rotten oranges

---

## Dataset
The dataset used in this project was provided as part of the **NVIDIA Deep Learning Institute (DLI)** course environment.

**Classes (6 total):**
- `freshapples`
- `freshbanana`
- `freshoranges`
- `rottenapples`
- `rottenbanana`
- `rottenoranges`

Due to licensing and GitHub size limitations, **raw image files are not included** in this repository.

### Dataset Structure
```text
data/
└── fruits/
    ├── train/
    │   ├── freshapples/
    │   ├── freshbanana/
    │   ├── freshoranges/
    │   ├── rottenapples/
    │   ├── rottenbanana/
    │   └── rottenoranges/
    └── valid/
        └── (same structure as train)
```
## 🔄 Data Augmentation

To improve generalization without distorting color cues critical to fruit freshness, the following augmentations were applied:

- Random resized crops
- Horizontal flips
- Mild rotations
- Normalization using ImageNet statistics

(Color jitter was intentionally avoided.)

---

## ⚙️ Training Results

- **Validation Accuracy:** ~96%
- **Training Stability:** Consistent convergence
- **Overfitting:** Minimal due to frozen base + augmentation

Fine-tuning the unfrozen backbone with a very low learning rate further improved performance.

---

## 🚀 Hardware & Performance

- Training was performed using **GPU acceleration (CUDA)**.
- The code automatically falls back to CPU if a GPU is unavailable.
- GPU is **recommended but not required** to run inference or inspect the notebook.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
