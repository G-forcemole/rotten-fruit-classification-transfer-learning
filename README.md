# Transfer Learning for Fresh vs Rotten Fruit Classification

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
