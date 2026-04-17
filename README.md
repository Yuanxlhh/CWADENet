# CWADE-Net

Official implementation of **CWADE-Net: A Deep Learning Framework for Vegetation Invasion and Brick Spalling Defect Detection on Nanjing Ming City Wall**.

## Introduction

This repository provides the official implementation of **CWADE-Net**, a deep learning framework designed for automated defect detection on the Nanjing Ming Dynasty City Wall.

CWADE-Net focuses on detecting three representative surface defects:

- Herbaceous/woody vegetation invasion
- Brick spalling
- Vine-type vegetation invasion

To address challenging real-world conditions such as uneven illumination, complex backgrounds, blurred defect edges, and large variations in defect scale, CWADE-Net integrates:

- **SCI-Net** for self-calibrated illumination enhancement
- **EIE** for edge information encoding
- **C3k2-FSM** for spatial-frequency feature extraction
- A **bidirectional feature fusion neck**
- A lightweight **LSCSBD detection head**

The proposed framework achieves a strong balance between detection accuracy, robustness, and inference efficiency, and supports intelligent inspection and conservation of the Nanjing Ming Dynasty City Wall.

---

## Highlights

- A specialized deep learning framework for cultural heritage defect detection
- Robust against low illumination, complex textures, and scale variation
- Supports detection of vegetation invasion and brick spalling
- Lightweight design for efficient inference and deployment
- Generalizes to unseen UAV imagery and also shows promising performance on crack detection

---

## Dataset

The dataset was collected from the northern section of the Nanjing Ming City Wall, from **Xuanwu Gate** through **Jiefang Gate** and **Taiping Gate** to **Fugui Mountain**.

### Image acquisition devices

- **Nikon D300**
- **iPhone 15 Pro Max**
- **DJI Matrice 4E UAV**

### Defect categories

- Herbaceous/woody vegetation invasion
- Brick spalling
- Vine-type vegetation invasion

### Dataset statistics

- Total valid images: **3206**
- Train/val/test split: **8:1:1**

After augmentation, the training set contained:

- **3473** herbaceous/woody vegetation instances
- **11356** brick-spalling instances
- **3518** vine-type vegetation instances

### Annotation

All images were annotated with ground-truth bounding boxes using **LabelImg**.

> Note: The dataset generated and analyzed during the current study is publicly available at:  
> **https://github.com/Yuanxlhh/CWADENet**

---

## Method Overview

CWADE-Net consists of three main components:

### 1. Backbone
The backbone integrates:

- **SCI-Net**: enhances low-light and non-uniform illumination conditions
- **EIE**: extracts and fuses multi-scale edge information
- **C3k2-FSM**: jointly extracts spatial and frequency-domain features for better texture and edge representation

### 2. Neck
The neck uses a **bidirectional feature fusion strategy** with **CBAM attention** to improve semantic and local-detail interaction across scales.

### 3. Detection Head
A lightweight **LSCSBD** detection head is adopted to reduce model complexity and improve inference efficiency while maintaining detection accuracy.

---

## Environment

The experiments were conducted with the following environment:

- Ubuntu 20.04
- Python 3.8
- PyTorch 1.1
- CUDA 11.3
- NVIDIA GeForce RTX 3090

Install dependencies with:

```bash
pip install -r requirements.txt
