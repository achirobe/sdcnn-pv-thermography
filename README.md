# PV-IRT-EAAI

**SDCNN-VGG16 Infrared Thermography PV Fault Detection**  
Resubmission to *Engineering Applications of Artificial Intelligence* (Elsevier Q1)

**Authors:** Reamy W. Achirobe, Lena Dzifa Mensah, Muyiwa S. Adaramola

---

## Overview

This repository contains the full experimental code for the EAAI resubmission.
It addresses 10 reviewer objections from the prior rejection via:

- 5-fold stratified cross-validation across 4 architectures (SDCNN, VGG-16, MobileNetV2, EfficientNet-B0)
- Statistical significance testing (paired t-test, Wilcoxon, Cohen's d)
- Grad-CAM interpretability heatmaps
- Data augmentation ablation
- Zero-shot cross-domain evaluation on external IRT-PV dataset

## Setup

```bash
git clone https://github.com/achirobe/pv-irt-eaai.git
cd pv-irt-eaai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Link your datasets:
```bash
mkdir -p data
ln -s /path/to/BC_dataset data/detection
ln -s /path/to/MC_dataset data/diagnosis
```

## Run

```bash
# Week 1 — all experiments
bash run_all.sh

# Week 2 — cross-domain (download Kaggle dataset first)
python experiments/exp06_cross_domain/run.py
```

## Hardware

Apple MacBook Air M1, 8 cores, 16 GB RAM, tensorflow-metal GPU acceleration.

## Results

All CSV tables and figures are saved to `results/` after running experiments.
