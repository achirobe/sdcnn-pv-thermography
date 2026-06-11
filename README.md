# SDCNN — PV Fault Detection from Infrared Thermography

> **"SDCNN: A Shallow Dense Convolutional Neural Network for Automated Fault Detection
> and Diagnosis in Photovoltaic Systems via Infrared Thermography — A Case Study from
> Navrongo, Ghana"**

This repository contains the complete code and per-fold results for an automated
photovoltaic (PV) fault-detection study using handheld infrared thermography (IRT)
images collected at the Navrongo VRA Solar Plant in Ghana's Upper East Region.

**This work is submitted in partial fulfilment of the requirements for the degree of
Master of Philosophy (MPhil) in Mechanical Engineering, Thermofluids and Energy Systems
option, at the Kwame Nkrumah University of Science and Technology (KNUST).**

**Author and sole contributor:** Reamy Womodowe Achirobe

---

## Overview

Code and results for automated PV fault detection from handheld-acquired infrared
thermography (IRT) images.

**Dataset:** The Ghana IRT-PV dataset is hosted in a separate private repository.
Access requires owner approval — request via
[github.com/achirobe/ghana-irt-pv-dataset](https://github.com/achirobe/ghana-irt-pv-dataset)
or email rachirobe@gmail.com.

**Two classification tasks:**
| Task | Images | Classes | Balance |
|------|--------|---------|---------|
| Detection | 228 | Faulty / Non-faulty | 50/50 |
| Diagnosis | 86 | Block / PatchWork | 50/50 |

**Four models compared:**
- **SDCNN** (proposed) — custom 3-block CNN, 11.2M params, trained from scratch
- **VGG-16** — frozen ImageNet backbone + Flatten head
- **MobileNetV2** — frozen ImageNet backbone + GAP head
- **EfficientNet-B0** — frozen ImageNet backbone + GAP head

A Global-Average-Pooling variant of SDCNN (0.11M params) is provided as an
efficiency ablation for memory-constrained edge hardware.

---

## Key Results (5-Fold Stratified CV)

### Fault Detection
| Model | F1 | Precision | Recall | Accuracy |
|-------|----|-----------|--------|----------|
| SDCNN (proposed) | 0.884 ± 0.064 | 0.931 ± 0.053 | 0.842 ± 0.073 | 0.890 ± 0.057 |
| VGG-16 | **0.943 ± 0.056** | **0.947 ± 0.044** | **0.939 ± 0.059** | **0.943 ± 0.056** |
| MobileNetV2 | 0.898 ± 0.065 | 0.942 ± 0.041 | 0.860 ± 0.094 | 0.903 ± 0.059 |
| EfficientNet-B0 | 0.909 ± 0.035 | 0.944 ± 0.039 | 0.878 ± 0.047 | 0.912 ± 0.035 |
| SDCNN-GAP (0.11M) | 0.839 ± 0.044 | 0.979 ± 0.043 | 0.737 ± 0.066 | 0.858 ± 0.039 |

### Fault Diagnosis
| Model | F1 | Precision | Recall | Accuracy |
|-------|----|-----------|--------|----------|
| SDCNN (proposed) | **1.000 ± 0.000** | **1.000** | **1.000** | **1.000** |
| VGG-16 | 0.989 ± 0.024 | 0.980 ± 0.045 | 1.000 ± 0.000 | 0.989 ± 0.025 |
| MobileNetV2 | 0.989 ± 0.024 | 0.980 ± 0.045 | 1.000 ± 0.000 | 0.989 ± 0.025 |
| EfficientNet-B0 | 0.989 ± 0.024 | 0.980 ± 0.045 | 1.000 ± 0.000 | 0.989 ± 0.025 |

No pairwise difference on detection is statistically significant at α = 0.05
(five paired folds; paired t-test and Wilcoxon). Zero-shot cross-domain transfer
to the Pierdicca Italian IRT-PV dataset is a representational transfer failure
(ROC-AUC ≈ 0.59) not recoverable by recalibration.

---

## Repo Structure

```
sdcnn-pv-thermography/
├── src/
│   ├── config.py                    # all paths + hyperparameters (SEED=42)
│   ├── utils/seeds.py               # set_seeds(42) for reproducibility
│   ├── data/loaders.py              # load_image_paths_and_labels, make_dataset
│   ├── models/
│   │   ├── sdcnn.py                 # proposed SDCNN
│   │   ├── vgg16_tl.py              # VGG-16 transfer learning
│   │   ├── mobilenetv2_tl.py        # MobileNetV2 transfer learning
│   │   └── efficientnet_tl.py       # EfficientNet-B0 transfer learning
│   ├── training/cv_runner.py        # 5-fold stratified CV orchestrator
│   └── evaluation/
│       ├── stats.py                 # paired t-test, Wilcoxon, Cohen's d, bootstrap CI
│       ├── gradcam.py               # Grad-CAM via tf.GradientTape (Keras 3 compatible)
│       └── cross_domain.py          # zero-shot Pierdicca IRT-PV evaluation
├── experiments/
│   ├── exp01_cv_baseline/           # SDCNN + VGG-16 5-fold CV
│   ├── exp02_modern_baselines/      # MobileNetV2 + EfficientNet-B0
│   ├── exp03_significance/          # statistical significance testing
│   ├── exp04_augment_ablation/      # with/without augmentation
│   ├── exp05_gradcam/               # Grad-CAM explainability maps
│   ├── exp06_cross_domain/          # zero-shot Pierdicca evaluation
│   ├── exp07_hyperparam_search/     # LR×Dropout grid
│   ├── exp08_ablation_extended/     # depth / dropout / fine-tuning / resolution
│   ├── exp09_sdcnn_gap/             # SDCNN Global-Average-Pooling head ablation
│   └── exp10_crossdomain_calibration/  # Platt recalibration of cross-domain scores
├── results/
│   ├── tables/                      # per-fold CSV results (committed to git)
│   └── logs/                        # .keras weights (git-ignored — too large)
└── run_all.sh                       # sequential experiment runner
```

---

## Reproducing the Experiments

### Requirements

```
Python >= 3.12
TensorFlow 2.18.0
tensorflow-metal (Apple Silicon GPU)
```

```bash
git clone https://github.com/achirobe/sdcnn-pv-thermography.git
cd sdcnn-pv-thermography
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

```
data/detection/faulty/      # 114 IRT images of faulty panels
data/detection/non-faulty/  # 114 IRT images of healthy panels
data/diagnosis/Block/       # 43 images — block-type fault pattern
data/diagnosis/PatchWork/   # 43 images — patchwork-type fault pattern
```

### Running All Experiments

```bash
bash run_all.sh
# Or individually:
python experiments/exp01_cv_baseline/run.py
python experiments/exp02_modern_baselines/run.py
python experiments/exp03_significance/run.py
python experiments/exp04_augment_ablation/run.py
python experiments/exp05_gradcam/run.py
python experiments/exp06_cross_domain/run.py
python experiments/exp08_ablation_extended/run.py
python experiments/exp09_sdcnn_gap/run.py
python experiments/exp10_crossdomain_calibration/run.py
```

Each experiment writes its per-fold metrics to `results/tables/` as CSV files.

---

## Cross-Domain Dataset (Pierdicca IRT-PV)

For zero-shot cross-domain evaluation, download the Pierdicca dataset
([Pierdicca et al. 2020](https://doi.org/10.3390/en13246496)) and place it at:

```
external/kaggle_irt_pv/kaggle_dataset/
  dataset_1/
    images/       *.jpg
    annotations/  *.json  {"instances": [{"defected_module": bool}]}
  dataset_2/
    images/
    annotations/
```

---

## Reproducibility

| Setting | Value |
|---------|-------|
| Global seed | 42 |
| Per-fold seed | `42 + fold_number` |
| TF op determinism | `enable_op_determinism()` |
| CV strategy | `StratifiedKFold(n_splits=5, shuffle=True)` |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Dropout | 0.5 |
| Input size | 224×224 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience=10, monitor=val_loss |

Deduplication was applied at the physical-panel level, so no panel appears in both
the training and validation partitions of any fold.

---

## License

MIT
