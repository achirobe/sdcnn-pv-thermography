# SDCNN-VGG16 IRT-PV — EAAI Submission

> **"SDCNN: A Shallow Deep Convolutional Neural Network for Automated Fault Detection
> and Diagnosis in Photovoltaic Systems via Infrared Thermography"**  
> Submitted to *Engineering Applications of Artificial Intelligence* (Elsevier Q1)

---

## Overview

Complete codebase and results for an EAAI journal paper on automated PV fault detection from UAV-acquired infrared thermography (IRT) images.

**Two classification tasks:**
| Task | Images | Classes | Balance |
|------|--------|---------|---------|
| Detection | 230 | Faulty / Non-faulty | 50/50 |
| Diagnosis | 88 | Block / PatchWork | 50/50 |

**Four models compared:**
- **SDCNN** (proposed) — custom 3-block CNN, 11.2M params, trained from scratch
- **VGG-16** — frozen ImageNet backbone + Flatten head
- **MobileNetV2** — frozen ImageNet backbone + GAP head
- **EfficientNet-B0** — frozen ImageNet backbone + GAP head

---

## Key Results (5-Fold Stratified CV)

### Fault Detection
| Model | F1 | Precision | Recall | Accuracy |
|-------|----|-----------|--------|----------|
| SDCNN (proposed) | 0.884 ± 0.064 | 0.931 ± 0.053 | 0.842 ± 0.073 | 0.890 ± 0.057 |
| VGG-16 | **0.943 ± 0.056** | **0.947 ± 0.044** | **0.939 ± 0.059** | **0.943 ± 0.056** |
| MobileNetV2 | 0.898 ± 0.065 | — | — | — |
| EfficientNet-B0 | *(pending exp02)* | — | — | — |

### Fault Diagnosis
| Model | F1 | Precision | Recall | Accuracy |
|-------|----|-----------|--------|----------|
| SDCNN (proposed) | **1.000 ± 0.000** | **1.000** | **1.000** | **1.000** |
| VGG-16 | 0.990 ± 0.024 | 0.980 ± 0.045 | 1.000 ± 0.000 | 0.989 ± 0.025 |

---

## Repo Structure

```
pv-irt-eaai/
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
│   ├── exp05_gradcam/               # Grad-CAM figure generation
│   ├── exp06_cross_domain/          # zero-shot Pierdicca evaluation
│   ├── exp07_hyperparam_search/     # LR×Dropout grid (conditional on baseline F1)
│   └── exp08_ablation_extended/     # depth / dropout / fine-tuning / resolution
├── results/
│   ├── tables/                      # CSV results (committed to git)
│   ├── figures/                     # PNG figures at 300 dpi (committed)
│   └── logs/                        # .keras weights (git-ignored — too large)
├── diagrams/
│   ├── methodology_flowchart.drawio # draw.io end-to-end pipeline
│   └── sdcnn_architecture.drawio    # draw.io SDCNN block diagram
├── manuscript/
│   ├── main.tex                     # EAAI LaTeX manuscript (elsarticle 5p)
│   ├── main_final.tex               # auto-filled version (via fill_results.py)
│   ├── references.bib               # BibTeX references
│   ├── compile.sh                   # one-command compilation
│   └── figures/                     # figures for LaTeX (auto-copied by compile.sh)
├── generate_figures.py              # generate all paper figures from results
├── fill_results.py                  # auto-fill LaTeX \todo{} with actual numbers
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
git clone https://github.com/achirobe/pv-irt-eaai.git
cd pv-irt-eaai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

```
data/detection/faulty/      # 115 IRT images of faulty panels
data/detection/non-faulty/  # 115 IRT images of healthy panels
data/diagnosis/Block/       # 44 images — block-type fault pattern
data/diagnosis/PatchWork/   # 44 images — patchwork-type fault pattern
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
```

### Generating Figures and Manuscript

```bash
python generate_figures.py       # generates all PNG figures
python fill_results.py           # fills \todo{} in LaTeX + prints Markdown summary
cd manuscript && bash compile.sh # copies figures + compiles PDF
```

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

---

## Citation

```bibtex
@article{achirobe2026sdcnn,
  author  = {Achirobe, Reamy Womodowe},
  title   = {{SDCNN}: A Shallow Deep Convolutional Neural Network for Automated
             Fault Detection and Diagnosis in Photovoltaic Systems via
             Infrared Thermography},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT
