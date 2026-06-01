#!/bin/bash
# End-to-end orchestration for Week 1 experiments.
# Run from project root: bash run_all.sh
# Week 2 (cross-domain) is separate — see experiments/exp06_cross_domain/run.py

set -e
PYTHON=".venv/bin/python"
cd "$(dirname "$0")"

echo "============================================"
echo " PV-IRT-EAAI — Week 1 Experiment Pipeline"
echo "============================================"

echo ""
echo "[1/5] Exp01: 5-fold CV — SDCNN + VGG16"
$PYTHON experiments/exp01_cv_baseline/run.py

echo ""
echo "[2/5] Exp02: 5-fold CV — MobileNetV2 + EfficientNet-B0"
$PYTHON experiments/exp02_modern_baselines/run.py

echo ""
echo "[3/5] Exp03: Statistical significance tests"
$PYTHON experiments/exp03_significance/run.py

echo ""
echo "[4/5] Exp04: Augmentation ablation"
$PYTHON experiments/exp04_augment_ablation/run.py

echo ""
echo "[5/5] Exp05: Grad-CAM figures + confusion matrices"
$PYTHON experiments/exp05_gradcam/run.py

echo ""
echo "Generating all publication figures..."
$PYTHON generate_figures.py

echo ""
echo "============================================"
echo " Week 1 complete. Results in results/"
echo " Run exp06 after downloading Kaggle dataset."
echo "============================================"
