#!/bin/bash
# End-to-end orchestration for all experiments.
# Usage: bash run_all.sh [--skip-done] [--no-ablation]
#
# --skip-done : skip experiments where output CSV already exists
# --no-ablation : skip the long exp08 extended ablation

set -e
PYTHON=".venv/bin/python"
cd "$(dirname "$0")"

SKIP_DONE=0
NO_ABLATION=0
for arg in "$@"; do
  case $arg in
    --skip-done)   SKIP_DONE=1 ;;
    --no-ablation) NO_ABLATION=1 ;;
  esac
done

run_exp() {
  local name="$1"; local script="$2"; local check_csv="$3"
  echo ""
  echo "════════════════════════════════════════════"
  echo " $name"
  echo "════════════════════════════════════════════"
  if [ "$SKIP_DONE" -eq 1 ] && [ -n "$check_csv" ] && [ -f "$check_csv" ]; then
    echo " [SKIPPED] $check_csv already exists"
    return
  fi
  $PYTHON "$script"
}

echo "════════════════════════════════════════════"
echo " PV-IRT Fault Detection Pipeline"
echo "════════════════════════════════════════════"

run_exp "Exp01: 5-fold CV — SDCNN + VGG-16" \
  experiments/exp01_cv_baseline/run.py \
  results/tables/cv_detection_sdcnn.csv

run_exp "Exp02: 5-fold CV — MobileNetV2 + EfficientNet-B0" \
  experiments/exp02_modern_baselines/run.py \
  results/tables/cv_detection_efficientnet_b0.csv

run_exp "Exp03: Statistical significance tests" \
  experiments/exp03_significance/run.py \
  results/tables/stats_detection.csv

run_exp "Exp04: Augmentation ablation (SDCNN with/without aug)" \
  experiments/exp04_augment_ablation/run.py \
  results/tables/cv_detection_sdcnn_noaug.csv

run_exp "Exp05: Grad-CAM figures + confusion matrices" \
  experiments/exp05_gradcam/run.py \
  results/figures/gradcam_grid.png

run_exp "Exp06: Zero-shot cross-domain (Pierdicca IRT-PV)" \
  experiments/exp06_cross_domain/run.py \
  results/tables/cross_domain.csv

if [ "$NO_ABLATION" -eq 0 ]; then
  run_exp "Exp08: Extended ablation study (depth/dropout/finetune/resolution)" \
    experiments/exp08_ablation_extended/run.py \
    results/tables/ablation_all.csv
fi

run_exp "Exp09: SDCNN Global-Average-Pooling head ablation" \
  experiments/exp09_sdcnn_gap/run.py \
  results/tables/cv_detection_sdcnn_gap.csv

run_exp "Exp10: Cross-domain Platt recalibration analysis" \
  experiments/exp10_crossdomain_calibration/run.py \
  results/tables/cross_domain_calibration.csv

echo ""
echo "════════════════════════════════════════════"
echo " ALL EXPERIMENTS DONE"
echo " Per-fold result tables: results/tables/"
echo "════════════════════════════════════════════"
