"""
Exp 02 — 5-fold CV: MobileNetV2 and EfficientNet-B0 on both tasks.
Addresses reviewer objection: VGG-16 is the only baseline (10+ years old).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.cv_runner import run_cv
from src.models.mobilenetv2_tl import build_mobilenetv2
from src.models.efficientnet_tl import build_efficientnet_b0

if __name__ == "__main__":
    for task in ("detection", "diagnosis"):
        run_cv(task=task, model_name="mobilenetv2",    build_fn=build_mobilenetv2)
        run_cv(task=task, model_name="efficientnet_b0", build_fn=build_efficientnet_b0)
    print("\nExp 02 complete.")
