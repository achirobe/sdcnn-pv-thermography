"""
Exp 01 — 5-fold stratified CV: SDCNN and VGG-16 on both tasks.
Addresses reviewer objections: single split unreliable, 100% F1 suspect.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.cv_runner import run_cv
from src.models.sdcnn import build_sdcnn
from src.models.vgg16_tl import build_vgg16

if __name__ == "__main__":
    for task in ("detection", "diagnosis"):
        run_cv(task=task, model_name="sdcnn",  build_fn=build_sdcnn)
        run_cv(task=task, model_name="vgg16",  build_fn=build_vgg16)
    print("\nExp 01 complete.")
