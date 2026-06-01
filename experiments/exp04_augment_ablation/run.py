"""
Exp 04 — Data augmentation ablation.
SDCNN trained once with augmentation (from exp01) and once without.
Reports the F1 delta to demonstrate methodological awareness.
Addresses reviewer objection: limited methodological contribution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.training.cv_runner import run_cv
from src.models.sdcnn import build_sdcnn
from src.config import RESULTS_DIR

if __name__ == "__main__":
    tables = RESULTS_DIR / "tables"

    for task in ("detection", "diagnosis"):
        run_cv(task=task, model_name="sdcnn_noaug",
               build_fn=build_sdcnn, augment_override=False)

    # Print delta
    print("\n=== Augmentation Ablation Summary ===")
    for task in ("detection", "diagnosis"):
        aug_csv   = tables / f"cv_{task}_sdcnn.csv"
        noaug_csv = tables / f"cv_{task}_sdcnn_noaug.csv"
        if aug_csv.exists() and noaug_csv.exists():
            aug_f1   = pd.read_csv(aug_csv)["f1"].mean()
            noaug_f1 = pd.read_csv(noaug_csv)["f1"].mean()
            print(f"  {task:10s}  With aug: {aug_f1:.4f}  "
                  f"Without: {noaug_f1:.4f}  Delta: {aug_f1 - noaug_f1:+.4f}")

    print("\nExp 04 complete.")
