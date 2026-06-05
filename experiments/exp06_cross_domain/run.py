"""
Exp 06 — Cross-domain zero-shot evaluation (Week 2).
Ghana-trained models evaluated on the Kaggle IRT-PV European dataset.
No retraining — pure zero-shot transfer.
Addresses reviewer objection: no external validation, single site.

BEFORE RUNNING:
  1. Download from: kaggle.com/datasets/marcosgabriel/photovoltaic-system-thermography
  2. Unzip to: ~/research/sdcnn-pv-thermography/external/kaggle_irt_pv/
     Expected structure: kaggle_irt_pv/<ClassName>/*.jpg
  3. Then run: python experiments/exp06_cross_domain/run.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.cross_domain import run_cross_domain

if __name__ == "__main__":
    run_cross_domain()
    print("\nExp 06 complete.")
