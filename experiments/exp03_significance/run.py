"""
Exp 03 — Statistical significance testing.
Paired t-test + Wilcoxon signed-rank + Cohen's d on per-fold F1 scores.
Addresses reviewer objection: no statistical significance testing between models.
Requires exp01 and exp02 to be complete first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.stats import paired_tests
from src.viz.plots import print_summary_table

if __name__ == "__main__":
    for task in ("detection", "diagnosis"):
        print(f"\n{'='*50}")
        print_summary_table(task)
        paired_tests(task, baseline="sdcnn")
    print("\nExp 03 complete.")
