"""
Generate all publication-ready figures from completed experiment results.
Run after exp01–exp05 are complete.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.viz.plots import (
    plot_cv_boxplot,
    plot_confusion_matrix,
    print_summary_table,
)

if __name__ == "__main__":
    print("Generating figures...\n")

    # Fig 10: F1 boxplots
    for task in ("detection", "diagnosis"):
        try:
            plot_cv_boxplot(task)
        except FileNotFoundError as e:
            print(f"  [skip boxplot {task}] {e}")

    # Figs 5–8: Best-fold confusion matrices
    for task in ("detection", "diagnosis"):
        for arch in ("sdcnn", "vgg16", "mobilenetv2", "efficientnet_b0"):
            plot_confusion_matrix(task, arch, save=True)

    # Summary tables to console + CSV
    for task in ("detection", "diagnosis"):
        try:
            print_summary_table(task)
        except FileNotFoundError as e:
            print(f"  [skip summary {task}] {e}")

    print("\nAll figures generated in results/figures/")
