"""
Generate all paper figures from CV results + saved model weights.

Run after all experiments (exp01–exp06) are complete:
  python generate_figures.py

Produces (in results/figures/):
  Fig 5-8:  cm_{task}_{model}.png            — confusion matrices (best fold)
  Fig 9:    gradcam_grid.png                 — Grad-CAM overlays
  Fig 10:   cv_boxplot_{task}.png            — F1 boxplots (all models)
  Fig 11:   learning_curves (from exp05 output)
  Fig 12:   cross_domain_cm.png              — Pierdicca zero-shot CM
  Extra:    summary_bar.png                  — bar chart mean±std F1 both tasks
             radar_{task}.png               — radar chart (P/R/F1/Acc)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import RESULTS_DIR
from src.viz.plots import (plot_cv_boxplot, plot_confusion_matrix,
                            print_summary_table, ARCH_LABELS, PALETTE)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIGS = RESULTS_DIR / "figures"
TABLES = RESULTS_DIR / "tables"
FIGS.mkdir(parents=True, exist_ok=True)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion matrices (Figs 5-8)
# ─────────────────────────────────────────────────────────────────────────────

def gen_confusion_matrices():
    print("\n[1] Confusion matrices")
    for task in ("detection", "diagnosis"):
        for model in ARCH_LABELS.keys():
            plot_confusion_matrix(task, model)


# ─────────────────────────────────────────────────────────────────────────────
# 2. F1 boxplots (Fig 10)
# ─────────────────────────────────────────────────────────────────────────────

def gen_boxplots():
    print("\n[2] F1 boxplots")
    for task in ("detection", "diagnosis"):
        try:
            plot_cv_boxplot(task)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Summary bar chart (mean ± std, all models × both tasks)
# ─────────────────────────────────────────────────────────────────────────────

def gen_summary_bar():
    print("\n[3] Summary bar chart")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, task in zip(axes, ("detection", "diagnosis")):
        rows = []
        for model in ARCH_LABELS.keys():
            csv = TABLES / f"cv_{task}_{model}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            rows.append({
                "label": ARCH_LABELS[model],
                "mean": df["f1"].mean(),
                "std": df["f1"].std(),
            })
        if not rows:
            continue

        rdf = pd.DataFrame(rows)
        colors = PALETTE[:len(rdf)]
        bars = ax.bar(rdf["label"], rdf["mean"], yerr=rdf["std"],
                      capsize=5, color=colors, alpha=0.85,
                      error_kw={"linewidth": 1.5, "ecolor": "black"})
        for bar, (_, row) in zip(bars, rdf.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + row["std"] + 0.008,
                    f"{row['mean']:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean F1 (5-fold CV)")
        ax.set_title(f"{task.capitalize()} Task", fontweight="bold")
        ax.set_ylim(max(0, rdf["mean"].min() - 0.15), 1.05)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model Comparison — 5-Fold Cross-Validation F1",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    save(fig, FIGS / "summary_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Radar chart (precision, recall, F1, accuracy — mean over folds)
# ─────────────────────────────────────────────────────────────────────────────

def gen_radar(task: str):
    print(f"\n[4] Radar chart — {task}")
    metrics = ["precision", "recall", "f1", "accuracy"]
    labels = ["Precision", "Recall", "F1", "Accuracy"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

    plotted = 0
    for i, model in enumerate(ARCH_LABELS.keys()):
        csv = TABLES / f"cv_{task}_{model}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        vals = [df[m].mean() for m in metrics] + [df[metrics[0]].mean()]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2,
                label=ARCH_LABELS[model])
        ax.fill(angles, vals, color=color, alpha=0.1)
        plotted += 1

    if plotted == 0:
        print(f"  [skip] No CSVs found for {task}")
        plt.close(fig)
        return

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title(f"{task.capitalize()} Task — Metrics Radar",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    save(fig, FIGS / f"radar_{task}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def gen_summary_tables():
    print("\n[5] Summary tables")
    for task in ("detection", "diagnosis"):
        try:
            print_summary_table(task)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Grad-CAM (requires best-fold SDCNN detection weights)
# ─────────────────────────────────────────────────────────────────────────────

def gen_gradcam():
    print("\n[6] Grad-CAM grid")
    out_path = FIGS / "gradcam_grid.png"
    if out_path.exists():
        print(f"  Already exists → {out_path}")
        return
    model_path = RESULTS_DIR / "logs" / "best_detection_sdcnn.keras"
    if not model_path.exists():
        print(f"  [skip] {model_path} not found — run exp01 first")
        return
    try:
        import tensorflow as tf
        from src.evaluation.gradcam import generate_gradcam_figure
        from src.data.loaders import load_image_paths_and_labels, make_dataset
        model = tf.keras.models.load_model(model_path)
        paths, labels = load_image_paths_and_labels("detection")
        ds = make_dataset(paths, labels, augment=False, training=False)
        y_prob = model.predict(ds, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        generate_gradcam_figure(
            model=model, val_paths=paths, y_true=labels, y_pred=y_pred,
            last_conv_name="conv3", save_path=out_path,
        )
    except Exception as e:
        print(f"  [error] Grad-CAM failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Generating all paper figures")
    print("=" * 60)

    gen_summary_tables()
    gen_confusion_matrices()
    gen_boxplots()
    gen_summary_bar()
    for task in ("detection", "diagnosis"):
        gen_radar(task)
    gen_gradcam()

    print(f"\nAll figures written to: {FIGS}")
    print("Done.")
