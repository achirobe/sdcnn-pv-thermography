import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

from src.config import RESULTS_DIR

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ARCH_LABELS = {
    "sdcnn": "SDCNN",
    "vgg16": "VGG-16",
    "mobilenetv2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved → {path}")


def _load_cv(task: str, models=None):
    if models is None:
        models = list(ARCH_LABELS.keys())
    frames = []
    for m in models:
        p = RESULTS_DIR / "tables" / f"cv_{task}_{m}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No CV CSVs found for task={task}")
    df = pd.concat(frames, ignore_index=True)
    df["arch_label"] = df["model"].map(ARCH_LABELS).fillna(df["model"])
    return df


# ── Figure 10: F1 boxplots ────────────────────────────────────────────────────

def plot_cv_boxplot(task: str, save: bool = True):
    df = _load_cv(task)
    order = [ARCH_LABELS[m] for m in ARCH_LABELS if m in df["model"].unique()]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.boxplot(data=df, x="arch_label", y="f1", order=order,
                palette=PALETTE[:len(order)], width=0.5,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 7},
                medianprops={"color": "black", "linewidth": 2.5},
                boxprops={"linewidth": 1.5},
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5},
                ax=ax)
    sns.stripplot(data=df, x="arch_label", y="f1", order=order,
                  color=".25", size=4, alpha=0.5, ax=ax)

    # Annotate boxes that collapse to a line (Q1 == Q3) so the reader
    # knows the median IS there — the box just has zero height.
    for i, label in enumerate(order):
        vals = df.loc[df["arch_label"] == label, "f1"].values
        q1, q3 = np.percentile(vals, [25, 75])
        if np.isclose(q1, q3, atol=1e-6):
            ax.text(i, np.median(vals) + 0.012, "med=Q1=Q3",
                    ha="center", va="bottom", fontsize=7, color="black",
                    fontstyle="italic")

    ax.set_ylabel("F1 Score (5-fold CV)")
    ax.set_xlabel("")
    ax.set_title(f"{task.capitalize()} task — per-fold F1")
    ax.set_ylim(max(0, df["f1"].min() - 0.05), min(1.08, df["f1"].max() + 0.08))
    plt.tight_layout()

    if save:
        _save(fig, RESULTS_DIR / "figures" / f"cv_boxplot_{task}.png")
    return fig


# ── Figure 5-8: Confusion matrices ───────────────────────────────────────────

def plot_confusion_matrix(task: str, model_name: str,
                           fold: int = None, save: bool = True):
    """
    Load best-fold predictions and plot confusion matrix.
    If fold is None, uses the fold with highest F1.
    """
    tables = RESULTS_DIR / "tables"
    csv = tables / f"cv_{task}_{model_name}.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found")
        return

    df = pd.read_csv(csv)
    best_fold = int(df.loc[df["f1"].idxmax(), "fold"])
    use_fold = fold if fold is not None else best_fold

    preds_file = tables / f"preds_{task}_{model_name}_fold{use_fold}.npy"
    if not preds_file.exists():
        print(f"  [skip] {preds_file} not found")
        return

    data = np.load(preds_file)
    y_true = data[:, 0].astype(int)
    y_pred = data[:, 2].astype(int)

    if task == "detection":
        labels = ["Non-faulty", "Faulty"]
    else:
        labels = ["Block", "PatchWork"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 13})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{ARCH_LABELS.get(model_name, model_name)} — {task.capitalize()}"
                 f"\n(Fold {use_fold}, F1={df.loc[df['fold']==use_fold,'f1'].values[0]:.3f})")
    plt.tight_layout()

    if save:
        fig_name = f"cm_{task}_{model_name}.png"
        _save(fig, RESULTS_DIR / "figures" / fig_name)
    return fig


# ── Figure 11: Learning curves ────────────────────────────────────────────────

def plot_learning_curves(history_dict: dict, task: str, model_name: str,
                          fold: int, save: bool = True):
    """
    history_dict: {"loss": [...], "val_loss": [...],
                   "acc": [...], "val_acc": [...]}
    """
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, title in zip(axes,
                                  [("loss", "val_loss"), ("acc", "val_acc")],
                                  ["Loss", "Accuracy"]):
        ax.plot(epochs, history_dict[metric[0]], label="Train", color=PALETTE[0])
        ax.plot(epochs, history_dict[metric[1]], label="Val",   color=PALETTE[1],
                linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{title} — {ARCH_LABELS.get(model_name, model_name)}"
                     f" [{task}] fold {fold}")
        ax.legend()

    plt.tight_layout()
    if save:
        _save(fig, RESULTS_DIR / "figures" /
              f"learning_curves_{task}_{model_name}_fold{fold}.png")
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(task: str):
    df = _load_cv(task)
    summary = (df.groupby("model")
                 .agg(mean_F1=("f1", "mean"), std_F1=("f1", "std"),
                      mean_Prec=("precision", "mean"),
                      mean_Rec=("recall", "mean"),
                      mean_Acc=("accuracy", "mean"))
                 .reset_index())
    summary["F1 (mean±std)"] = (
        summary["mean_F1"].map("{:.4f}".format) + " ± " +
        summary["std_F1"].map("{:.4f}".format)
    )
    summary["model_label"] = summary["model"].map(ARCH_LABELS).fillna(summary["model"])
    print(f"\n=== {task.upper()} — 5-fold CV Summary ===")
    print(summary[["model_label", "F1 (mean±std)",
                   "mean_Prec", "mean_Rec", "mean_Acc"]].to_string(index=False))
    # save as CSV too
    out = RESULTS_DIR / "tables" / f"summary_{task}.csv"
    summary.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return summary
