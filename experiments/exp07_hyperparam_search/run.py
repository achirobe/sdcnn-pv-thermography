"""
Exp 07 — Hyperparameter grid search for SDCNN.
Grid: LR × Dropout, evaluated by 5-fold CV mean F1.
Only run this if exp01 results are below threshold (detection <0.85, diagnosis <0.80).

Produces:
  results/tables/hyperparam_grid_{task}.csv
  results/figures/hyperparam_heatmap_{task}.png   (LR vs Dropout F1 heatmap)
  results/figures/hyperparam_lines_{task}.png      (F1 vs LR, F1 vs Dropout lines)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.config import N_FOLDS, SEED, EPOCHS, PATIENCE, RESULTS_DIR, IMG_SIZE, BATCH_SIZE, N_CHANNELS, DROPOUT
from src.data.loaders import load_image_paths_and_labels, make_dataset
from src.utils.seeds import set_seeds

# ── Grid definition ───────────────────────────────────────────────────────────
LR_VALUES      = [1e-3, 5e-4, 1e-4, 5e-5]
DROPOUT_VALUES = [0.3, 0.5, 0.6, 0.7]
BATCH_VALUES   = [16, 32]          # secondary axis — only used if LR/DO search passes

# ── Model builder with custom HP ──────────────────────────────────────────────

def build_sdcnn_hp(lr: float, dropout: float) -> tf.keras.Model:
    from tensorflow.keras import layers, Model
    inputs = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(name="pool1")(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(name="pool2")(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(name="pool3")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="sdcnn_hp")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model


def cv_mean_f1(task: str, lr: float, dropout: float,
               batch_size: int = 32) -> tuple[float, float]:
    """Return (mean_f1, std_f1) for given HP on 5-fold CV."""
    set_seeds(SEED)
    paths, labels = load_image_paths_and_labels(task)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), 1):
        set_seeds(SEED + fold)

        # rebuild dataset with current batch size
        train_ds = tf.data.Dataset.from_tensor_slices(
            (list(paths[tr_idx]), list(labels[tr_idx]))).shuffle(len(tr_idx), seed=SEED+fold)
        val_ds   = tf.data.Dataset.from_tensor_slices(
            (list(paths[va_idx]), list(labels[va_idx])))

        def _load(p, l, aug):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            if aug:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.1)
                img = tf.image.random_contrast(img, 0.9, 1.1)
            return img, tf.cast(l, tf.float32)

        train_ds = (train_ds
                    .map(lambda p, l: _load(p, l, True), num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE))
        val_ds   = (val_ds
                    .map(lambda p, l: _load(p, l, False), num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE))

        model = build_sdcnn_hp(lr, dropout)
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0)],
                  verbose=0)

        y_prob = model.predict(val_ds, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = np.concatenate([y.numpy() for _, y in val_ds])
        fold_f1s.append(f1_score(y_true, y_pred, zero_division=0))
        tf.keras.backend.clear_session()

    return float(np.mean(fold_f1s)), float(np.std(fold_f1s))


def run_grid_search(task: str):
    print(f"\n{'='*55}", flush=True)
    print(f"Hyperparameter grid search — {task}", flush=True)
    print(f"Grid: {len(LR_VALUES)} LR × {len(DROPOUT_VALUES)} dropout"
          f" = {len(LR_VALUES)*len(DROPOUT_VALUES)} configs", flush=True)
    print(f"{'='*55}", flush=True)

    tables = RESULTS_DIR / "tables"
    figs   = RESULTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    rows = []
    best_f1, best_cfg = -1, {}

    for lr in LR_VALUES:
        for do in DROPOUT_VALUES:
            mean_f1, std_f1 = cv_mean_f1(task, lr, do)
            rows.append({"task": task, "lr": lr, "dropout": do,
                         "mean_f1": mean_f1, "std_f1": std_f1})
            print(f"  LR={lr:.0e}  DO={do:.1f}  →  F1={mean_f1:.4f} ± {std_f1:.4f}",
                  flush=True)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_cfg = {"lr": lr, "dropout": do}

    df = pd.DataFrame(rows)
    df.to_csv(tables / f"hyperparam_grid_{task}.csv", index=False)
    print(f"\nBest: LR={best_cfg['lr']:.0e}  DO={best_cfg['dropout']:.1f}"
          f"  →  F1={best_f1:.4f}", flush=True)

    _plot_heatmap(df, task, figs)
    _plot_lines(df, task, figs)
    return df, best_cfg


def _plot_heatmap(df: pd.DataFrame, task: str, figs: Path):
    pivot = df.pivot_table(index="dropout", columns="lr", values="mean_f1")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, vmin=pivot.values.min() - 0.02,
                annot_kws={"size": 11})
    ax.set_title(f"Mean CV F1 — SDCNN ({task.capitalize()})\nLR × Dropout grid search")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Dropout")
    ax.set_xticklabels([f"{v:.0e}" for v in df["lr"].unique()])
    plt.tight_layout()
    out = figs / f"hyperparam_heatmap_{task}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap → {out}", flush=True)


def _plot_lines(df: pd.DataFrame, task: str, figs: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # F1 vs LR (one line per dropout)
    for do, grp in df.groupby("dropout"):
        grp_sorted = grp.sort_values("lr")
        axes[0].plot(grp_sorted["lr"], grp_sorted["mean_f1"],
                     marker="o", label=f"DO={do}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Mean F1 (5-fold CV)")
    axes[0].set_title(f"F1 vs Learning Rate ({task})")
    axes[0].legend(fontsize=8)

    # F1 vs Dropout (one line per LR)
    for lr, grp in df.groupby("lr"):
        grp_sorted = grp.sort_values("dropout")
        axes[1].plot(grp_sorted["dropout"], grp_sorted["mean_f1"],
                     marker="o", label=f"LR={lr:.0e}")
    axes[1].set_xlabel("Dropout Rate")
    axes[1].set_ylabel("Mean F1 (5-fold CV)")
    axes[1].set_title(f"F1 vs Dropout Rate ({task})")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = figs / f"hyperparam_lines_{task}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Lines plot → {out}", flush=True)


if __name__ == "__main__":
    for task in ("detection", "diagnosis"):
        run_grid_search(task)
    print("\nExp 07 complete.", flush=True)
