"""
Exp 08 — Extended ablation study for the paper.

Each ablation tests one design decision in isolation (all others fixed at best config):

A. SDCNN architecture depth: 2-block vs 3-block (current) vs 4-block
B. Dropout sensitivity: 0.3 / 0.4 / 0.5 / 0.6 / 0.7
C. Transfer-learning strategy: frozen vs unfreeze-last-2 vs unfreeze-last-4 (VGG16)
D. Input resolution: 128×128 vs 224×224 (speed/accuracy trade-off)

Produces:
  results/tables/ablation_{A-D}_{task}.csv
  results/figures/ablation_bar_{task}.png  — grouped bar chart of all ablations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

from src.config import (N_FOLDS, SEED, EPOCHS, PATIENCE, RESULTS_DIR,
                        IMG_SIZE, BATCH_SIZE, N_CHANNELS, DROPOUT, LR)
from src.data.loaders import load_image_paths_and_labels, make_dataset
from src.utils.seeds import set_seeds

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 11})


# ── Generic CV runner for ablations ──────────────────────────────────────────

def ablation_cv(task: str, build_fn, label: str,
                img_size=IMG_SIZE, batch_size=BATCH_SIZE) -> dict:
    set_seeds(SEED)
    paths, labels = load_image_paths_and_labels(task)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1s, precs, recs, accs = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), 1):
        set_seeds(SEED + fold)

        def _load(p, l, aug, size):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, size)
            img = tf.cast(img, tf.float32) / 255.0
            if aug:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.1)
                img = tf.image.random_contrast(img, 0.9, 1.1)
            return img, tf.cast(l, tf.float32)

        size = img_size
        tr_ds = (tf.data.Dataset.from_tensor_slices(
                     (list(paths[tr_idx]), list(labels[tr_idx])))
                 .shuffle(len(tr_idx), seed=SEED+fold)
                 .map(lambda p, l: _load(p, l, True, size), num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(batch_size).prefetch(tf.data.AUTOTUNE))
        va_ds = (tf.data.Dataset.from_tensor_slices(
                     (list(paths[va_idx]), list(labels[va_idx])))
                 .map(lambda p, l: _load(p, l, False, size), num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(batch_size).prefetch(tf.data.AUTOTUNE))

        model = build_fn()
        model.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0)],
                  verbose=0)

        y_prob = model.predict(va_ds, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = np.concatenate([y.numpy() for _, y in va_ds])

        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
        f1s.append(f); precs.append(p); recs.append(r)
        accs.append(accuracy_score(y_true, y_pred))
        tf.keras.backend.clear_session()

    result = {"label": label, "task": task,
               "mean_f1": np.mean(f1s), "std_f1": np.std(f1s),
               "mean_prec": np.mean(precs), "mean_rec": np.mean(recs),
               "mean_acc": np.mean(accs)}
    print(f"  {label:40s}  F1={result['mean_f1']:.4f} ± {result['std_f1']:.4f}",
          flush=True)
    return result


# ── A. Architecture depth ─────────────────────────────────────────────────────

def build_sdcnn_2block():
    from tensorflow.keras import layers, Model
    inp = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = layers.Conv2D(32, 3, activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="sdcnn_2block")
    m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
              loss="binary_crossentropy")
    return m

def build_sdcnn_3block():    # standard SDCNN
    from src.models.sdcnn import build_sdcnn
    return build_sdcnn()

def build_sdcnn_4block():
    from tensorflow.keras import layers, Model
    inp = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = layers.Conv2D(32, 3, activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = Model(inp, out, name="sdcnn_4block")
    m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
              loss="binary_crossentropy")
    return m

DEPTH_VARIANTS = [
    ("SDCNN-2block (shallow)", build_sdcnn_2block),
    ("SDCNN-3block (proposed)", build_sdcnn_3block),
    ("SDCNN-4block (deep)", build_sdcnn_4block),
]


# ── B. Dropout sensitivity ─────────────────────────────────────────────────────

def _make_sdcnn_dropout(do):
    def builder():
        from tensorflow.keras import layers, Model
        inp = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
        x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(do)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = Model(inp, out, name=f"sdcnn_do{int(do*10)}")
        m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
                  loss="binary_crossentropy")
        return m
    return builder

DROPOUT_VARIANTS = [(f"Dropout={do}", _make_sdcnn_dropout(do))
                    for do in [0.3, 0.4, 0.5, 0.6, 0.7]]


# ── C. Fine-tuning strategy (VGG16) ──────────────────────────────────────────

def _make_vgg16_finetune(unfreeze_blocks: int):
    def builder():
        from tensorflow.keras import layers, Model
        from tensorflow.keras.applications import VGG16
        base = VGG16(include_top=False, weights="imagenet",
                     input_shape=(*IMG_SIZE, N_CHANNELS))
        # Freeze all first, then unfreeze last N blocks
        base.trainable = False
        if unfreeze_blocks > 0:
            for layer in base.layers[-unfreeze_blocks * 3:]:
                layer.trainable = True
        inp = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
        x = tf.keras.applications.vgg16.preprocess_input(inp * 255.0)
        x = base(x, training=(unfreeze_blocks > 0))
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = Model(inp, out, name=f"vgg16_unfreeze{unfreeze_blocks}")
        # lower LR when fine-tuning
        lr = LR / 10 if unfreeze_blocks > 0 else LR
        m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
                  loss="binary_crossentropy")
        return m
    return builder

FINETUNE_VARIANTS = [
    ("VGG16 frozen (baseline)", _make_vgg16_finetune(0)),
    ("VGG16 unfreeze block5", _make_vgg16_finetune(1)),
    ("VGG16 unfreeze blocks 4+5", _make_vgg16_finetune(2)),
]


# ── D. Input resolution ───────────────────────────────────────────────────────

def _make_sdcnn_resolution(size):
    from src.models.sdcnn import build_sdcnn
    h, w = size
    def builder():
        from tensorflow.keras import layers, Model
        inp = tf.keras.Input(shape=(h, w, N_CHANNELS))
        x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = Model(inp, out, name=f"sdcnn_{h}x{w}")
        m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
                  loss="binary_crossentropy")
        return m
    return builder, size

RESOLUTION_VARIANTS = [
    ("128×128 (fast)", *_make_sdcnn_resolution((128, 128))),
    ("224×224 (standard)", *_make_sdcnn_resolution((224, 224))),
]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ablation_bar(all_results: list[dict], task: str, figs: Path):
    df = pd.DataFrame(all_results)
    df = df[df["task"] == task].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    bars = ax.bar(df["label"], df["mean_f1"], yerr=df["std_f1"],
                  capsize=4, color=colors, alpha=0.85, error_kw={"linewidth": 1.5})

    # annotate with F1 value
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["std_f1"] + 0.005,
                f"{row['mean_f1']:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean F1 (5-fold CV)")
    ax.set_title(f"Ablation Study — {task.capitalize()} Task")
    ax.set_ylim(max(0, df["mean_f1"].min() - 0.10), 1.05)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()

    out = figs / f"ablation_bar_{task}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Ablation bar chart → {out}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tables = RESULTS_DIR / "tables"
    figs   = RESULTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    all_results = []

    for task in ("detection", "diagnosis"):
        print(f"\n{'='*55}", flush=True)
        print(f"ABLATION STUDY — {task.upper()}", flush=True)
        print(f"{'='*55}", flush=True)

        # A. Depth
        print("\n[A] Architecture depth", flush=True)
        rows_A = []
        for label, fn in DEPTH_VARIANTS:
            r = ablation_cv(task, fn, label)
            rows_A.append(r); all_results.append(r)
        pd.DataFrame(rows_A).to_csv(tables / f"ablation_depth_{task}.csv", index=False)

        # B. Dropout
        print("\n[B] Dropout sensitivity", flush=True)
        rows_B = []
        for label, fn in DROPOUT_VARIANTS:
            r = ablation_cv(task, fn, label)
            rows_B.append(r); all_results.append(r)
        pd.DataFrame(rows_B).to_csv(tables / f"ablation_dropout_{task}.csv", index=False)

        # C. Fine-tuning strategy
        print("\n[C] VGG16 fine-tuning strategy", flush=True)
        rows_C = []
        for label, fn in FINETUNE_VARIANTS:
            r = ablation_cv(task, fn, label)
            rows_C.append(r); all_results.append(r)
        pd.DataFrame(rows_C).to_csv(tables / f"ablation_finetune_{task}.csv", index=False)

        # D. Resolution
        print("\n[D] Input resolution", flush=True)
        rows_D = []
        for label, fn, size in RESOLUTION_VARIANTS:
            r = ablation_cv(task, fn, label, img_size=size)
            rows_D.append(r); all_results.append(r)
        pd.DataFrame(rows_D).to_csv(tables / f"ablation_resolution_{task}.csv", index=False)

        # Plot combined bar chart
        task_results = [r for r in all_results if r["task"] == task]
        plot_ablation_bar(task_results, task, figs)

    pd.DataFrame(all_results).to_csv(tables / "ablation_all.csv", index=False)
    print("\nExp 08 complete.", flush=True)
