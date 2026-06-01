"""
Week 2: Zero-shot cross-domain evaluation on the Kaggle IRT-PV dataset.
The Ghana-trained best-fold models are evaluated without any retraining.
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from src.config import EXTERNAL_DIR, RESULTS_DIR, IMG_SIZE, BATCH_SIZE

# Kaggle dataset multi-class label → binary (1=fault, 0=no-fault)
# Adjust if the folder layout differs after download.
FAULT_CLASSES    = {"Bird-drop", "Dusty", "Electrical-damage",
                    "Physical-damage", "Snow-covered"}
NONFAULT_CLASSES = {"Clean"}


def _check_dataset_present():
    if not EXTERNAL_DIR.exists() or not any(EXTERNAL_DIR.iterdir()):
        print("\n" + "=" * 60)
        print("WEEK 2 — CROSS-DOMAIN DATASET NOT FOUND")
        print("=" * 60)
        print(f"Expected path: {EXTERNAL_DIR}")
        print("\nDownload instructions:")
        print("  1. Visit: kaggle.com/datasets/marcosgabriel/photovoltaic-system-thermography")
        print("  2. Download and unzip the dataset")
        print(f"  3. Place images under: {EXTERNAL_DIR}/")
        print("     Folder structure should be: kaggle_irt_pv/<ClassName>/*.jpg")
        print("  4. Re-run: python experiments/exp06_cross_domain/run.py")
        print("=" * 60 + "\n")
        return False
    return True


def load_external_dataset():
    """Scan EXTERNAL_DIR, map labels to binary, return (paths, labels)."""
    paths, labels = [], []
    for class_dir in sorted(EXTERNAL_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name
        if name in FAULT_CLASSES:
            label = 1
        elif name in NONFAULT_CLASSES:
            label = 0
        else:
            print(f"  [cross_domain] Skipping unknown class folder: {name}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in class_dir.glob(ext):
                paths.append(str(p))
                labels.append(label)

    print(f"[cross_domain] Loaded {len(paths)} external images "
          f"({sum(labels)} fault / {len(labels)-sum(labels)} no-fault)")
    return np.array(paths), np.array(labels, dtype=np.int32)


def _make_external_ds(paths, labels):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_on_external(model_path: Path, name: str,
                          ext_paths, ext_labels) -> dict:
    model = tf.keras.models.load_model(model_path)
    ds = _make_external_ds(ext_paths, ext_labels)

    y_prob = model.predict(ds, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = ext_labels

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"  {name:20s}  F1={f:.4f}  Prec={p:.4f}  Rec={r:.4f}  Acc={acc:.4f}")
    return {"model": name, "precision": p, "recall": r, "f1": f, "accuracy": acc,
            "y_true": y_true, "y_pred": y_pred}


def run_cross_domain():
    if not _check_dataset_present():
        sys.exit(0)

    ext_paths, ext_labels = load_external_dataset()
    logs_dir   = RESULTS_DIR / "logs"
    tables_dir = RESULTS_DIR / "tables"
    figs_dir   = RESULTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    archs = ["sdcnn", "vgg16", "mobilenetv2", "efficientnet_b0"]
    rows, best_result = [], None

    for arch in archs:
        mp = logs_dir / f"best_detection_{arch}.keras"
        if not mp.exists():
            print(f"  [skip] {mp} not found — run exp01/exp02 first.")
            continue
        result = evaluate_on_external(mp, arch, ext_paths, ext_labels)
        rows.append({k: v for k, v in result.items()
                     if k not in ("y_true", "y_pred")})
        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = result

    pd.DataFrame(rows).to_csv(tables_dir / "cross_domain.csv", index=False)
    print(f"\n[cross_domain] Results saved to {tables_dir/'cross_domain.csv'}")

    # plot confusion matrix for best model
    if best_result:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(best_result["y_true"], best_result["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No-fault", "Fault"],
                    yticklabels=["No-fault", "Fault"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Cross-domain — {best_result['model']} (zero-shot)")
        plt.tight_layout()
        plt.savefig(figs_dir / "cross_domain_cm.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[cross_domain] Confusion matrix saved → {figs_dir/'cross_domain_cm.png'}")
