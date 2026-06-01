"""
Week 2: Zero-shot cross-domain evaluation on the Pierdicca IRT-PV dataset.
(Also compatible with the Bommes 2021 / Kaggle photovoltaic-system-thermography dataset.)

Dataset structure expected (Pierdicca format):
  external/kaggle_irt_pv/kaggle_dataset/
    dataset_1/
      images/       *.jpg
      annotations/  *.json  — {"instances": [{"defected_module": bool, ...}, ...]}
    dataset_2/
      images/
      annotations/

Label mapping:
  Any image where at least one module has defected_module=True  → fault   (1)
  All modules defected_module=False                             → no-fault (0)

Ghana-trained best-fold models are evaluated without any retraining.
"""
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, confusion_matrix)

from src.config import EXTERNAL_DIR, RESULTS_DIR, IMG_SIZE, BATCH_SIZE

_KAGGLE_ROOT = EXTERNAL_DIR / "kaggle_dataset"


def _check_dataset_present() -> bool:
    if not _KAGGLE_ROOT.exists():
        print("\n" + "=" * 60)
        print("WEEK 2 — CROSS-DOMAIN DATASET NOT FOUND")
        print("=" * 60)
        print(f"Expected path: {_KAGGLE_ROOT}")
        print("\nPlace the Pierdicca IRT-PV dataset at:")
        print(f"  {EXTERNAL_DIR}/kaggle_dataset/dataset_1/images/ ...")
        print("=" * 60 + "\n")
        return False
    return True


def load_external_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Parse Pierdicca-format annotations across dataset_1 and dataset_2.
    Returns (paths_array, labels_array) where label=1 means at least one
    defective module is present in the image.
    """
    paths, labels = [], []

    for sub in ("dataset_1", "dataset_2"):
        img_dir = _KAGGLE_ROOT / sub / "images"
        ann_dir = _KAGGLE_ROOT / sub / "annotations"
        if not img_dir.exists():
            print(f"  [cross_domain] Skipping {sub} — images dir not found")
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            ann_path = ann_dir / (img_path.stem + ".json")
            if not ann_path.exists():
                print(f"  [cross_domain] No annotation for {img_path.name}, skipping")
                continue

            ann = json.loads(ann_path.read_text())
            instances = ann.get("instances", [])
            has_defect = any(inst.get("defected_module", False)
                             for inst in instances)
            paths.append(str(img_path))
            labels.append(1 if has_defect else 0)

    labels_arr = np.array(labels, dtype=np.int32)
    n_fault    = int(labels_arr.sum())
    n_clean    = int(len(labels_arr) - n_fault)
    print(f"[cross_domain] Loaded {len(paths)} images from Pierdicca dataset "
          f"({n_fault} fault / {n_clean} no-fault)")
    return np.array(paths), labels_arr


def _make_external_ds(paths: np.ndarray, labels: np.ndarray):
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
                          ext_paths: np.ndarray,
                          ext_labels: np.ndarray) -> dict:
    model = tf.keras.models.load_model(model_path)
    ds    = _make_external_ds(ext_paths, ext_labels)

    y_prob = model.predict(ds, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = ext_labels

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"  {name:25s}  F1={f:.4f}  Prec={p:.4f}  Rec={r:.4f}  Acc={acc:.4f}")
    return {"model": name, "precision": p, "recall": r, "f1": f,
            "accuracy": acc, "y_true": y_true, "y_pred": y_pred}


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

    print("\n[cross_domain] Zero-shot evaluation on Pierdicca IRT-PV dataset")
    print("-" * 60)
    for arch in archs:
        mp = logs_dir / f"best_detection_{arch}.keras"
        if not mp.exists():
            print(f"  [skip] {mp.name} not found — run exp01/exp02 first.")
            continue
        result = evaluate_on_external(mp, arch, ext_paths, ext_labels)
        rows.append({k: v for k, v in result.items()
                     if k not in ("y_true", "y_pred")})
        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = result

    if not rows:
        print("[cross_domain] No models evaluated — check logs/ folder.")
        return

    # also record in-domain CV mean for comparison
    for arch in archs:
        csv = tables_dir / f"cv_detection_{arch}.csv"
        if csv.exists():
            import pandas as pd_inner
            indomain_f1 = pd_inner.read_csv(csv)["f1"].mean()
            for row in rows:
                if row["model"] == arch:
                    row["indomain_cv_f1"] = indomain_f1
                    row["domain_gap"] = indomain_f1 - row["f1"]

    pd.DataFrame(rows).to_csv(tables_dir / "cross_domain.csv", index=False)
    print(f"\n[cross_domain] Results → {tables_dir / 'cross_domain.csv'}")

    # confusion matrix for best model
    if best_result:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(best_result["y_true"], best_result["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No-fault", "Fault"],
                    yticklabels=["No-fault", "Fault"],
                    annot_kws={"size": 13})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Cross-domain (Pierdicca) — {best_result['model']}\n"
                     f"zero-shot, F1={best_result['f1']:.3f}")
        plt.tight_layout()
        out = figs_dir / "cross_domain_cm.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[cross_domain] Confusion matrix → {out}")
