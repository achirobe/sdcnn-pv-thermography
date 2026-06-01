import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.config import N_FOLDS, SEED, EPOCHS, PATIENCE, RESULTS_DIR
from src.data.loaders import load_image_paths_and_labels, make_dataset
from src.utils.seeds import set_seeds


def run_cv(task: str, model_name: str, build_fn, augment_override: bool = True):
    """
    Run 5-fold stratified cross-validation.

    Saves:
      results/tables/cv_{task}_{model_name}.csv   — per-fold metrics
      results/tables/preds_{task}_{model_name}_fold{k}.npy  — (y_true, y_prob, y_pred)
      results/logs/best_{task}_{model_name}.keras — weights from best-F1 fold
    """
    set_seeds(SEED)

    paths, labels = load_image_paths_and_labels(task)
    print(f"\n[{task}/{model_name}] Dataset: {len(paths)} images, "
          f"class balance: {labels.sum()}/{len(labels)-labels.sum()}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_rows = []
    best_f1 = -1.0

    tables_dir = RESULTS_DIR / "tables"
    logs_dir   = RESULTS_DIR / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), start=1):
        print(f"\n  Fold {fold}/{N_FOLDS} — train={len(tr_idx)}, val={len(va_idx)}", flush=True)
        set_seeds(SEED + fold)

        train_ds = make_dataset(paths[tr_idx], labels[tr_idx],
                                augment=augment_override, training=True)
        val_ds   = make_dataset(paths[va_idx], labels[va_idx],
                                augment=False, training=False)

        model = build_fn()
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, verbose=0),
        ]

        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS, callbacks=cb, verbose=0,
        )
        stopped = len(history.history["loss"])
        print(f"    Stopped at epoch {stopped}", flush=True)

        # --- collect predictions (single predict call avoids tf.function retracing) ---
        y_prob = model.predict(val_ds, verbose=0).flatten()
        y_true = np.concatenate([y_batch.numpy() for _, y_batch in val_ds])
        y_pred = (y_prob >= 0.5).astype(int)

        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        print(f"    F1={f:.4f}  Prec={p:.4f}  Rec={r:.4f}  Acc={acc:.4f}", flush=True)

        fold_rows.append({
            "task": task, "model": model_name, "fold": fold,
            "precision": p, "recall": r, "f1": f, "accuracy": acc,
        })

        # save predictions
        np.save(
            tables_dir / f"preds_{task}_{model_name}_fold{fold}.npy",
            np.column_stack([y_true, y_prob, y_pred]),
        )

        # save best-fold model
        if f > best_f1:
            best_f1 = f
            model.save(logs_dir / f"best_{task}_{model_name}.keras")
            print(f"    *** New best model saved (F1={best_f1:.4f})", flush=True)

        tf.keras.backend.clear_session()

    df = pd.DataFrame(fold_rows)
    out_csv = tables_dir / f"cv_{task}_{model_name}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n[{task}/{model_name}] CV complete — "
          f"mean F1={df['f1'].mean():.4f} ± {df['f1'].std():.4f}", flush=True)
    print(f"  Results saved to {out_csv}", flush=True)
    return df
