"""
Exp 10 — Cross-domain calibration on the Pierdicca dataset.

Reviewer response: the zero-shot cross-domain result (SDCNN F1=0.452,
precision=0.292, recall=1.000) is the trivial "predict-all-faulty" baseline.
We claim this is decision-threshold miscalibration under prior shift
(training prevalence 50% vs Pierdicca 29%), not a representational failure.
This experiment tests that claim directly.

Method: take the Ghana-trained best-fold SDCNN detection model, obtain its
predicted probabilities on Pierdicca, hold out a small labelled calibration
subset, fit Platt recalibration (logistic on the model logit), and evaluate
on the remaining images. Temperature scaling alone cannot change a 0.5-
threshold decision (monotonic), so Platt's bias term is the operative fix.
We average over many random calibration splits to avoid lucky-split bias.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.config import RESULTS_DIR, SEED
from src.evaluation.cross_domain import load_external_dataset
from src.data.loaders import make_dataset

MODEL_PATH = RESULTS_DIR / "logs" / "best_detection_sdcnn.keras"
N_CAL = 20          # labelled images used for calibration
N_SPLITS = 50       # random calibration splits to average over


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return p, r, f, accuracy_score(y_true, y_pred)


def main():
    print(f"Loading model: {MODEL_PATH}", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    paths, y = load_external_dataset()
    ds = make_dataset(paths, y, augment=False, training=False)
    prob = model.predict(ds, verbose=0).flatten()
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob)).reshape(-1, 1)

    # ---- Uncalibrated baseline (threshold 0.5) on the full set ----
    base_pred = (prob >= 0.5).astype(int)
    bp, br, bf, ba = metrics(y, base_pred)
    print(f"\nUncalibrated (zero-shot, t=0.5) on all {len(y)} images:")
    print(f"  F1={bf:.3f}  Prec={bp:.3f}  Rec={br:.3f}  Acc={ba:.3f}", flush=True)

    # ---- Platt recalibration averaged over random calibration splits ----
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, train_size=N_CAL,
                                 random_state=SEED)
    cal_f1, cal_p, cal_r, cal_a = [], [], [], []
    base_f1_test = []
    for cal_idx, test_idx in sss.split(logit, y):
        clf = LogisticRegression(C=1.0, solver="lbfgs")
        clf.fit(logit[cal_idx], y[cal_idx])
        pred = clf.predict(logit[test_idx])
        p, r, f, a = metrics(y[test_idx], pred)
        cal_p.append(p); cal_r.append(r); cal_f1.append(f); cal_a.append(a)
        # uncalibrated baseline on the SAME test subset (fair comparison)
        _, _, bf_t, _ = metrics(y[test_idx], (prob[test_idx] >= 0.5).astype(int))
        base_f1_test.append(bf_t)

    print(f"\nPlatt recalibration ({N_CAL} calibration images, "
          f"{N_SPLITS} random splits, evaluated on remaining {len(y)-N_CAL}):")
    print(f"  F1   = {np.mean(cal_f1):.3f} +/- {np.std(cal_f1):.3f}")
    print(f"  Prec = {np.mean(cal_p):.3f} +/- {np.std(cal_p):.3f}")
    print(f"  Rec  = {np.mean(cal_r):.3f} +/- {np.std(cal_r):.3f}")
    print(f"  Acc  = {np.mean(cal_a):.3f} +/- {np.std(cal_a):.3f}")
    print(f"\n  Uncalibrated F1 on the same held-out test folds: "
          f"{np.mean(base_f1_test):.3f} +/- {np.std(base_f1_test):.3f}")
    print(f"  --> mean F1 improvement: "
          f"{np.mean(cal_f1) - np.mean(base_f1_test):+.3f}", flush=True)

    # save summary
    import pandas as pd
    out = RESULTS_DIR / "tables" / "cross_domain_calibration.csv"
    pd.DataFrame([{
        "uncalibrated_f1_full": bf, "uncalibrated_prec_full": bp,
        "uncalibrated_rec_full": br, "uncalibrated_acc_full": ba,
        "n_cal": N_CAL, "n_splits": N_SPLITS,
        "platt_f1_mean": np.mean(cal_f1), "platt_f1_std": np.std(cal_f1),
        "platt_prec_mean": np.mean(cal_p), "platt_rec_mean": np.mean(cal_r),
        "platt_acc_mean": np.mean(cal_a),
        "base_f1_test_mean": np.mean(base_f1_test),
        "f1_improvement": np.mean(cal_f1) - np.mean(base_f1_test),
    }]).to_csv(out, index=False)
    print(f"\nSaved -> {out}", flush=True)
    print("\nExp 10 complete.", flush=True)


if __name__ == "__main__":
    main()
