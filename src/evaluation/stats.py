import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from src.config import RESULTS_DIR


def bootstrap_ci(values, n_boot: int = 10_000, ci: float = 0.95, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lo), float(hi)


def paired_tests(task: str, baseline: str = "sdcnn"):
    tables_dir = RESULTS_DIR / "tables"
    models = ["sdcnn", "vgg16", "mobilenetv2", "efficientnet_b0"]

    dfs = []
    for m in models:
        csv = tables_dir / f"cv_{task}_{m}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing {csv} — run CV experiments first.")
        df = pd.read_csv(csv)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    pivot = combined.pivot_table(index="fold", columns="model", values="f1")

    rows = []
    for other in models:
        if other == baseline:
            continue
        a = pivot[baseline].values
        b = pivot[other].values

        t_stat, t_p = stats.ttest_rel(a, b)
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="zsplit",
                                     alternative="two-sided", method="approx")

        diff = a - b
        cohen_d = diff.mean() / (diff.std(ddof=1) + 1e-12)

        mean_a, ci_lo_a, ci_hi_a = bootstrap_ci(a)
        mean_b, ci_lo_b, ci_hi_b = bootstrap_ci(b)

        rows.append({
            "task": task,
            "model_A": baseline, "model_B": other,
            "mean_F1_A": mean_a, "CI95_lo_A": ci_lo_a, "CI95_hi_A": ci_hi_a,
            "mean_F1_B": mean_b, "CI95_lo_B": ci_lo_b, "CI95_hi_B": ci_hi_b,
            "delta": mean_a - mean_b,
            "t_stat": t_stat, "t_p": t_p,
            "w_stat": w_stat, "w_p": w_p,
            "cohen_d": cohen_d,
            "significant": min(t_p, w_p) < 0.05,
        })

    out = pd.DataFrame(rows)
    out_path = tables_dir / f"stats_{task}.csv"
    out.to_csv(out_path, index=False)
    print(f"[stats/{task}] Saved to {out_path}")
    print(out[["model_B", "mean_F1_A", "mean_F1_B", "delta",
               "t_p", "w_p", "cohen_d", "significant"]].to_string(index=False))
    return out
