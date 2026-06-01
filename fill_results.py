"""
Fill LaTeX \todo{} placeholders in main.tex with actual results.
Run after all experiments complete:
  python fill_results.py

Reads: results/tables/cv_*.csv, stats_*.csv, cross_domain.csv
Writes: manuscript/main_final.tex  (original left untouched)
Also prints a full Markdown summary to stdout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

RESULTS = Path("results/tables")
MANUSCRIPT = Path("manuscript")

ARCH_LABELS = {
    "sdcnn":            "SDCNN (ours)",
    "vgg16":            "VGG-16",
    "mobilenetv2":      "MobileNetV2",
    "efficientnet_b0":  "EfficientNet-B0",
}
ARCH_ORDER = list(ARCH_LABELS.keys())


def load_cv(task: str, model: str):
    csv = RESULTS / f"cv_{task}_{model}.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def fmt(mean, std):
    return f"${mean:.3f} \\pm {std:.3f}$"


def fmt_val(v):
    return f"${v:.3f}$"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown summary
# ─────────────────────────────────────────────────────────────────────────────

def print_markdown_summary():
    print("# Experiment Results Summary\n")

    for task in ("detection", "diagnosis"):
        print(f"## {task.capitalize()} Task — 5-Fold CV\n")
        header = "| Model | F1 | Precision | Recall | Accuracy |"
        sep    = "|---|---|---|---|---|"
        rows = [header, sep]
        for model in ARCH_ORDER:
            df = load_cv(task, model)
            if df is None:
                rows.append(f"| {ARCH_LABELS[model]} | — | — | — | — |")
                continue
            rows.append(
                f"| {ARCH_LABELS[model]} "
                f"| {df['f1'].mean():.4f} ± {df['f1'].std():.4f} "
                f"| {df['precision'].mean():.4f} ± {df['precision'].std():.4f} "
                f"| {df['recall'].mean():.4f} ± {df['recall'].std():.4f} "
                f"| {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f} |"
            )
        print("\n".join(rows))
        print()

    # Stats
    for task in ("detection", "diagnosis"):
        stats_csv = RESULTS / f"stats_{task}.csv"
        if not stats_csv.exists():
            continue
        print(f"## Statistical Tests — {task.capitalize()} (SDCNN vs baselines)\n")
        df = pd.read_csv(stats_csv)
        print(df[["model_B", "mean_F1_A", "mean_F1_B", "delta",
                   "CI95_lo_A", "CI95_hi_A", "t_p", "w_p",
                   "cohen_d", "significant"]].to_markdown(index=False))
        print()

    # Cross-domain
    cd_csv = RESULTS / "cross_domain.csv"
    if cd_csv.exists():
        print("## Cross-Domain Evaluation (Pierdicca IRT-PV)\n")
        df = pd.read_csv(cd_csv)
        print(df.to_markdown(index=False))
        print()

    # Augmentation ablation
    for task in ("detection", "diagnosis"):
        aug_csv   = RESULTS / f"cv_{task}_sdcnn.csv"
        noaug_csv = RESULTS / f"cv_{task}_sdcnn_noaug.csv"
        if aug_csv.exists() and noaug_csv.exists():
            aug_f1   = pd.read_csv(aug_csv)["f1"].mean()
            noaug_f1 = pd.read_csv(noaug_csv)["f1"].mean()
            print(f"## Augmentation Ablation — {task.capitalize()}\n")
            print(f"With augmentation: {aug_f1:.4f}")
            print(f"Without augmentation: {noaug_f1:.4f}")
            print(f"Delta: {aug_f1 - noaug_f1:+.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generation
# ─────────────────────────────────────────────────────────────────────────────

def _latex_cv_table(task: str) -> str:
    rows = []
    best_f1 = {
        m: (load_cv(task, m)["f1"].mean() if load_cv(task, m) is not None else -1)
        for m in ARCH_ORDER
    }
    best_model = max(best_f1, key=best_f1.get)

    for model in ARCH_ORDER:
        df = load_cv(task, model)
        label = ARCH_LABELS[model]
        if df is None:
            rows.append(f"    {label} & — & — & — & — \\\\")
            continue
        f1_m, f1_s = df["f1"].mean(), df["f1"].std()
        p_m,  p_s  = df["precision"].mean(), df["precision"].std()
        r_m,  r_s  = df["recall"].mean(), df["recall"].std()
        a_m,  a_s  = df["accuracy"].mean(), df["accuracy"].std()

        bold = model == best_model
        def _fmt(m, s):
            v = f"{m:.3f} \\pm {s:.3f}"
            return f"$\\mathbf{{{v}}}$" if bold else f"${v}$"

        rows.append(
            f"    {label} & {_fmt(f1_m,f1_s)} & {_fmt(p_m,p_s)} & "
            f"{_fmt(r_m,r_s)} & {_fmt(a_m,a_s)} \\\\"
        )
    return "\n".join(rows)


def _latex_stats_table(task: str) -> str:
    stats_csv = RESULTS / f"stats_{task}.csv"
    if not stats_csv.exists():
        return "    % stats results not yet available"
    df = pd.read_csv(stats_csv)
    rows = []
    for _, row in df.iterrows():
        sig = "\\checkmark" if row["significant"] else "\\texttimes"
        rows.append(
            f"    {ARCH_LABELS.get(row['model_B'], row['model_B'])} & "
            f"${row['delta']:+.3f}$ & "
            f"${row['t_p']:.3f}$ & "
            f"${row['w_p']:.3f}$ & "
            f"${row['cohen_d']:.3f}$ & {sig} \\\\"
        )
    return "\n".join(rows)


def _latex_cross_domain_table() -> str:
    cd_csv = RESULTS / "cross_domain.csv"
    if not cd_csv.exists():
        return "    % cross-domain results not yet available"
    df = pd.read_csv(cd_csv)
    rows = []
    for _, row in df.iterrows():
        label = ARCH_LABELS.get(row["model"], row["model"])
        gap = row.get("domain_gap", "—")
        gap_str = f"${gap:.3f}$" if isinstance(gap, float) and not np.isnan(gap) else "—"
        rows.append(
            f"    {label} & ${row['f1']:.3f}$ & ${row['precision']:.3f}$ & "
            f"${row['recall']:.3f}$ & {gap_str} \\\\"
        )
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fill main.tex placeholders
# ─────────────────────────────────────────────────────────────────────────────

def fill_latex():
    main_tex = MANUSCRIPT / "main.tex"
    if not main_tex.exists():
        print("manuscript/main.tex not found — skipping LaTeX fill")
        return

    content = main_tex.read_text()

    # ── Detection table rows ─────────────────────────────────────────────────
    for task in ("detection", "diagnosis"):
        table_body = _latex_cv_table(task)
        # Replace placeholder comments in the tabular body
        old_marker = (
            "    SDCNN (ours)     & \\todo{fill} & \\todo{fill} & \\todo{fill} & \\todo{fill} \\\\"
            if task == "detection" else ""
        )
        # We directly regenerate the two tabular bodies
        # Find the table for this task and replace its MobileNetV2/EfficientNet rows
        mob_label   = "MobileNetV2      & \\todo{fill} & \\todo{fill} & \\todo{fill} & \\todo{fill}"
        eff_label   = "EfficientNet-B0  & \\todo{fill} & \\todo{fill} & \\todo{fill} & \\todo{fill}"

        for model in ("mobilenetv2", "efficientnet_b0"):
            df = load_cv(task, model)
            if df is None:
                continue
            label = ARCH_LABELS[model]
            f1_m, f1_s = df["f1"].mean(), df["f1"].std()
            p_m,  p_s  = df["precision"].mean(), df["precision"].std()
            r_m,  r_s  = df["recall"].mean(), df["recall"].std()
            a_m,  a_s  = df["accuracy"].mean(), df["accuracy"].std()
            new_row = (
                f"{label:<16} & ${f1_m:.3f} \\pm {f1_s:.3f}$ & "
                f"${p_m:.3f} \\pm {p_s:.3f}$ & "
                f"${r_m:.3f} \\pm {r_s:.3f}$ & "
                f"${a_m:.3f} \\pm {a_s:.3f}$"
            )
            if model == "mobilenetv2":
                content = content.replace(
                    f"MobileNetV2      & \\todo{{fill}} & \\todo{{fill}} & \\todo{{fill}} & \\todo{{fill}}",
                    new_row, 1)
            else:
                content = content.replace(
                    f"EfficientNet-B0  & \\todo{{fill}} & \\todo{{fill}} & \\todo{{fill}} & \\todo{{fill}}",
                    new_row, 1)

    # ── Stats table rows ─────────────────────────────────────────────────────
    for task in ("detection",):
        stats_csv = RESULTS / f"stats_{task}.csv"
        if not stats_csv.exists():
            continue
        df = pd.read_csv(stats_csv)
        for _, row in df.iterrows():
            model_key = row["model_B"]
            label = ARCH_LABELS.get(model_key, model_key)
            sig = "\\checkmark" if row["significant"] else "\\texttimes"
            new_row = (
                f"{label:<15} & ${row['delta']:+.3f}$ & "
                f"${row['t_p']:.3f}$ & ${row['w_p']:.3f}$ & "
                f"${row['cohen_d']:.3f}$ & {sig}"
            )
            content = content.replace(
                f"VGG-16          & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}"
                    if model_key == "vgg16" else
                f"MobileNetV2     & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}"
                    if model_key == "mobilenetv2" else
                f"EfficientNet-B0 & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}",
                new_row, 1)

    # ── Cross-domain table ───────────────────────────────────────────────────
    cd_csv = RESULTS / "cross_domain.csv"
    if cd_csv.exists():
        df = pd.read_csv(cd_csv)
        for _, row in df.iterrows():
            model_key = row["model"]
            label = ARCH_LABELS.get(model_key, model_key)
            gap = row.get("domain_gap", None)
            gap_str = f"${gap:.3f}$" if gap is not None and not np.isnan(float(gap)) else "—"
            new_row = (
                f"{label:<15} & ${row['f1']:.3f}$ & ${row['precision']:.3f}$ & "
                f"${row['recall']:.3f}$ & {gap_str}"
            )
            placeholder = (
                f"SDCNN           & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}"
                    if model_key == "sdcnn" else
                f"VGG-16          & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}"
                    if model_key == "vgg16" else
                f"MobileNetV2     & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}"
                    if model_key == "mobilenetv2" else
                f"EfficientNet-B0 & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}"
            )
            content = content.replace(placeholder, new_row, 1)

    # ── Augmentation delta ───────────────────────────────────────────────────
    aug_csv = RESULTS / "cv_detection_sdcnn.csv"
    noaug_csv = RESULTS / "cv_detection_sdcnn_noaug.csv"
    if aug_csv.exists() and noaug_csv.exists():
        aug_f1   = pd.read_csv(aug_csv)["f1"].mean()
        noaug_f1 = pd.read_csv(noaug_csv)["f1"].mean()
        content = content.replace(
            "Training SDCNN without augmentation yields a mean F1 of \\todo{fill} on\ndetection (vs.\\ \\numres{0.884} with augmentation)",
            f"Training SDCNN without augmentation yields a mean F1 of \\numres{{{noaug_f1:.3f}}} on\ndetection (vs.\\ \\numres{{0.884}} with augmentation)"
        )

    out = MANUSCRIPT / "main_final.tex"
    out.write_text(content)
    print(f"\nFilled LaTeX written to: {out}")


if __name__ == "__main__":
    print_markdown_summary()
    fill_latex()
    print("\nDone.")
