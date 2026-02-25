"""
Publication Regression Plot
===========================

Creates a Springer Nature Computer Science compliant regression scatter
plot for the best-performing model configuration with a session-level
95% confidence interval lower bound above zero.

Configuration
-------------
- Target: WHO-5 (``who_5_percentage_score_corrected``)
- Audio: noisy (no denoising)
- Features: eGeMAPS (``eGeMAPSv02``)
- Speech task: read-happy (``speechtasks-nilago-happy``)
- Model: SVR
- Session-level CCC: 0.350 (95% CI: 0.020 - 0.515)

Session-level aggregation
-------------------------
Segment-level predictions are averaged within each (participant,
session) pair, while the target value (identical across segments)
is taken as-is. This is identical to the aggregation in
``notebooks/mwas/vad_average_regression.ipynb``.

Figure specifications
---------------------
- Springer Nature CS single-column width (84 mm / 3.31 in)
- Fonts: DejaVu Sans, embedded as TrueType (pdf.fonttype = 42)
- DPI: 300
- Seaborn regression plot with 95% CI band

Input
-----
results/mwas/composed/compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv
    Paper results index pointing to per-model prediction parquet files.

Output (saved to notebooks/mwas/figures/)
------
regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.pdf
regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.svg
regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.png
"""

import os

import numpy as np
import pandas as pd

import audbenchmark

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# CCC metric - aligned with the paper (population std, ddof=0)
ccc_metric = audbenchmark.metric.concordance_cc


def configure_plot_style():
    """Set Springer Nature CS compliant rcParams."""
    sns.set_context("paper")
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica",
                "Arial",
                "sans-serif",
            ],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,  # TrueType - ensures fonts are embedded
            "ps.fonttype": 42,
            "svg.fonttype": "none",  # text as text in SVG
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def main():
    # ===== Paths =====
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    path_results_root = os.path.join(project_root, "results", "mwas", "modelling")
    path_paper_csv = os.path.join(
        project_root,
        "results",
        "mwas",
        "composed",
        "compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv",
    )
    output_dir = os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # ===== Load paper results index =====
    df_paper = pd.read_csv(path_paper_csv)

    # ===== Identify the target configuration =====
    # Best model with session-level CI lower bound > 0:
    # WHO-5, noisy, eGeMAPS, read-happy, SVR
    mask = (
        (df_paper["Target"] == "who_5_percentage_score_corrected")
        & (df_paper["Features"] == "eGeMAPSv02")
        & (df_paper["Survey"].str.contains("noisy"))
    )
    matches = df_paper[mask]
    if len(matches) == 0:
        raise RuntimeError(
            "Target configuration not found in paper CSV. "
            "Expected: WHO-5, eGeMAPSv02, noisy."
        )
    row = matches.iloc[0]

    print(f"Target:   {row['Target']}")
    print(f"Features: {row['Features']}")
    print(f"Task:     {row['Task']}")
    print(f"Survey:   {row['Survey']}")
    print()

    # ===== Load segment-level predictions =====
    data_path = row["path"].replace(
        "models/results-compiled.yaml",
        "data/df_results_test.parquet.zstd",
    )
    full_path = path_results_root + data_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Prediction file not found: {full_path}")

    df_pred = pd.read_parquet(full_path)
    print(f"Loaded {len(df_pred)} segments")

    # ===== Aggregate to session level =====
    target_col = "who_5_percentage_score_corrected"
    df_sess = (
        df_pred.groupby(["participant_code", "session"])
        .agg(
            target_value=(target_col, "first"),
            predictions=("predictions", "mean"),
        )
        .reset_index()
    )
    n_sessions = len(df_sess)
    n_participants = df_sess["participant_code"].nunique()
    print(f"Aggregated to {n_sessions} sessions " f"({n_participants} participants)")

    # ===== Verify CCC =====
    ccc_val = ccc_metric(
        df_sess["target_value"].values,
        df_sess["predictions"].values,
    )
    print(f"Session-level CCC = {ccc_val:.6f}")

    # Cross-check against expected value
    expected_ccc = 0.350369
    if abs(ccc_val - expected_ccc) > 1e-4:
        print(
            f"WARNING: CCC {ccc_val:.6f} differs from expected " f"{expected_ccc:.6f}"
        )
    else:
        print(f"CCC matches expected value ({expected_ccc:.6f})")

    # Also verify model type from path
    model_path = row["path"]
    if "SVR" in model_path:
        print("Model type: SVR (confirmed from path)")
    else:
        print(f"WARNING: Expected SVR but path contains: {model_path}")

    print()

    # ===== Create figure =====
    configure_plot_style()

    # Single-column width: ~84 mm ~ 3.31 in; using 3.5 in for margins
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    sns.regplot(
        x=df_sess["predictions"].values,
        y=df_sess["target_value"].values,
        ax=ax,
        scatter_kws={"alpha": 0.5, "s": 15, "edgecolor": "none"},
        line_kws={"linewidth": 1.5},
        ci=95,
        color="#4C72B0",
    )

    ax.set_xlim(0.570, 0.96)
    ax.set_ylim(0.3, 1.03)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    ax.set_title("Regression Plot")

    sns.despine(ax=ax)
    plt.tight_layout()

    # ===== Save in three formats =====
    basename = (
        "regression_who_5_percentage_score_corrected" "_noisy_eGeMAPSv02-publication"
    )

    for cur_ext in ["svg", "png", "pdf"]:
        cur_plot_path = os.path.join(output_dir, f"{basename}.{cur_ext}")
        plt.savefig(cur_plot_path, format=cur_ext, bbox_inches="tight")
        print(f"Saved: {cur_plot_path}")

        # ===== Verify PDF font embedding =====
        if cur_ext == "pdf":
            with open(cur_plot_path, "rb") as f:
                pdf_bytes = f.read()
            has_truetype = b"/Type1" not in pdf_bytes or b"TrueType" in pdf_bytes
            print(
                f"Font embedding: "
                f"{'OK (TrueType)' if has_truetype else 'WARNING: may contain Type1'}"
            )

    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
