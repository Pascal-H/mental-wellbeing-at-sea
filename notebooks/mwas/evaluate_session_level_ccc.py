"""
Session-Level CCC Evaluation
=============================

Aggregates segment-level predictions to session level and computes the
Concordance Correlation Coefficient (CCC) with 95% bootstrap confidence
intervals for all 20 best-performing models' configurations.

Background
----------
The modelling pipeline produces predictions at the VAD-segment level
(0.76-6.0s segments from voice-activity detection). Within a recording
session, multiple segments share the same target label. To obtain one
prediction per session, segment-level predictions are averaged within
each (participant, session) pair, while the target value (identical
across segments) is taken as-is.

As described in the publication:

    "Those segment-level predictions were then averaged within each
     recording session to yield a single prediction per session, as
     the recording session represents a generalisable unit of
     observation and session-level aggregation eliminates
     within-session dependencies among predictions derived from the
     same recording."

CCC Computation
---------------
Aligned with the paper's methodology:

- CCC: ``audbenchmark.metric.concordance_cc`` (population std, ddof=0)
- 95% CI: ``confidence_intervals.evaluate_with_conf_int`` with
  participant-clustered bootstrap (num_bootstraps=1,000, alpha=5)

Validation
----------
- Segment-level CCC is validated against the paper's ``ccc_conf_mean``.
- Session-level CCC is validated against the existing
  ``concordance_cc-test-agg-average`` column (computed in
  ``notebooks/mwas/vad_average_regression.ipynb``).

Input
-----
results/mwas/composed/compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv

Output (saved to results/mwas/composed/session_level_analysis/)
------
session_level_results.csv
    Detailed results for all 20 configurations at segment and session level.
compiled-session_level-paper_configs.csv
    Publication-ready CSV with session-level CCC and bootstrap CIs.
session_level_results.txt
    Human-readable text log of the full evaluation output.
"""

import os
import warnings

import numpy as np
import pandas as pd

import audbenchmark
from confidence_intervals import evaluate_with_conf_int


# CCC metric - aligned with the paper (population std, ddof=0)
ccc_metric = audbenchmark.metric.concordance_cc

# Collects all log output for writing to text file
_output_lines = []


def log(msg=""):
    """Print to stdout and collect for text file output."""
    print(msg)
    _output_lines.append(msg)


def compute_ccc_with_ci(labels, predictions, participant_codes,
                        num_bootstraps=1000, alpha=5):
    """Compute CCC with 95% CI via participant-clustered bootstrap.

    Args:
        labels: array of ground truth values
        predictions: array of model predictions
        participant_codes: array of participant identifiers (bootstrap
            clusters)
        num_bootstraps: number of bootstrap iterations
        alpha: significance level for CI (5 => 95% CI)

    Returns:
        dict with keys: ccc, ci_low, ci_high, significant, n
    """
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    n = len(labels)

    if n < 2:
        return {
            "ccc": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "significant": False, "n": n,
        }

    tpl = evaluate_with_conf_int(
        samples=predictions,
        metric=ccc_metric,
        labels=labels,
        conditions=np.asarray(participant_codes),
        num_bootstraps=num_bootstraps,
        alpha=alpha,
    )

    return {
        "ccc": tpl[0],
        "ci_low": tpl[1][0],
        "ci_high": tpl[1][1],
        "significant": tpl[1][0] > 0,
        "n": n,
    }


def aggregate_to_sessions(df, target):
    """Aggregate segment-level predictions to session level.

    For each (participant, session) pair, predictions are averaged
    and the target value is taken (identical for all segments in a
    session).

    Args:
        df: DataFrame with columns
            [participant_code, session, <target>, predictions]
        target: name of the target column

    Returns:
        DataFrame with columns
            [participant_code, session, target_value, predictions]
    """
    return (
        df.groupby(["participant_code", "session"])
        .agg(
            target_value=(target, "first"),
            predictions=("predictions", "mean"),
        )
        .reset_index()
    )


def main():
    # ===== Paths =====
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    path_results_root = os.path.join(
        project_root, "results", "mwas", "modelling"
    )
    path_paper_csv = os.path.join(
        project_root, "results", "mwas", "composed",
        "compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv",
    )
    output_dir = os.path.join(
        project_root, "results", "mwas", "composed",
        "session_level_analysis",
    )
    os.makedirs(output_dir, exist_ok=True)

    # ===== Load paper results index =====
    df_paper = pd.read_csv(path_paper_csv)
    has_agg_avg = "concordance_cc-test-agg-average" in df_paper.columns

    log(f"Paper configurations: {len(df_paper)}")
    if has_agg_avg:
        log(
            "Existing session-level CCC "
            "(concordance_cc-test-agg-average) found "
            "- will validate against it."
        )
    log()

    # ===== Process each configuration =====
    all_results = []

    for idx, row in df_paper.iterrows():
        target = row["Target"]
        features = row["Features"]
        task = row["Task"]
        survey_label = row["Survey"]
        audio_quality = (
            "noisy" if "noisy" in survey_label.lower() else "denoised"
        )

        # Load segment-level predictions
        data_path = row["path"].replace(
            "models/results-compiled.yaml",
            "data/df_results_test.parquet.zstd",
        )
        full_path = path_results_root + data_path
        if not os.path.exists(full_path):
            warnings.warn(f"Prediction file not found: {full_path}")
            continue

        df_pred = pd.read_parquet(full_path)

        log(f"{'=' * 80}")
        log(
            f"Config {idx + 1}/{len(df_paper)}: {target} | "
            f"{features[:30]}... | {task} | {audio_quality}"
        )
        log(
            f"  Segments: {len(df_pred)}, "
            f"Participants: {df_pred['participant_code'].nunique()}"
        )

        # ---- 1. Segment-level CCC + CI (validate against paper) ----
        m_seg = compute_ccc_with_ci(
            df_pred[target].values,
            df_pred["predictions"].values,
            df_pred["participant_code"].values,
        )
        paper_ccc = row["ccc_conf_mean"]
        seg_match = abs(m_seg["ccc"] - paper_ccc) < 1e-6
        log(
            f"  Segment:  CCC={m_seg['ccc']:.6f}  "
            f"CI=[{m_seg['ci_low']:.6f}, {m_seg['ci_high']:.6f}]  "
            f"N={m_seg['n']}  "
            f"Paper={'MATCH' if seg_match else 'MISMATCH'}"
        )

        # ---- 2. Session-level CCC + CI ----
        df_sess = aggregate_to_sessions(df_pred, target)
        m_sess = compute_ccc_with_ci(
            df_sess["target_value"].values,
            df_sess["predictions"].values,
            df_sess["participant_code"].values,
        )

        # Validate against existing concordance_cc-test-agg-average
        sess_match = True
        if has_agg_avg and not pd.isna(
            row["concordance_cc-test-agg-average"]
        ):
            existing_agg = row["concordance_cc-test-agg-average"]
            sess_match = abs(m_sess["ccc"] - existing_agg) < 1e-6
            log(
                f"  Session:  CCC={m_sess['ccc']:.6f}  "
                f"CI=[{m_sess['ci_low']:.6f}, {m_sess['ci_high']:.6f}]  "
                f"N={m_sess['n']}  "
                f"vs existing={existing_agg:.6f} "
                f"{'MATCH' if sess_match else 'MISMATCH'}"
            )
        else:
            log(
                f"  Session:  CCC={m_sess['ccc']:.6f}  "
                f"CI=[{m_sess['ci_low']:.6f}, {m_sess['ci_high']:.6f}]  "
                f"N={m_sess['n']}"
            )

        delta = m_sess["ccc"] - m_seg["ccc"]
        log(f"  Delta session-segment: {delta:+.6f}")

        result = {
            "target": target,
            "features": features,
            "task": task,
            "audio_quality": audio_quality,
            "survey_label": survey_label,
            "path": row["path"],
            # Segment-level
            "ccc_segment": m_seg["ccc"],
            "ccc_segment_ci_low": m_seg["ci_low"],
            "ccc_segment_ci_high": m_seg["ci_high"],
            "ccc_segment_significant": m_seg["significant"],
            "n_segments": m_seg["n"],
            "paper_ccc_conf_mean": paper_ccc,
            "segment_matches_paper": seg_match,
            # Session-level
            "ccc_session": m_sess["ccc"],
            "ccc_session_ci_low": m_sess["ci_low"],
            "ccc_session_ci_high": m_sess["ci_high"],
            "ccc_session_significant": m_sess["significant"],
            "n_sessions": m_sess["n"],
            "existing_agg_average": (
                row["concordance_cc-test-agg-average"]
                if has_agg_avg else np.nan
            ),
            "session_matches_existing": sess_match,
            "delta_session_segment": delta,
        }
        all_results.append(result)

    # ===== Save detailed results =====
    df_results = pd.DataFrame(all_results)
    detail_csv = os.path.join(output_dir, "session_level_results.csv")
    df_results.to_csv(detail_csv, index=False)
    log(f"\nDetailed results: {detail_csv}")

    # ===== Save publication-ready CSV =====
    # Format: ccc_conf_mean = session-level CCC (with bootstrap CIs)
    pub_data = []
    for _, r in df_results.iterrows():
        pub_data.append({
            "ccc_conf_mean": r["ccc_session"],
            "ccc_conf_low": r["ccc_session_ci_low"],
            "ccc_conf_high": r["ccc_session_ci_high"],
            "lower_bound_larger_null": r["ccc_session_significant"],
            "task": r["task"],
            "Features": r["features"],
            "ccc_segment_mean": r["ccc_segment"],
            "ccc_segment_low": r["ccc_segment_ci_low"],
            "ccc_segment_high": r["ccc_segment_ci_high"],
            "concordance_cc-test-agg-average": r["ccc_session"],
            "path": r["path"],
            "Target": r["target"],
            "Task": r["task"],
            "Survey": r["survey_label"],
        })

    df_pub = pd.DataFrame(pub_data)
    pub_csv = os.path.join(
        output_dir, "compiled-session_level-paper_configs.csv"
    )
    df_pub.to_csv(pub_csv, index=False)
    log(f"Publication CSV: {pub_csv}")

    # ===== Validation summary =====
    log(f"\n{'=' * 80}")
    log("VALIDATION")
    log(f"{'=' * 80}")

    n_seg_match = df_results["segment_matches_paper"].sum()
    log(
        f"  Segment CCC vs paper: "
        f"{n_seg_match}/{len(df_results)} match"
    )
    if has_agg_avg:
        n_sess_match = df_results["session_matches_existing"].sum()
        log(
            f"  Session CCC vs existing agg-average: "
            f"{n_sess_match}/{len(df_results)} match"
        )

    # ===== Analysis summary =====
    log(f"\n{'=' * 80}")
    log("SUMMARY: SEGMENT vs SESSION LEVEL CCC")
    log(f"{'=' * 80}\n")

    seg_sig = df_results["ccc_segment_significant"].sum()
    sess_sig = df_results["ccc_session_significant"].sum()
    n_total = len(df_results)

    log(
        f"  Segment: Mean CCC = "
        f"{df_results['ccc_segment'].mean():.4f}  "
        f"(SD: {df_results['ccc_segment'].std():.4f})  "
        f"Significant: {seg_sig}/{n_total}"
    )
    log(
        f"  Session: Mean CCC = "
        f"{df_results['ccc_session'].mean():.4f}  "
        f"(SD: {df_results['ccc_session'].std():.4f})  "
        f"Significant: {sess_sig}/{n_total}"
    )

    delta_col = df_results["delta_session_segment"]
    log(
        f"\n  Delta session-segment: "
        f"mean={delta_col.mean():+.4f}  "
        f"(range: {delta_col.min():+.4f} to {delta_col.max():+.4f})"
    )
    log(
        f"    Session > Segment: "
        f"{(delta_col > 0).sum()}/{n_total} configs"
    )

    log(f"\n--- Per target ---")
    for tgt in df_results["target"].unique():
        df_t = df_results[df_results["target"] == tgt]
        log(f"\n  {tgt}:")
        log(
            f"    Segment: {df_t['ccc_segment'].mean():.4f}  "
            f"(sig: {df_t['ccc_segment_significant'].sum()}"
            f"/{len(df_t)})"
        )
        log(
            f"    Session: {df_t['ccc_session'].mean():.4f}  "
            f"(sig: {df_t['ccc_session_significant'].sum()}"
            f"/{len(df_t)})  "
            f"delta: {df_t['delta_session_segment'].mean():+.4f}"
        )
        for _, r in df_t.iterrows():
            sig_seg = "*" if r["ccc_segment_significant"] else " "
            sig_sess = "*" if r["ccc_session_significant"] else " "
            log(
                f"      [{r['audio_quality']:8s} | "
                f"{r['features'][:15]:15s}]  "
                f"seg={r['ccc_segment']:.4f}{sig_seg}  "
                f"sess={r['ccc_session']:.4f}{sig_sess}  "
                f"CI=[{r['ccc_session_ci_low']:.4f}, "
                f"{r['ccc_session_ci_high']:.4f}]"
            )

    log(f"\nAll output files saved to: {output_dir}")
    log("Done.")

    # ===== Save text log =====
    txt_path = os.path.join(output_dir, "session_level_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(_output_lines))
    print(f"Text log: {txt_path}")


if __name__ == "__main__":
    main()
