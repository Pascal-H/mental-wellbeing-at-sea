"""
Retrospective Label Validation (Assessment-Only Session Comparison)
====================================================================

Validates that retrospectively assigning weekly survey scores to
recordings from the preceding week does not inflate predictive
performance.

Background
----------
Self-reported wellbeing scores from weekly surveys (WHO-5, PSS-10,
PHQ-8) were retrospectively assigned to all recording sessions
conducted within the preceding week. This means that daily sessions
carry labels from the next weekly assessment, even though no
questionnaire was completed during the daily session.

To verify this approach, we restrict the evaluation to
"assessment-only" sessions - sessions where a questionnaire was
actually completed (survey types: weekly_01, weekly_02, baseline,
final) - and compare against the session-level CCC obtained from
all sessions (including daily sessions with retrospective labels).

As described in the publication:

    "To verify that this retrospective assignment does not inflate
     predictive performance, we conducted a supplementary analysis
     restricted to assessment-only sessions - i.e., sessions during
     which a survey was actually completed - and compared their
     session-level CCC against the values obtained from all sessions."

Session-level aggregation
-------------------------
For each (participant, session) pair, segment-level predictions are
averaged and the target label is taken (identical across segments
within a session).

Applicable targets
------------------
Only the three retrospective targets are analysed:

- WHO-5 (``who_5_percentage_score_corrected``)
- PSS-10 (``pss_10_total_score``)
- PHQ-8 (``phq_8_total_score``)

Daily self-report targets (stress_current, stress_work_tasks) are not
retrospective and are excluded from this analysis.

CCC Computation
---------------
- CCC: ``audbenchmark.metric.concordance_cc`` (population std, ddof=0)
- 95% CI: ``confidence_intervals.evaluate_with_conf_int`` with
  participant-clustered bootstrap (num_bootstraps=1000, alpha=5)

Input
-----
results/mwas/composed/compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv
Metadata CSV with survey type information (path_meta_csv below).

Output (saved to results/mwas/composed/session_level_analysis/)
------
retrospective_label_validation_results.csv
    Per-configuration CCC for all sessions vs assessment-only sessions.
retrospective_label_validation.txt
    Human-readable summary of the comparison.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import audbenchmark
from confidence_intervals import evaluate_with_conf_int


# CCC metric - aligned with the paper (population std, ddof=0)
ccc_metric = audbenchmark.metric.concordance_cc

# Collects per-configuration analysis output for text file
_analysis_lines = []


def log(msg=""):
    """Print to stdout and collect for text file output."""
    print(msg)
    _analysis_lines.append(msg)


# Assessment session survey types (questionnaire actually completed)
ASSESSMENT_SURVEY_TYPES = {"weekly_01", "weekly_02", "baseline", "final"}

# Retrospective targets (weekly questionnaires assigned retroactively)
RETROSPECTIVE_TARGETS = [
    "who_5_percentage_score_corrected",
    "phq_8_total_score",
    "pss_10_total_score",
]


def compute_ccc_with_ci(
    labels, predictions, participant_codes, num_bootstraps=1000, alpha=5
):
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
            "ccc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "significant": False,
            "n": n,
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
        # "Singificant" in this context means if the lower confidence interval is > 0
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


def resolve_survey_type(df_pred, df_meta):
    """Match prediction rows to their survey types using metadata.

    Handles type mismatches in session identifiers by trying multiple
    type coercions until a successful match is found.

    Args:
        df_pred: DataFrame of segment-level predictions with columns
            [participant_code, session, ...]
        df_meta: metadata DataFrame with columns
            [participant_code, session, survey]

    Returns:
        Copy of df_pred with an added ``survey_type`` column.
    """
    df_pred = df_pred.copy()
    # Base lookup: MultiIndex Series keyed by (participant_code, session)
    meta_lookup = df_meta.groupby(["participant_code", "session"])["survey"].first()
    # Vectorized initial match without type coercion
    idx = pd.MultiIndex.from_arrays(
        [df_pred["participant_code"], df_pred["session"]],
        names=["participant_code", "session"],
    )
    mapped = meta_lookup.reindex(idx)
    df_pred["survey_type"] = mapped.values
    df_pred["survey_type"] = df_pred["survey_type"].fillna("unknown")
    # If more than half of the rows are still unknown, try coercing session types
    if (df_pred["survey_type"] == "unknown").sum() > len(df_pred) * 0.5:
        for cast_fn in [str, int, float]:
            try:
                # Build a coerced lookup keyed by (participant_code, cast_fn(session))
                coerced_dict = {
                    (pc, cast_fn(s)): sv for (pc, s), sv in meta_lookup.items()
                }
                meta_coerced = pd.Series(coerced_dict)
                meta_coerced.index = pd.MultiIndex.from_tuples(
                    meta_coerced.index, names=["participant_code", "session"]
                )
                # Coerce df_pred["session"] with the same cast_fn
                coerced_session = df_pred["session"].map(cast_fn)
                idx_coerced = pd.MultiIndex.from_arrays(
                    [df_pred["participant_code"], coerced_session],
                    names=["participant_code", "session"],
                )
                mapped_coerced = meta_coerced.reindex(idx_coerced)
                df_pred["survey_type"] = mapped_coerced.values
                df_pred["survey_type"] = df_pred["survey_type"].fillna("unknown")
                if (df_pred["survey_type"] == "unknown").sum() < len(df_pred) * 0.5:
                    break
            except (ValueError, TypeError):
                # If casting fails for any value, skip this cast_fn entirely
                continue

    return df_pred


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
    path_meta_csv = (
        "/data/share/aisoundlab-mental_wellbeing_at_sea/"
        "data_mwas_processed-final_data/final_data-df_files.csv"
    )
    output_dir = os.path.join(
        project_root,
        "results",
        "mwas",
        "composed",
        "session_level_analysis",
    )
    os.makedirs(output_dir, exist_ok=True)

    # ===== Load data =====
    df_paper = pd.read_csv(path_paper_csv)
    df_meta = pd.read_csv(path_meta_csv)

    # Filter to retrospective targets only
    df_retro_configs = df_paper[df_paper["Target"].isin(RETROSPECTIVE_TARGETS)]
    log(f"Retrospective configurations: " f"{len(df_retro_configs)} / {len(df_paper)}")
    log()

    # ===== Process each retrospective configuration =====
    all_results = []

    for idx, row in df_retro_configs.iterrows():
        target = row["Target"]
        features = row["Features"]
        task = row["Task"]
        survey_label = row["Survey"]
        audio_quality = "noisy" if "noisy" in survey_label.lower() else "denoised"

        # Load predictions and resolve survey types
        data_path = row["path"].replace(
            "models/results-compiled.yaml",
            "data/df_results_test.parquet.zstd",
        )
        full_path = path_results_root + data_path
        if not os.path.exists(full_path):
            warnings.warn(f"Prediction file not found: {full_path}")
            continue

        df_pred = pd.read_parquet(full_path)
        df_pred = resolve_survey_type(df_pred, df_meta)

        log(f"{'=' * 80}")
        log(f"{target} | {features[:30]}... | {audio_quality}")

        # ---- Session-level CCC on ALL sessions ----
        df_sess_all = aggregate_to_sessions(df_pred, target)
        m_all = compute_ccc_with_ci(
            df_sess_all["target_value"].values,
            df_sess_all["predictions"].values,
            df_sess_all["participant_code"].values,
        )
        log(
            f"  All sessions:       "
            f"CCC={m_all['ccc']:.6f}  "
            f"CI=[{m_all['ci_low']:.6f}, {m_all['ci_high']:.6f}]  "
            f"N={m_all['n']}"
        )

        # ---- Session-level CCC on ASSESSMENT-ONLY sessions ----
        df_assess = df_pred[df_pred["survey_type"].isin(ASSESSMENT_SURVEY_TYPES)]
        m_assess = {
            "ccc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "significant": False,
            "n": 0,
        }
        n_assess_sessions = 0

        if len(df_assess) > 0:
            df_sess_assess = aggregate_to_sessions(df_assess, target)
            n_assess_sessions = len(df_sess_assess)
            n_assess_participants = df_sess_assess["participant_code"].nunique()
            if n_assess_sessions >= 2:
                m_assess = compute_ccc_with_ci(
                    df_sess_assess["target_value"].values,
                    df_sess_assess["predictions"].values,
                    df_sess_assess["participant_code"].values,
                )
            log(
                f"  Assessment-only:    "
                f"CCC={m_assess['ccc']:.6f}  "
                f"CI=[{m_assess['ci_low']:.6f}, "
                f"{m_assess['ci_high']:.6f}]  "
                f"N={n_assess_sessions} sessions, "
                f"{n_assess_participants} participants"
            )
        else:
            log("  Assessment-only:    No assessment sessions found")

        delta = (
            m_assess["ccc"] - m_all["ccc"] if not np.isnan(m_assess["ccc"]) else np.nan
        )
        if not np.isnan(delta):
            log(f"  Delta (assess - all): {delta:+.6f}")

        result = {
            "target": target,
            "features": features,
            "task": task,
            "audio_quality": audio_quality,
            "survey_label": survey_label,
            "path": row["path"],
            # All sessions
            "ccc_session_all": m_all["ccc"],
            "ccc_session_all_ci_low": m_all["ci_low"],
            "ccc_session_all_ci_high": m_all["ci_high"],
            "ccc_session_all_significant": m_all["significant"],
            "n_sessions_all": m_all["n"],
            # Assessment-only sessions
            "ccc_session_assess": m_assess["ccc"],
            "ccc_session_assess_ci_low": m_assess["ci_low"],
            "ccc_session_assess_ci_high": m_assess["ci_high"],
            "ccc_session_assess_significant": m_assess["significant"],
            "n_sessions_assess": n_assess_sessions,
            # Delta
            "delta_assess_minus_all": delta,
        }
        all_results.append(result)

    # ===== Save results CSV =====
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "retrospective_label_validation_results.csv")
    df_results.to_csv(csv_path, index=False)
    log(f"\nResults CSV: {csv_path}")

    # ===== Build summary (requires all results) =====
    mean_all = df_results["ccc_session_all"].mean()
    mean_assess = df_results["ccc_session_assess"].mean()
    deltas = df_results["delta_assess_minus_all"].dropna()
    delta_mean = deltas.mean()
    delta_min = deltas.min()
    delta_max = deltas.max()
    n_higher = (deltas > 0).sum()
    n_all_sig = df_results["ccc_session_all_significant"].sum()
    n_assess_sig = df_results["ccc_session_assess_significant"].sum()

    summary_lines = []
    summary_lines.append("Retrospective Label Validation: Session-Level Analysis")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    summary_lines.append("PURPOSE")
    summary_lines.append("-" * 60)
    summary_lines.append("Compare session-level CCC computed on ALL sessions")
    summary_lines.append(
        "(including those with retrospectively assigned weekly labels)"
    )
    summary_lines.append("against CCC computed only on ASSESSMENT-ONLY sessions")
    summary_lines.append("(where a questionnaire was actually completed).")
    summary_lines.append("")
    summary_lines.append(
        "Assessment session types: weekly_01, weekly_02, " "baseline, final"
    )
    summary_lines.append("Retrospective targets: WHO-5, PSS-10, PHQ-8")
    summary_lines.append(
        "Daily targets (stress_current, stress_work_tasks) are "
        "not retrospective and are excluded."
    )
    summary_lines.append("")

    summary_lines.append("OVERALL SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"  Configurations:            {len(df_results)}")
    summary_lines.append(f"  Mean CCC (all sessions):   {mean_all:.4f}")
    summary_lines.append(f"  Mean CCC (assess-only):    {mean_assess:.4f}")
    summary_lines.append(f"  Mean delta (assess - all): {delta_mean:+.4f}")
    summary_lines.append(
        f"  Delta range:               " f"[{delta_min:+.4f}, {delta_max:+.4f}]"
    )
    summary_lines.append(f"  Assessment > All:          " f"{n_higher}/{len(deltas)}")
    summary_lines.append(f"  Significant (all):         {n_all_sig}/{len(df_results)}")
    summary_lines.append(
        f"  Significant (assess-only): " f"{n_assess_sig}/{len(df_results)}"
    )
    summary_lines.append("")

    # Per-target breakdown
    for tgt in RETROSPECTIVE_TARGETS:
        df_t = df_results[df_results["target"] == tgt]
        if len(df_t) == 0:
            continue

        summary_lines.append(f"TARGET: {tgt}")
        summary_lines.append("-" * 60)
        summary_lines.append(
            f"  Mean CCC (all):    " f"{df_t['ccc_session_all'].mean():.4f}"
        )
        summary_lines.append(
            f"  Mean CCC (assess): " f"{df_t['ccc_session_assess'].mean():.4f}"
        )
        summary_lines.append(
            f"  Mean delta:        " f"{df_t['delta_assess_minus_all'].mean():+.4f}"
        )
        summary_lines.append("")

    # ===== Write text file: summary first, then detailed analysis =====
    txt_path = os.path.join(output_dir, "retrospective_label_validation.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines))
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("DETAILED PER-CONFIGURATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(_analysis_lines))
    print(f"Text log: {txt_path}")

    # ===== Console summary =====
    log(f"\n{'=' * 80}")
    log("SUMMARY")
    log(f"{'=' * 80}")
    log(f"  Mean CCC (all sessions):    {mean_all:.4f}")
    log(f"  Mean CCC (assessment-only): {mean_assess:.4f}")
    log(f"  Mean delta:                 {delta_mean:+.4f}")
    log(f"  Range:                      " f"[{delta_min:+.4f}, {delta_max:+.4f}]")
    log(f"  Assessment > All:           " f"{n_higher}/{len(deltas)}")
    log("\nDone.")


if __name__ == "__main__":
    main()
