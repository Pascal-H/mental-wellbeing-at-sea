"""
Confounder analysis: Wind speed, SNR, and denoising effects on emotion predictions.

Evaluates two potential confounders for passively recorded speech emotion analysis:
  1. Wind-SNR relationship: whether stronger winds degrade audio quality (lower SNR)
  2. Denoising effect: whether noise reduction systematically shifts emotion estimates
     and whether that shift depends on wind speed or SNR

Saves structured results as YAML and a verbose text report with conditional
interpretation.

Addresses reviewer comments:
  Reviewer 1, Point 4 -- Methodological considerations on wind, noise, and
      denoising confounding.
  Reviewer 2, Point 5 -- Confounding factors in passive data analysis.

References:
  Leem, S.-G., Fulford, D., Onnela, J.-P., Gard, D., & Busso, C. (2023).
      Selective speech enhancement for recognition of whispered and
      lombard speech. IEEE/ACM Trans. Audio, Speech, and Lang. Process., 31,
      1620-1634.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_BLOCK = "3h"
BRIDGE_MICS = ("M1", "M2", "M3", "M6")

# Paths -- adjust to local setup
DIR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DIR_EVALUATED = os.path.join(DIR_ROOT, "data", "evaluated")
DIR_OUT = os.path.join(DIR_EVALUATED, "confounder-noise_denoising")
PATH_EVENTS_YAML = os.path.join(
    DIR_ROOT, "src", "evaluate_results", "evaluate_time_course_events.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _cohens_d_label(d):
    """Return a label for paired Cohen's d."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def filter_duration(df, min_duration=3):
    """Keep only segments whose duration >= min_duration seconds."""
    durations = (
        df.index.get_level_values("end") - df.index.get_level_values("start")
    ).total_seconds()
    return df[durations >= min_duration]


def assign_location(timestamp, event_timings):
    """
    Assign operational context to a timestamp.

    Returns one of: 'operations', 'sea', 'port', 'unknown'.
    Loading / discharge ("operations") takes precedence over the land/sea flag.
    """
    ts = pd.Timestamp(timestamp)
    for op in ("loading", "discharge"):
        for entry in event_timings.get("loading_and_discharge", {}).get(op, []):
            if pd.Timestamp(entry["start"]) <= ts <= pd.Timestamp(entry["end"]):
                return "operations"
    for entry in event_timings.get("land_and_sea", {}).get("sea", []):
        if pd.Timestamp(entry["start"]) <= ts <= pd.Timestamp(entry["end"]):
            return "sea"
    for entry in event_timings.get("land_and_sea", {}).get("land", []):
        if pd.Timestamp(entry["start"]) <= ts <= pd.Timestamp(entry["end"]):
            return "port"
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load emotion (denoised + non-denoised), SNR, wind, and event data."""
    print("Loading data ...")

    df_emotion_den = pd.read_pickle(
        os.path.join(DIR_EVALUATED, "df_all_predictions_emotion.pkl")
    )
    print(f"  Denoised emotion predictions: {len(df_emotion_den):,} segments")

    df_emotion_noden = pd.read_pickle(
        os.path.join(
            DIR_EVALUATED,
            "df_all_predictions_emotion-no_denoising-auvad-mobilenet-no_transcripts.pkl",
        )
    )
    print(f"  Non-denoised emotion predictions: {len(df_emotion_noden):,} segments")

    df_snr = pd.read_pickle(os.path.join(DIR_EVALUATED, "df_all_predictions_snr.pkl"))
    print(f"  SNR predictions: {len(df_snr):,} segments")

    df_wind = pd.read_csv(
        os.path.join(DIR_EVALUATED, "wind_speed", "true_wind_speed.csv")
    )
    df_wind["time"] = pd.to_datetime(df_wind["time"])
    print(f"  Wind speed measurements: {len(df_wind):,} rows")

    with open(PATH_EVENTS_YAML, "r") as fh:
        event_timings = yaml.safe_load(fh)

    return df_emotion_den, df_emotion_noden, df_snr, df_wind, event_timings


# ---------------------------------------------------------------------------
# Part 1 -- Wind speed vs SNR
# ---------------------------------------------------------------------------


def analyse_wind_snr(df_snr, df_wind, event_timings):
    """Correlate wind speed with bridge-microphone SNR in 3 h blocks."""
    print("\n" + "=" * 72)
    print("PART 1: WIND SPEED vs SNR")
    print("=" * 72)

    df_snr_3s = filter_duration(df_snr, min_duration=3)
    df_snr_3s = df_snr_3s.copy()
    df_snr_3s["time"] = pd.to_datetime(df_snr_3s["time"])
    df_snr_3s["time_block"] = df_snr_3s["time"].dt.floor(TIME_BLOCK)
    print(f"SNR segments >= 3 s: {len(df_snr_3s):,}")

    # Aggregate bridge microphones
    bridge = df_snr_3s[df_snr_3s["microphone"].isin(BRIDGE_MICS)]
    snr_agg = (
        bridge.groupby("time_block")["snr"].agg(["mean", "std", "count"]).reset_index()
    )
    snr_agg.columns = ["time_block", "snr_mean", "snr_std", "snr_count"]
    print(f"{TIME_BLOCK} blocks with SNR data: {len(snr_agg)}")

    # Aggregate wind speed
    wind = df_wind.copy()
    wind["time_block"] = wind["time"].dt.floor(TIME_BLOCK)
    wind_agg = (
        wind.groupby("time_block")["true_wind_speed_gps"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    wind_agg.columns = ["time_block", "wind_mean", "wind_std", "wind_count"]
    print(f"{TIME_BLOCK} blocks with wind data: {len(wind_agg)}")

    # Merge
    merged = pd.merge(snr_agg, wind_agg, on="time_block", how="inner")
    merged = merged.dropna(subset=["wind_mean", "snr_mean"])
    merged["location"] = merged["time_block"].apply(
        lambda t: assign_location(t, event_timings)
    )
    print(f"Aligned {TIME_BLOCK} blocks: {len(merged)}")
    print(f"\nLocation distribution:\n{merged['location'].value_counts().to_string()}")

    # Overall correlation
    r_all, p_all = stats.pearsonr(merged["wind_mean"], merged["snr_mean"])
    r2_all = r_all**2
    n_all = len(merged)
    print(
        f"\nOverall: r = {r_all:.4f}, p = {p_all:.6f}, r^2 = {r2_all:.4f}, N = {n_all}"
    )
    if r_all < 0 and p_all < 0.05:
        print("  -> Higher wind speed associated with lower SNR (more ambient noise)")
    elif r_all > 0 and p_all < 0.05:
        print("  -> Higher wind speed associated with higher SNR")
    else:
        print("  -> No significant wind-SNR relationship")

    # Stratified
    strat = {}
    print("\nStratified by location:")
    for loc in ("sea", "port", "operations"):
        sub = merged[merged["location"] == loc]
        if len(sub) >= 5:
            r, p = stats.pearsonr(sub["wind_mean"], sub["snr_mean"])
            strat[loc] = {
                "r": float(r),
                "p": float(p),
                "n": int(len(sub)),
                "r_squared": float(r**2),
            }
            print(
                f"  {loc:12s}: r = {r:+.4f}, p = {p:.4f} {_sig_stars(p):>3s}, N = {len(sub)}"
            )
        else:
            strat[loc] = {"r": None, "p": None, "n": int(len(sub)), "r_squared": None}
            print(f"  {loc:12s}: insufficient data (N = {len(sub)})")

    results = {
        "overall_r": float(r_all),
        "overall_p": float(p_all),
        "overall_r_squared": float(r2_all),
        "n_blocks": int(n_all),
        "wind_mean_avg": float(merged["wind_mean"].mean()),
        "wind_mean_sd": float(merged["wind_mean"].std()),
        "snr_mean_avg": float(merged["snr_mean"].mean()),
        "snr_mean_sd": float(merged["snr_mean"].std()),
        "stratified": strat,
    }
    return merged, results


# ---------------------------------------------------------------------------
# Part 2 -- Denoising effect on emotion predictions
# ---------------------------------------------------------------------------


def analyse_denoising(df_den, df_noden, event_timings):
    """Compare denoised and non-denoised emotion predictions in paired 3 h blocks."""
    print("\n" + "=" * 72)
    print("PART 2: DENOISING EFFECT ON EMOTION PREDICTIONS")
    print("=" * 72)

    df_den_3s = filter_duration(df_den, min_duration=3).copy()
    df_noden_3s = filter_duration(df_noden, min_duration=3).copy()
    print(f"Denoised segments >= 3 s:     {len(df_den_3s):,}")
    print(f"Non-denoised segments >= 3 s: {len(df_noden_3s):,}")

    # Time blocks on bridge microphones
    df_den_3s["time"] = pd.to_datetime(df_den_3s["time"])
    df_noden_3s["time"] = pd.to_datetime(df_noden_3s["time"])
    den_br = df_den_3s[df_den_3s["microphone"].isin(BRIDGE_MICS)].copy()
    noden_br = df_noden_3s[df_noden_3s["microphone"].isin(BRIDGE_MICS)].copy()
    den_br["time_block"] = den_br["time"].dt.floor(TIME_BLOCK)
    noden_br["time_block"] = noden_br["time"].dt.floor(TIME_BLOCK)

    emo_cols = ["prediction_arousal", "prediction_dominance", "prediction_valence"]

    den_agg = den_br.groupby("time_block")[emo_cols].mean().reset_index()
    den_agg.columns = ["time_block"] + [f"{c}_den" for c in emo_cols]

    noden_agg = noden_br.groupby("time_block")[emo_cols].mean().reset_index()
    noden_agg.columns = ["time_block"] + [f"{c}_noden" for c in emo_cols]

    comp = pd.merge(den_agg, noden_agg, on="time_block", how="inner")
    print(f"Aligned {TIME_BLOCK} blocks: {len(comp)}")

    # Differences
    for c in emo_cols:
        comp[f"{c}_diff"] = comp[f"{c}_den"] - comp[f"{c}_noden"]
    comp["intensity_den"] = (
        comp["prediction_arousal_den"] + comp["prediction_dominance_den"]
    ) / 2
    comp["intensity_noden"] = (
        comp["prediction_arousal_noden"] + comp["prediction_dominance_noden"]
    ) / 2
    comp["intensity_diff"] = comp["intensity_den"] - comp["intensity_noden"]
    comp["location"] = comp["time_block"].apply(
        lambda t: assign_location(t, event_timings)
    )

    # Paired comparisons
    dim_labels = {
        "prediction_arousal": "arousal",
        "prediction_dominance": "dominance",
        "prediction_valence": "valence",
        "intensity": "intensity",
    }

    results = {}
    print("\nPaired comparisons (denoised vs non-denoised):")
    for col, label in dim_labels.items():
        if col == "intensity":
            d_vals = comp["intensity_den"].values
            nd_vals = comp["intensity_noden"].values
            diff = comp["intensity_diff"].values
        else:
            d_vals = comp[f"{col}_den"].values
            nd_vals = comp[f"{col}_noden"].values
            diff = comp[f"{col}_diff"].values

        t, p_t = stats.ttest_rel(d_vals, nd_vals)
        r, p_r = stats.pearsonr(d_vals, nd_vals)
        d_mean = float(np.mean(diff))
        d_sd = float(np.std(diff, ddof=1))
        cohens_d = d_mean / d_sd if d_sd > 0 else 0.0

        print(f"\n  {label.capitalize()}:")
        print(f"    Denoised:     M = {np.mean(d_vals):.4f}, SD = {np.std(d_vals):.4f}")
        print(
            f"    Non-denoised: M = {np.mean(nd_vals):.4f}, SD = {np.std(nd_vals):.4f}"
        )
        print(f"    Diff:         M = {d_mean:.4f}, SD = {d_sd:.4f}")
        print(f"    t = {t:.3f}, p = {p_t:.6f} {_sig_stars(p_t)}")
        print(f"    Cohen's d = {cohens_d:.4f} ({_cohens_d_label(cohens_d)})")
        print(f"    Correlation (den vs noden): r = {r:.4f}, p = {p_r:.6f}")

        results[label] = {
            "den_mean": float(np.mean(d_vals)),
            "den_sd": float(np.std(d_vals)),
            "noden_mean": float(np.mean(nd_vals)),
            "noden_sd": float(np.std(nd_vals)),
            "diff_mean": d_mean,
            "diff_sd": d_sd,
            "t_stat": float(t),
            "p_ttest": float(p_t),
            "cohens_d": float(cohens_d),
            "correlation_r": float(r),
            "correlation_p": float(p_r),
        }

    return comp, results


# ---------------------------------------------------------------------------
# Part 3 -- Wind / SNR dependent denoising bias
# ---------------------------------------------------------------------------


def analyse_denoising_bias(df_comp, df_wind_snr, event_timings):
    """
    Test whether the denoising effect on emotion predictions depends
    systematically on wind speed or SNR.
    """
    print("\n" + "=" * 72)
    print("PART 3: WIND- AND SNR-DEPENDENT DENOISING BIAS")
    print("=" * 72)

    diff_cols = [
        ("intensity_diff", "intensity"),
        ("prediction_arousal_diff", "arousal"),
        ("prediction_dominance_diff", "dominance"),
        ("prediction_valence_diff", "valence"),
    ]

    merged = pd.merge(
        df_comp[["time_block", "location"] + [c for c, _ in diff_cols]],
        df_wind_snr[["time_block", "wind_mean", "snr_mean"]],
        on="time_block",
        how="inner",
    )
    merged = merged.dropna(subset=["wind_mean"])
    n = len(merged)
    print(f"Aligned blocks for bias analysis: {n}")

    # Wind vs denoising effect
    results_wind = {}
    print("\nWind speed vs denoising effect:")
    for diff_col, label in diff_cols:
        valid = merged[diff_col].notna()
        r, p = stats.pearsonr(
            merged.loc[valid, "wind_mean"], merged.loc[valid, diff_col]
        )
        r2 = r**2
        print(
            f"  {label:12s} diff vs wind: r = {r:+.4f}, p = {p:.4f} {_sig_stars(p):>3s},"
            f" r^2 = {r2:.4f} ({r2*100:.1f}% variance)"
        )
        results_wind[label] = {
            "r": float(r),
            "p": float(p),
            "r_squared": float(r2),
            "n": int(valid.sum()),
        }

    # SNR vs denoising effect
    results_snr = {}
    print("\nSNR vs denoising effect:")
    valid_snr = merged["snr_mean"].notna()
    for diff_col, label in diff_cols:
        valid = valid_snr & merged[diff_col].notna()
        r, p = stats.pearsonr(
            merged.loc[valid, "snr_mean"], merged.loc[valid, diff_col]
        )
        r2 = r**2
        print(
            f"  {label:12s} diff vs SNR:  r = {r:+.4f}, p = {p:.4f} {_sig_stars(p):>3s},"
            f" r^2 = {r2:.4f} ({r2*100:.1f}% variance)"
        )
        results_snr[label] = {
            "r": float(r),
            "p": float(p),
            "r_squared": float(r2),
            "n": int(valid.sum()),
        }

    # Stratified: intensity diff vs wind by location
    strat = {}
    print("\nStratified (intensity diff vs wind):")
    for loc in ("sea", "port"):
        sub = merged[merged["location"] == loc]
        if len(sub) >= 5:
            r, p = stats.pearsonr(sub["wind_mean"], sub["intensity_diff"])
            strat[loc] = {"r": float(r), "p": float(p), "n": int(len(sub))}
            print(
                f"  {loc:12s}: r = {r:+.4f}, p = {p:.4f} {_sig_stars(p):>3s}, N = {len(sub)}"
            )
        else:
            strat[loc] = {"r": None, "p": None, "n": int(len(sub))}
            print(f"  {loc:12s}: insufficient data (N = {len(sub)})")

    results = {
        "n_blocks": int(n),
        "wind_vs_diff": results_wind,
        "snr_vs_diff": results_snr,
        "stratified_intensity_wind": strat,
    }
    return merged, results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def build_yaml(wind_snr, denoising, bias):
    """Assemble a YAML-friendly results dictionary."""
    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "confounder_noise_denoising_main.py",
            "description": (
                "Confounder analysis: wind-SNR relationship, denoising effect "
                "on emotion predictions, and wind/SNR-dependent denoising bias"
            ),
            "time_block": TIME_BLOCK,
        },
        "wind_snr": wind_snr,
        "denoising_effect": denoising,
        "denoising_bias": bias,
    }


def build_verbose(wind_snr, denoising, bias):
    """Build a verbose text report with conditional interpretation."""
    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("CONFOUNDER ANALYSIS REPORT")
    lines.append("Wind Speed, SNR, and Denoising Effects on Emotion Predictions")
    lines.append(sep)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Script: confounder_noise_denoising_main.py")
    lines.append(f"Time block: {TIME_BLOCK}")

    # --- Part 1 ----------------------------------------------------------
    lines.append("")
    lines.append(sep)
    lines.append("PART 1: WIND SPEED vs SNR")
    lines.append(sep)
    r = wind_snr["overall_r"]
    p = wind_snr["overall_p"]
    r2 = wind_snr["overall_r_squared"]
    n = wind_snr["n_blocks"]
    lines.append(f"  Pearson r:  {r:.4f}")
    lines.append(f"  p-value:    {p:.6f} {_sig_stars(p)}")
    lines.append(f"  r-squared:  {r2:.4f} ({r2*100:.1f}% variance explained)")
    lines.append(f"  N blocks:   {n}")
    lines.append(
        f"  Wind speed: M = {wind_snr['wind_mean_avg']:.2f}, SD = {wind_snr['wind_mean_sd']:.2f} m/s"
    )
    lines.append(
        f"  SNR:        M = {wind_snr['snr_mean_avg']:.2f}, SD = {wind_snr['snr_mean_sd']:.2f} dB"
    )
    lines.append("")

    # Conditional interpretation
    if p < 0.05 and r < 0:
        lines.append("  Interpretation:")
        lines.append(
            "    Higher wind speeds are associated with lower SNR (more ambient noise)."
        )
        lines.append(
            f"    Wind speed explains approximately {r2*100:.0f}% of SNR variance."
        )
    elif p < 0.05 and r > 0:
        lines.append("  Interpretation:")
        lines.append("    Higher wind speeds are associated with higher SNR.")
    else:
        lines.append("  Interpretation:")
        lines.append("    No significant relationship between wind speed and SNR.")

    lines.append("")
    lines.append("  Stratified by operational context:")
    for loc in ("sea", "port", "operations"):
        s = wind_snr["stratified"].get(loc, {})
        if s.get("r") is not None:
            r_loc = s["r"]
            p_loc = s["p"]
            n_loc = s["n"]
            r2_loc = s["r_squared"]
            lines.append(
                f"    {loc:12s}: r = {r_loc:+.4f}, p = {p_loc:.4f} {_sig_stars(p_loc):>3s},"
                f" r^2 = {r2_loc:.4f}, N = {n_loc}"
            )
        else:
            lines.append(f"    {loc:12s}: insufficient data (N = {s.get('n', 0)})")

    # Conditional interpretation of stratified results
    sea_s = wind_snr["stratified"].get("sea", {})
    port_s = wind_snr["stratified"].get("port", {})
    ops_s = wind_snr["stratified"].get("operations", {})
    lines.append("")
    if sea_s.get("r") is not None and sea_s["p"] < 0.05 and sea_s["r"] < 0:
        lines.append(
            f"    At sea: wind substantially degrades audio quality"
            f" (r = {sea_s['r']:.2f})."
        )
    if port_s.get("r") is not None and port_s["p"] >= 0.05:
        lines.append(
            "    In port: wind-SNR relationship not significant"
            " (structural shielding)."
        )
    if ops_s.get("r") is not None and ops_s["p"] < 0.05 and ops_s["r"] > 0:
        lines.append(
            f"    Operations: positive correlation (r = {ops_s['r']:.2f})"
            " suggests machinery noise dominates."
        )

    # --- Part 2 ----------------------------------------------------------
    lines.append("")
    lines.append(sep)
    lines.append("PART 2: DENOISING EFFECT ON EMOTION PREDICTIONS")
    lines.append(sep)

    for label in ("intensity", "arousal", "dominance", "valence"):
        d = denoising[label]
        lines.append("")
        lines.append(f"  {label.capitalize()}:")
        lines.append(
            f"    Denoised:     M = {d['den_mean']:.4f}, SD = {d['den_sd']:.4f}"
        )
        lines.append(
            f"    Non-denoised: M = {d['noden_mean']:.4f}, SD = {d['noden_sd']:.4f}"
        )
        lines.append(
            f"    Diff (den - noden): {d['diff_mean']:+.4f} (SD = {d['diff_sd']:.4f})"
        )
        lines.append(
            f"    Paired t-test: t = {d['t_stat']:.3f},"
            f" p = {d['p_ttest']:.6f} {_sig_stars(d['p_ttest'])}"
        )
        lines.append(
            f"    Cohen's d: {d['cohens_d']:.4f} ({_cohens_d_label(d['cohens_d'])})"
        )
        lines.append(
            f"    Correlation: r = {d['correlation_r']:.4f},"
            f" p = {d['correlation_p']:.6f}"
        )

    # Conditional interpretation
    lines.append("")
    lines.append("  Interpretation:")
    int_d = denoising["intensity"]
    if int_d["p_ttest"] < 0.05 and int_d["diff_mean"] > 0:
        lines.append("    Denoising slightly INCREASES predicted emotion intensity.")
    elif int_d["p_ttest"] < 0.05 and int_d["diff_mean"] < 0:
        lines.append("    Denoising slightly DECREASES predicted emotion intensity.")
    else:
        lines.append(
            "    No significant systematic difference between denoised and"
            " non-denoised predictions."
        )

    den_sd = denoising["arousal"]["den_sd"]
    noden_sd = denoising["arousal"]["noden_sd"]
    if den_sd < noden_sd:
        reduction_pct = (1 - den_sd / noden_sd) * 100
        lines.append(
            f"    Denoising reduces arousal prediction variance"
            f" (SD: {noden_sd:.4f} -> {den_sd:.4f}, ~{reduction_pct:.0f}% reduction)."
        )
    lines.append(
        f"    Block-wise correlation (den vs noden, intensity):"
        f" r = {int_d['correlation_r']:.2f}, suggesting both versions"
        " capture similar temporal patterns."
    )

    # --- Part 3 ----------------------------------------------------------
    lines.append("")
    lines.append(sep)
    lines.append("PART 3: WIND- AND SNR-DEPENDENT DENOISING BIAS")
    lines.append(sep)

    lines.append("")
    lines.append("  Wind speed vs denoising effect:")
    for label in ("intensity", "arousal", "dominance", "valence"):
        w = bias["wind_vs_diff"][label]
        lines.append(
            f"    {label:12s}: r = {w['r']:+.4f}, p = {w['p']:.4f} {_sig_stars(w['p']):>3s},"
            f" r^2 = {w['r_squared']:.4f} ({w['r_squared']*100:.1f}%)"
        )

    lines.append("")
    lines.append("  SNR vs denoising effect:")
    for label in ("intensity", "arousal", "dominance", "valence"):
        s = bias["snr_vs_diff"][label]
        lines.append(
            f"    {label:12s}: r = {s['r']:+.4f}, p = {s['p']:.4f} {_sig_stars(s['p']):>3s},"
            f" r^2 = {s['r_squared']:.4f} ({s['r_squared']*100:.1f}%)"
        )

    lines.append("")
    lines.append("  Stratified: intensity diff vs wind by location:")
    for loc in ("sea", "port"):
        st = bias["stratified_intensity_wind"].get(loc, {})
        if st.get("r") is not None:
            lines.append(
                f"    {loc:12s}: r = {st['r']:+.4f}, p = {st['p']:.4f}"
                f" {_sig_stars(st['p']):>3s}, N = {st['n']}"
            )
        else:
            lines.append(f"    {loc:12s}: insufficient data (N = {st.get('n', 0)})")

    # Conditional interpretation
    lines.append("")
    lines.append("  Interpretation:")
    w_int = bias["wind_vs_diff"]["intensity"]
    r_w = w_int["r"]
    p_w = w_int["p"]
    r2_w = w_int["r_squared"]

    if p_w < 0.05 and abs(r_w) >= 0.3:
        lines.append(
            f"    Wind speed shows a significant and medium-or-larger correlation"
            f" with the denoising effect (r = {r_w:.2f}, r^2 = {r2_w*100:.1f}%)."
        )
        lines.append(
            "    This suggests potential wind-dependent bias in the denoising step."
        )
    elif p_w < 0.05:
        lines.append(
            f"    Wind speed shows a statistically significant but small"
            f" correlation with the denoising effect"
            f" (r = {r_w:.2f}, r^2 = {r2_w*100:.1f}%)."
        )
        lines.append(
            f"    This explains only {r2_w*100:.1f}% of variance in the"
            " denoising effect and is unlikely to meaningfully confound results."
        )
    else:
        lines.append(
            "    No significant correlation between wind speed and the"
            " denoising effect."
        )

    # Stratified interpretation
    sea_st = bias["stratified_intensity_wind"].get("sea", {})
    port_st = bias["stratified_intensity_wind"].get("port", {})
    sea_nonsig = sea_st.get("p") is not None and sea_st["p"] >= 0.05
    port_nonsig = port_st.get("p") is not None and port_st["p"] >= 0.05
    if sea_nonsig and port_nonsig:
        lines.append(
            "    Stratified analyses show no significant wind-dependent bias"
            " within sea or port contexts, suggesting that the weak overall"
            " association is not driven by one operational state."
        )

    # SNR interpretation
    s_int = bias["snr_vs_diff"]["intensity"]
    r_s = s_int["r"]
    p_s = s_int["p"]
    if p_s < 0.05 and r_s < 0:
        lines.append(
            f"    SNR shows a negative correlation with the denoising"
            f" effect (r = {r_s:.2f}), indicating that denoising has a"
            " larger impact in low-SNR conditions. This is consistent with"
            " intended noise-reduction behaviour."
        )

    lines.append("")
    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(DIR_OUT, exist_ok=True)
    print(f"Output directory: {DIR_OUT}")

    df_den, df_noden, df_snr, df_wind, events = load_data()

    # Part 1
    df_wind_snr, wind_snr_res = analyse_wind_snr(df_snr, df_wind, events)

    # Part 2
    df_comp, denoising_res = analyse_denoising(df_den, df_noden, events)

    # Part 3
    df_bias, bias_res = analyse_denoising_bias(df_comp, df_wind_snr, events)

    # Write YAML
    yaml_data = build_yaml(wind_snr_res, denoising_res, bias_res)
    yaml_path = os.path.join(DIR_OUT, "confounder_results.yaml")
    with open(yaml_path, "w") as fh:
        yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)
    print(f"\nSaved: {yaml_path}")

    # Write verbose text
    verbose = build_verbose(wind_snr_res, denoising_res, bias_res)
    txt_path = os.path.join(DIR_OUT, "confounder_results_verbose.txt")
    with open(txt_path, "w") as fh:
        fh.write(verbose)
    print(f"Saved: {txt_path}")

    print(f"\nAll results saved to: {DIR_OUT}")


if __name__ == "__main__":
    main()
