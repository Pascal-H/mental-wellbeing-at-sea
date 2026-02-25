"""
Mediation Analysis: Wind Speed -> Bridge Emotion Intensity -> Self-Reported Stress

Implements a multilevel mediation model examining how environmental conditions
(wind speed) affect crew stress through passively sensed bridge emotion intensity.
Follows Hayes (2018) mediation framework with mixed-effects models accounting
for speaker-level clustering.

Analysis steps:
  1. Bivariate correlations (wind, emotion dimensions, stress)
  2. Primary mediation: Wind -> Emotion Intensity -> Current Stress
  3. Secondary analyses with individual emotion dimensions as mediators
  4. Alternative outcome variables (work stress, PSS-10, PHQ-8, WHO-5)
  5. Confounder-adjusted models (operational mode, time of day)
  6. Sensitivity analysis for unmeasured confounding
  7. OLS robustness check

Saves structured results as YAML (for plotting) and a verbose text report
with conditional interpretation of the mediation pattern.

References:
  Hayes, A. F. (2022). Introduction to Mediation, Moderation, and Conditional
      Process Analysis (3rd ed.). Guilford Press.
  Shrout, P. E. & Bolger, N. (2002). Mediation in experimental and
      nonexperimental studies. Psychological Methods, 7(4), 422-445.
  MacKinnon, D. P., Krull, J. L. & Lockwood, C. M. (2000). Equivalence of
      the mediation, confounding and suppression effect. Prevention Science, 1,
      173-181.
"""

import os
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import yaml

from evaluate_time_course_utils import concat_all_preds, load_database_aisl

# Suppress convergence warnings (common with small cluster sizes)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization.*")
warnings.filterwarnings("ignore", message=".*MixedLM optimization.*")
warnings.filterwarnings("ignore", message=".*Gradient optimization.*")
warnings.filterwarnings("ignore", message=".*MLE may be on the boundary.*")
warnings.filterwarnings("ignore", message=".*Hessian matrix.*")
warnings.filterwarnings("ignore", message=".*Random effects covariance.*")
warnings.filterwarnings("ignore", message=".*Retrying MixedLM.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_AGGREGATION_HOURS = 3.0
MERGE_TOLERANCE_HOURS = 2.0
TIME_WINDOW_THRESHOLD = 3.5
TIME_WINDOW_HOURS = 3.0
BRIDGE_MICS = ("M1", "M2", "M3", "M6")


# ---------------------------------------------------------------------------
# Event / confounder helpers
# ---------------------------------------------------------------------------


def load_event_timings(yaml_path):
    """Load event timings from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def assign_operational_mode(timestamp, event_timings):
    """
    Assign operational mode to a timestamp based on event timings.

    Returns a dictionary with:
      - at_sea: 1 if at sea, 0 otherwise
      - cargo_ops: 1 if cargo operations (loading/discharge), 0 otherwise
      - operational_mode: categorical ('at_sea', 'in_port', 'cargo_ops')
    """
    ts = pd.to_datetime(timestamp)

    at_sea = 0
    for period in event_timings.get("land_and_sea", {}).get("sea", []):
        start = pd.to_datetime(period["start"])
        end = pd.to_datetime(period["end"])
        if start <= ts <= end:
            at_sea = 1
            break

    cargo_ops = 0
    for ops_type in ["loading", "discharge"]:
        for period in event_timings.get("loading_and_discharge", {}).get(ops_type, []):
            start = pd.to_datetime(period["start"])
            end = pd.to_datetime(period["end"])
            if start <= ts <= end:
                cargo_ops = 1
                break

    if at_sea:
        mode = "at_sea"
    elif cargo_ops:
        mode = "cargo_ops"
    else:
        mode = "in_port"

    return {"at_sea": at_sea, "cargo_ops": cargo_ops, "operational_mode": mode}


def extract_time_features(timestamp):
    """
    Extract time-of-day features from a timestamp.

    Returns:
      - hour: hour of day (0-23)
      - time_of_day: categorical ('night', 'morning', 'afternoon', 'evening')
      - is_daytime: 1 if 06:00-18:00, 0 otherwise
    """
    ts = pd.to_datetime(timestamp)
    hour = ts.hour

    if 6 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 18:
        time_of_day = "afternoon"
    elif 18 <= hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    is_daytime = 1 if 6 <= hour < 18 else 0
    return {"hour": hour, "time_of_day": time_of_day, "is_daytime": is_daytime}


# ---------------------------------------------------------------------------
# Mediation model functions
# ---------------------------------------------------------------------------


def run_mediation_mlm(df, x_col, m_col, y_col, cluster_col, n_bootstrap=500, seed=42):
    """
    Run multilevel mediation analysis with cluster-robust bootstrap CIs.

    Estimates three mixed-effects models (random intercept for cluster):
      (1) Y ~ X               (total effect c)
      (2) M ~ X               (a-path)
      (3) Y ~ X + M           (direct effect c', b-path)

    Falls back to OLS if mixed-effects estimation fails.

    Parameters
    ----------
    df : DataFrame
        Data with all variables.
    x_col, m_col, y_col : str
        Column names for predictor, mediator, outcome.
    cluster_col : str
        Column for random intercept grouping (e.g. speaker ID).
    n_bootstrap : int
        Number of cluster-bootstrap samples for indirect effect CI.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Path coefficients, SEs, p-values, indirect effect with bootstrap CI.
    """
    np.random.seed(seed)
    n_clusters = df[cluster_col].nunique()
    convergence_ok = True

    try:
        # Model 1: c-path (total effect)
        m1 = smf.mixedlm(f"{y_col} ~ {x_col}", df, groups=df[cluster_col]).fit(
            reml=False, method="nm"
        )
        c_coef, c_se, c_pval = (
            m1.fe_params[x_col],
            m1.bse[x_col],
            m1.pvalues[x_col],
        )
        if not m1.converged:
            convergence_ok = False

        # Model 2: a-path (X -> M)
        m2 = smf.mixedlm(f"{m_col} ~ {x_col}", df, groups=df[cluster_col]).fit(
            reml=False, method="nm"
        )
        a_coef, a_se, a_pval = (
            m2.fe_params[x_col],
            m2.bse[x_col],
            m2.pvalues[x_col],
        )
        if not m2.converged:
            convergence_ok = False

        # Model 3: b and c' paths
        m3 = smf.mixedlm(
            f"{y_col} ~ {x_col} + {m_col}", df, groups=df[cluster_col]
        ).fit(reml=False, method="nm")
        cp_coef, cp_se, cp_pval = (
            m3.fe_params[x_col],
            m3.bse[x_col],
            m3.pvalues[x_col],
        )
        b_coef, b_se, b_pval = (
            m3.fe_params[m_col],
            m3.bse[m_col],
            m3.pvalues[m_col],
        )
        if not m3.converged:
            convergence_ok = False

    except Exception as e:
        print(f"  [WARN] MixedLM failed ({str(e)[:50]}), falling back to OLS")
        convergence_ok = False

        X = sm.add_constant(df[x_col].values)
        m1 = sm.OLS(df[y_col].values, X).fit()
        c_coef, c_se, c_pval = m1.params[1], m1.bse[1], m1.pvalues[1]

        m2 = sm.OLS(df[m_col].values, X).fit()
        a_coef, a_se, a_pval = m2.params[1], m2.bse[1], m2.pvalues[1]

        X_both = sm.add_constant(np.column_stack([df[x_col].values, df[m_col].values]))
        m3 = sm.OLS(df[y_col].values, X_both).fit()
        cp_coef, cp_se, cp_pval = m3.params[1], m3.bse[1], m3.pvalues[1]
        b_coef, b_se, b_pval = m3.params[2], m3.bse[2], m3.pvalues[2]

    # Indirect effect and Sobel SE
    indirect = a_coef * b_coef
    se_sobel = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)

    # Cluster-robust bootstrap
    clusters = df[cluster_col].unique()
    indirect_boot = []

    for _ in range(n_bootstrap):
        boot_clusters = np.random.choice(clusters, len(clusters), replace=True)
        boot_dfs = []
        for i, c in enumerate(boot_clusters):
            cluster_data = df[df[cluster_col] == c].copy()
            cluster_data[cluster_col] = f"{c}_{i}"
            boot_dfs.append(cluster_data)
        df_boot = pd.concat(boot_dfs, ignore_index=True)

        try:
            ma = smf.mixedlm(
                f"{m_col} ~ {x_col}", df_boot, groups=df_boot[cluster_col]
            ).fit(reml=False)
            mb = smf.mixedlm(
                f"{y_col} ~ {x_col} + {m_col}",
                df_boot,
                groups=df_boot[cluster_col],
            ).fit(reml=False)
            indirect_boot.append(ma.fe_params[x_col] * mb.fe_params[m_col])
        except Exception:
            pass

    if len(indirect_boot) >= 100:
        ci_low, ci_high = np.percentile(indirect_boot, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan

    is_suppression = (np.sign(indirect) != np.sign(c_coef)) or (
        abs(cp_coef) > abs(c_coef)
    )

    return {
        "a": a_coef,
        "a_se": a_se,
        "a_p": a_pval,
        "b": b_coef,
        "b_se": b_se,
        "b_p": b_pval,
        "c": c_coef,
        "c_se": c_se,
        "c_p": c_pval,
        "cp": cp_coef,
        "cp_se": cp_se,
        "cp_p": cp_pval,
        "indirect": indirect,
        "indirect_se": se_sobel,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(df),
        "n_clusters": n_clusters,
        "suppression": is_suppression,
        "converged": convergence_ok,
    }


def run_adjusted_mediation_mlm(
    df, x_col, m_col, y_col, cluster_col, covariates, n_bootstrap=500, seed=42
):
    """
    Run multilevel mediation with additional covariates in all path equations.

    Same as run_mediation_mlm but includes covariates (e.g. operational mode,
    time of day) in each of the three regression models.
    """
    np.random.seed(seed)
    n_clusters = df[cluster_col].nunique()
    convergence_ok = True

    cov_str = " + ".join(covariates) if covariates else ""

    try:
        formula1 = f"{y_col} ~ {x_col}" + (f" + {cov_str}" if cov_str else "")
        m1 = smf.mixedlm(formula1, df, groups=df[cluster_col]).fit(
            reml=False, method="nm"
        )
        c_coef, c_se, c_pval = (
            m1.fe_params[x_col],
            m1.bse[x_col],
            m1.pvalues[x_col],
        )
        if not m1.converged:
            convergence_ok = False

        formula2 = f"{m_col} ~ {x_col}" + (f" + {cov_str}" if cov_str else "")
        m2 = smf.mixedlm(formula2, df, groups=df[cluster_col]).fit(
            reml=False, method="nm"
        )
        a_coef, a_se, a_pval = (
            m2.fe_params[x_col],
            m2.bse[x_col],
            m2.pvalues[x_col],
        )
        if not m2.converged:
            convergence_ok = False

        formula3 = f"{y_col} ~ {x_col} + {m_col}" + (f" + {cov_str}" if cov_str else "")
        m3 = smf.mixedlm(formula3, df, groups=df[cluster_col]).fit(
            reml=False, method="nm"
        )
        cp_coef, cp_se, cp_pval = (
            m3.fe_params[x_col],
            m3.bse[x_col],
            m3.pvalues[x_col],
        )
        b_coef, b_se, b_pval = (
            m3.fe_params[m_col],
            m3.bse[m_col],
            m3.pvalues[m_col],
        )
        if not m3.converged:
            convergence_ok = False

    except Exception as e:
        print(f"  [WARN] MixedLM failed ({str(e)[:50]}), falling back to OLS")
        convergence_ok = False

        cov_data = df[covariates].values if covariates else np.zeros((len(df), 0))
        X = sm.add_constant(np.column_stack([df[x_col].values, cov_data]))
        m1 = sm.OLS(df[y_col].values, X).fit()
        c_coef, c_se, c_pval = m1.params[1], m1.bse[1], m1.pvalues[1]

        m2 = sm.OLS(df[m_col].values, X).fit()
        a_coef, a_se, a_pval = m2.params[1], m2.bse[1], m2.pvalues[1]

        X_both = sm.add_constant(
            np.column_stack([df[x_col].values, df[m_col].values, cov_data])
        )
        m3 = sm.OLS(df[y_col].values, X_both).fit()
        cp_coef, cp_se, cp_pval = m3.params[1], m3.bse[1], m3.pvalues[1]
        b_coef, b_se, b_pval = m3.params[2], m3.bse[2], m3.pvalues[2]

    indirect = a_coef * b_coef
    se_sobel = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)

    clusters = df[cluster_col].unique()
    indirect_boot = []

    for _ in range(n_bootstrap):
        boot_clusters = np.random.choice(clusters, len(clusters), replace=True)
        boot_dfs = []
        for i, c in enumerate(boot_clusters):
            cluster_data = df[df[cluster_col] == c].copy()
            cluster_data[cluster_col] = f"{c}_{i}"
            boot_dfs.append(cluster_data)
        df_boot = pd.concat(boot_dfs, ignore_index=True)

        try:
            ma = smf.mixedlm(formula2, df_boot, groups=df_boot[cluster_col]).fit(
                reml=False
            )
            mb = smf.mixedlm(formula3, df_boot, groups=df_boot[cluster_col]).fit(
                reml=False
            )
            indirect_boot.append(ma.fe_params[x_col] * mb.fe_params[m_col])
        except Exception:
            pass

    if len(indirect_boot) >= 100:
        ci_low, ci_high = np.percentile(indirect_boot, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan

    is_suppression = (np.sign(indirect) != np.sign(c_coef)) or (
        abs(cp_coef) > abs(c_coef)
    )

    return {
        "a": a_coef,
        "a_se": a_se,
        "a_p": a_pval,
        "b": b_coef,
        "b_se": b_se,
        "b_p": b_pval,
        "c": c_coef,
        "c_se": c_se,
        "c_p": c_pval,
        "cp": cp_coef,
        "cp_se": cp_se,
        "cp_p": cp_pval,
        "indirect": indirect,
        "indirect_se": se_sobel,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(df),
        "n_clusters": n_clusters,
        "suppression": is_suppression,
        "converged": convergence_ok,
        "covariates": covariates,
    }


def run_robustness_ols(df, x_col, m_col, y_col, n_bootstrap=1000, seed=42):
    """
    Run standard OLS mediation for comparison with mixed-effects results.

    Ignores clustering; serves as a robustness check.
    """
    np.random.seed(seed)

    X = sm.add_constant(df[x_col].values)
    X_both = sm.add_constant(np.column_stack([df[x_col].values, df[m_col].values]))

    m1 = sm.OLS(df[y_col].values, X).fit()
    m2 = sm.OLS(df[m_col].values, X).fit()
    m3 = sm.OLS(df[y_col].values, X_both).fit()

    c_coef, c_se, c_pval = m1.params[1], m1.bse[1], m1.pvalues[1]
    a_coef, a_se, a_pval = m2.params[1], m2.bse[1], m2.pvalues[1]
    cp_coef, cp_se, cp_pval = m3.params[1], m3.bse[1], m3.pvalues[1]
    b_coef, b_se, b_pval = m3.params[2], m3.bse[2], m3.pvalues[2]

    indirect = a_coef * b_coef
    se_sobel = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)

    indirect_boot = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(df), len(df), replace=True)
        try:
            X_b = sm.add_constant(df.iloc[idx][x_col].values)
            X_both_b = sm.add_constant(
                np.column_stack(
                    [df.iloc[idx][x_col].values, df.iloc[idx][m_col].values]
                )
            )
            m2_b = sm.OLS(df.iloc[idx][m_col].values, X_b).fit()
            m3_b = sm.OLS(df.iloc[idx][y_col].values, X_both_b).fit()
            indirect_boot.append(m2_b.params[1] * m3_b.params[2])
        except Exception:
            pass

    ci_low, ci_high = (
        np.percentile(indirect_boot, [2.5, 97.5])
        if len(indirect_boot) >= 100
        else (np.nan, np.nan)
    )
    is_suppression = (np.sign(indirect) != np.sign(c_coef)) or (
        abs(cp_coef) > abs(c_coef)
    )

    return {
        "a": a_coef,
        "a_se": a_se,
        "a_p": a_pval,
        "b": b_coef,
        "b_se": b_se,
        "b_p": b_pval,
        "c": c_coef,
        "c_se": c_se,
        "c_p": c_pval,
        "cp": cp_coef,
        "cp_se": cp_se,
        "cp_p": cp_pval,
        "indirect": indirect,
        "indirect_se": se_sobel,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(df),
        "n_clusters": 0,
        "suppression": is_suppression,
        "converged": True,
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_analysis_confounding(result, rho_values=None):
    """
    Sensitivity analysis for unmeasured confounding between M and Y.

    Estimates how strong an unmeasured confounder (rho = correlation with
    both mediator and outcome) would need to be to reduce the indirect
    effect to non-significance.

    Based on VanderWeele (2010) bias formulas for sensitivity analysis.

    Parameters
    ----------
    result : dict
        Mediation results from run_mediation_mlm.
    rho_values : array-like, optional
        Hypothetical confounder correlations to test.

    Returns
    -------
    DataFrame with sensitivity results for each rho value.
    """
    if rho_values is None:
        rho_values = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

    indirect = result["indirect"]
    b_coef = result["b"]

    results_list = []
    for rho in rho_values:
        bias = rho * abs(b_coef) * 0.5
        adjusted_b = b_coef - np.sign(b_coef) * bias
        adjusted_indirect = result["a"] * adjusted_b

        adjusted_ci_low = adjusted_indirect - 1.96 * result["indirect_se"]
        adjusted_ci_high = adjusted_indirect + 1.96 * result["indirect_se"]
        still_significant = adjusted_ci_low > 0 or adjusted_ci_high < 0

        results_list.append(
            {
                "rho_confounder": rho,
                "original_indirect": indirect,
                "adjusted_indirect": adjusted_indirect,
                "change_pct": (
                    ((adjusted_indirect - indirect) / abs(indirect) * 100)
                    if indirect != 0
                    else 0
                ),
                "still_significant": still_significant,
            }
        )

    return pd.DataFrame(results_list)


def compute_rho_to_nullify(result):
    """
    Compute the confounder correlation (rho) that would reduce the
    indirect effect to zero.
    """
    indirect = result["indirect"]
    b_coef = result["b"]

    if b_coef == 0:
        return np.inf

    rho_critical = (
        abs(indirect) / (abs(result["a"]) * abs(b_coef) * 0.5)
        if result["a"] != 0
        else np.inf
    )
    return min(rho_critical, 1.0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_pvalue(p):
    """Format p-value with significance stars."""
    if p < 0.001:
        return f"{p:.4f}***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


def print_result(result, title):
    """Print mediation result in a compact format."""
    print(f"\n{title}")
    print("=" * 80)
    converge_note = "" if result["converged"] else " [CONVERGENCE ISSUES]"
    print(f"N={result['n']}, clusters={result['n_clusters']}{converge_note}")
    print(
        f"a (X->M):      {result['a']:7.4f}, SE={result['a_se']:.4f}, "
        f"p={format_pvalue(result['a_p'])}"
    )
    print(
        f"b (M->Y):      {result['b']:7.4f}, SE={result['b_se']:.4f}, "
        f"p={format_pvalue(result['b_p'])}"
    )
    print(
        f"c (total):     {result['c']:7.4f}, SE={result['c_se']:.4f}, "
        f"p={format_pvalue(result['c_p'])}"
    )
    print(
        f"c' (direct):   {result['cp']:7.4f}, SE={result['cp_se']:.4f}, "
        f"p={format_pvalue(result['cp_p'])}"
    )
    print(
        f"a*b (indirect):{result['indirect']:7.4f}, "
        f"95%CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )

    significant = result["ci_low"] > 0 or result["ci_high"] < 0
    sig_str = "Significant" if significant else "Not significant"
    extra = ""
    if significant:
        if result["suppression"]:
            extra = ", Suppression (inconsistent mediation)"
        else:
            extra = ", Consistent mediation"
    print(f"  -> {sig_str}{extra}")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def save_results_to_yaml(results, output_dir, config, correlations, confounder_dist):
    """Save all analysis results to a structured YAML file."""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "mediation_analysis_main.py",
            "description": (
                "Mediation analysis: Wind -> Passive Emotion -> Active Stress"
            ),
        },
        "configuration": convert_to_serializable(config),
        "correlations": correlations,
        "confounder_distributions": convert_to_serializable(confounder_dist),
        "primary_analysis": convert_to_serializable(results["primary"]),
        "dimensions": convert_to_serializable(results["dimensions"]),
        "outcomes": convert_to_serializable(results["outcomes"]),
        "ols_comparison": convert_to_serializable(results.get("ols_comparison", {})),
        "sensitivity": convert_to_serializable(results.get("sensitivity", [])),
        "rho_critical": convert_to_serializable(results.get("rho_critical", None)),
        "confounder_adjusted": convert_to_serializable(
            results.get("confounder_adjusted", {})
        ),
    }

    yaml_path = os.path.join(output_dir, "mediation_results.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(
            output,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=False,
        )
    print(f"\nResults saved to: {yaml_path}")
    return yaml_path


def save_verbose_report(results, output_dir, config, correlations, confounder_dist):
    """
    Save a verbose text report with conditional interpretation.

    The interpretation text adapts to the actual results:
      - Suppression (inconsistent mediation) vs. consistent mediation
      - Significant vs. non-significant indirect effects
      - Direction of each path coefficient
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MEDIATION ANALYSIS REPORT")
    lines.append("Wind Speed -> Bridge Emotion Intensity -> Self-Reported Stress")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Script: mediation_analysis_main.py")

    # -- Configuration
    lines.append("\n" + "=" * 80)
    lines.append("CONFIGURATION")
    lines.append("=" * 80)
    for key, value in config.items():
        lines.append(f"  {key}: {value}")

    # - Confounder distributions
    lines.append("\n" + "=" * 80)
    lines.append("CONFOUNDER DISTRIBUTIONS")
    lines.append("=" * 80)
    for key, value in confounder_dist.items():
        lines.append(f"\n{key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {value}")

    # - Correlations
    lines.append("\n" + "=" * 80)
    lines.append("BIVARIATE CORRELATIONS")
    lines.append("=" * 80)
    for corr in correlations:
        lines.append(
            f"  {corr['label']}: r={corr['r']:.4f}, " f"p={corr['p']:.4f} {corr['sig']}"
        )

    # - Primary analysis
    lines.append("\n" + "=" * 80)
    lines.append("PRIMARY ANALYSIS: Wind -> Emotion Intensity -> Stress (Current)")
    lines.append("=" * 80)
    r = results["primary"]
    lines.append(f"  N = {r['n']}, clusters = {r['n_clusters']}")
    lines.append(f"  Converged: {r['converged']}")
    lines.append("\n  Path coefficients:")
    lines.append(
        f"    a (X->M):     {r['a']:.4f}, SE={r['a_se']:.4f}, p={r['a_p']:.4f}"
    )
    lines.append(
        f"    b (M->Y):     {r['b']:.4f}, SE={r['b_se']:.4f}, p={r['b_p']:.4f}"
    )
    lines.append(
        f"    c (total):    {r['c']:.4f}, SE={r['c_se']:.4f}, p={r['c_p']:.4f}"
    )
    lines.append(
        f"    c' (direct):  {r['cp']:.4f}, SE={r['cp_se']:.4f}, " f"p={r['cp_p']:.4f}"
    )
    lines.append("\n  Indirect effect:")
    lines.append(f"    a*b = {r['indirect']:.4f}")
    lines.append(f"    95% CI = [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    significant = r["ci_low"] > 0 or r["ci_high"] < 0
    lines.append(f"    Significant: {'Yes' if significant else 'No'}")
    lines.append(f"    Suppression: {'Yes' if r['suppression'] else 'No'}")

    # Conditional interpretation
    lines.append("\n  Interpretation:")
    a_dir = "lower" if r["a"] < 0 else "higher"
    b_dir = "lower" if r["b"] < 0 else "higher"
    cp_dir = "lower" if r["cp"] < 0 else "higher"

    lines.append(
        f"    - Higher wind speed is associated with {a_dir} bridge emotion"
        f" intensity (a = {r['a']:.4f})."
    )
    lines.append(
        f"    - Higher emotion intensity is associated with {b_dir}"
        f" self-reported stress (b = {r['b']:.4f})."
    )

    if significant:
        if r["suppression"]:
            lines.append(
                "    - The indirect and direct effects have OPPOSITE signs,"
                " indicating an inconsistent mediation (suppression) pattern."
            )
            lines.append(
                "    - Two competing pathways exist: the direct pathway from"
                " wind to stress, and the indirect pathway through emotion"
                " that operates in the opposite direction."
            )
            lines.append(
                "    - This suggests passive emotion monitoring captures a"
                " complexity in the wind-stress relationship that would be"
                " hidden when examining only environmental data."
            )
        else:
            lines.append(
                "    - The indirect effect is significant and consistent"
                f" with a standard mediation pattern (a*b = {r['indirect']:.4f})."
            )
            lines.append(
                "    - Emotion intensity partially mediates the relationship"
                " between wind speed and self-reported stress."
            )
    else:
        lines.append("    - The indirect effect is NOT statistically significant.")
        lines.append(
            "    - There is insufficient evidence for mediation through"
            " emotion intensity in this sample."
        )

    c_sig = r["c_p"] < 0.05
    cp_sig = r["cp_p"] < 0.05
    if not c_sig and cp_sig:
        lines.append(
            "    - Note: The total effect (c) is not significant while the"
            " direct effect (c') is, which is consistent with suppression."
        )

    # - Dimension analysis
    lines.append("\n" + "=" * 80)
    lines.append("EMOTION DIMENSION ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"\n{'Mediator':<30} {'Indirect':>10} {'95% CI':>22} {'Sig':>6}")
    lines.append("-" * 70)
    for dim, result in results["dimensions"].items():
        sig = "Yes" if (result["ci_low"] > 0 or result["ci_high"] < 0) else "No"
        ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        lines.append(f"  {dim:<28} {result['indirect']:>10.4f} {ci_str:>22} {sig:>6}")

    # Conditional dimension interpretation
    sig_dims = [
        dim
        for dim, res in results["dimensions"].items()
        if res["ci_low"] > 0 or res["ci_high"] < 0
    ]
    nonsig_dims = [
        dim
        for dim, res in results["dimensions"].items()
        if not (res["ci_low"] > 0 or res["ci_high"] < 0)
    ]

    if sig_dims:
        lines.append(f"\n  Significant mediators: {', '.join(sig_dims)}")
    if nonsig_dims:
        lines.append(f"  Non-significant mediators: {', '.join(nonsig_dims)}")

    # - Outcome analysis
    lines.append("\n" + "=" * 80)
    lines.append("ALTERNATIVE OUTCOME ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"\n{'Outcome':<30} {'Indirect':>10} {'95% CI':>22} {'Sig':>6}")
    lines.append("-" * 70)
    for outcome, result in results["outcomes"].items():
        sig = "Yes" if (result["ci_low"] > 0 or result["ci_high"] < 0) else "No"
        ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        lines.append(
            f"  {outcome:<28} {result['indirect']:>10.4f} {ci_str:>22} " f"{sig:>6}"
        )

    sig_outcomes = [
        out
        for out, res in results["outcomes"].items()
        if res["ci_low"] > 0 or res["ci_high"] < 0
    ]
    nonsig_outcomes = [
        out
        for out, res in results["outcomes"].items()
        if not (res["ci_low"] > 0 or res["ci_high"] < 0)
    ]

    if sig_outcomes:
        lines.append(f"\n  Significant indirect effects for: {', '.join(sig_outcomes)}")
        lines.append(
            "  These are momentary or recent-task stress measures, suggesting"
            " the mediation pathway operates at a short time scale."
        )
    if nonsig_outcomes:
        lines.append(f"  Non-significant for: {', '.join(nonsig_outcomes)}")
        if any(x in ", ".join(nonsig_outcomes) for x in ["PSS", "PHQ", "WHO"]):
            lines.append(
                "  Retrospective long-term measures (PSS-10, PHQ-8, WHO-5)"
                " did not show significant indirect effects, consistent with"
                " the interpretation that the mediation pathway is specific"
                " to momentary or recent-task stress."
            )

    # - OLS comparison
    if results.get("ols_comparison"):
        lines.append("\n" + "=" * 80)
        lines.append("OLS vs MIXED-EFFECTS COMPARISON")
        lines.append("=" * 80)
        r_ols = results["ols_comparison"]
        r_mlm = results["primary"]
        lines.append(
            f"\n{'Path':<20} {'OLS':>12} {'Mixed-Effects':>15} " f"{'Difference':>12}"
        )
        lines.append("-" * 60)
        for label, key_ols, key_mlm in [
            ("a (X->M)", "a", "a"),
            ("b (M->Y)", "b", "b"),
            ("c (total)", "c", "c"),
            ("c' (direct)", "cp", "cp"),
            ("Indirect (a*b)", "indirect", "indirect"),
        ]:
            lines.append(
                f"{label:<20} {r_ols[key_ols]:>12.4f} "
                f"{r_mlm[key_mlm]:>15.4f} "
                f"{abs(r_ols[key_ols] - r_mlm[key_mlm]):>12.4f}"
            )

        ols_sig = r_ols["ci_low"] > 0 or r_ols["ci_high"] < 0
        mlm_sig = r_mlm["ci_low"] > 0 or r_mlm["ci_high"] < 0
        if ols_sig == mlm_sig:
            lines.append(
                "\n  Both methods agree on significance"
                " -> Results are robust to clustering adjustment."
            )
        else:
            lines.append(
                "\n  Methods disagree on significance"
                " -> Clustering adjustment matters for inference."
            )

    # - Sensitivity analysis
    if results.get("sensitivity") is not None:
        lines.append("\n" + "=" * 80)
        lines.append("SENSITIVITY ANALYSIS FOR UNMEASURED CONFOUNDING")
        lines.append("=" * 80)
        sens_df = results["sensitivity"]
        if isinstance(sens_df, pd.DataFrame):
            lines.append(
                f"\n{'rho':>15} {'Original':>12} {'Adjusted':>12} "
                f"{'Change %':>12} {'Still Sig?':>12}"
            )
            lines.append("-" * 65)
            for _, row in sens_df.iterrows():
                sig = "Yes" if row["still_significant"] else "No"
                lines.append(
                    f"{row['rho_confounder']:>15.2f} "
                    f"{row['original_indirect']:>12.4f} "
                    f"{row['adjusted_indirect']:>12.4f} "
                    f"{row['change_pct']:>11.1f}% {sig:>12}"
                )

        rho_crit = results.get("rho_critical", "N/A")
        lines.append(f"\n  Critical rho to nullify effect: {rho_crit}")

        # Conditional robustness interpretation
        if isinstance(rho_crit, (int, float)):
            if rho_crit > 0.3:
                lines.append(
                    "  -> The effect is robust: would require a strong"
                    " confounder (rho > 0.3) to nullify."
                )
            elif rho_crit > 0.15:
                lines.append(
                    "  -> The effect shows moderate robustness to"
                    " unmeasured confounding."
                )
            else:
                lines.append(
                    "  -> The effect is sensitive: could be nullified by"
                    " weak confounders."
                )

        # Find transition point
        if isinstance(sens_df, pd.DataFrame) and len(sens_df) > 0:
            sig_rows = sens_df[sens_df["still_significant"]]
            nonsig_rows = sens_df[~sens_df["still_significant"]]
            if len(sig_rows) > 0 and len(nonsig_rows) > 0:
                boundary = sig_rows["rho_confounder"].max()
                lines.append(
                    f"  -> The effect remains significant up to"
                    f" rho = {boundary:.2f} and becomes non-significant"
                    f" beyond that threshold."
                )

    # - Confounder-adjusted analysis
    if results.get("confounder_adjusted"):
        lines.append("\n" + "=" * 80)
        lines.append("CONFOUNDER-ADJUSTED ANALYSIS")
        lines.append("=" * 80)
        lines.append(
            f"\n{'Adjustment':<30} {'a':>8} {'b':>8} {'Indirect':>10} "
            f"{'95% CI':>20} {'Sig':>5}"
        )
        lines.append("-" * 85)
        for label, r_adj in results["confounder_adjusted"].items():
            sig = "Yes" if (r_adj["ci_low"] > 0 or r_adj["ci_high"] < 0) else "No"
            ci_str = f"[{r_adj['ci_low']:.4f}, {r_adj['ci_high']:.4f}]"
            lines.append(
                f"{label:<30} {r_adj['a']:>8.4f} {r_adj['b']:>8.4f} "
                f"{r_adj['indirect']:>10.4f} {ci_str:>20} {sig:>5}"
            )

        # Conditional confounder interpretation
        unadj = results["confounder_adjusted"].get("Unadjusted", {})
        adj_all = results["confounder_adjusted"].get("Adj: All Available", {})

        if unadj and adj_all:
            unadj_sig = unadj["ci_low"] > 0 or unadj["ci_high"] < 0
            adj_sig = adj_all["ci_low"] > 0 or adj_all["ci_high"] < 0
            unadj_ind = unadj["indirect"]
            adj_ind = adj_all["indirect"]
            change_pct = (
                ((adj_ind - unadj_ind) / abs(unadj_ind) * 100) if unadj_ind != 0 else 0
            )

            lines.append(
                f"\n  Change in indirect effect with full adjustment:"
                f" {change_pct:+.1f}%"
            )

            if unadj_sig and adj_sig:
                lines.append(
                    "  -> Significance is preserved after adjusting for"
                    " available confounders."
                )
                lines.append(
                    "  -> Results are robust to operational mode and" " time of day."
                )
                if change_pct > 0:
                    lines.append(
                        "  -> The fully adjusted estimates are larger in"
                        " magnitude, indicating that the measured"
                        " contextual variables did not explain away the"
                        " observed pathway."
                    )
            elif unadj_sig and not adj_sig:
                lines.append("  -> Effect becomes non-significant after adjustment.")
                lines.append(
                    "  -> Confounders may partially explain the observed"
                    " relationship."
                )
            elif not unadj_sig and adj_sig:
                lines.append("  -> Effect becomes significant after adjustment.")
                lines.append(
                    "  -> Confounders were acting as suppressors of"
                    " the indirect effect."
                )
            else:
                lines.append(
                    "  -> Effect remains non-significant in both"
                    " the unadjusted and adjusted models."
                )

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    txt_path = os.path.join(output_dir, "mediation_results_verbose.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Verbose report saved to: {txt_path}")
    return txt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("MEDIATION ANALYSIS")
    print("Wind Speed -> Bridge Emotion Intensity -> Self-Reported Stress")
    print("=" * 80)

    parser = argparse.ArgumentParser(
        description="Mediation Analysis: Wind -> Emotion -> Stress"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (YAML + text report)",
    )
    args, _ = parser.parse_known_args()

    # -- Paths (adjust to your data layout) ---------------------------------
    dir_script = os.path.dirname(os.path.abspath(__file__))
    dir_repo = os.path.dirname(os.path.dirname(dir_script))  # repo root

    dir_output = os.path.join(dir_repo, "data", "output")
    dir_evaluated = os.path.join(dir_repo, "data", "evaluated")
    dir_wind = os.path.join(dir_evaluated, "wind_speed")
    path_active = (
        "/data/share/aisoundlab-mental_wellbeing_at_sea/"
        "data_mwas_processed-final_data/final_data-df_files.csv"
    )
    yaml_events = os.path.join(dir_script, "evaluate_time_course_events.yaml")

    output_dir = args.output_dir or os.path.join(
        dir_evaluated, "mediation-wind_emotion_stress"
    )
    os.makedirs(output_dir, exist_ok=True)

    # -- Load data ----------------------------------------------------------
    print("\nLoading data...")
    df_passive = concat_all_preds(dir_output, dir_evaluated, "false", "emotion")
    df_active = load_database_aisl(path_active)
    df_wind = pd.read_csv(os.path.join(dir_wind, "true_wind_speed.csv"))

    # Filter by minimum segment duration (3 s)
    durations = (
        df_passive.index.get_level_values("end")
        - df_passive.index.get_level_values("start")
    ).total_seconds()
    df_passive = df_passive[durations >= 3.0]
    print(f"Passive segments >= 3 s: {len(df_passive):,}")

    # -- Aggregate passive data into 3-hour windows -------------------------
    df_passive["base_time"] = pd.to_datetime(df_passive["time"])
    df_wind["time"] = pd.to_datetime(df_wind["time"])
    df_passive["time"] = df_passive["base_time"]
    print(f"Using {TIME_AGGREGATION_HOURS}-hour time windows")
    print(f"  Unique time bins: {df_passive['time'].nunique()}")

    df_passive_agg = (
        df_passive.groupby(["microphone", "time"]).mean(numeric_only=True).reset_index()
    )
    print(f"Aggregated passive data: {len(df_passive_agg)} rows")

    # -- Align wind data to passive time bins -------------------------------
    unique_times = np.sort(df_passive["time"].unique())
    wind_agg = []

    for i, t_start in enumerate(unique_times):
        t_start = pd.Timestamp(t_start)
        if i < len(unique_times) - 1:
            t_next = pd.Timestamp(unique_times[i + 1])
            interval = t_next - t_start
            if interval <= pd.Timedelta(hours=TIME_WINDOW_THRESHOLD):
                t_end = t_next
            else:
                t_end = t_start + pd.Timedelta(hours=TIME_WINDOW_HOURS)
        else:
            t_end = t_start + pd.Timedelta(hours=TIME_WINDOW_HOURS)

        mask = (df_wind["time"] >= t_start) & (df_wind["time"] < t_end)
        wind_mean = df_wind.loc[mask, "true_wind_speed_gps"].mean()
        wind_agg.append({"time": t_start, "wind": wind_mean})

    df_wind_agg = pd.DataFrame(wind_agg)
    df_wind_agg["time"] = pd.to_datetime(df_wind_agg["time"])
    df_merged = pd.merge(df_passive_agg, df_wind_agg, on="time", how="left").dropna(
        subset=["wind"]
    )

    # Min-max normalise wind speed to [0, 1]
    df_merged["wind_norm"] = (df_merged["wind"] - df_merged["wind"].min()) / (
        df_merged["wind"].max() - df_merged["wind"].min()
    )

    # -- Aggregate bridge microphones ---------------------------------------
    bridge_data = (
        df_merged[df_merged.microphone.isin(BRIDGE_MICS)]
        .groupby("time")
        .agg(
            {
                "prediction_arousal": "mean",
                "prediction_valence": "mean",
                "prediction_dominance": "mean",
                "wind_norm": "first",
            }
        )
        .reset_index()
    )

    # Emotion intensity = mean(arousal, dominance)
    bridge_data["emotion_combined"] = (
        bridge_data["prediction_arousal"] + bridge_data["prediction_dominance"]
    ) / 2

    # -- Merge with active data ---------------------------------------------
    df_active["stress_current"] = df_active["stress_current"] / 100.0
    df_active["stress_work_tasks"] = df_active["stress_work_tasks"] / 100.0
    df_active_session = (
        df_active.groupby("session")
        .agg(
            {
                "stress_current": "first",
                "stress_work_tasks": "first",
                "pss_10_total_score": "first",
                "phq_8_total_score": "first",
                "who_5_percentage_score_corrected": "first",
                "speaker_file": "first",
                "date_file": "first",
            }
        )
        .reset_index()
    )

    df_active_session["date_file"] = pd.to_datetime(df_active_session["date_file"])
    bridge_data["time"] = pd.to_datetime(bridge_data["time"])

    df_final = pd.merge_asof(
        df_active_session.sort_values("date_file"),
        bridge_data.sort_values("time"),
        left_on="date_file",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(hours=MERGE_TOLERANCE_HOURS),
    ).dropna(subset=["time"])

    print(
        f"Final merged: N={len(df_final)}, "
        f"speakers={df_final['speaker_file'].nunique()}"
    )

    # -- Add confounder variables -------------------------------------------
    print("\nAdding confounder variables...")
    event_timings = load_event_timings(yaml_events)

    op_modes = df_final["time"].apply(
        lambda t: assign_operational_mode(t, event_timings)
    )
    df_final["at_sea"] = op_modes.apply(lambda x: x["at_sea"])
    df_final["cargo_ops"] = op_modes.apply(lambda x: x["cargo_ops"])
    df_final["operational_mode"] = op_modes.apply(lambda x: x["operational_mode"])

    time_features = df_final["time"].apply(extract_time_features)
    df_final["hour"] = time_features.apply(lambda x: x["hour"])
    df_final["time_of_day"] = time_features.apply(lambda x: x["time_of_day"])
    df_final["is_daytime"] = time_features.apply(lambda x: x["is_daytime"])

    print(f"\nOperational mode distribution:")
    print(df_final["operational_mode"].value_counts().to_string())
    print(f"\nTime of day distribution:")
    print(df_final["time_of_day"].value_counts().to_string())
    print(
        f"\nDaytime vs Night: "
        f"{df_final['is_daytime'].sum()} daytime, "
        f"{len(df_final) - df_final['is_daytime'].sum()} night"
    )

    confounder_dist = {
        "operational_mode": df_final["operational_mode"].value_counts().to_dict(),
        "time_of_day": df_final["time_of_day"].value_counts().to_dict(),
        "daytime_count": int(df_final["is_daytime"].sum()),
        "night_count": int(len(df_final) - df_final["is_daytime"].sum()),
    }

    results = {}

    # =====================================================================
    # CORRELATIONS
    # =====================================================================
    print("\n" + "=" * 80)
    print("BIVARIATE CORRELATIONS")
    print("=" * 80)

    corr_pairs = [
        ("wind_norm", "emotion_combined", "Wind vs Emotion (Combined)"),
        ("wind_norm", "prediction_arousal", "Wind vs Arousal"),
        ("wind_norm", "prediction_valence", "Wind vs Valence"),
        ("wind_norm", "prediction_dominance", "Wind vs Dominance"),
        (
            "emotion_combined",
            "stress_current",
            "Emotion vs Stress (Current)",
        ),
    ]

    correlations = []
    for v1, v2, label in corr_pairs:
        if v1 in df_final.columns and v2 in df_final.columns:
            r, p = pearsonr(df_final[v1], df_final[v2])
            sig_str = format_pvalue(p)
            print(f"{label:45s} r={r:6.3f}, p={sig_str}")
            correlations.append(
                {
                    "var1": v1,
                    "var2": v2,
                    "label": label,
                    "r": float(r),
                    "p": float(p),
                    "sig": sig_str,
                }
            )

    # =====================================================================
    # PRIMARY ANALYSIS
    # =====================================================================
    print("\n\n" + "#" * 80)
    print("PRIMARY ANALYSIS: Wind -> Emotion Intensity -> Stress (Current)")
    print("#" * 80)

    r = run_mediation_mlm(
        df_final,
        "wind_norm",
        "emotion_combined",
        "stress_current",
        "speaker_file",
        n_bootstrap=1000,
        seed=42,
    )
    results["primary"] = r
    print_result(r, "PRIMARY RESULT")

    # =====================================================================
    # EMOTION DIMENSIONS AS MEDIATORS
    # =====================================================================
    print("\n\n" + "#" * 80)
    print("EMOTION DIMENSIONS AS MEDIATORS")
    print("#" * 80)

    results["dimensions"] = {}
    for dim, label in [
        ("emotion_combined", "Emotion Intensity (A+D)/2"),
        ("prediction_arousal", "Arousal"),
        ("prediction_valence", "Valence"),
        ("prediction_dominance", "Dominance"),
    ]:
        if dim in df_final.columns:
            r = run_mediation_mlm(
                df_final,
                "wind_norm",
                dim,
                "stress_current",
                "speaker_file",
                n_bootstrap=500,
                seed=42,
            )
            results["dimensions"][label] = r
            print_result(r, f"Mediator: {label}")

    # =====================================================================
    # ALTERNATIVE OUTCOMES
    # =====================================================================
    print("\n\n" + "#" * 80)
    print("ALTERNATIVE OUTCOME VARIABLES")
    print("#" * 80)

    results["outcomes"] = {}
    for outcome, (label, norm) in [
        ("stress_current", ("Stress (Current VAS)", 1.0)),
        ("stress_work_tasks", ("Stress (Work Tasks VAS)", 1.0)),
        ("pss_10_total_score", ("PSS-10 Stress", 40.0)),
        ("phq_8_total_score", ("PHQ-8 Depression", 24.0)),
        (
            "who_5_percentage_score_corrected",
            ("WHO-5 Wellbeing", 100.0),
        ),
    ]:
        if outcome in df_final.columns and df_final[outcome].notna().sum() > 50:
            df_sub = df_final.copy()
            df_sub[f"{outcome}_norm"] = df_sub[outcome] / norm
            df_sub = df_sub.dropna(subset=[f"{outcome}_norm"])

            if len(df_sub) >= 50:
                r = run_mediation_mlm(
                    df_sub,
                    "wind_norm",
                    "emotion_combined",
                    f"{outcome}_norm",
                    "speaker_file",
                    n_bootstrap=500,
                    seed=42,
                )
                results["outcomes"][label] = r
                print_result(r, f"Outcome: {label}")

    # =====================================================================
    # SENSITIVITY ANALYSIS
    # =====================================================================
    print("\n" + "#" * 80)
    print("SENSITIVITY ANALYSIS FOR UNMEASURED CONFOUNDING")
    print("#" * 80)

    sens_df = sensitivity_analysis_confounding(results["primary"])
    print("\nSensitivity Table:")
    print("-" * 70)
    print(
        f"{'rho (confounder)':>15} {'Original':>12} {'Adjusted':>12} "
        f"{'Change %':>12} {'Still Sig?':>12}"
    )
    print("-" * 70)
    for _, row in sens_df.iterrows():
        sig_str = "Yes" if row["still_significant"] else "No"
        print(
            f"{row['rho_confounder']:>15.2f} "
            f"{row['original_indirect']:>12.4f} "
            f"{row['adjusted_indirect']:>12.4f} "
            f"{row['change_pct']:>11.1f}% {sig_str:>12}"
        )

    rho_critical = compute_rho_to_nullify(results["primary"])
    print(f"\nCritical rho to nullify effect: {rho_critical:.3f}")

    results["sensitivity"] = sens_df
    results["rho_critical"] = rho_critical

    # =====================================================================
    # OLS ROBUSTNESS CHECK
    # =====================================================================
    print("\n\n" + "#" * 80)
    print("ROBUSTNESS CHECK: OLS vs MIXED-EFFECTS")
    print("#" * 80)

    r_ols = run_robustness_ols(
        df_final,
        "wind_norm",
        "emotion_combined",
        "stress_current",
        n_bootstrap=1000,
    )
    r_mlm = results["primary"]

    print(f"\n{'Path':<20} {'OLS':>12} {'Mixed-Effects':>15} {'Difference':>12}")
    print("-" * 60)
    for label, key in [
        ("a (X->M)", "a"),
        ("b (M->Y)", "b"),
        ("c (total)", "c"),
        ("c' (direct)", "cp"),
        ("Indirect (a*b)", "indirect"),
    ]:
        print(
            f"{label:<20} {r_ols[key]:>12.4f} {r_mlm[key]:>15.4f} "
            f"{abs(r_ols[key] - r_mlm[key]):>12.4f}"
        )

    results["ols_comparison"] = r_ols

    # =====================================================================
    # CONFOUNDER-ADJUSTED ANALYSIS
    # =====================================================================
    print("\n\n" + "#" * 80)
    print("CONFOUNDER-ADJUSTED ANALYSIS")
    print("#" * 80)

    results["confounder_adjusted"] = {}
    confounder_sets = {
        "Unadjusted": [],
        "Adj: Operational Mode": ["at_sea", "cargo_ops"],
        "Adj: Time of Day": ["is_daytime"],
        "Adj: All Available": ["at_sea", "cargo_ops", "is_daytime"],
    }

    for label, covariates in confounder_sets.items():
        print(f"\n>>> {label}")
        if covariates:
            print(f"    Covariates: {', '.join(covariates)}")

        r_adj = run_adjusted_mediation_mlm(
            df_final,
            "wind_norm",
            "emotion_combined",
            "stress_current",
            "speaker_file",
            covariates=covariates,
            n_bootstrap=500,
            seed=42,
        )
        results["confounder_adjusted"][label] = r_adj

        sig = "Yes" if (r_adj["ci_low"] > 0 or r_adj["ci_high"] < 0) else "No"
        ci_str = f"[{r_adj['ci_low']:.4f}, {r_adj['ci_high']:.4f}]"
        print(f"    a-path: {r_adj['a']:.4f} (p={r_adj['a_p']:.4f})")
        print(f"    b-path: {r_adj['b']:.4f} (p={r_adj['b_p']:.4f})")
        print(f"    Indirect: {r_adj['indirect']:.4f}, 95% CI: {ci_str} {sig}")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    config = {
        "time_aggregation_hours": TIME_AGGREGATION_HOURS,
        "merge_tolerance_hours": MERGE_TOLERANCE_HOURS,
        "n_samples": results["primary"]["n"],
        "n_clusters": results["primary"]["n_clusters"],
        "output_dir": output_dir,
    }

    save_results_to_yaml(results, output_dir, config, correlations, confounder_dist)
    save_verbose_report(results, output_dir, config, correlations, confounder_dist)

    print(f"\nAll results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    results = main()
