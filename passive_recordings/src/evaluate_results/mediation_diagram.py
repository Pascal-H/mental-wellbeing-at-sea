"""
Mediation Diagram: Publication-Quality PGF/LaTeX Causal Path Diagram

Generates a mediation path diagram for the analysis:
  Wind Speed (X) -> Bridge Emotion Intensity (M) -> Self-Reported Stress (Y)

Uses the PGF/LaTeX backend (pdflatex) for crisp text rendering suitable
for journal submission.  Reads structured results from a YAML file generated
by mediation_analysis_main.py.

Springer Nature Computer Science figure specs:
  - Single column: 84 mm (3.31 in)
  - 1.5 column:   120 mm (4.72 in)
  - Full page:    174 mm (6.85 in)
  - Min font size: 6 pt (recommended 8+ pt)
  - Vector PDF preferred, Type 42 (TrueType) fonts

Visual conventions (Psychology / Health publications):
  - Observed variables: rectangles
  - Causal paths: solid arrows with path coefficients
  - Direct effect (c'): dashed line
  - Positive coefficients: dark green
  - Negative coefficients: dark red
  - Significance: *p<.05, **p<.01, ***p<.001
  - Confounders: gray boxes with dashed connectors
"""

import os
import sys
import importlib
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from typing import Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MM_PER_INCH = 25.4
SPRINGER_FULL_PAGE_MM = 174

_POS_COLOR = "#1B7A3D"  # dark green
_NEG_COLOR = "#B71C1C"  # dark red
_ZERO_COLOR = "#424242"  # near-black (neutral, zero effect)
_CONF_COLOR = "#757575"  # gray


def mm_to_inch(mm: float) -> float:
    """Convert millimetres to inches."""
    return mm / MM_PER_INCH


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _fmt_coef(coef: float, p: float) -> str:
    """Format a path coefficient with significance stars."""
    return f"{coef:+.3f}{_sig_stars(p)}"


def _coef_color(coef: float) -> str:
    """Return colour string based on coefficient sign."""
    if coef > 0:
        return _POS_COLOR
    if coef < 0:
        return _NEG_COLOR
    return _ZERO_COLOR  # exactly zero: neutral colour


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _draw_rect(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    label: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "black",
    linestyle: str = "-",
    lw: float = 0.8,
    fontsize: float = 9,
    fontweight: str = "bold",
    text_color: str = "black",
):
    """Draw a rounded rectangle with a centred label."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        linespacing=1.35,
        zorder=4,
    )
    return box


def _draw_arrow(
    ax,
    xy_start: Tuple[float, float],
    xy_end: Tuple[float, float],
    *,
    color: str = "black",
    linestyle: str = "-",
    lw: float = 1.0,
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 12,
):
    """Draw a directed arrow between two points."""
    arrow = FancyArrowPatch(
        xy_start,
        xy_end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        linestyle=linestyle,
        linewidth=lw,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_results_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load mediation analysis results from YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Results file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        results = yaml.safe_load(f)

    print(f"Loaded results from: {yaml_path}")
    print(f"  Generated: {results['metadata']['generated_at']}")
    print(f"  N samples: {results['configuration']['n_samples']}")
    print(f"  N clusters: {results['configuration']['n_clusters']}")
    return results


# ---------------------------------------------------------------------------
# Main diagram function
# ---------------------------------------------------------------------------


def create_mediation_diagram(
    results: Dict[str, Any],
    output_path: str,
    show_confounders: bool = True,
    dpi: int = 300,
) -> Optional[str]:
    """
    Render the mediation path diagram via the PGF/LaTeX backend.

    Produces a PDF with LaTeX-rendered text (Palatino via mathpazo).
    Falls back gracefully if pdflatex is not installed.

    Parameters
    ----------
    results : dict
        Analysis results loaded from YAML.
    output_path : str
        Where to save the PDF figure.
    show_confounders : bool
        Whether to include the control-variable box and arrows.
    dpi : int
        Resolution (mainly for any raster elements).

    Returns
    -------
    str or None
        Path to the saved PDF, or None if pdflatex is unavailable.
    """
    if not shutil.which("pdflatex"):
        print("pdflatex not found -- skipping PGF diagram.")
        return None

    # Switch to PGF backend temporarily; save state for guaranteed restoration.
    prev_backend = matplotlib.get_backend()
    _rc_keys_to_restore = [
        "pgf.texsystem",
        "pgf.rcfonts",
        "pgf.preamble",
        "pdf.fonttype",
        "font.family",
        "font.size",
    ]
    prev_rc = {k: matplotlib.rcParams[k] for k in _rc_keys_to_restore}

    matplotlib.use("pgf")
    importlib.reload(plt)

    fig = None
    try:
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
                "pgf.preamble": (r"\usepackage[T1]{fontenc}" r"\usepackage{mathpazo}"),
                "pdf.fonttype": 42,
                "font.family": "serif",
                "font.size": 8,
            }
        )

        primary = results["primary_analysis"]

        # ---- sizing -------------------------------------------------------
        fig_w = mm_to_inch(SPRINGER_FULL_PAGE_MM)
        fig_h = fig_w * 0.58 if show_confounders else fig_w * 0.48
        fs_var, fs_coef, fs_note = 9, 8, 7
        lw = 0.9

        bh = 0.35
        bw_x = 1.80  # "Wind Speed (X)"
        bw_m = 2.40  # "Emotion Intensity (M)"
        bw_y = 2.50  # "Self-Reported Stress (Y)"
        bw_c = 4.00  # "Controls: ..."

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(-0.3, 8.3)
        y_lo = -0.65 if show_confounders else 0.3
        ax.set_ylim(y_lo, 4.85)
        ax.set_aspect("equal")
        ax.axis("off")

        # ---- title --------------------------------------------------------
        ax.text(
            4.0,
            4.70,
            "Mediation Model",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="top",
        )

        # ---- node positions -----------------------------------------------
        x_pos = np.array([1.3, 1.4])
        m_pos = np.array([4.0, 3.2])
        y_pos = np.array([6.7, 1.4])

        x_hw, x_hh = bw_x / 2, bh / 2
        m_hw, m_hh = bw_m / 2, bh / 2
        y_hw, y_hh = bw_y / 2, bh / 2

        # ---- draw variable boxes ------------------------------------------
        _draw_rect(ax, *x_pos, bw_x, bh, "Wind Speed (X)", fontsize=fs_var, lw=lw)
        _draw_rect(
            ax, *m_pos, bw_m, bh, "Emotion Intensity (M)", fontsize=fs_var, lw=lw
        )
        _draw_rect(
            ax, *y_pos, bw_y, bh, "Self-Reported Stress (Y)", fontsize=fs_var, lw=lw
        )

        if show_confounders:
            c_pos = np.array([4.0, -0.1])
            _draw_rect(
                ax,
                *c_pos,
                bw_c,
                bh,
                "Controls: Operational Mode, Time of Day",
                facecolor="#F5F5F5",
                edgecolor=_CONF_COLOR,
                linestyle="--",
                lw=0.5,
                fontsize=fs_note,
                fontweight="normal",
            )

        # ---- edge-point helper --------------------------------------------
        def _edge_pt(pos, half_w, half_h, angle_deg):
            rad = np.deg2rad(angle_deg)
            dx, dy = np.cos(rad), np.sin(rad)
            if abs(dx) < 1e-9:
                t = half_h / abs(dy) if abs(dy) > 1e-9 else 1e6
            elif abs(dy) < 1e-9:
                t = half_w / abs(dx)
            else:
                t = min(half_w / abs(dx), half_h / abs(dy))
            return (pos[0] + dx * t, pos[1] + dy * t)

        # Natural centre-to-centre angles
        angle_a = np.degrees(np.arctan2(m_pos[1] - x_pos[1], m_pos[0] - x_pos[0]))
        angle_b = np.degrees(np.arctan2(y_pos[1] - m_pos[1], y_pos[0] - m_pos[0]))

        # ---- path a: X -> M ----------------------------------------------
        a_coef, a_p = primary["a"], primary["a_p"]
        a_s = _edge_pt(x_pos, x_hw, x_hh, angle_a)
        a_e = _edge_pt(m_pos, m_hw, m_hh, angle_a + 180)
        _draw_arrow(ax, a_s, a_e, color=_coef_color(a_coef), lw=lw)

        a_perp = np.array([-np.sin(np.radians(angle_a)), np.cos(np.radians(angle_a))])
        a_mid = np.array([(a_s[0] + a_e[0]) / 2, (a_s[1] + a_e[1]) / 2])
        a_lbl = a_mid + a_perp * 0.22
        ax.text(
            a_lbl[0],
            a_lbl[1],
            f"a = {_fmt_coef(a_coef, a_p)}",
            fontsize=fs_coef,
            color=_coef_color(a_coef),
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=0.5),
        )

        # ---- path b: M -> Y ----------------------------------------------
        b_coef, b_p = primary["b"], primary["b_p"]
        b_s = _edge_pt(m_pos, m_hw, m_hh, angle_b)
        b_e = _edge_pt(y_pos, y_hw, y_hh, angle_b + 180)
        _draw_arrow(ax, b_s, b_e, color=_coef_color(b_coef), lw=lw)

        b_perp = np.array([-np.sin(np.radians(angle_b)), np.cos(np.radians(angle_b))])
        b_mid = np.array([(b_s[0] + b_e[0]) / 2, (b_s[1] + b_e[1]) / 2])
        b_lbl = b_mid + b_perp * 0.22
        ax.text(
            b_lbl[0],
            b_lbl[1],
            f"b = {_fmt_coef(b_coef, b_p)}",
            fontsize=fs_coef,
            color=_coef_color(b_coef),
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=0.5),
        )

        # ---- path c' (direct): X -> Y, dashed ----------------------------
        cp_coef, cp_p = primary["cp"], primary["cp_p"]
        cp_s = _edge_pt(x_pos, x_hw, x_hh, 0)
        cp_e = _edge_pt(y_pos, y_hw, y_hh, 180)
        _draw_arrow(
            ax,
            cp_s,
            cp_e,
            color=_coef_color(cp_coef),
            linestyle="--",
            lw=lw,
        )
        cprime = r"$c'$"
        ax.text(
            (cp_s[0] + cp_e[0]) / 2,
            cp_s[1] - 0.25,
            f"{cprime} = {_fmt_coef(cp_coef, cp_p)}  (direct)",
            fontsize=fs_coef,
            color=_coef_color(cp_coef),
            fontweight="bold",
            ha="center",
            va="top",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=0.5),
        )

        # ---- confounder arrows --------------------------------------------
        if show_confounders:
            c_hw = bw_c / 2
            c_hh = bh / 2

            _draw_arrow(
                ax,
                (c_pos[0] - c_hw, c_pos[1]),
                _edge_pt(x_pos, x_hw, x_hh, -90),
                color=_CONF_COLOR,
                linestyle="--",
                lw=0.5,
                connectionstyle="arc3,rad=-0.30",
                mutation_scale=8,
            )
            _draw_arrow(
                ax,
                (c_pos[0], c_pos[1] + c_hh),
                _edge_pt(m_pos, m_hw, m_hh, -90),
                color=_CONF_COLOR,
                linestyle="--",
                lw=0.5,
                mutation_scale=8,
            )
            _draw_arrow(
                ax,
                (c_pos[0] + c_hw, c_pos[1]),
                _edge_pt(y_pos, y_hw, y_hh, -90),
                color=_CONF_COLOR,
                linestyle="--",
                lw=0.5,
                connectionstyle="arc3,rad=0.30",
                mutation_scale=8,
            )

        # ---- indirect effect annotation box -------------------------------
        indirect = primary["indirect"]
        ci_lo, ci_hi = primary["ci_low"], primary["ci_high"]
        indirect_se = primary["indirect_se"]

        z = abs(indirect / indirect_se) if indirect_se > 0 else 0
        if z > 3.291:
            stars = "***"
        elif z > 2.576:
            stars = "**"
        elif z > 1.96:
            stars = "*"
        else:
            stars = ""

        ann_lines = [
            f"Indirect (a $\\times$ b) = {indirect:+.3f}{stars}",
            f"95\\% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        ]
        ann_text = "\n".join(ann_lines)

        ax.text(
            4.0,
            4.1,
            ann_text,
            fontsize=fs_note,
            ha="center",
            va="top",
            linespacing=1.4,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="#FFFDE7",
                ec="#FBC02D",
                lw=0.5,
                alpha=0.92,
            ),
        )

        # ---- legend -------------------------------------------------------
        legend_handles = [
            Line2D([0], [0], color=_NEG_COLOR, lw=lw, label="Negative effect"),
            Line2D([0], [0], color=_POS_COLOR, lw=lw, label="Positive effect"),
            Line2D(
                [0], [0], color="gray", lw=lw, ls="--", label=r"Direct effect ($c'$)"
            ),
        ]
        if show_confounders:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=_CONF_COLOR,
                    lw=0.5,
                    ls="--",
                    label="Control variables",
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=7,
            framealpha=0.9,
            edgecolor="#BDBDBD",
            borderpad=0.4,
            handlelength=1.6,
        )

        # ---- save ---------------------------------------------------------
        if not output_path.lower().endswith(".pdf"):
            output_path = os.path.splitext(output_path)[0] + ".pdf"

        plt.savefig(
            output_path,
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
        fig = None  # mark as closed so finally doesn't double-close

        print(f"Mediation diagram saved: {output_path}")
        return output_path

    finally:
        # Guarantee cleanup: close figure, restore rcParams, restore backend
        if fig is not None:
            plt.close(fig)
        for k, v in prev_rc.items():
            matplotlib.rcParams[k] = v
        matplotlib.use(prev_backend)
        importlib.reload(plt)


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------


def main():
    """Load results from YAML and generate the mediation diagram."""
    dir_script = os.path.dirname(os.path.abspath(__file__))
    dir_repo = os.path.dirname(os.path.dirname(dir_script))

    default_yaml = os.path.join(
        dir_repo,
        "data",
        "evaluated",
        "mediation-wind_emotion_stress",
        "mediation_results.yaml",
    )
    default_output = os.path.join(
        dir_repo,
        "data",
        "evaluated",
        "mediation-wind_emotion_stress",
        "causal_diagram_pgf.pdf",
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate mediation path diagram (PGF/LaTeX)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=default_yaml,
        help="Path to YAML file with mediation results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Path for the output PDF",
    )
    parser.add_argument(
        "--no-confounders",
        action="store_true",
        help="Hide confounder control box",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster elements",
    )
    args = parser.parse_args()

    try:
        results = load_results_from_yaml(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun mediation_analysis_main.py first to generate results.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    create_mediation_diagram(
        results,
        args.output,
        show_confounders=not args.no_confounders,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
