"""
Advanced ML figure templates collected from the user's reference libraries.

These functions are intentionally lightweight and notebook-friendly. They cover
figure types not already handled by the main regression/SAC templates:
classification ROC comparison, label-group checks, feature-selection consensus,
optimizer radar/traces, Williams applicability-domain plots, Sobol/Monte Carlo
uncertainty, and DML/Causal Forest effect summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


NATURE_COLORS = {
    "blue": "#4C78A8",
    "teal": "#2A9D8F",
    "green": "#59A14F",
    "coral": "#E76F51",
    "gold": "#F2C14E",
    "navy": "#2F3E5C",
    "gray": "#6C757D",
    "light_gray": "#E9ECEF",
}


def configure_advanced_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "savefig.dpi": 600,
            "figure.dpi": 160,
            "axes.linewidth": 1.55,
            "axes.labelweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "legend.frameon": False,
        }
    )


def style_axis(ax, grid: bool = False) -> None:
    ax.grid(grid, color="#E6E8EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.55)
    ax.spines["bottom"].set_linewidth(1.55)
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.5, bottom=True, left=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def plot_roc_auc_comparison(y_true, score_dict: dict[str, np.ndarray], ax=None):
    """ROC curves plus AUC labels for classification papers.

    Use this when the reference task is classification. For formal model
    comparison, compute DeLong or bootstrap p values separately and place them
    in the caption/table rather than overloading the ROC panel.
    """
    from sklearn.metrics import auc, roc_curve

    configure_advanced_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5.0))
    colors = [NATURE_COLORS[k] for k in ["blue", "teal", "coral", "green", "navy", "gold"]]
    for i, (name, scores) in enumerate(score_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, lw=2.1, color=colors[i % len(colors)], label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color="#9AA4B2")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    style_axis(ax)
    return ax


def plot_label_group_significance(summary_df: pd.DataFrame, ax=None):
    """Bubble/lollipop view for testing whether classification labels differ.

    Expected columns: feature, effect, p_value. Optional column: group.
    """
    configure_advanced_style()
    data = summary_df.copy().sort_values("effect")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, max(3.6, 0.38 * len(data))))
    y = np.arange(len(data))
    sig = -np.log10(data["p_value"].clip(lower=1e-300))
    sizes = 36 + 42 * np.clip(sig / max(sig.max(), 1), 0, 1)
    colors = np.where(data["p_value"] < 0.05, NATURE_COLORS["coral"], NATURE_COLORS["gray"])
    ax.axvline(0, color="#AAB2BD", lw=1.3, ls="--")
    ax.scatter(data["effect"], y, s=sizes, facecolors="white", edgecolors=colors, linewidths=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Standardized group difference")
    ax.set_ylabel("")
    style_axis(ax)
    return ax


def plot_feature_selection_consensus(selection_df: pd.DataFrame, ax=None):
    """Consensus plot for Boruta/RFE/GA/LGBM feature-selection results.

    Expected columns: feature and one binary/score column per selection method.
    """
    configure_advanced_style()
    data = selection_df.copy()
    method_cols = [c for c in data.columns if c != "feature"]
    data["consensus"] = data[method_cols].sum(axis=1)
    data = data.sort_values(["consensus", "feature"], ascending=[True, True])
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, max(4.0, 0.36 * len(data))))
    y = np.arange(len(data))
    ax.barh(y, data["consensus"], color=NATURE_COLORS["teal"], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Selection consensus score")
    ax.set_xlim(0, max(len(method_cols), 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    style_axis(ax)
    return ax


def plot_optimizer_trace(trace_df: pd.DataFrame, ax=None):
    """Bayesian/GA/PSO tuning trace.

    Expected columns: iteration, score, optimizer.
    """
    configure_advanced_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = [NATURE_COLORS[k] for k in ["blue", "teal", "coral", "navy"]]
    for i, (name, sub) in enumerate(trace_df.groupby("optimizer", sort=False)):
        sub = sub.sort_values("iteration")
        best = sub["score"].cummax()
        ax.plot(sub["iteration"], best, lw=2.0, marker="o", ms=4.5, color=colors[i % len(colors)], label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best validation score")
    ax.legend(loc="best")
    style_axis(ax)
    return ax


def plot_metric_radar(metric_df: pd.DataFrame, ax=None):
    """Radar chart for model or optimizer performance comparison.

    Rows are models/optimizers, columns are normalized metrics where higher is
    better. Keep to 3-6 metrics and 2-5 methods.
    """
    configure_advanced_style()
    labels = list(metric_df.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.r_[angles, angles[0]]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw={"polar": True})
    colors = [NATURE_COLORS[k] for k in ["blue", "teal", "coral", "navy", "gold"]]
    for i, (name, row) in enumerate(metric_df.iterrows()):
        values = np.r_[row.to_numpy(dtype=float), row.iloc[0]]
        ax.plot(angles, values, lw=2.0, color=colors[i % len(colors)], label=str(name))
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold", fontsize=11)
    ax.set_yticklabels([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    return ax


def plot_williams_applicability_domain(leverage, std_residual, split=None, h_star=None, ax=None):
    """Williams plot: leverage vs standardized residual.

    Use for applicability-domain checks in QSAR/QSPR-like regression. Negative
    residual values are expected and are not a plotting error.
    """
    configure_advanced_style()
    leverage = np.asarray(leverage, dtype=float)
    std_residual = np.asarray(std_residual, dtype=float)
    if split is None:
        split = np.array(["Samples"] * len(leverage))
    if h_star is None:
        h_star = 3 * np.nanmean(leverage)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.8))
    colors = {"Train": NATURE_COLORS["blue"], "Test": NATURE_COLORS["coral"], "Samples": NATURE_COLORS["teal"]}
    for name in pd.unique(split):
        mask = np.asarray(split) == name
        ax.scatter(leverage[mask], std_residual[mask], s=58, facecolors="white",
                   edgecolors=colors.get(str(name), NATURE_COLORS["gray"]), linewidths=1.7, label=str(name))
    ax.axhline(3, color="#AAB2BD", lw=1.4, ls="--")
    ax.axhline(-3, color="#AAB2BD", lw=1.4, ls="--")
    ax.axvline(h_star, color=NATURE_COLORS["coral"], lw=1.4, ls="--")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residual")
    ax.legend(loc="best")
    style_axis(ax)
    return ax


def plot_sobol_sensitivity(sobol_df: pd.DataFrame, ax=None):
    """Sobol sensitivity bar plot.

    Expected columns: feature, S1, ST. Optional columns: S1_err, ST_err.
    """
    configure_advanced_style()
    data = sobol_df.copy().sort_values("ST")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, max(4.0, 0.36 * len(data))))
    y = np.arange(len(data))
    ax.barh(y - 0.18, data["S1"], height=0.34, color=NATURE_COLORS["blue"], label="First-order")
    ax.barh(y + 0.18, data["ST"], height=0.34, color=NATURE_COLORS["teal"], label="Total")
    ax.set_yticks(y)
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Sobol sensitivity index")
    ax.legend(loc="lower right")
    style_axis(ax)
    return ax


def plot_monte_carlo_uncertainty(x, y_mean, y_low, y_high, ax=None):
    """Mean prediction with Monte Carlo uncertainty band."""
    configure_advanced_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.fill_between(x, y_low, y_high, color=NATURE_COLORS["blue"], alpha=0.18, linewidth=0)
    ax.plot(x, y_mean, color=NATURE_COLORS["blue"], lw=2.2)
    ax.set_xlabel("Input")
    ax.set_ylabel("Predicted response")
    style_axis(ax)
    return ax


def plot_causal_effect_forest(effect_df: pd.DataFrame, ax=None):
    """ATE/CATE forest plot for DML/Causal Forest results.

    Expected columns: subgroup, effect, ci_low, ci_high.
    """
    configure_advanced_style()
    data = effect_df.copy().sort_values("effect")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, max(3.8, 0.42 * len(data))))
    y = np.arange(len(data))
    xerr = np.vstack([data["effect"] - data["ci_low"], data["ci_high"] - data["effect"]])
    ax.axvline(0, color="#AAB2BD", lw=1.4, ls="--")
    ax.errorbar(data["effect"], y, xerr=xerr, fmt="o", ms=6.5, mfc="white",
                mec=NATURE_COLORS["navy"], ecolor=NATURE_COLORS["navy"],
                elinewidth=1.7, capsize=3.2)
    ax.set_yticks(y)
    ax.set_yticklabels(data["subgroup"])
    ax.set_xlabel("Estimated treatment effect")
    ax.set_ylabel("")
    style_axis(ax)
    return ax


ADVANCED_TEMPLATE_MAP = {
    "classification_roc_delong": ["plot_roc_auc_comparison"],
    "label_group_significance": ["plot_label_group_significance"],
    "boruta_rfe_ga_consensus": ["plot_feature_selection_consensus"],
    "bayes_ga_pso_tuning": ["plot_optimizer_trace", "plot_metric_radar"],
    "williams_applicability_domain": ["plot_williams_applicability_domain"],
    "shap_sobol_monte_carlo": ["plot_sobol_sensitivity", "plot_monte_carlo_uncertainty"],
    "dml_causal_forest": ["plot_causal_effect_forest"],
}
