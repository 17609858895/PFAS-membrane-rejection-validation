"""Draw Figs S5-S7 from the saved Round-2 tables (no model fitting).

Inputs: 04_Tables/Round2/R2_G*, R2_H01, R2_I*; 04_Tables/group_conformal_fold_metrics.csv;
04_Tables/Revision/Table_R3_12_study_unit_conformal_folds.csv.
Outputs: 03_Figures/Supplementary/FigS05_AFT_Diagnostics, FigS06_Censoring_Simulation,
FigS07_Out_of_Fold_Applicability (or under $PFAS_OUTPUT_ROOT).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("PFAS_OUTPUT_ROOT", ROOT))
TABLES = OUTPUT_ROOT / "04_Tables"
R2 = TABLES / "Round2"
FIG_ROOT = OUTPUT_ROOT / "03_Figures" / "Supplementary"
sys.path.insert(0, str(ROOT / "07_Scripts"))
sys.path.insert(0, str(ROOT / "07_Scripts" / "figure_style"))
import generate_pfas_paper_figures as core  # noqa: E402
from ml_publication_style import configure_publication_style, style_axis  # noqa: E402

BLUE, CORAL, GREEN, TEAL, PURPLE, GRAY = core.BLUE, core.CORAL, core.GREEN, core.TEAL, core.PURPLE, core.GRAY
INK, MUTED = core.INK, core.MUTED
TICK, LABEL = 11.5, 14.0


def nelson_aalen(resid: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(resid)
    r, e = resid[order], event[order]
    at_risk = len(r) - np.arange(len(r))
    return r, np.cumsum(e / at_risk)


def fig_s5() -> None:
    rows = pd.read_csv(R2 / "R2_G03_aft_row_residuals.csv")
    diag = pd.read_csv(R2 / "R2_G01_aft_likelihood_diagnostics.csv")
    row_iv = pd.read_csv(TABLES / "group_conformal_fold_metrics.csv").groupby("nominal")["coverage_exact"].mean()
    study_iv = pd.read_csv(TABLES / "Revision" / "Table_R3_12_study_unit_conformal_folds.csv").groupby("nominal")["row_exact_coverage"].mean()
    cens = rows["is_censored"].astype(bool).to_numpy()
    nominal = np.array([0.80, 0.90, 0.95])
    labels = {"nominal": ("Nominal AFT scale (σ = 1.2)", CORAL), "recal": ("Fold-recalibrated scale", BLUE)}

    def f_a(ax):
        for key, (lab, col) in labels.items():
            r = np.sort(rows[f"rqr_{key}"].to_numpy())
            q = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
            ax.plot(q, r, ".", ms=2.2, color=col, alpha=0.6, label=lab)
        lim = [-4, 4]
        ax.plot(lim, lim, "--", color=MUTED, lw=1)
        ax.set_xlim(lim); ax.set_ylim(-5, 6)
        style_axis(ax, xlabel="Normal quantile", ylabel="Randomized quantile residual", tick_size=TICK, label_size=LABEL)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left", markerscale=4)

    def f_b(ax):
        for key, (lab, col) in labels.items():
            r, h = nelson_aalen(rows[f"cox_snell_{key}"].to_numpy(), ~cens)
            keep = r <= np.quantile(r, 0.99)
            ax.plot(r[keep], h[keep], color=col, lw=2)
        ax.plot([0, 4], [0, 4], "--", color=MUTED, lw=1)
        ax.set_xlim(0, 4); ax.set_ylim(0, 4)
        style_axis(ax, xlabel="Cox–Snell residual", ylabel="Cumulative hazard", tick_size=TICK, label_size=LABEL)

    def f_c(ax):
        series = [
            ("AFT, σ = 1.2", [diag.loc[0, f"exact_coverage_{int(n*100)}"] for n in nominal], CORAL, "o"),
            ("AFT, recalibrated σ", [diag.loc[1, f"exact_coverage_{int(n*100)}"] for n in nominal], BLUE, "s"),
            ("Row-score empirical", [row_iv.loc[n] for n in nominal], TEAL, "^"),
            ("Study-score empirical", [study_iv.loc[n] for n in nominal], PURPLE, "D"),
        ]
        for lab, cov, col, mk in series:
            ax.plot(nominal * 100, np.asarray(cov) * 100, marker=mk, color=col, lw=2, label=lab)
        ax.plot([78, 97], [78, 97], "--", color=MUTED, lw=1)
        ax.set_xticks([80, 90, 95])
        style_axis(ax, xlabel="Nominal coverage (%)", ylabel="Exact-row coverage (%)", tick_size=TICK, label_size=LABEL)
        ax.legend(frameon=False, fontsize=9.0, loc="lower right")

    def f_d(ax):
        for key, (lab, col) in labels.items():
            z = (rows["logit_observed_or_bound"] - rows["margin_f"]) / (1.2 if key == "nominal" else rows["sigma_star_fold"])
            p = 1 - stats.norm.cdf(z.to_numpy()[cens])
            ax.hist(p, bins=np.linspace(0, 1, 21), color=col, alpha=0.55, label=lab, edgecolor="white")
        ax.axvline(0.5, color=INK, lw=1, ls="--")
        style_axis(ax, xlabel="P(rejection ≥ bound)", ylabel="Censored rows", tick_size=TICK, label_size=LABEL)

    panels = [f_a, f_b, f_c, f_d]
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.42, bottom=0.18)
    core.save_publication_figure(fig, FIG_ROOT / "FigS05_AFT_Diagnostics", "FigS05_aft_likelihood_diagnostics", panels,
                                 [(5.8, 4.8)] * 4, source_script=Path(__file__), combined_axes=axes)


def fig_s6() -> None:
    sim = pd.read_csv(R2 / "R2_H01_censoring_mechanism_simulation.csv")
    methods = ["AFT (bounds as lower bounds)", "Bounds treated as exact", "Censored rows excluded"]
    short = {"AFT (bounds as lower bounds)": "AFT bounds", "Bounds treated as exact": "Bounds as exact", "Censored rows excluded": "Excluded"}
    colors = [BLUE, CORAL, GREEN]
    levels = sorted(sim["target_censoring"].unique())

    def grouped(ax, column, ylabel):
        width = 0.26
        x = np.arange(len(levels))
        for k, (m, col) in enumerate(zip(methods, colors)):
            vals = [sim.loc[(sim.method == m) & (sim.target_censoring == lv), column].iloc[0] for lv in levels]
            ax.bar(x + (k - 1) * width, vals, width, color=col, label=short[m], edgecolor="white")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [f"{lv*100:.1f}%".replace(".0%", "%") for lv in levels])
        style_axis(ax, xlabel="Censored share", ylabel=ylabel, tick_size=TICK, label_size=LABEL)

    def f_a(ax):
        grouped(ax, "bias_vs_true_censored_pp", "Bias on censored rows (pp)")
        ax.legend(frameon=False, fontsize=9.5, loc="lower left")

    def f_b(ax):
        grouped(ax, "rmse_vs_true_censored_pp", "RMSE on censored rows (pp)")

    def f_c(ax):
        part = sim.loc[np.isclose(sim.target_censoring, 0.127)]
        for k, (m, col) in enumerate(zip(methods, colors)):
            row = part.loc[part.method == m].iloc[0]
            ax.plot([0, 1], [row["naive_rmse_vs_reported_all_pp"], row["rmse_vs_true_all_pp"]], marker="o", color=col, lw=2, label=short[m])
        ax.set_xticks([0, 1], ["Bounds as\nobserved", "Known\nvalues"])
        ax.set_xlim(-0.3, 1.3)
        style_axis(ax, xlabel="Evaluation target (12.7%)", ylabel="RMSE, all rows (pp)", tick_size=TICK, label_size=LABEL)

    panels = [f_a, f_b, f_c]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.42, bottom=0.2)
    core.save_publication_figure(fig, FIG_ROOT / "FigS06_Censoring_Simulation", "FigS06_censoring_mechanism_simulation", panels,
                                 [(5.8, 4.8)] * 3, source_script=Path(__file__), combined_axes=axes)


def fig_s7() -> None:
    rows = pd.read_csv(R2 / "R2_I04_row_distances_out_of_fold.csv")
    curve = pd.read_csv(R2 / "R2_I03_out_of_fold_reject_curve.csv")
    exact = ~rows["is_censored"].astype(bool)
    e = rows.loc[exact]

    def f_a(ax):
        ratio = (e["knn8_distance"] / e["fold_threshold"]).to_numpy()
        err = 100 * e["abs_error"].to_numpy()
        ax.scatter(ratio, err, s=5, color=BLUE, alpha=0.35, linewidths=0)
        edges = np.array([0, 0.25, 0.5, 0.7, 0.85, 1.0, 1.25, np.inf])
        idx = np.digitize(ratio, edges[1:-1])
        keep = [b for b in range(len(edges) - 1) if np.sum(idx == b) >= 20]
        centers = [np.median(ratio[idx == b]) for b in keep]
        means = [np.mean(err[idx == b]) for b in keep]
        ax.plot(centers, means, "-o", color=CORAL, lw=2, ms=4)
        ax.axvline(1.0, color=INK, lw=1, ls="--")
        ax.set_xscale("log")
        style_axis(ax, xlabel="Distance / training-only threshold", ylabel="Absolute error (pp)", tick_size=TICK, label_size=LABEL)

    def f_b(ax):
        ax.plot(100 * curve["withheld_fraction"], curve["rmse_retained_pp"], "-o", color=TEAL, lw=2, ms=4)
        ax.axhline(curve["rmse_retained_pp"].iloc[0], color=MUTED, lw=1, ls="--")
        style_axis(ax, xlabel="Withheld exact rows (%)", ylabel="Retained RMSE (pp)", tick_size=TICK, label_size=LABEL)

    def f_c(ax):
        flagged = e["outside_training_only_rule"].astype(bool)
        large = e["abs_error"] > 0.10
        vals = np.array([[np.sum(~flagged & ~large), np.sum(~flagged & large)], [np.sum(flagged & ~large), np.sum(flagged & large)]])
        frac = vals / vals.sum(axis=1, keepdims=True) * 100
        ax.bar([0, 1], frac[:, 0], color=GREEN, label="Error ≤ 10 pp", edgecolor="white")
        ax.bar([0, 1], frac[:, 1], bottom=frac[:, 0], color=CORAL, label="Error > 10 pp", edgecolor="white")
        for k in range(2):
            ax.text(k, 101, f"n = {vals[k].sum()}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks([0, 1], ["Retained", "Flagged"])
        ax.set_ylim(0, 112)
        style_axis(ax, xlabel="Training-only rule", ylabel="Exact rows (%)", tick_size=TICK, label_size=LABEL)
        ax.legend(frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    panels = [f_a, f_b, f_c]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.42, bottom=0.18)
    core.save_publication_figure(fig, FIG_ROOT / "FigS07_Out_of_Fold_Applicability", "FigS07_out_of_fold_applicability_screening", panels,
                                 [(5.8, 4.8)] * 3, source_script=Path(__file__), combined_axes=axes)


def main() -> None:
    configure_publication_style()
    fig_s5()
    fig_s6()
    fig_s7()
    print(f"Figs S5-S7 written to {FIG_ROOT}")


if __name__ == "__main__":
    main()
