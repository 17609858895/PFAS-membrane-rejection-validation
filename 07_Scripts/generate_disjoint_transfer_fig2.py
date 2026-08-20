from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "07_Scripts"
STYLE_ROOT = SCRIPT_ROOT / "figure_style"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(STYLE_ROOT))

import audit_internal_high_accuracy_claim as audit  # noqa: E402
import generate_pfas_paper_figures as core  # noqa: E402
from ml_publication_style import (  # noqa: E402
    configure_publication_style,
    numeric_ticks,
    style_axis,
)
from scientific_palettes import get_palette  # noqa: E402


OUT = ROOT / "03_Figures" / "Fig02_Disjoint_familiar_vs_unseen_study_transfer"
AUDIT_OUT = ROOT / "99_Working_Files" / "high_accuracy_claim_audit"
STEM = "Fig02_Disjoint_familiar_vs_unseen_study_transfer"

configure_publication_style()
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
    }
)

COLORS = get_palette("nature_new", n=9)
BLUE, CORAL, GREEN, TEAL, LAVENDER, PINK, GOLD, PURPLE, GRAY = COLORS
INK = "#1F2933"
MUTED = "#65737E"
LIGHT = "#E6EAF0"


def exact_metrics(df: pd.DataFrame, idx: np.ndarray, pred: np.ndarray) -> dict:
    return core.evaluation(
        df.loc[idx, "Rejection"].to_numpy(float),
        pred,
        df.loc[idx, "is_censored"].to_numpy(bool),
    )


def group_overlap(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, column: str
) -> float:
    training = set(df.loc[train_idx, column].astype(str))
    return float(df.loc[test_idx, column].astype(str).isin(training).mean())


def build_predictions():
    _, df = core.load_data()
    x = core.feature_matrix(df)
    groups = df["ref_id"].astype(str).to_numpy()

    development_idx, unseen_idx = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=audit.SEED
        ).split(x, groups=groups)
    )
    familiar_train_idx, familiar_test_idx = train_test_split(
        development_idx,
        test_size=0.15,
        random_state=audit.SEED + 20,
        shuffle=True,
    )

    best_name = json.loads(
        (AUDIT_OUT / "disjoint_claim_audit.json").read_text(encoding="utf-8")
    )["best_candidate"]
    cfg = next(item for item in audit.CANDIDATES if item["name"] == best_name)
    model = audit.fit("aft", x, df, familiar_train_idx, cfg, audit.SEED + 900)
    familiar_pred = audit.predict("aft", model, x[familiar_test_idx])
    unseen_pred = audit.predict("aft", model, x[unseen_idx])

    familiar_metrics = exact_metrics(df, familiar_test_idx, familiar_pred)
    unseen_metrics = exact_metrics(df, unseen_idx, unseen_pred)
    overlaps = {
        "Familiar-row test": {
            "Study": group_overlap(df, familiar_train_idx, familiar_test_idx, "ref_id"),
            "Membrane": group_overlap(df, familiar_train_idx, familiar_test_idx, "mem"),
            "PFAS": group_overlap(df, familiar_train_idx, familiar_test_idx, "PFAS"),
        },
        "Unseen-study test": {
            "Study": group_overlap(df, familiar_train_idx, unseen_idx, "ref_id"),
            "Membrane": group_overlap(df, familiar_train_idx, unseen_idx, "mem"),
            "PFAS": group_overlap(df, familiar_train_idx, unseen_idx, "PFAS"),
        },
    }

    rows = []
    for label, idx, pred in (
        ("Familiar-row test", familiar_test_idx, familiar_pred),
        ("Unseen-study test", unseen_idx, unseen_pred),
    ):
        for pos, row_idx in enumerate(idx):
            rows.append(
                {
                    "evaluation": label,
                    "row_index": int(row_idx),
                    "study": str(df.loc[row_idx, "ref_id"]),
                    "membrane": str(df.loc[row_idx, "mem"]),
                    "PFAS": str(df.loc[row_idx, "PFAS"]),
                    "observed_or_lower_bound": float(df.loc[row_idx, "Rejection"]),
                    "predicted_P50": float(pred[pos]),
                    "is_censored": bool(df.loc[row_idx, "is_censored"]),
                }
            )
    prediction_frame = pd.DataFrame(rows)
    prediction_frame.to_csv(
        AUDIT_OUT / "disjoint_row_predictions.csv", index=False, encoding="utf-8-sig"
    )

    metric_frame = pd.DataFrame(
        [
            {"evaluation": "Familiar-row test", **familiar_metrics},
            {"evaluation": "Unseen-study test", **unseen_metrics},
        ]
    )
    metric_frame.to_csv(
        AUDIT_OUT / "disjoint_metric_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"evaluation": evaluation, "group": group, "overlap": value}
            for evaluation, values in overlaps.items()
            for group, value in values.items()
        ]
    ).to_csv(AUDIT_OUT / "disjoint_group_overlap.csv", index=False, encoding="utf-8-sig")

    return (
        df,
        familiar_train_idx,
        familiar_test_idx,
        unseen_idx,
        familiar_pred,
        unseen_pred,
        familiar_metrics,
        unseen_metrics,
        overlaps,
        cfg,
    )


def compact_axis(ax, xlabel=None, ylabel=None):
    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        tick_size=9.5,
        label_size=11.5,
    )


def split_composition_panel(ax, n_train, n_familiar, n_unseen, cfg):
    study_segments = [
        ("Development", 48, GREEN),
        ("Unseen", 13, CORAL),
    ]
    row_segments = [
        ("Fit + tune", n_train, GREEN),
        ("Familiar", n_familiar, BLUE),
        ("Unseen", n_unseen, CORAL),
    ]
    for ypos, segments in zip([1, 0], [study_segments, row_segments]):
        total = float(sum(value for _, value, _ in segments))
        left = 0.0
        for label, value, color in segments:
            width = 100.0 * value / total
            ax.barh(
                ypos,
                width,
                left=left,
                height=0.56,
                color=color,
                edgecolor=INK,
                linewidth=0.8,
            )
            ax.text(
                left + width / 2,
                ypos,
                f"{label}\n{value:,}",
                ha="center",
                va="center",
                fontsize=7.2 if width < 18 else 7.8,
                color=INK,
            )
            left += width
    ax.set_yticks([1, 0], ["Studies", "PFAS rows"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.62, 1.62)
    compact_axis(ax, xlabel="Allocation (%)")
    numeric_ticks(ax, x=True, y=False, n=5)


def scatter_panel(ax, df, idx, pred, metrics, title, color):
    exact = ~df.loc[idx, "is_censored"].to_numpy(bool)
    observed = df.loc[idx, "Rejection"].to_numpy(float)
    ax.scatter(
        observed[exact],
        pred[exact],
        s=15,
        alpha=0.50,
        color=color,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0.5, 1.0], [0.5, 1.0], color="#52616B", lw=1.1, ls="--", zorder=0)
    ax.set_xlim(0.49, 1.01)
    ax.set_ylim(0.49, 1.01)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    compact_axis(ax, xlabel="Observed rejection", ylabel="Predicted rejection")
    ax.set_title(title, pad=5, color=color)
    ax.text(
        0.03,
        0.97,
        f"R² (logit) = {metrics['logit_r2']:.3f}\n"
        f"R² (rejection) = {metrics['r2']:.3f}\n"
        f"C-index = {metrics['c_index']:.3f}\n"
        f"n = {metrics['n_exact']} exact",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": LIGHT, "alpha": 0.90},
    )


def metric_panel(ax, familiar, unseen):
    labels = ["Logit R²", "Rejection R²", "C-index"]
    familiar_values = np.array(
        [familiar["logit_r2"], familiar["r2"], familiar["c_index"]]
    )
    unseen_values = np.array([unseen["logit_r2"], unseen["r2"], unseen["c_index"]])
    y = np.arange(len(labels))[::-1]

    for ypos, familiar_value, unseen_value in zip(y, familiar_values, unseen_values):
        ax.plot(
            [unseen_value, familiar_value],
            [ypos, ypos],
            color="#CBD2D9",
            lw=2.0,
            zorder=1,
        )
        ax.scatter(familiar_value, ypos, s=58, color=BLUE, edgecolor="white", lw=0.7, zorder=3)
        ax.scatter(unseen_value, ypos, s=58, color=CORAL, edgecolor="white", lw=0.7, zorder=3)
        ax.text(
            familiar_value + 0.025,
            ypos,
            f"{familiar_value:.2f}",
            ha="left",
            va="center",
            fontsize=7.3,
            color=INK,
        )
        ax.text(
            unseen_value - 0.025,
            ypos,
            f"{unseen_value:.2f}",
            ha="right",
            va="center",
            fontsize=7.3,
            color=INK,
        )
        ax.text(
            (familiar_value + unseen_value) / 2,
            ypos + 0.17,
            f"Δ {unseen_value - familiar_value:+.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=MUTED,
        )

    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.45, 2.45)
    ax.set_yticks(y, labels)
    compact_axis(ax, xlabel="Metric value")
    numeric_ticks(ax, x=True, y=False, n=5)
    ax.text(0.23, 1.035, "Familiar rows", transform=ax.transAxes, color=BLUE, fontsize=7.7)
    ax.text(0.68, 1.035, "Unseen studies", transform=ax.transAxes, color=CORAL, fontsize=7.7)


def error_panel(ax, df, familiar_idx, familiar_pred, unseen_idx, unseen_pred, familiar, unseen):
    series = []
    for label, idx, pred, color in (
        ("Familiar", familiar_idx, familiar_pred, BLUE),
        ("Unseen", unseen_idx, unseen_pred, CORAL),
    ):
        exact = ~df.loc[idx, "is_censored"].to_numpy(bool)
        observed = df.loc[idx, "Rejection"].to_numpy(float)[exact]
        errors = 100 * np.abs(np.asarray(pred)[exact] - observed)
        ordered = np.sort(errors)
        cumulative = np.arange(1, ordered.size + 1) / ordered.size
        ax.step(ordered, cumulative, where="post", lw=2.2, color=color, label=label)
        median = float(np.median(errors))
        ax.scatter([median], [0.5], s=34, color=color, edgecolor="white", lw=0.6, zorder=4)
        series.append((label, errors, color))

    maximum = max(float(np.max(errors)) for _, errors, _ in series)
    upper = 5 * np.ceil(maximum / 5)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, 1.02)
    compact_axis(
        ax,
        xlabel="Absolute error (percentage points)",
        ylabel="Cumulative fraction",
    )
    numeric_ticks(ax, x=True, y=True, n=5)
    ax.text(
        0.97,
        0.20,
        f"Familiar: RMSE {100 * familiar['rmse']:.2f} | MAE {100 * familiar['mae']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.2,
        color=BLUE,
    )
    ax.text(
        0.97,
        0.09,
        f"Unseen: RMSE {100 * unseen['rmse']:.2f} | MAE {100 * unseen['mae']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.2,
        color=CORAL,
    )


def overlap_panel(ax, overlaps):
    labels = ["Study", "Membrane", "PFAS"]
    familiar = [100 * overlaps["Familiar-row test"][label] for label in labels]
    unseen = [100 * overlaps["Unseen-study test"][label] for label in labels]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, familiar, width, label="Familiar rows", color=BLUE)
    ax.bar(x + width / 2, unseen, width, label="Unseen studies", color=CORAL)
    for offset, values in ((-width / 2, familiar), (width / 2, unseen)):
        for pos, value in zip(x + offset, values):
            ax.text(pos, value + 2.2, f"{value:.0f}%", ha="center", va="bottom", fontsize=7.2, color=INK)
    ax.set_ylim(0, 112)
    ax.set_xticks(x, labels)
    compact_axis(ax, ylabel="Test rows overlapping training (%)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    (
        df,
        train_idx,
        familiar_idx,
        unseen_idx,
        familiar_pred,
        unseen_pred,
        familiar_metrics,
        unseen_metrics,
        overlaps,
        cfg,
    ) = build_predictions()

    panels = [
        lambda ax: split_composition_panel(ax, len(train_idx), len(familiar_idx), len(unseen_idx), cfg),
        lambda ax: scatter_panel(ax, df, familiar_idx, familiar_pred, familiar_metrics, "Familiar rows", BLUE),
        lambda ax: scatter_panel(ax, df, unseen_idx, unseen_pred, unseen_metrics, "Unseen studies", CORAL),
        lambda ax: metric_panel(ax, familiar_metrics, unseen_metrics),
        lambda ax: error_panel(
            ax,
            df,
            familiar_idx,
            familiar_pred,
            unseen_idx,
            unseen_pred,
            familiar_metrics,
            unseen_metrics,
        ),
        lambda ax: overlap_panel(ax, overlaps),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4))
    for drawer, ax in zip(panels, axes.flat):
        drawer(ax)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.085, top=0.945, wspace=0.39, hspace=0.42)

    core.save_publication_figure(
        fig,
        OUT,
        STEM,
        panels,
        [(7.0, 5.2), (6.0, 5.2), (6.0, 5.2), (6.4, 4.8), (6.4, 4.8), (6.2, 4.8)],
        source_script=Path(__file__),
        combined_axes=axes,
    )
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
