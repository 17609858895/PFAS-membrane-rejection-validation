from __future__ import annotations

import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.startswith("Fig"):
    ROOT = SCRIPT_PATH.parents[2]
else:
    ROOT = SCRIPT_PATH.parents[1]
FIG_ROOT = ROOT / "03_Figures"
TABLE_ROOT = ROOT / "04_Tables"
TMP_ROOT = ROOT / "99_Working_Files"
JSON_PATH = TMP_ROOT / "sheets" / "pfas_data.json"
WORKBOOK_PATH = ROOT / "05_Source_Data" / "es6c02287_si_002.xlsx"
STYLE_ROOT = ROOT / "07_Scripts" / "figure_style"

import sys

sys.path.insert(0, str(STYLE_ROOT))
from ml_publication_style import (  # noqa: E402
    AXIS_LABEL_PAD,
    MAJOR_TICK_PAD,
    configure_publication_style,
    numeric_ticks,
    save_figure_with_subfigures,
    style_axis,
)
from scientific_palettes import PLOT_COLORS, STYLE_COLORS, get_palette  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
configure_publication_style()

# Multi-panel figures need slightly smaller type than standalone panels, but
# axis labels, ticks and panel keys must remain readable after journal scaling.
FIGURE_TEXT_SCALE = 0.90

PALETTE = get_palette("nature_new", n=9)
BLUE, CORAL, GREEN, TEAL, LAVENDER, PINK, GOLD, PURPLE, GRAY = PALETTE
INK = STYLE_COLORS["ink"]
MUTED = STYLE_COLORS["muted"]
LIGHT = STYLE_COLORS["light"]
DIVERGING = LinearSegmentedColormap.from_list(
    "pfas_diverging", ["#4E85AC", "#F7F7F7", "#D8292C"]
)

FEATURES = [
    "pwp_LMH/bar",
    "ON",
    "pH",
    "volume_Å3",
    "dipole_debye",
    "pKa",
    "temp_ºC",
    "flux_LMH",
    "pfas_feed_conc_µg/L",
    "mon_cat_mM",
    "di_cat_mM",
    "tri_cat_mM",
    "doc_mg/L",
]
LOG_FEATURES = {
    "flux_LMH",
    "pfas_feed_conc_µg/L",
    "mon_cat_mM",
    "di_cat_mM",
    "tri_cat_mM",
    "doc_mg/L",
}
FEATURE_LABELS = {
    "pwp_LMH/bar": "Water permeance",
    "ON": "O:N ratio",
    "pH": "pH",
    "volume_Å3": "PFAS volume",
    "dipole_debye": "Dipole moment",
    "pKa": "pKa",
    "temp_ºC": "Temperature",
    "flux_LMH": "Permeate flux",
    "pfas_feed_conc_µg/L": "PFAS feed concentration",
    "mon_cat_mM": "Monovalent cations",
    "di_cat_mM": "Divalent cations",
    "tri_cat_mM": "Trivalent cations",
    "doc_mg/L": "DOC",
}
COMPACT_FEATURE_LABELS = {
    **FEATURE_LABELS,
    "pwp_LMH/bar": "Permeance",
    "pfas_feed_conc_µg/L": "Feed concentration",
    "mon_cat_mM": "Monovalent ions",
    "di_cat_mM": "Divalent ions",
    "tri_cat_mM": "Trivalent ions",
}


def logistic(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def clean_axis(ax, xlabel=None, ylabel=None, tick_size=12.5, label_size=15.5):
    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        tick_size=tick_size,
        label_size=label_size,
    )
    ax.set_facecolor("white")


def add_metric_box(
    ax,
    lines: list[str],
    loc=(0.04, 0.96),
    *,
    ha="left",
    va="top",
):
    ax.text(
        loc[0],
        loc[1],
        "\n".join(lines),
        transform=ax.transAxes,
        va=va,
        ha=ha,
        fontsize=11.5,
        fontweight="normal",
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CDD3D8",
            "alpha": 0.92,
        },
    )


def scale_figure_text(fig: plt.Figure, factor: float = FIGURE_TEXT_SCALE) -> None:
    """Scale text while enforcing the user's regular-weight figure style."""
    if getattr(fig, "_pfas_text_scaled", False):
        return
    for artist in fig.findobj(match=Text):
        size = artist.get_fontsize()
        if np.isfinite(size):
            artist.set_fontsize(max(1.0, size * factor))
        artist.set_fontweight("normal")
    for ax in fig.axes:
        tick_pad = getattr(ax, "_publication_tick_pad", MAJOR_TICK_PAD)
        ax.tick_params(axis="both", which="major", pad=tick_pad)
        for label in ax.get_xticklabels(minor=False) + ax.get_xticklabels(minor=True):
            label.set_horizontalalignment("center")
            label.set_rotation_mode("anchor")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("normal")
            label.set_fontsize(max(11.2, label.get_fontsize()))
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontweight("normal")
            label.set_fontsize(max(13.8, label.get_fontsize()))
        if ax.get_xlabel().strip():
            ax.xaxis.labelpad = getattr(
                ax, "_publication_xlabel_pad", AXIS_LABEL_PAD
            )
        if ax.get_ylabel().strip():
            ax.yaxis.labelpad = getattr(
                ax, "_publication_ylabel_pad", AXIS_LABEL_PAD
            )
    fig._pfas_text_scaled = True


def align_composite_panel_grid(fig: plt.Figure, axes: list) -> None:
    """Record the intended grid and snap only the plotting-area boundaries."""
    if len(axes) < 2:
        return
    fig.canvas.draw()
    records = []
    for ax in axes:
        pos = ax.get_position()
        records.append(
            {
                "axis_on": bool(ax.axison),
                "ax": ax,
                "x0": pos.x0,
                "x1": pos.x1,
                "y0": pos.y0,
                "y1": pos.y1,
                "left_group": round(pos.x0, 6),
                "right_group": round(pos.x1, 6),
                "bottom_group": round(pos.y0, 6),
                "top_group": round(pos.y1, 6),
                "row_group": (round(pos.y0, 6), round(pos.y1, 6)),
                "col_group": (round(pos.x0, 6), round(pos.x1, 6)),
            }
        )
    if len(records) < 2:
        return

    for record in records:
        ax = record["ax"]
        ax._publication_left_group = record["left_group"]
        ax._publication_right_group = record["right_group"]
        ax._publication_bottom_group = record["bottom_group"]
        ax._publication_top_group = record["top_group"]

    def targets(group_key: str, value_key: str) -> dict[float, float]:
        grouped: dict[float, list[float]] = {}
        for record in records:
            grouped.setdefault(record[group_key], []).append(record[value_key])
        return {key: float(np.median(values)) for key, values in grouped.items()}

    left = targets("left_group", "x0")
    right = targets("right_group", "x1")
    bottom = targets("bottom_group", "y0")
    top = targets("top_group", "y1")
    for record in records:
        x0 = left[record["left_group"]]
        x1 = right[record["right_group"]]
        y0 = bottom[record["bottom_group"]]
        y1 = top[record["top_group"]]
        if x1 > x0 and y1 > y0:
            record["ax"].set_position([x0, y0, x1 - x0, y1 - y0])
    fig.canvas.draw()


def compact_requested_ylabel_gaps(fig: plt.Figure, axes: list) -> dict[str, float]:
    """Place selected y-axis titles at a measured gap from their tick labels.

    Panels opt in with ``_publication_ylabel_gap_pt``.  The label pad is
    corrected in points, so the requested gap survives later axis movement and
    resizing during visible-boundary alignment.
    """
    measured: dict[str, float] = {}
    for panel_index, ax in enumerate(axes):
        requested_gap = getattr(ax, "_publication_ylabel_gap_pt", None)
        if requested_gap is None or not ax.get_ylabel().strip():
            continue
        ax.yaxis._autolabelpos = True
        for _ in range(3):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            tick_boxes = [
                label.get_window_extent(renderer)
                for label in ax.get_yticklabels()
                if label.get_visible() and label.get_text().strip()
            ]
            if not tick_boxes:
                break
            label_box = ax.yaxis.label.get_window_extent(renderer)
            tick_left = min(box.x0 for box in tick_boxes)
            actual_gap_pt = (tick_left - label_box.x1) * 72.0 / fig.dpi
            correction_pt = actual_gap_pt - float(requested_gap)
            if abs(correction_pt) <= 0.02:
                break
            ax.yaxis.labelpad -= correction_pt
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_boxes = [
            label.get_window_extent(renderer)
            for label in ax.get_yticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        if tick_boxes:
            label_box = ax.yaxis.label.get_window_extent(renderer)
            measured[f"panel_{panel_index + 1}"] = (
                min(box.x0 for box in tick_boxes) - label_box.x1
            ) * 72.0 / fig.dpi
    return measured


def _owned_colorbar_axes(fig: plt.Figure, parent_ax) -> list:
    owned = []
    for candidate in fig.axes:
        if candidate is parent_ax:
            continue
        colorbar = getattr(candidate, "_colorbar", None)
        mappable = getattr(colorbar, "mappable", None)
        if getattr(mappable, "axes", None) is parent_ax:
            owned.append(candidate)
    return owned


def _panel_tightbbox(fig: plt.Figure, ax, renderer) -> Bbox:
    boxes = [ax.get_tightbbox(renderer)]
    boxes.extend(
        colorbar_ax.get_tightbbox(renderer)
        for colorbar_ax in _owned_colorbar_axes(fig, ax)
    )
    return Bbox.union(boxes).transformed(fig.transFigure.inverted())


def _remap_owned_axes(old_parent: Bbox, new_parent: Bbox, owned_axes: list) -> None:
    if old_parent.width <= 0 or old_parent.height <= 0:
        return
    for owned_ax in owned_axes:
        child = owned_ax.get_position()
        rel_x0 = (child.x0 - old_parent.x0) / old_parent.width
        rel_x1 = (child.x1 - old_parent.x0) / old_parent.width
        rel_y0 = (child.y0 - old_parent.y0) / old_parent.height
        rel_y1 = (child.y1 - old_parent.y0) / old_parent.height
        x0 = new_parent.x0 + rel_x0 * new_parent.width
        x1 = new_parent.x0 + rel_x1 * new_parent.width
        y0 = new_parent.y0 + rel_y0 * new_parent.height
        y1 = new_parent.y0 + rel_y1 * new_parent.height
        if x1 > x0 and y1 > y0:
            owned_ax.set_position([x0, y0, x1 - x0, y1 - y0])


def align_visible_panel_boundaries(
    fig: plt.Figure,
    axes: list,
    *,
    tolerance_pt: float = 0.20,
    max_iterations: int = 10,
) -> dict[str, float | int | bool | str]:
    """Align rendered panel envelopes by moving the real axes.

    Axisless flow/framework panels are excluded.  No transparent or invisible
    artist is added; every reported boundary is the actual rendered content.
    """
    active_axes = [ax for ax in axes if ax.axison]
    if len(active_axes) < 2:
        return {
            "method": "rendered-tightbbox-positioning",
            "iterations": 0,
            "converged": True,
            "excluded_axisless_panels": len(axes) - len(active_axes),
        }

    def group(ax, edge: str) -> float:
        pos = ax.get_position()
        fallback = {
            "left": pos.x0,
            "right": pos.x1,
            "bottom": pos.y0,
            "top": pos.y1,
        }[edge]
        return float(
            getattr(ax, f"_publication_{edge}_group", round(fallback, 6))
        )

    converged = False
    max_spread_pt = float("inf")
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        compact_requested_ylabel_gaps(fig, active_axes)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        records = []
        for ax in active_axes:
            pos = ax.get_position()
            tight = _panel_tightbbox(fig, ax, renderer)
            records.append(
                {
                    "ax": ax,
                    "pos": pos,
                    "tight": tight,
                    "left_group": group(ax, "left"),
                    "right_group": group(ax, "right"),
                    "bottom_group": group(ax, "bottom"),
                    "top_group": group(ax, "top"),
                    "left_margin": pos.x0 - tight.x0,
                    "right_margin": tight.x1 - pos.x1,
                    "bottom_margin": pos.y0 - tight.y0,
                    "top_margin": tight.y1 - pos.y1,
                }
            )

        def targets(group_key: str, edge: str) -> dict[float, float]:
            grouped: dict[float, list[float]] = {}
            for record in records:
                grouped.setdefault(record[group_key], []).append(
                    getattr(record["tight"], edge)
                )
            return {
                key: float(np.median(values))
                for key, values in grouped.items()
                if len(values) > 1
            }

        left_targets = targets("left_group", "x0")
        right_targets = targets("right_group", "x1")
        bottom_targets = targets("bottom_group", "y0")
        top_targets = targets("top_group", "y1")

        spreads = []
        for target_map, group_key, edge in [
            (left_targets, "left_group", "x0"),
            (right_targets, "right_group", "x1"),
            (bottom_targets, "bottom_group", "y0"),
            (top_targets, "top_group", "y1"),
        ]:
            for group_key_value in target_map:
                values = [
                    getattr(record["tight"], edge)
                    for record in records
                    if record[group_key] == group_key_value
                ]
                scale = (
                    fig.get_figwidth() * 72.0
                    if edge in {"x0", "x1"}
                    else fig.get_figheight() * 72.0
                )
                spreads.append((max(values) - min(values)) * scale)
        max_spread_pt = float(max(spreads, default=0.0))
        if max_spread_pt <= tolerance_pt:
            converged = True
            break

        for record in records:
            old_pos = record["pos"]
            x0 = (
                left_targets[record["left_group"]] + record["left_margin"]
                if record["left_group"] in left_targets
                else old_pos.x0
            )
            x1 = (
                right_targets[record["right_group"]] - record["right_margin"]
                if record["right_group"] in right_targets
                else old_pos.x1
            )
            y0 = (
                bottom_targets[record["bottom_group"]]
                + record["bottom_margin"]
                if record["bottom_group"] in bottom_targets
                else old_pos.y0
            )
            y1 = (
                top_targets[record["top_group"]] - record["top_margin"]
                if record["top_group"] in top_targets
                else old_pos.y1
            )
            minimum_width = max(0.035, old_pos.width * 0.45)
            minimum_height = max(0.035, old_pos.height * 0.45)
            if x1 - x0 < minimum_width:
                x0, x1 = old_pos.x0, old_pos.x1
            if y1 - y0 < minimum_height:
                y0, y1 = old_pos.y0, old_pos.y1
            new_pos = Bbox.from_extents(x0, y0, x1, y1)
            owned_axes = _owned_colorbar_axes(fig, record["ax"])
            record["ax"].set_position(new_pos)
            _remap_owned_axes(old_pos, new_pos, owned_axes)

    compact_requested_ylabel_gaps(fig, active_axes)
    fig.canvas.draw()
    return {
        "method": "rendered-tightbbox-positioning",
        "iterations": completed_iterations,
        "converged": converged,
        "max_visual_boundary_spread_pt": max_spread_pt,
        "excluded_axisless_panels": len(axes) - len(active_axes),
    }


def assert_no_alignment_spacers(fig: plt.Figure) -> None:
    """Fail if an invisible alignment substitute is accidentally reintroduced."""
    for artist in fig.findobj():
        if getattr(artist, "get_gid", lambda: None)() == "alignment-spacer":
            raise RuntimeError("Invisible alignment spacer detected")


def composite_alignment_audit(fig: plt.Figure, axes: list) -> dict[str, object]:
    """Measure plotting and full visual-boundary spread at shared grid edges."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    records = []
    for ax in axes:
        pos = ax.get_position()
        tight = _panel_tightbbox(fig, ax, renderer)
        records.append(
            {
                "axis_on": bool(ax.axison),
                "x0": pos.x0,
                "x1": pos.x1,
                "y0": pos.y0,
                "y1": pos.y1,
                "left_group": getattr(
                    ax, "_publication_left_group", round(pos.x0, 6)
                ),
                "right_group": getattr(
                    ax, "_publication_right_group", round(pos.x1, 6)
                ),
                "bottom_group": getattr(
                    ax, "_publication_bottom_group", round(pos.y0, 6)
                ),
                "top_group": getattr(
                    ax, "_publication_top_group", round(pos.y1, 6)
                ),
                "tight_x0": tight.x0,
                "tight_x1": tight.x1,
                "tight_y0": tight.y0,
                "tight_y1": tight.y1,
            }
        )

    def spread(
        group_key: str,
        value_key: str,
        scale_points: float,
        *,
        axis_only: bool = False,
    ) -> float:
        grouped: dict[float, list[float]] = {}
        for record in records:
            if axis_only and not record["axis_on"]:
                continue
            grouped.setdefault(record[group_key], []).append(record[value_key])
        values = [
            (max(group) - min(group)) * scale_points
            for group in grouped.values()
            if len(group) > 1
        ]
        return float(max(values, default=0.0))

    width_points = fig.get_figwidth() * 72.0
    height_points = fig.get_figheight() * 72.0
    return {
        "alignment_method": "rendered-tightbbox-positioning",
        "transparent_alignment_spacers": 0,
        "panel_count": len(records),
        "axisless_panel_count": sum(not record["axis_on"] for record in records),
        "plot_left_spread_pt": spread("left_group", "x0", width_points),
        "plot_right_spread_pt": spread("right_group", "x1", width_points),
        "plot_top_spread_pt": spread("top_group", "y1", height_points),
        "plot_bottom_spread_pt": spread("bottom_group", "y0", height_points),
        "visual_left_spread_pt": spread(
            "left_group", "tight_x0", width_points, axis_only=True
        ),
        "visual_right_spread_pt": spread(
            "right_group", "tight_x1", width_points, axis_only=True
        ),
        "visual_top_spread_pt": spread(
            "top_group", "tight_y1", height_points, axis_only=True
        ),
        "visual_bottom_spread_pt": spread(
            "bottom_group", "tight_y0", height_points, axis_only=True
        ),
    }


def save_publication_figure(
    fig: plt.Figure,
    folder: Path,
    stem: str,
    panel_drawers: list,
    standalone_sizes: list[tuple[float, float]] | None = None,
    source_script: Path | None = None,
    combined_axes: list | tuple | np.ndarray | None = None,
):
    """Save a combined figure, then independently rerender every panel."""
    folder.mkdir(parents=True, exist_ok=True)
    subdir = folder / "subfigures"
    if subdir.exists():
        shutil.rmtree(subdir)
    letters = "abcdefghijklmnopqrstuvwxyz"
    panel_axes = (
        list(np.asarray(combined_axes, dtype=object).reshape(-1))
        if combined_axes is not None
        else fig.axes[: len(panel_drawers)]
    )
    if len(panel_axes) != len(panel_drawers):
        raise ValueError(
            f"{stem}: expected {len(panel_drawers)} panel axes, got {len(panel_axes)}"
        )
    scale_figure_text(fig)
    align_composite_panel_grid(fig, panel_axes)
    compact_requested_ylabel_gaps(fig, panel_axes)
    for i, ax in enumerate(panel_axes):
        ax.text(
            -0.13,
            1.05,
            letters[i],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="normal",
            color=INK,
            clip_on=False,
        )
    alignment_result = align_visible_panel_boundaries(fig, panel_axes)
    assert_no_alignment_spacers(fig)
    ylabel_gap_audit = compact_requested_ylabel_gaps(fig, panel_axes)
    # The point-based label-pad correction does not change grid membership, but
    # one final rendered pass ensures its glyph edge is part of the alignment.
    alignment_result = align_visible_panel_boundaries(fig, panel_axes)
    assert_no_alignment_spacers(fig)
    ylabel_gap_audit = compact_requested_ylabel_gaps(fig, panel_axes)
    audit_root = TMP_ROOT / "figure_qa" / "alignment_audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    audit = composite_alignment_audit(fig, panel_axes)
    audit["alignment_run"] = alignment_result
    audit["measured_ylabel_gaps_pt"] = ylabel_gap_audit
    (audit_root / f"{stem}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    combined_exports = [
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ]
    for extension, extra_kwargs in combined_exports:
        target = folder / f"{stem}.{extension}"
        fallback = folder / f"{stem}_updated.{extension}"
        try:
            fig.savefig(
                target,
                bbox_inches="tight",
                facecolor="white",
                **extra_kwargs,
            )
            if fallback.exists():
                fallback.unlink()
        except PermissionError:
            fig.savefig(
                fallback,
                bbox_inches="tight",
                facecolor="white",
                **extra_kwargs,
            )
            print(f"Locked export retained at {target}; current figure saved to {fallback}")
    plt.close(fig)
    subdir.mkdir(parents=True, exist_ok=True)
    standalone_sizes = standalone_sizes or [(6.2, 5.0)] * len(panel_drawers)
    for i, (drawer, size) in enumerate(zip(panel_drawers, standalone_sizes)):
        sfig, sax = plt.subplots(figsize=size)
        drawer(sax)
        scale_figure_text(sfig)
        sfig.tight_layout(pad=1.25)
        compact_requested_ylabel_gaps(sfig, [sax])
        suffix = letters[i]
        sfig.savefig(
            subdir / f"{stem}_{suffix}.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        sfig.savefig(
            subdir / f"{stem}_{suffix}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(sfig)
    readme = [
        "# Subfigures",
        "",
        "Each panel was independently rerendered on a standalone canvas from the same final drawing function used by the combined figure.",
        "",
    ]
    for i in range(len(panel_drawers)):
        readme.append(
            f"- {letters[i]}: `{stem}_{letters[i]}.png`, `{stem}_{letters[i]}.pdf`"
        )
    (subdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    code_copy = folder / f"{stem}_code.py"
    source_script = Path(source_script) if source_script is not None else SCRIPT_PATH
    if source_script.resolve() != code_copy.resolve():
        shutil.copy2(source_script, code_copy)
    (folder / "CODE_README.md").write_text(
        "# Figure code\n\n"
        f"`{code_copy.name}` is a synchronized copy of the figure-generation pipeline. "
        "It uses the project-root `07_Scripts/figure_style/ml_publication_style.py` and "
        "`07_Scripts/figure_style/scientific_palettes.py` helpers. "
        "Run it from the project root to regenerate all dependent figures, subfigures, tables, and captions together.\n",
        encoding="utf-8",
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if JSON_PATH.exists():
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        sheet = next(
            s for s in payload["sheets"] if s["name"] == "PFAS_literature_dataset_2b"
        )
        raw = pd.DataFrame(sheet["values"][1:], columns=sheet["values"][0])
    else:
        raw = pd.read_excel(
            WORKBOOK_PATH,
            sheet_name="PFAS_literature_dataset_2b",
            engine="openpyxl",
        )
    numeric = [
        "ref_year",
        "Rejection",
        "Transmission",
        "pfas_feed_conc_µg/L",
        "total_PFAS_feed_conc_µg/L",
        "pH",
        "mon_cat_mM",
        "di_cat_mM",
        "tri_cat_mM",
        "mon_an_mM",
        "di_an",
        "tri_an",
        "doc_mg/L",
        "ios_mM",
        "mw_g/mol",
        "logP",
        "volume_Å3",
        "pKa",
        "tpsa_Å2",
        "total_partial_charge",
        "polarizability_Å3",
        "dipole_debye",
        "min_PA",
        "max_PA",
        "min_PR",
        "max_PR",
        "length",
        "carbons",
        "pwp_LMH/bar",
        "zp_mV",
        "wca_º",
        "ON",
        "temp_ºC",
        "flux_LMH",
        "tmp_bar",
    ]
    for col in numeric:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["is_censored"] = raw["LOQ"].astype(str).str.lower().eq("right")
    filtered = raw.loc[
        raw["Rejection"].ge(0.5)
        & raw["mem_class"].astype(str).str.upper().isin(["NF", "RO"])
    ].copy()
    filtered = filtered.reset_index(drop=True)
    r = filtered["Rejection"].clip(0.500001, 0.999999)
    filtered["logit_R"] = np.log(r / (1 - r))
    filtered["odds_R"] = r / (1 - r)
    filtered["feed_group"] = (
        filtered["feed_type"]
        .fillna("Unreported")
        .replace({"Raw": "Real water", "Lab": "Synthetic", "Pure water": "Pure water"})
    )
    filtered["chain_group"] = pd.cut(
        filtered["carbons"],
        bins=[-np.inf, 3, 5, 7, np.inf],
        labels=["C≤3", "C4–5", "C6–7", "C≥8"],
    )
    return raw, filtered


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        x[col] = np.log1p(x[col].clip(lower=0))
    return x.to_numpy(dtype=float)


def fit_aft(X, odds, censored, seed=42, rounds=150):
    dtrain = xgb.DMatrix(X, missing=np.nan)
    lower = np.asarray(odds, dtype=float)
    upper = np.where(np.asarray(censored, dtype=bool), np.inf, lower)
    dtrain.set_float_info("label_lower_bound", lower)
    dtrain.set_float_info("label_upper_bound", upper)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist",
        "max_depth": 3,
        "min_child_weight": 14,
        "eta": 0.045,
        "subsample": 0.82,
        "colsample_bytree": 0.82,
        "lambda": 7.0,
        "alpha": 0.15,
        "seed": int(seed),
        "nthread": 4,
    }
    return xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)


def predict_aft(model, X) -> np.ndarray:
    odds = np.clip(model.predict(xgb.DMatrix(X, missing=np.nan)), 1e-6, 1e8)
    return odds / (1.0 + odds)


def fit_naive(X, rejection, seed=42, rounds=170, sample_weight=None):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=rounds,
        max_depth=3,
        min_child_weight=14,
        learning_rate=0.045,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=7.0,
        reg_alpha=0.15,
        random_state=int(seed),
        n_jobs=4,
    )
    model.fit(X, rejection, sample_weight=sample_weight)
    return model


def censored_concordance(y, pred, censored) -> float:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    censored = np.asarray(censored, bool)
    concordant = 0.0
    comparable = 0.0
    n = len(y)
    for i in range(n - 1):
        yi, pi, ci = y[i], pred[i], censored[i]
        yj = y[i + 1 :]
        pj = pred[i + 1 :]
        cj = censored[i + 1 :]
        exact = (~ci) & (~cj) & (np.abs(yi - yj) > 1e-12)
        i_lower = ci & (~cj) & (yi > yj)
        j_lower = (~ci) & cj & (yj > yi)
        comp = exact | i_lower | j_lower
        if not np.any(comp):
            continue
        direction = np.sign(yi - yj[comp])
        pdirection = np.sign(pi - pj[comp])
        concordant += np.sum(direction == pdirection) + 0.5 * np.sum(pdirection == 0)
        comparable += len(direction)
    return float(concordant / comparable) if comparable else np.nan


def evaluation(y, pred, censored) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    censored = np.asarray(censored, bool)
    exact = ~censored
    if exact.sum() >= 2:
        r2 = r2_score(y[exact], pred[exact])
        exact_y = np.clip(y[exact], 0.500001, 0.999999)
        exact_pred = np.clip(pred[exact], 0.500001, 0.999999)
        logit_y = np.log(exact_y / (1.0 - exact_y))
        logit_pred = np.log(exact_pred / (1.0 - exact_pred))
        logit_r2 = r2_score(logit_y, logit_pred)
        rmse = mean_squared_error(y[exact], pred[exact]) ** 0.5
        mae = mean_absolute_error(y[exact], pred[exact])
    else:
        r2 = logit_r2 = rmse = mae = np.nan
    bound_violation = np.mean(pred[censored] < y[censored]) if censored.any() else np.nan
    return {
        "r2": float(r2),
        "logit_r2": float(logit_r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "c_index": censored_concordance(y, pred, censored),
        "bound_violation": float(bound_violation),
        "n": len(y),
        "n_exact": int(exact.sum()),
    }


def fit_predict(method, train_idx, test_idx, X, df, seed=42, rounds=150):
    if method == "aft":
        model = fit_aft(
            X[train_idx],
            df.loc[train_idx, "odds_R"].to_numpy(),
            df.loc[train_idx, "is_censored"].to_numpy(),
            seed=seed,
            rounds=rounds,
        )
        return model, predict_aft(model, X[test_idx])
    if method == "naive":
        model = fit_naive(
            X[train_idx], df.loc[train_idx, "Rejection"].to_numpy(), seed, rounds
        )
        return model, np.clip(model.predict(X[test_idx]), 0.5, 1.0)
    if method == "uncensored":
        keep = train_idx[~df.loc[train_idx, "is_censored"].to_numpy()]
        model = fit_naive(X[keep], df.loc[keep, "Rejection"].to_numpy(), seed, rounds)
        return model, np.clip(model.predict(X[test_idx]), 0.5, 1.0)
    raise ValueError(method)


def strict_condition_key(df: pd.DataFrame) -> pd.Series:
    cols = [
        "ref_id",
        "mem",
        "feed_type",
        "pH",
        "mix_single_pfas",
        "comp_descrip",
        "pfas_feed_conc_µg/L",
        "total_PFAS_feed_conc_µg/L",
        "mon_cat_mM",
        "di_cat_mM",
        "tri_cat_mM",
        "mon_an_mM",
        "di_an",
        "tri_an",
        "doc_mg/L",
        "ios_mM",
        "temp_ºC",
        "flux_LMH",
        "tmp_bar",
        "mode",
        "system",
    ]
    key = df[cols].copy()
    for col in key.columns:
        key[col] = key[col].fillna("<NA>").astype(str)
    return key.agg("|".join, axis=1)


@dataclass
class SplitResults:
    summary: pd.DataFrame
    folds: pd.DataFrame
    predictions: dict[str, np.ndarray]
    study_distance: np.ndarray
    study_new_membrane: np.ndarray
    study_new_pfas: np.ndarray
    overlap: pd.DataFrame


def run_split_audit(df: pd.DataFrame, X: np.ndarray) -> SplitResults:
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    predictions: dict[str, np.ndarray] = {}
    fold_records = []
    summary_records = []
    overlap_records = []

    imp = SimpleImputer(strategy="median")
    scaled = StandardScaler().fit_transform(imp.fit_transform(X))
    clusters = KMeans(n_clusters=10, random_state=42, n_init=20).fit_predict(scaled)

    for strategy, stratify in [("Cluster-stratified", clusters), ("Row-random", None)]:
        pooled_pred = np.full((12, len(df)), np.nan)
        for rep in range(12):
            tr, te = train_test_split(
                np.arange(len(df)),
                test_size=0.15,
                random_state=110 + rep,
                stratify=stratify,
            )
            split_imputer = SimpleImputer(strategy="median")
            x_train = split_imputer.fit_transform(X[tr])
            x_test = split_imputer.transform(X[te])
            split_scaler = StandardScaler().fit(x_train)
            x_train = split_scaler.transform(x_train)
            x_test = split_scaler.transform(x_test)
            nearest_distance = (
                NearestNeighbors(n_neighbors=1)
                .fit(x_train)
                .kneighbors(x_test, return_distance=True)[0][:, 0]
            )
            train_studies = set(df.loc[tr, "ref_id"].fillna("<NA>").astype(str))
            train_membranes = set(df.loc[tr, "mem"].fillna("<NA>").astype(str))
            train_pfas = set(df.loc[tr, "PFAS"].fillna("<NA>").astype(str))
            overlap_records.append(
                {
                    "strategy": strategy,
                    "repeat": rep + 1,
                    "same_study": float(
                        df.loc[te, "ref_id"]
                        .fillna("<NA>")
                        .astype(str)
                        .isin(train_studies)
                        .mean()
                    ),
                    "same_membrane": float(
                        df.loc[te, "mem"]
                        .fillna("<NA>")
                        .astype(str)
                        .isin(train_membranes)
                        .mean()
                    ),
                    "same_pfas": float(
                        df.loc[te, "PFAS"]
                        .fillna("<NA>")
                        .astype(str)
                        .isin(train_pfas)
                        .mean()
                    ),
                    "exact_feature_duplicate": float(
                        np.mean(nearest_distance < 1e-9)
                    ),
                    "median_nearest_distance": float(
                        np.median(nearest_distance)
                    ),
                }
            )
            model, pred = fit_predict("aft", tr, te, X, df, seed=110 + rep, rounds=130)
            pooled_pred[rep, te] = pred
            m = evaluation(y[te], pred, c[te])
            _, train_pred = fit_predict(
                "aft", tr, tr, X, df, seed=110 + rep, rounds=130
            )
            train_m = evaluation(y[tr], train_pred, c[tr])
            fold_records.append(
                {
                    "strategy": strategy,
                    "fold": rep + 1,
                    **m,
                    "train_r2": train_m["r2"],
                    "train_test_gap": train_m["r2"] - m["r2"],
                }
            )
        mean_pred = np.full(len(df), np.nan, dtype=float)
        tested = np.isfinite(pooled_pred).any(axis=0)
        mean_pred[tested] = np.nanmean(pooled_pred[:, tested], axis=0)
        predictions[strategy] = mean_pred
        valid = np.isfinite(mean_pred)
        summary_records.append(
            {"strategy": strategy, **evaluation(y[valid], mean_pred[valid], c[valid])}
        )

    study_distance = np.full(len(df), np.nan)
    new_mem = np.zeros(len(df), bool)
    new_pfas = np.zeros(len(df), bool)
    grouped = [
        ("Study-blocked", "ref_id"),
        ("Membrane-blocked", "mem"),
        ("PFAS-blocked", "PFAS"),
    ]
    for strategy, group_col in grouped:
        pred_all = np.full(len(df), np.nan)
        groups = df[group_col].fillna("<NA>").astype(str).to_numpy()
        for fold_no, group in enumerate(pd.unique(groups), 1):
            te = np.flatnonzero(groups == group)
            tr = np.flatnonzero(groups != group)
            _, pred = fit_predict(
                "aft", tr, te, X, df, seed=500 + fold_no, rounds=115
            )
            pred_all[te] = pred
            m = evaluation(y[te], pred, c[te])
            train_r2 = np.nan
            if m["n_exact"] >= 5:
                fold_records.append(
                    {
                        "strategy": strategy,
                        "fold": str(group),
                        **m,
                        "train_r2": train_r2,
                        "train_test_gap": np.nan,
                    }
                )
            if strategy == "Study-blocked":
                tr_imp = SimpleImputer(strategy="median")
                xtr = tr_imp.fit_transform(X[tr])
                xte = tr_imp.transform(X[te])
                scaler = StandardScaler().fit(xtr)
                xtr = scaler.transform(xtr)
                xte = scaler.transform(xte)
                k = min(8, len(tr))
                nn = NearestNeighbors(n_neighbors=k).fit(xtr)
                study_distance[te] = nn.kneighbors(xte)[0].mean(axis=1)
                train_mem = set(df.loc[tr, "mem"].fillna("<NA>").astype(str))
                train_pfas = set(df.loc[tr, "PFAS"].fillna("<NA>").astype(str))
                new_mem[te] = ~df.loc[te, "mem"].fillna("<NA>").astype(str).isin(train_mem)
                new_pfas[te] = ~df.loc[te, "PFAS"].fillna("<NA>").astype(str).isin(train_pfas)
        predictions[strategy] = pred_all
        summary_records.append({"strategy": strategy, **evaluation(y, pred_all, c)})

    summary = pd.DataFrame(summary_records)
    folds = pd.DataFrame(fold_records)
    overlap = pd.DataFrame(overlap_records)
    return SplitResults(
        summary,
        folds,
        predictions,
        study_distance,
        new_mem,
        new_pfas,
        overlap,
    )


def model_ladder(df: pd.DataFrame, X: np.ndarray):
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=10)
    rows = []
    predictions = {}
    for method, label in [
        ("naive", "Censored as exact"),
        ("uncensored", "Censored excluded"),
        ("aft", "Censor-aware AFT"),
    ]:
        pred = np.full(len(df), np.nan)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
            _, p = fit_predict(method, tr, te, X, df, seed=700 + fold, rounds=145)
            pred[te] = p
            rows.append({"method": label, "fold": fold, **evaluation(y[te], p, c[te])})
        predictions[label] = pred
    fold_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"method": label, **evaluation(y, pred, c)}
            for label, pred in predictions.items()
        ]
    )
    aft_resid = y - predictions["Censor-aware AFT"]
    exact = ~c
    study_means = (
        pd.DataFrame({"study": groups[exact], "residual": aft_resid[exact]})
        .groupby("study")
        .agg(mean_residual=("residual", "mean"), n=("residual", "size"))
        .reset_index()
    )
    total_var = np.var(aft_resid[exact], ddof=1)
    between_var = np.average(
        (study_means["mean_residual"] - np.average(study_means["mean_residual"], weights=study_means["n"])) ** 2,
        weights=study_means["n"],
    )
    residual_icc = float(np.clip(between_var / total_var, 0, 1))
    return summary, fold_df, predictions, study_means, residual_icc


def grouped_conformal(df: pd.DataFrame, X: np.ndarray):
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    outer = GroupKFold(n_splits=10)
    pred_all = np.full(len(df), np.nan)
    scale_all = np.full(len(df), np.nan)
    q_by_nominal = {0.80: np.full(len(df), np.nan), 0.90: np.full(len(df), np.nan), 0.95: np.full(len(df), np.nan)}
    fold_records = []
    rng = np.random.default_rng(20260717)

    for fold, (train_outer, test_idx) in enumerate(outer.split(X, y, groups), 1):
        train_groups = np.unique(groups[train_outer])
        rng.shuffle(train_groups)
        n_cal = max(4, int(np.ceil(len(train_groups) * 0.22)))
        cal_groups = set(train_groups[:n_cal])
        cal_idx = train_outer[np.array([g in cal_groups for g in groups[train_outer]])]
        fit_idx = train_outer[np.array([g not in cal_groups for g in groups[train_outer]])]

        cal_preds = []
        test_preds = []
        unique_fit_groups = np.unique(groups[fit_idx])
        for b in range(8):
            sampled = rng.choice(unique_fit_groups, size=len(unique_fit_groups), replace=True)
            boot_idx = np.concatenate([fit_idx[groups[fit_idx] == g] for g in sampled])
            model = fit_aft(
                X[boot_idx],
                df.loc[boot_idx, "odds_R"].to_numpy(),
                df.loc[boot_idx, "is_censored"].to_numpy(),
                seed=9000 + fold * 20 + b,
                rounds=105,
            )
            cal_preds.append(predict_aft(model, X[cal_idx]))
            test_preds.append(predict_aft(model, X[test_idx]))
        cal_preds = np.asarray(cal_preds)
        test_preds = np.asarray(test_preds)
        cal_mean = cal_preds.mean(axis=0)
        test_mean = test_preds.mean(axis=0)
        cal_scale = np.maximum(cal_preds.std(axis=0, ddof=1), 0.008)
        test_scale = np.maximum(test_preds.std(axis=0, ddof=1), 0.008)
        exact_cal = ~c[cal_idx]
        scores = np.abs(y[cal_idx][exact_cal] - cal_mean[exact_cal]) / cal_scale[exact_cal]
        pred_all[test_idx] = test_mean
        scale_all[test_idx] = test_scale
        for nominal in q_by_nominal:
            q = np.quantile(scores, min(1, (math.ceil((len(scores) + 1) * nominal) / len(scores))))
            q_by_nominal[nominal][test_idx] = q
            width = 2 * q * test_scale
            lo = np.clip(test_mean - q * test_scale, 0.5, 1)
            hi = np.clip(test_mean + q * test_scale, 0.5, 1)
            exact_test = ~c[test_idx]
            coverage = np.mean((y[test_idx][exact_test] >= lo[exact_test]) & (y[test_idx][exact_test] <= hi[exact_test]))
            bound_compatible = np.mean(hi[c[test_idx]] >= y[test_idx][c[test_idx]]) if c[test_idx].any() else np.nan
            fold_records.append(
                {
                    "fold": fold,
                    "nominal": nominal,
                    "coverage_exact": coverage,
                    "bound_compatible": bound_compatible,
                    "mean_width": np.mean(width),
                    "n_exact": int(exact_test.sum()),
                }
            )

    interval = pd.DataFrame({"prediction": pred_all, "scale": scale_all})
    for nominal, q_arr in q_by_nominal.items():
        interval[f"q_{int(nominal * 100)}"] = q_arr
        interval[f"lo_{int(nominal * 100)}"] = np.clip(pred_all - q_arr * scale_all, 0.5, 1)
        interval[f"hi_{int(nominal * 100)}"] = np.clip(pred_all + q_arr * scale_all, 0.5, 1)
    return interval, pd.DataFrame(fold_records)


def shap_stability_and_ale(df: pd.DataFrame, X: np.ndarray):
    y = df["Rejection"].to_numpy(float)
    groups = df["ref_id"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold_importance = []
    sign_records = []
    models = []
    train_sets = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        model = fit_aft(
            X[tr],
            df.loc[tr, "odds_R"].to_numpy(),
            df.loc[tr, "is_censored"].to_numpy(),
            seed=1200 + fold,
            rounds=150,
        )
        contrib = model.predict(xgb.DMatrix(X[te], missing=np.nan), pred_contribs=True)[:, :-1]
        fold_importance.append(np.mean(np.abs(contrib), axis=0))
        for j, feature in enumerate(FEATURES):
            valid = np.isfinite(X[te, j]) & np.isfinite(contrib[:, j])
            xv = X[te, j][valid]
            sv = contrib[:, j][valid]
            rho = (
                spearmanr(xv, sv).statistic
                if valid.sum() > 5 and np.nanstd(xv) > 0 and np.nanstd(sv) > 0
                else np.nan
            )
            sign_records.append({"fold": fold, "feature": feature, "rho": rho})
        models.append(model)
        train_sets.append(tr)
    importance = np.asarray(fold_importance)
    imp_df = pd.DataFrame(importance, columns=FEATURES)
    summary = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap": imp_df.mean().to_numpy(),
            "sd_abs_shap": imp_df.std(ddof=1).to_numpy(),
            "mean_rank": imp_df.rank(axis=1, ascending=False).mean().to_numpy(),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    signs = pd.DataFrame(sign_records)
    sign_pivot = signs.pivot(index="feature", columns="fold", values="rho").reindex(FEATURES)

    top_features = summary.head(4)["feature"].tolist()
    ale_curves = {}
    for feature in top_features:
        j = FEATURES.index(feature)
        curves = []
        common_x = np.linspace(0.05, 0.95, 15)
        for model, tr in zip(models, train_sets):
            vals = X[tr, j]
            valid_vals = vals[np.isfinite(vals)]
            if len(np.unique(valid_vals)) < 5:
                continue
            edges = np.unique(np.quantile(valid_vals, np.linspace(0, 1, 16)))
            if len(edges) < 5:
                continue
            effects, centers, weights = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = np.isfinite(vals) & (vals >= lo) & (vals <= hi if hi == edges[-1] else vals < hi)
                idx = tr[mask]
                if len(idx) == 0:
                    continue
                xlo = X[idx].copy()
                xhi = X[idx].copy()
                xlo[:, j] = lo
                xhi[:, j] = hi
                diff = predict_aft(model, xhi) - predict_aft(model, xlo)
                effects.append(np.mean(diff))
                centers.append((lo + hi) / 2)
                weights.append(len(idx))
            effects = np.asarray(effects)
            centers = np.asarray(centers)
            if len(effects) < 4:
                continue
            cum = np.cumsum(effects)
            cum = cum - np.average(cum, weights=weights)
            qpos = np.array([np.mean(valid_vals <= c) for c in centers])
            curves.append(np.interp(common_x, qpos, cum))
        if curves:
            display_x = np.quantile(X[:, j][np.isfinite(X[:, j])], common_x)
            ale_curves[feature] = (display_x, np.asarray(curves))
    return summary, imp_df, sign_pivot, ale_curves


def within_study_associations(df: pd.DataFrame):
    """Restricted second-tier association layer with explicit failure boundaries."""
    rng = np.random.default_rng(404)
    treatments = ["mon_cat_mM", "di_cat_mM", "doc_mg/L"]
    labels = {
        "mon_cat_mM": "Monovalent cations",
        "di_cat_mM": "Divalent cations",
        "doc_mg/L": "DOC",
    }
    roles = {
        "mon_cat_mM": "Primary",
        "di_cat_mM": "Exploratory",
        "doc_mg/L": "Exploratory",
    }
    covariates = [
        "pH",
        "pwp_LMH/bar",
        "volume_Å3",
        "dipole_debye",
        "pKa",
        "flux_LMH",
        "pfas_feed_conc_µg/L",
        "temp_ºC",
    ]
    base = df.loc[~df["is_censored"]].copy()
    results = []
    boot_store = {}
    placebo_store = {}
    refutation_records = []

    def fit_effect(
        data,
        treatment,
        permute=False,
        already_transformed=False,
        extra_columns=None,
    ):
        extra_columns = list(extra_columns or [])
        cols_to_read = ["ref_id", "logit_R", treatment] + covariates + extra_columns
        work = data[cols_to_read].copy()
        if not already_transformed:
            work[treatment] = np.log1p(
                pd.to_numeric(work[treatment], errors="coerce").clip(lower=0)
            )
            for col in ["flux_LMH", "pfas_feed_conc_µg/L"]:
                work[col] = np.log1p(
                    pd.to_numeric(work[col], errors="coerce").clip(lower=0)
                )
        variable_studies = work.groupby("ref_id")[treatment].nunique(dropna=True)
        variable_studies = variable_studies[variable_studies > 1].index
        work = work[work["ref_id"].isin(variable_studies)].copy()
        model_columns = [treatment] + covariates + extra_columns
        work[model_columns] = work[model_columns].fillna(
            work[model_columns].median(numeric_only=True)
        )
        if permute:
            work[treatment] = work.groupby("ref_id")[treatment].transform(
                lambda s: rng.permutation(s.to_numpy())
            )
        demeaned = work[model_columns + ["logit_R"]] - work.groupby("ref_id")[
            model_columns + ["logit_R"]
        ].transform("mean")
        Xw = demeaned[model_columns].to_numpy(float)
        yw = demeaned["logit_R"].to_numpy(float)
        Xs = StandardScaler().fit_transform(Xw)
        model = Ridge(alpha=8.0).fit(Xs, yw)
        return float(model.coef_[0]), work, len(variable_studies)

    def approximate_e_value(standardized_effect):
        """Approximate E-value after converting a continuous-outcome SMD to RR."""
        rr = float(np.exp(0.91 * abs(float(standardized_effect))))
        return float(rr + np.sqrt(max(0.0, rr * (rr - 1.0))))

    for treatment in treatments:
        coef, support, n_studies = fit_effect(base, treatment)
        studies = support["ref_id"].unique()
        boots = []
        for _ in range(300):
            sampled = rng.choice(studies, size=len(studies), replace=True)
            pieces = []
            for k, study in enumerate(sampled):
                piece = support[support["ref_id"] == study].copy()
                piece["ref_id"] = f"{study}__{k}"
                pieces.append(piece)
            try:
                bcoef, _, _ = fit_effect(
                    pd.concat(pieces, ignore_index=True),
                    treatment,
                    already_transformed=True,
                )
                boots.append(bcoef)
            except Exception:
                continue

        placebo = []
        for _ in range(180):
            try:
                pcoef, _, _ = fit_effect(
                    support,
                    treatment,
                    permute=True,
                    already_transformed=True,
                )
                placebo.append(pcoef)
            except Exception:
                continue

        boots = np.asarray(boots, dtype=float)
        placebo = np.asarray(placebo, dtype=float)
        boot_store[treatment] = boots
        placebo_store[treatment] = placebo
        median_logit = float(base["logit_R"].median())
        effect_pp = float(
            (logistic(median_logit + coef) - logistic(median_logit)) * 100
        )
        ci = np.quantile(boots, [0.025, 0.975]) if len(boots) else np.array([np.nan, np.nan])
        ci_pp = (logistic(median_logit + ci) - logistic(median_logit)) * 100
        within_y = support["logit_R"] - support.groupby("ref_id")["logit_R"].transform("mean")
        within_y_sd = float(within_y.std(ddof=1))
        standardized = coef / within_y_sd if within_y_sd > 0 else np.nan
        if ci[0] <= 0 <= ci[1]:
            ci_nearest = 0.0
        else:
            ci_nearest = min(abs(ci[0]), abs(ci[1]))
        results.append(
            {
                "treatment": treatment,
                "label": labels[treatment],
                "role": roles[treatment],
                "effect_logit_per_within_sd": coef,
                "effect_pp_at_median": effect_pp,
                "ci_low_pp": ci_pp[0],
                "ci_high_pp": ci_pp[1],
                "variable_studies": n_studies,
                "rows": len(support),
                "placebo_two_sided_p": (
                    (np.sum(np.abs(placebo) >= abs(coef)) + 1) / (len(placebo) + 1)
                    if len(placebo)
                    else np.nan
                ),
                "approx_e_value": approximate_e_value(standardized),
                "approx_e_value_ci_limit": approximate_e_value(
                    ci_nearest / within_y_sd if within_y_sd > 0 else 0
                ),
            }
        )

        if treatment == "mon_cat_mM":
            random_common = []
            data_subset = []
            for _ in range(180):
                try:
                    perturbed = support.copy()
                    perturbed["__random_common"] = rng.normal(size=len(perturbed))
                    rcoef, _, _ = fit_effect(
                        perturbed,
                        treatment,
                        already_transformed=True,
                        extra_columns=["__random_common"],
                    )
                    random_common.append(rcoef)
                except Exception:
                    continue
                try:
                    keep_n = max(3, int(np.ceil(0.8 * len(studies))))
                    kept = rng.choice(studies, size=keep_n, replace=False)
                    scoef, _, _ = fit_effect(
                        support[support["ref_id"].isin(kept)].copy(),
                        treatment,
                        already_transformed=True,
                    )
                    data_subset.append(scoef)
                except Exception:
                    continue

            def add_refutation(label, values, original=False):
                values = np.asarray(values, dtype=float)
                if original:
                    point, low, high = coef, ci[0], ci[1]
                else:
                    point = float(np.median(values))
                    low, high = np.quantile(values, [0.025, 0.975])
                refutation_records.append(
                    {
                        "test": label,
                        "estimate": point,
                        "ci_low": low,
                        "ci_high": high,
                        "replicates": 1 if original else len(values),
                    }
                )

            add_refutation("Original", boots, original=True)
            add_refutation("Placebo treatment", placebo)
            add_refutation("Random common cause", random_common)
            add_refutation("80% study subset", data_subset)

    # Cross-fitted binary-treatment propensity is used only as an overlap diagnostic
    # for presence versus absence of monovalent cations. The effect model remains
    # continuous and study-demeaned.
    mon = base[["ref_id", "mon_cat_mM"] + covariates].dropna(subset=["mon_cat_mM"]).copy()
    variable_studies = mon.groupby("ref_id")["mon_cat_mM"].nunique()
    variable_studies = variable_studies[variable_studies > 1].index
    mon = mon[mon["ref_id"].isin(variable_studies)].copy()
    mon["exposed"] = mon["mon_cat_mM"].gt(0)
    overlap_x = mon[covariates].copy()
    for col in ["flux_LMH", "pfas_feed_conc_µg/L"]:
        overlap_x[col] = np.log1p(
            pd.to_numeric(overlap_x[col], errors="coerce").clip(lower=0)
        )
    propensity = np.full(len(mon), np.nan)
    groups = mon["ref_id"].astype(str).to_numpy()
    y_treat = mon["exposed"].astype(int).to_numpy()
    n_splits = min(5, pd.Series(groups).nunique())
    for train, test in GroupKFold(n_splits=n_splits).split(overlap_x, y_treat, groups):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(overlap_x.iloc[train]))
        x_test = scaler.transform(imputer.transform(overlap_x.iloc[test]))
        if np.unique(y_treat[train]).size < 2:
            propensity[test] = y_treat[train].mean()
            continue
        propensity_model = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=404,
        ).fit(x_train, y_treat[train])
        propensity[test] = propensity_model.predict_proba(x_test)[:, 1]
    mon["propensity"] = np.clip(propensity, 1e-4, 1 - 1e-4)
    p0 = mon.loc[~mon["exposed"], "propensity"]
    p1 = mon.loc[mon["exposed"], "propensity"]
    common_low = float(max(p0.min(), p1.min()))
    common_high = float(min(p0.max(), p1.max()))
    mon["in_common_support"] = mon["propensity"].between(common_low, common_high)
    overlap_summary = pd.DataFrame(
        [
            {
                "diagnostic": "Non-zero versus zero monovalent cation propensity",
                "rows": len(mon),
                "studies": mon["ref_id"].nunique(),
                "nonzero_rows": int(mon["exposed"].sum()),
                "zero_rows": int((~mon["exposed"]).sum()),
                "common_support_low": common_low,
                "common_support_high": common_high,
                "retained_fraction": float(mon["in_common_support"].mean()),
            }
        ]
    )

    slopes = []
    slope_data = base.copy()
    slope_data["log_mon"] = np.log1p(slope_data["mon_cat_mM"].clip(lower=0))
    for study, group in slope_data.groupby("ref_id"):
        group = group.dropna(subset=["log_mon", "logit_R"])
        if len(group) < 10 or group["log_mon"].nunique() < 2:
            continue
        slope = np.polyfit(group["log_mon"].to_numpy(), group["logit_R"].to_numpy(), 1)[0]
        slopes.append({"study": study, "slope": slope, "n": len(group)})
    slopes = pd.DataFrame(slopes).sort_values("slope")

    # Audited causal-eligibility counts from the manuscript framework. They refer
    # to the causal-condition audit, not the 2,424-row prediction view.
    support_audit = pd.DataFrame(
        [
            ["Monovalent", 203, 34, 22, 45, "Primary"],
            ["Divalent", 221, 33, 23, 92, "Exploratory"],
            ["DOC", 129, 25, 20, 89, "Exploratory"],
            ["Trivalent", 24, 6, 6, 100, "Excluded"],
        ],
        columns=[
            "treatment",
            "independent_conditions",
            "reporting_studies",
            "within_study_varying_studies",
            "between_study_variance_pct",
            "role",
        ],
    )
    return (
        pd.DataFrame(results),
        boot_store,
        placebo_store,
        slopes,
        mon,
        overlap_summary,
        pd.DataFrame(refutation_records),
        support_audit,
    )


def temporal_validation(df: pd.DataFrame, X: np.ndarray, split_results: SplitResults):
    cutoff = 2023
    train_idx = np.flatnonzero(df["ref_year"].to_numpy() <= cutoff)
    test_idx = np.flatnonzero(df["ref_year"].to_numpy() > cutoff)
    _, pred = fit_predict("aft", train_idx, test_idx, X, df, seed=1600, rounds=180)
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    temporal_metrics = evaluation(y[test_idx], pred, c[test_idx])
    temporal_pred = np.full(len(df), np.nan)
    temporal_pred[test_idx] = pred

    imp = SimpleImputer(strategy="median")
    train_x = imp.fit_transform(X[train_idx])
    test_x = imp.transform(X[test_idx])
    shift = []
    for j, feature in enumerate(FEATURES):
        pooled_sd = np.std(train_x[:, j], ddof=1)
        smd = (np.mean(test_x[:, j]) - np.mean(train_x[:, j])) / pooled_sd if pooled_sd > 0 else 0
        shift.append({"feature": feature, "smd": smd, "abs_smd": abs(smd)})
    shift_df = pd.DataFrame(shift).sort_values("abs_smd", ascending=False)

    comparisons = split_results.summary[["strategy", "r2", "rmse", "c_index"]].copy()
    comparisons = pd.concat(
        [
            comparisons,
            pd.DataFrame([{"strategy": "Publication-year holdout", **temporal_metrics}]),
        ],
        ignore_index=True,
    )
    return cutoff, test_idx, temporal_pred, temporal_metrics, shift_df, comparisons


def draw_figures(
    raw,
    df,
    split_results,
    ladder_summary,
    ladder_folds,
    ladder_preds,
    study_means,
    residual_icc,
    interval,
    conformal_folds,
    shap_summary,
    shap_fold,
    sign_pivot,
    ale_curves,
    assoc,
    boot_store,
    placebo_store,
    study_slopes,
    overlap_df,
    overlap_summary,
    refutation_summary,
    support_audit,
    temporal,
):
    def display_study(value):
        return str(value).replace("√©", "é")

    def compact_study_label(value, max_chars=24):
        label = display_study(value)
        if len(label) <= max_chars:
            return label
        if "," in label:
            author, year = label.rsplit(",", 1)
            year = year.strip()
            if year.isdigit():
                author = author.replace(" et al.", "").replace(" et al", "")
                available = max_chars - len(year) - 2
                return f"{author[:available].rstrip()}, {year}"
        return label[:max_chars].rstrip()

    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    exact = ~c
    study_pred = split_results.predictions["Study-blocked"]
    study_error = np.abs(y - study_pred)
    condition_id = strict_condition_key(df)
    multiplicity = condition_id.value_counts()

    # Figure 1
    def f1a(ax):
        labels = ["Raw", "R ≥ 0.50", "NF / RO", "Published"]
        counts = np.array(
            [len(raw), int(raw["Rejection"].ge(0.50).sum()), len(df), 2353],
            dtype=float,
        )
        x = np.arange(len(labels))
        colors = [BLUE, GREEN, TEAL, CORAL]
        ax.plot(x, counts, color=MUTED, linewidth=1.8, zorder=1)
        ax.scatter(
            x,
            counts,
            s=92,
            c=colors,
            edgecolor=INK,
            linewidth=0.9,
            zorder=3,
        )
        for xpos, value in zip(x, counts):
            ax.text(
                xpos,
                value + 12,
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=11.2,
                fontweight="normal",
                color=INK,
            )
        for i, loss in enumerate(np.diff(counts), start=1):
            ax.text(
                i - 0.5,
                (counts[i - 1] + counts[i]) / 2 - 13,
                f"−{abs(int(loss))}",
                ha="center",
                va="top",
                fontsize=10.2,
                fontweight="normal",
                color=CORAL if i == 3 else MUTED,
            )
        clean_axis(ax, None, "Records retained")
        ax.set_xticks(x, labels, rotation=0, ha="center")
        ax.set_xlim(-0.35, len(labels) - 0.65)
        ax.set_ylim(counts.min() - 65, counts.max() + 62)
        numeric_ticks(ax, x=False, n=5)

    def f1b(ax):
        condition_counts = (
            pd.DataFrame(
                {
                    "study": df["ref_id"].astype(str),
                    "condition": condition_id.astype(str),
                }
            )
            .drop_duplicates()
            .groupby("study", observed=True)
            .size()
            .sort_values(ascending=False)
        )
        rank = np.arange(1, len(condition_counts) + 1)
        values = condition_counts.to_numpy(float)
        ax.plot(rank, values, color=TEAL, linewidth=2.0, zorder=1)
        ax.scatter(
            rank,
            values,
            s=24,
            facecolor=GREEN,
            edgecolor=INK,
            linewidth=0.55,
            alpha=0.88,
            zorder=2,
        )
        clean_axis(ax, "Study rank", "Strict conditions")
        ax.set_yscale("log")
        ax.set_xlim(0.5, len(values) + 0.5)
        add_metric_box(
            ax,
            [
                f"Studies = {len(values)}",
                f"Conditions = {int(values.sum()):,}",
                f"Median = {np.median(values):.0f}",
            ],
            loc=(0.96, 0.96),
            ha="right",
        )
        numeric_ticks(ax, x=True, y=False, n=4)

    def f1c(ax):
        bins = np.arange(1, min(13, multiplicity.max() + 2)) - 0.5
        ax.hist(multiplicity.clip(upper=12), bins=bins, color=GREEN, edgecolor=INK, linewidth=0.8)
        clean_axis(ax, "PFAS records per condition", "Number of conditions")
        ax._publication_tick_pad = 0.5
        ax._publication_xlabel_pad = 1.0
        ax._publication_ylabel_pad = -2.0
        ax._publication_ylabel_gap_pt = 4.0
        numeric_ticks(ax, n=5)
        ax.axvline(multiplicity.mean(), color=PLOT_COLORS["red"], linestyle="--", linewidth=1.8)
        add_metric_box(ax, [f"Mean = {multiplicity.mean():.2f}", f"Strict conditions = {condition_id.nunique():,}"], loc=(0.58, 0.95))

    def f1d(ax):
        group_order = ["Study", "Membrane", "PFAS"]
        group_sizes = pd.concat(
            [
                pd.DataFrame({"Grouping level": "Study", "Records per group": df.groupby("ref_id", observed=True).size().to_numpy()}),
                pd.DataFrame({"Grouping level": "Membrane", "Records per group": df.groupby("mem", observed=True).size().to_numpy()}),
                pd.DataFrame({"Grouping level": "PFAS", "Records per group": df.groupby("PFAS", observed=True).size().to_numpy()}),
            ],
            ignore_index=True,
        )
        sns.boxplot(
            data=group_sizes,
            x="Grouping level",
            y="Records per group",
            order=group_order,
            palette=[BLUE, GREEN, TEAL],
            width=0.48,
            showfliers=False,
            linewidth=1.0,
            ax=ax,
        )
        sns.stripplot(
            data=group_sizes,
            x="Grouping level",
            y="Records per group",
            order=group_order,
            color=INK,
            alpha=0.34,
            jitter=0.16,
            size=3.3,
            ax=ax,
        )
        ax.set_yscale("log")
        clean_axis(ax, "Grouping level", "PFAS-level records per group")
        for i, level in enumerate(group_order):
            values = group_sizes.loc[group_sizes["Grouping level"].eq(level), "Records per group"]
            ax.text(
                i,
                0.97,
                f"n = {len(values)}\nmedian = {values.median():.0f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9.8,
                fontweight="normal",
                color=INK,
            )

    def f1e(ax):
        coverage = pd.Series({"Studies": df.ref_id.nunique(), "Membranes": df.mem.nunique(), "PFAS": df.PFAS.nunique(), "Censored rows": int(c.sum()), "Real-water rows": int(df.feed_group.eq("Real water").sum())})
        colors = [BLUE, GREEN, TEAL, CORAL, GOLD]
        bars = ax.barh(coverage.index[::-1], coverage.values[::-1], color=colors[::-1], edgecolor=INK, linewidth=0.8)
        for bar, value in zip(bars, coverage.values[::-1]):
            ax.text(bar.get_width() + max(coverage.values) * 0.02, bar.get_y() + bar.get_height()/2, f"{value:,}", va="center", fontweight="normal", fontsize=11.5)
        clean_axis(ax, "Count", None)
        ax.set_xlim(0, max(coverage.values) * 1.18)
        numeric_ticks(ax, y=False, n=4)

    def f1f(ax):
        miss = df[FEATURES].isna().mean().sort_values()
        labels = [COMPACT_FEATURE_LABELS[x] for x in miss.index]
        colors = [GREEN if v < 0.1 else GOLD if v < 0.4 else CORAL for v in miss.values]
        ax.barh(labels, miss.values * 100, color=colors, edgecolor=INK, linewidth=0.7)
        clean_axis(ax, "Missing values (%)", None, tick_size=11.5, label_size=14.5)
        ax.set_xlim(0, max(70, miss.max()*115))
        numeric_ticks(ax, y=False, n=4)

    panels = [f1a, f1b, f1c, f1d, f1e, f1f]
    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 1.14], hspace=0.48, wspace=0.62)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for drawer, ax in zip(panels, axes): drawer(ax)
    save_publication_figure(fig, FIG_ROOT / "Fig01_Data_provenance_and_hierarchy", "Fig01_Data_provenance_and_hierarchy", panels, [(8,4.5),(7,4.5),(7,5.5),(7,5.5),(7,5.5),(7.5,6)], combined_axes=axes)

    # Figure 2
    strategy_order = ["Cluster-stratified", "Row-random", "Study-blocked", "Membrane-blocked", "PFAS-blocked"]
    strategy_colors = dict(zip(strategy_order, [BLUE, GREEN, CORAL, TEAL, LAVENDER]))
    fold_data = split_results.folds.copy()
    fold_data["r2_plot"] = fold_data["r2"].clip(-1.5, 1.0)
    _, _, _, temporal_metrics, _, _ = temporal

    def f2a(ax):
        labels = ["Training", "Five-fold\nCV", "Test"]
        x = np.arange(len(labels))
        values = np.array([0.97, 0.80, 0.86])
        errors = np.array([0.00, 0.04, 0.00])
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            color=[GRAY, BLUE, GREEN],
            edgecolor=INK,
            linewidth=0.8,
            error_kw={"ecolor": INK, "elinewidth": 1.0},
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11.5,
            )
        clean_axis(ax, None, "Benchmark R² (logit scale)")
        ax.set_ylim(0.68, 1.02)
        ax.set_xticks(x, labels, rotation=0, ha="center")
        numeric_ticks(ax, x=False, n=4)

    def f2b(ax):
        overlap = split_results.overlap.loc[
            split_results.overlap["strategy"].eq("Cluster-stratified")
        ].copy()
        columns = [
            "same_study",
            "same_membrane",
            "same_pfas",
            "exact_feature_duplicate",
        ]
        labels = ["Study", "Membrane", "PFAS", "Exact feature\nvector"]
        colors = [CORAL, TEAL, LAVENDER, GOLD]
        means = overlap[columns].mean().to_numpy() * 100
        bars = ax.bar(
            np.arange(len(columns)),
            means,
            color=colors,
            edgecolor=INK,
            linewidth=0.8,
            width=0.62,
        )
        for x_pos, column in enumerate(columns):
            values = overlap[column].to_numpy() * 100
            jitter = np.linspace(-0.11, 0.11, len(values))
            ax.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                s=15,
                color=INK,
                alpha=0.35,
                zorder=3,
            )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11.0,
            )
        clean_axis(ax, None, "Overlap with training (%)")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=0, ha="center")
        ax.set_ylim(0, 108)
        numeric_ticks(ax, x=False, n=5)

    def f2c(ax):
        agg = split_results.summary.set_index("strategy").reindex(strategy_order)
        labels = ["Benchmark", "Cluster", "Study", "Temporal"]
        values = [
            0.89,
            agg.loc["Cluster-stratified", "c_index"],
            agg.loc["Study-blocked", "c_index"],
            temporal_metrics["c_index"],
        ]
        bars = ax.bar(
            np.arange(len(values)),
            values,
            width=.56,
            color=[GRAY, BLUE, CORAL, LAVENDER],
            edgecolor=INK,
            linewidth=0.8,
        )
        for b, v in zip(bars, values):
            ax.text(
                b.get_x()+b.get_width()/2,
                v+0.012,
                f"{v:.2f}",
                ha="center",
                fontsize=11.5,
            )
        clean_axis(ax, None, "C-index")
        ax.set_xticks(np.arange(len(values)), labels, rotation=0, ha="center")
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.1)
        ax.set_ylim(0.45, 1.0)

    def prediction_panel(ax, strategy, color, marker_size=22, marker_alpha=0.55):
        pred = split_results.predictions[strategy]
        valid = exact & np.isfinite(pred)
        ax.scatter(
            y[valid]*100,
            pred[valid]*100,
            s=marker_size,
            facecolors="none",
            edgecolors=color,
            alpha=marker_alpha,
            linewidth=0.65,
        )
        ax.plot([50,100],[50,100], linestyle="--", color=MUTED, linewidth=1.4)
        m = evaluation(y[np.isfinite(pred)], pred[np.isfinite(pred)], c[np.isfinite(pred)])
        clean_axis(ax, "Observed rejection (%)", "Predicted rejection (%)")
        ax.set_xlim(49,101); ax.set_ylim(49,101); numeric_ticks(ax,n=5)
        add_metric_box(
            ax,
            [
                strategy,
                f"R² (rejection) = {m['r2']:.2f}",
                f"R² (logit) = {m['logit_r2']:.2f}",
                f"RMSE = {m['rmse']*100:.1f} pp",
            ],
            loc=(0.96, 0.06),
            ha="right",
            va="bottom",
        )

    def f2d(ax): prediction_panel(ax, "Cluster-stratified", GREEN)
    def f2e(ax): prediction_panel(ax, "Study-blocked", CORAL)
    def f2f(ax):
        agg = split_results.summary.set_index("strategy")
        delta = (agg.loc["Row-random", "r2"] - agg.loc[["Study-blocked","Membrane-blocked","PFAS-blocked"], "r2"])
        labels = ["Study", "Membrane", "PFAS"]
        ax.barh(
            labels,
            delta.values,
            height=.52,
            color=[CORAL,TEAL,LAVENDER],
            edgecolor=INK,
            linewidth=0.8,
        )
        clean_axis(ax, "R² optimism (random − blocked)", None)
        ax.axvline(0,color=MUTED,linewidth=1.1)
        numeric_ticks(ax,y=False,n=4)
        for i,v in enumerate(delta.values): ax.text(v+0.02*np.sign(v if v else 1),i,f"{v:+.2f}",va="center",fontweight="normal",fontsize=11.5)

    panels=[f2a,f2b,f2c,f2d,f2e,f2f]
    fig,axes=plt.subplots(2,3,figsize=(12.2,7.3))
    for drawer,ax in zip(panels,axes.flat): drawer(ax)
    fig.subplots_adjust(hspace=0.43,wspace=0.34)
    # Historical Figure 2 is retained on disk, but the current manuscript uses
    # the independently generated familiar-row versus unseen-study transfer
    # figure. Do not overwrite that active figure during the main pipeline run.
    plt.close(fig)

    # Figure 3
    method_order=["Censored as exact","Censored excluded","Censor-aware AFT"]
    method_colors={method_order[0]:BLUE,method_order[1]:GOLD,method_order[2]:CORAL}
    def f3a(ax):
        sns.boxplot(
            data=ladder_folds,
            x="method",
            y=ladder_folds["rmse"]*100,
            order=method_order,
            palette=method_colors,
            width=.48,
            showfliers=False,
            ax=ax,
        )
        clean_axis(ax,"Censoring treatment","Study-blocked RMSE (pp)")
        ax.set_xticklabels(["Bound as exact","Exact only","AFT"],rotation=12,ha="center")
        numeric_ticks(ax,x=False,n=5)
    def f3b(ax):
        bins=pd.qcut(y[exact],q=6,duplicates="drop")
        centers=df.loc[exact].groupby(bins,observed=True)["Rejection"].mean()*100
        for label,color,marker in [("Censored as exact",BLUE,"o"),("Censor-aware AFT",CORAL,"s")]:
            resid=(ladder_preds[label][exact]-y[exact])*100
            mean=pd.Series(resid).groupby(bins,observed=True).mean()
            ax.plot(centers.values,mean.values,marker=marker,color=color,linewidth=2,label=label)
        ax.axhline(0,color=MUTED,linestyle="--",linewidth=1.2)
        clean_axis(ax,"Observed rejection bin mean (%)","Mean prediction bias (pp)")
        numeric_ticks(ax,n=5)
    def f3c(ax):
        vals=study_means["mean_residual"]*100
        ax.hist(vals,bins=16,color=TEAL,edgecolor=INK,linewidth=0.8)
        ax.axvline(0,color=MUTED,linestyle="--",linewidth=1.3)
        clean_axis(ax,"Study mean residual (pp)","Number of studies")
        add_metric_box(ax, [f"Residual ICC ≈ {residual_icc:.2f}", "Diagnostic only"], loc=(0.04, 0.96))
    def f3d(ax):
        sub=study_means.assign(abs_res=lambda x:x.mean_residual.abs()).nlargest(14,"abs_res").sort_values("mean_residual")
        ax.scatter(sub["mean_residual"]*100,np.arange(len(sub)),s=45,c=np.where(sub.mean_residual>0,CORAL,BLUE),edgecolor=INK,linewidth=0.6)
        ax.hlines(np.arange(len(sub)),0,sub["mean_residual"]*100,color=MUTED,linewidth=1)
        ax.set_yticks(np.arange(len(sub)), [compact_study_label(s) for s in sub.study])
        ax.axvline(0,color=INK,linewidth=1)
        clean_axis(ax,"Mean residual (pp)",None,tick_size=10.8,label_size=14.5)
    def f3e(ax):
        methods=method_order
        vals=[ladder_summary.set_index("method").loc[m,"bound_violation"]*100 for m in methods]
        x=np.arange(len(methods))
        bars=ax.bar(x,vals,color=[method_colors[m] for m in methods],edgecolor=INK,linewidth=0.8)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.1f}%",ha="center",fontweight="normal",fontsize=11.5)
        clean_axis(ax,None,"Lower-bound violation (%)")
        ax.set_xticks(x,["As exact","Excluded","AFT"],rotation=0,ha="center")
    panels=[f3a,f3b,f3c,f3d,f3e]
    fig=plt.figure(figsize=(13.4,7.4)); gs=fig.add_gridspec(2,3,hspace=.52,wspace=.55)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,:2]),fig.add_subplot(gs[1,2])]
    for drawer,ax in zip(panels,axes):drawer(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=10.5, frameon=False)
    save_publication_figure(fig,FIG_ROOT/"Fig03_Censoring_and_study_heterogeneity","Fig03_Censoring_and_study_heterogeneity",panels,[(6.5,5.2),(6.5,5.2),(6.5,5.2),(9,6),(6.5,5.2)], combined_axes=axes)

    # Figure 4
    def f4a(ax):
        valid=np.flatnonzero(exact & np.isfinite(interval["prediction"]))
        chosen=valid[np.argsort(interval.loc[valid,"prediction"].to_numpy())]
        if len(chosen)>90: chosen=chosen[np.linspace(0,len(chosen)-1,90).astype(int)]
        x=np.arange(len(chosen)); pred=interval.loc[chosen,"prediction"].to_numpy()*100
        lo=interval.loc[chosen,"lo_90"].to_numpy()*100; hi=interval.loc[chosen,"hi_90"].to_numpy()*100
        ax.fill_between(x,lo,hi,color=BLUE,alpha=.35,label="90% interval")
        ax.plot(x,pred,color=PLOT_COLORS["navy"],linewidth=1.7,label="Prediction")
        ax.scatter(x,y[chosen]*100,s=18,facecolor=CORAL,edgecolor=INK,linewidth=.4,label="Observed",zorder=4)
        clean_axis(ax,"Held-out observations (sorted)","Rejection (%)")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=11.0,
            frameon=False,
            borderaxespad=0.0,
        )
    def f4b(ax):
        nom=[.8,.9,.95]
        means=conformal_folds.groupby("nominal")["coverage_exact"].mean().reindex(nom)
        lo=conformal_folds.groupby("nominal")["coverage_exact"].quantile(.1).reindex(nom)
        hi=conformal_folds.groupby("nominal")["coverage_exact"].quantile(.9).reindex(nom)
        ax.plot(np.array(nom)*100,np.array(nom)*100,color=MUTED,linestyle="--",linewidth=1.3)
        ax.errorbar(np.array(nom)*100,means.values*100,yerr=[(means-lo).values*100,(hi-means).values*100],fmt="o-",color=TEAL,markerfacecolor="white",markeredgecolor=INK,capsize=4,linewidth=2)
        clean_axis(ax,"Nominal coverage (%)","Empirical coverage (%)")
        ax.set_xlim(77,97);ax.set_ylim(55,102);numeric_ticks(ax,n=4)
    def f4c(ax):
        width=(interval["hi_90"]-interval["lo_90"])*100
        err=np.abs(interval["prediction"]-y)*100
        valid=exact & width.notna().to_numpy()
        q=pd.qcut(width[valid],5,duplicates="drop")
        temp=pd.DataFrame({"width":width[valid],"error":err[valid]}).groupby(q,observed=True).agg(width=("width","mean"),error=("error","mean"),n=("error","size"))
        ax.scatter(width[valid],err[valid],s=14,facecolors="none",edgecolors=GRAY,alpha=.25)
        ax.plot(temp.width,temp.error,"o-",color=CORAL,markeredgecolor=INK,linewidth=2,markersize=7)
        clean_axis(ax,"90% interval width (pp)","Absolute error (pp)")
        numeric_ticks(ax,n=5)
    def f4d(ax):
        rows=[]
        groups={"All exact":exact,"Real water":exact & df.feed_group.eq("Real water").to_numpy(),"Short chain":exact & df.carbons.le(5).fillna(False).to_numpy(),"High rejection":exact & df.Rejection.ge(.95).to_numpy()}
        for label,mask in groups.items():
            lo=interval["lo_90"].to_numpy();hi=interval["hi_90"].to_numpy();valid=mask & np.isfinite(lo)
            rows.append((label,np.mean((y[valid]>=lo[valid])&(y[valid]<=hi[valid]))*100,valid.sum()))
        labels=[r[0] for r in rows];vals=[r[1] for r in rows]
        bars=ax.barh(labels[::-1],vals[::-1],color=[BLUE,GREEN,GOLD,CORAL][::-1],edgecolor=INK,linewidth=.8)
        ax.axvline(90,color=MUTED,linestyle="--",linewidth=1.2)
        for b,v in zip(bars,vals[::-1]):ax.text(v+1,b.get_y()+b.get_height()/2,f"{v:.1f}%",va="center",fontweight="normal",fontsize=11.5)
        clean_axis(ax,"Empirical 90% coverage (%)",None)
        ax.set_xlim(0,105)
    def f4e(ax):
        sub=conformal_folds[conformal_folds.nominal.eq(.9)]
        ax.scatter(sub.coverage_exact*100,sub.mean_width*100,s=70,c=np.arange(len(sub)),cmap="viridis",edgecolor=INK,linewidth=.7)
        clean_axis(ax,"Fold coverage (%)","Interval width (pp)")
        numeric_ticks(ax,n=5)
        add_metric_box(ax,["Each point = held-out study fold","Intervals calibrated on separate studies"],loc=(.05,.95))
    panels=[f4a,f4b,f4c,f4d,f4e]
    fig=plt.figure(figsize=(12.2,7.1));gs=fig.add_gridspec(2,3,hspace=.44,wspace=.39)
    axes=[fig.add_subplot(gs[0,:2]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2])]
    for drawer,ax in zip(panels,axes):drawer(ax)
    save_publication_figure(fig,FIG_ROOT/"Fig04_Group_calibrated_uncertainty","Fig04_Group_calibrated_uncertainty",panels,[(9,5.2),(6,5),(6,5),(6,5),(6,5)], combined_axes=axes)

    # Figure 5
    distance=split_results.study_distance
    dnorm=distance/np.nanmedian(distance)
    domain_df=pd.DataFrame({"distance":dnorm,"error":study_error*100,"exact":exact,"new_mem":split_results.study_new_membrane,"new_pfas":split_results.study_new_pfas,"real":df.feed_group.eq("Real water").to_numpy(),"short":df.carbons.le(5).fillna(False).to_numpy()})
    def f5a(ax):
        valid=exact & np.isfinite(dnorm)
        ax.scatter(dnorm[valid],study_error[valid]*100,s=17,facecolors="none",edgecolors=TEAL,alpha=.35,linewidth=.65)
        bins=pd.qcut(dnorm[valid],6,duplicates="drop")
        temp=pd.DataFrame({"d":dnorm[valid],"e":study_error[valid]*100}).groupby(bins,observed=True).agg(d=("d","mean"),e=("e","mean"))
        ax.plot(temp.d,temp.e,"o-",color=CORAL,markeredgecolor=INK,linewidth=2.2)
        clean_axis(ax,"Relative kNN distance to training domain","Absolute error (pp)")
        ax.set_xscale("symlog", linthresh=0.25, linscale=1.0, base=10)
        ax.set_xticks([0, 1, 5, 30])
        ax.set_xticklabels(["0", "1", "5", "30"])
        numeric_ticks(ax,x=False,n=5)
    def f5b(ax):
        valid=np.flatnonzero(exact & np.isfinite(dnorm))
        order=valid[np.argsort(dnorm[valid])]
        coverages=np.linspace(.35,1,14);rmse=[]
        for cov in coverages:
            keep=order[:max(5,int(len(order)*cov))]
            rmse.append(mean_squared_error(y[keep],study_pred[keep])**.5*100)
        ax.plot(coverages*100,rmse,"o-",color=PLOT_COLORS["navy"],markerfacecolor=BLUE,markeredgecolor=INK,linewidth=2.2)
        clean_axis(ax,"Predictions retained (%)","RMSE after rejection (pp)")
        ax._publication_tick_pad = 0.5
        ax._publication_xlabel_pad = 1.0
        ax._publication_ylabel_pad = -2.0
        ax._publication_ylabel_gap_pt = 4.0
        numeric_ticks(ax,n=5)
    def f5c(ax):
        cats=[("All exact",np.ones(len(df),bool)),("New membrane",domain_df.new_mem.to_numpy()),("New PFAS",domain_df.new_pfas.to_numpy()),("Real water",domain_df.real.to_numpy()),("Short chain",domain_df.short.to_numpy())]
        vals=[];ns=[]
        for label,mask in cats:
            valid=exact & mask & np.isfinite(study_pred)
            vals.append(mean_absolute_error(y[valid],study_pred[valid])*100 if valid.any() else np.nan);ns.append(valid.sum())
        bars=ax.barh([x[0] for x in cats][::-1],vals[::-1],color=[BLUE,CORAL,LAVENDER,GREEN,GOLD][::-1],edgecolor=INK,linewidth=.8)
        for b,v,n in zip(bars,vals[::-1],ns[::-1]):ax.text(v+.2,b.get_y()+b.get_height()/2,f"{v:.1f} (n={n})",va="center",fontsize=11.3,fontweight="normal")
        clean_axis(ax,"Mean absolute error (pp)",None,tick_size=11.3,label_size=14.5)
        numeric_ticks(ax,y=False,n=5)
    def f5d(ax):
        miss=df[FEATURES].isna().mean(axis=1)*100
        valid=exact & np.isfinite(study_pred)
        plot_df=pd.DataFrame({
            "missing":np.rint(miss[valid]).astype(int),
            "error":study_error[valid]*100,
        })
        order=sorted(plot_df["missing"].unique())
        sns.boxplot(
            data=plot_df,
            x="missing",
            y="error",
            order=order,
            color=LAVENDER,
            width=.58,
            showfliers=False,
            linewidth=1.0,
            ax=ax,
        )
        jitter_df=plot_df.sample(min(len(plot_df), 900), random_state=42)
        sns.stripplot(
            data=jitter_df,
            x="missing",
            y="error",
            order=order,
            color=PURPLE,
            alpha=.38,
            size=3.2,
            jitter=.20,
            ax=ax,
        )
        clean_axis(ax,"Missing predictors (%)","Absolute error (pp)")
        ax.set_xticklabels([str(v) for v in order])
        numeric_ticks(ax,x=False,n=5)
    def f5e(ax):
        valid=exact & np.isfinite(dnorm)
        threshold=np.nanquantile(dnorm[valid],.9)
        out=dnorm>=threshold
        matrix=pd.crosstab(df.feed_group,pd.Series(out,index=df.index,name="Outside AD"),normalize="index")
        vals=matrix.get(True,pd.Series(0,index=matrix.index))*100
        feed_colors={"Pure water":BLUE,"Real water":GREEN,"Synthetic":CORAL,"Synthetic water":CORAL,"Unreported":MUTED}
        feed_labels={"Pure water":"Pure\nwater","Real water":"Real\nwater","Synthetic":"Synthetic","Synthetic water":"Synthetic","Unreported":"Unreported"}
        x=np.arange(len(vals))
        bars=ax.bar(x,vals.values,color=[feed_colors.get(str(v),MUTED) for v in vals.index],edgecolor=INK,linewidth=.8)
        for bar,value in zip(bars,vals.values):
            ax.text(
                bar.get_x()+bar.get_width()/2,
                value+0.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11.3,
                fontweight="normal",
            )
        clean_axis(ax,None,"Rows outside distance AD (%)")
        ax._publication_tick_pad = 0.5
        ax._publication_ylabel_pad = -2.0
        ax._publication_ylabel_gap_pt = 4.0
        ax.set_xticks(x,[feed_labels.get(str(v),str(v)) for v in vals.index],rotation=0,ha="center")
        ax.set_ylim(0, max(5, float(vals.max())*1.22))
    panels=[f5a,f5b,f5c,f5d,f5e]
    fig=plt.figure(figsize=(12.2,7.1));gs=fig.add_gridspec(2,3,hspace=.44,wspace=.40)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,:2]),fig.add_subplot(gs[1,2])]
    for drawer,ax in zip(panels,axes):drawer(ax)
    save_publication_figure(fig,FIG_ROOT/"Fig05_Applicability_domain_and_reject_option","Fig05_Applicability_domain_and_reject_option",panels,[(6.3,5)]*3+[(9,5.5),(6.3,5)], combined_axes=axes)

    # Figure 6
    failure=df.assign(abs_error=study_error*100).pivot_table(index="chain_group",columns="feed_group",values="abs_error",aggfunc="mean",observed=True)
    def f6a(ax):
        ax.set_facecolor("#F1F3F5")
        sns.heatmap(
            failure,
            annot=True,
            annot_kws={"fontsize": 11.5, "fontweight": "normal"},
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=.8,
            linecolor="white",
            cbar_kws={
                "label": "MAE (pp)",
                "shrink": .80,
                "pad": .04,
                "aspect": 18,
            },
            ax=ax,
        )
        for row_no,col_no in np.argwhere(failure.isna().to_numpy()):
            ax.add_patch(
                plt.Rectangle(
                    (col_no, row_no),
                    1,
                    1,
                    facecolor="#F1F3F5",
                    edgecolor="white",
                    linewidth=.8,
                    zorder=2,
                )
            )
            ax.text(
                col_no+.5,
                row_no+.5,
                "—",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="normal",
                color=MUTED,
                zorder=3,
            )
        matrix_labels = {
            "Pure water": "Pure\nwater",
            "Real water": "Real\nwater",
            "Synthetic": "Synthetic\nwater",
            "Synthetic water": "Synthetic\nwater",
            "Unreported": "Not\nreported",
        }
        ax.set_xticklabels(
            [matrix_labels.get(str(value), str(value)) for value in failure.columns],
            rotation=0,
            ha="center",
            va="top",
        )
        ax.set_yticklabels(
            [str(value) for value in failure.index],
            rotation=0,
            ha="right",
            va="center",
        )
        ax.set_xlabel("Feed matrix",fontweight="normal",fontsize=14.5,labelpad=7)
        ax.set_ylabel("PFAS chain group",fontweight="normal",fontsize=14.5,labelpad=7)
        ax.tick_params(axis="x",bottom=True,labelsize=11.2,pad=5)
        ax.tick_params(axis="y",left=True,labelsize=11.2,pad=5)
        for lab in ax.get_xticklabels()+ax.get_yticklabels():
            lab.set_fontweight("normal")
        colorbar = ax.collections[0].colorbar
        colorbar.set_label("MAE (pp)",fontsize=11.5,fontweight="normal",labelpad=7)
        colorbar.ax.tick_params(labelsize=10.5,pad=3)
        for lab in colorbar.ax.get_yticklabels():
            lab.set_fontweight("normal")
    def f6b(ax):
        sub=shap_summary.head(10).sort_values("mean_abs_shap")
        ax.barh([COMPACT_FEATURE_LABELS[x] for x in sub.feature],sub.mean_abs_shap,xerr=sub.sd_abs_shap,color=TEAL,edgecolor=INK,linewidth=.7,capsize=3)
        clean_axis(ax,"Mean |SHAP| across study folds",None,tick_size=11.2,label_size=14.5)
        numeric_ticks(ax,y=False,n=4)
    def f6c(ax):
        top=shap_summary.head(10).feature.tolist();ranks=shap_fold[top].rank(axis=1,ascending=False).T
        sns.heatmap(ranks,cmap="Blues_r",annot=True,annot_kws={"fontsize": 10.8, "fontweight": "normal"},fmt=".0f",linewidths=.6,linecolor="white",cbar_kws={"label":"Within-fold rank"},ax=ax)
        ax.set_yticklabels([COMPACT_FEATURE_LABELS[x] for x in top],rotation=0)
        ax.set_xlabel("Study-blocked fold",fontweight="normal",fontsize=15);ax.set_ylabel("")
        ax.tick_params(bottom=True,left=True,labelsize=11.2)
        for lab in ax.get_xticklabels()+ax.get_yticklabels():lab.set_fontweight("normal")
    def f6d(ax):
        top=shap_summary.head(10).feature.tolist();rho=sign_pivot.loc[top]
        sns.heatmap(
            rho,
            cmap=DIVERGING,
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            annot_kws={"fontsize": 10.8, "fontweight": "normal"},
            fmt=".2f",
            linewidths=.6,
            linecolor="white",
            cbar_kws={"label":"Spearman ρ","shrink":.84,"pad":.018,"aspect":22},
            ax=ax,
        )
        ax.set_yticklabels([COMPACT_FEATURE_LABELS[x] for x in top],rotation=0)
        ax.set_xlabel("Study-blocked fold",fontweight="normal",fontsize=15);ax.set_ylabel("")
        ax.tick_params(bottom=True,left=True,labelsize=11.2)
        for lab in ax.get_xticklabels()+ax.get_yticklabels():lab.set_fontweight("normal")
    def f6e(ax):
        colors=[BLUE,CORAL,GREEN,LAVENDER]
        for (feature,(xvals,curves)),color in zip(ale_curves.items(),colors):
            mean=curves.mean(axis=0)*100;lo=np.quantile(curves,.1,axis=0)*100;hi=np.quantile(curves,.9,axis=0)*100
            q=np.linspace(0,100,len(mean))
            ax.plot(q,mean,color=color,linewidth=2,label=COMPACT_FEATURE_LABELS[feature])
            ax.fill_between(q,lo,hi,color=color,alpha=.18)
        ax.axhline(0,color=MUTED,linestyle="--",linewidth=1.1)
        clean_axis(ax,"Feature percentile","Centered ALE (pp)")
        legend = ax.legend(
            loc="upper left",
            ncol=2,
            fontsize=9.5,
            frameon=True,
            borderpad=0.35,
            handlelength=1.6,
            columnspacing=0.8,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#CDD3D8")
        legend.get_frame().set_alpha(0.92)
        numeric_ticks(ax,n=5)
    panels=[f6a,f6b,f6c,f6d,f6e]
    fig=plt.figure(figsize=(14.4,8.0))
    outer=fig.add_gridspec(2,1,hspace=.42)
    top=outer[0].subgridspec(1,3,width_ratios=[1.08,1.06,1.06],wspace=.68)
    bottom=outer[1].subgridspec(1,2,width_ratios=[2.05,1.0],wspace=.20)
    axes=[fig.add_subplot(top[0,0]),fig.add_subplot(top[0,1]),fig.add_subplot(top[0,2]),fig.add_subplot(bottom[0,0]),fig.add_subplot(bottom[0,1])]
    for drawer,ax in zip(panels,axes):drawer(ax)
    save_publication_figure(fig,FIG_ROOT/"Fig06_Failure_map_and_stable_associations","Fig06_Failure_map_and_stable_associations",panels,[(7,5.5),(7,6),(7,6),(10,7),(7,5.5)], combined_axes=axes)

    # Figure 7
    assoc_indexed = assoc.set_index("treatment").loc[
        ["mon_cat_mM", "di_cat_mM", "doc_mg/L"]
    ].reset_index()

    def f7a(ax):
        order = ["Trivalent", "DOC", "Divalent", "Monovalent"]
        support = support_audit.set_index("treatment").reindex(order)
        y_pos = np.arange(len(order))
        colors = [MUTED, GOLD, BLUE, CORAL]
        for ypos, (_, row), color in zip(y_pos, support.iterrows(), colors):
            reporting = float(row.reporting_studies)
            varying = float(row.within_study_varying_studies)
            ax.hlines(ypos, varying, reporting, color="#CBD2D9", linewidth=3.0, zorder=1)
            ax.scatter(
                reporting,
                ypos,
                s=76,
                facecolor="white",
                edgecolor=color,
                linewidth=2.0,
                zorder=3,
            )
            ax.scatter(
                varying,
                ypos,
                s=76,
                facecolor=color,
                edgecolor=INK,
                linewidth=0.7,
                zorder=4,
            )
            ax.text(
                reporting + 0.9,
                ypos,
                f"{int(varying)}/{int(reporting)}",
                ha="left",
                va="center",
                fontsize=10.8,
                fontweight="normal",
                color=INK,
            )
        ax.scatter([], [], s=64, facecolor=TEAL, edgecolor=INK, linewidth=0.7, label="Varying")
        ax.scatter([], [], s=64, facecolor="white", edgecolor=TEAL, linewidth=2.0, label="Reporting")
        ax.set_yticks(y_pos, order)
        clean_axis(ax, "Number of studies", None, tick_size=11.0, label_size=14.5)
        ax.set_xlim(0, float(support.reporting_studies.max()) + 5.5)
        numeric_ticks(ax, y=False, n=5)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            fontsize=10.2,
            frameon=False,
            borderaxespad=0.0,
        )

    def f7b(ax):
        for exposed, color, label in [
            (False, BLUE, "Zero"),
            (True, CORAL, "Non-zero"),
        ]:
            values = overlap_df.loc[overlap_df["exposed"].eq(exposed), "propensity"]
            sns.kdeplot(
                values,
                ax=ax,
                color=color,
                fill=True,
                alpha=0.22,
                linewidth=2,
                label=label,
                warn_singular=False,
            )
        summary_row = overlap_summary.iloc[0]
        ax.axvspan(
            summary_row.common_support_low,
            summary_row.common_support_high,
            color=GREEN,
            alpha=0.10,
            label="Common support",
        )
        ax.axvline(summary_row.common_support_low, color=MUTED, linestyle="--", linewidth=1)
        ax.axvline(summary_row.common_support_high, color=MUTED, linestyle="--", linewidth=1)
        clean_axis(
            ax,
            "Cross-fitted P(monovalent cations > 0)",
            "Density",
            label_size=14.5,
        )
        ax.set_xlim(0, 1)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=9.8,
            frameon=False,
            borderaxespad=0.0,
        )
        numeric_ticks(ax, n=5)

    def f7c(ax):
        y_pos = np.arange(len(assoc_indexed))
        labels = ["Monovalent", "Divalent", "DOC"]
        colors = [CORAL, BLUE, GOLD]
        for i, (_, row) in enumerate(assoc_indexed.iterrows()):
            ax.errorbar(
                row.effect_pp_at_median,
                i,
                xerr=[[row.effect_pp_at_median - row.ci_low_pp], [row.ci_high_pp - row.effect_pp_at_median]],
                fmt="o",
                markersize=7,
                color=colors[i],
                markerfacecolor="white",
                markeredgecolor=INK,
                capsize=5,
                linewidth=2,
            )
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos, labels)
        clean_axis(
            ax,
            "Adjusted rejection change\n(pp / within-study SD)",
            None,
            tick_size=10.8,
            label_size=13.8,
        )
        numeric_ticks(ax, y=False, n=5)

    def f7d(ax):
        labels = ["Monovalent", "Divalent", "DOC"]
        values = assoc_indexed["approx_e_value"].to_numpy(float)
        limits = assoc_indexed["approx_e_value_ci_limit"].to_numpy(float)
        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos,
            values - 1,
            left=1,
            color=[CORAL, BLUE, GOLD],
            edgecolor=INK,
            linewidth=0.8,
        )
        ax.scatter(
            limits,
            y_pos,
            marker="D",
            s=45,
            facecolor="white",
            edgecolor=INK,
            zorder=4,
            label="CI-limit",
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontweight="normal",
                fontsize=11.2,
            )
        ax.axvline(1, color=MUTED, linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos, labels)
        clean_axis(ax, "Approximate E-value", None, tick_size=10.8, label_size=14.0)
        ax.set_xlim(0.98, max(values) + 0.07)
        numeric_ticks(ax, y=False, n=4)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            fontsize=10.8,
            frameon=False,
            borderaxespad=0.0,
        )

    def f7e(ax):
        order = ["Original", "Placebo treatment", "Random common cause", "80% study subset"]
        ref = refutation_summary.set_index("test").reindex(order).reset_index()
        y_pos = np.arange(len(ref))
        colors = [CORAL, GOLD, GREEN, BLUE]
        for i, row in ref.iterrows():
            ax.errorbar(
                row.estimate,
                i,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt="o",
                color=colors[i],
                markerfacecolor="white",
                markeredgecolor=INK,
                capsize=5,
                linewidth=2,
            )
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.2)
        original_estimate = ref.loc[ref["test"].eq("Original"), "estimate"].iloc[0]
        ax.axvline(original_estimate, color=BLUE, linestyle=":", linewidth=1.4)
        ax.text(
            original_estimate,
            1.02,
            "Original estimate",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10.8,
            fontweight="normal",
            color=BLUE,
        )
        ax.set_yticks(y_pos, ["Original", "Placebo", "Common cause", "80% subset"])
        clean_axis(ax, "Adjusted coefficient (logit scale)", None, tick_size=10.8, label_size=14.0)
        numeric_ticks(ax, y=False, n=5)

    panels = [f7a, f7b, f7c, f7d, f7e]
    fig = plt.figure(figsize=(14.8, 7.9))
    gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.54)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    save_publication_figure(
        fig,
        FIG_ROOT / "Fig07_Within_study_adjusted_associations",
        "Fig07_Within_study_adjusted_associations",
        panels,
        [(7, 5.5)] * 5,
        combined_axes=axes,
    )

    # Figure 8
    cutoff,test_idx,temporal_pred,temporal_metrics,shift_df,comparisons=temporal
    def f8a(ax):
        counts=df.groupby("ref_year")["ref_id"].nunique()
        colors=[BLUE if year<=cutoff else CORAL for year in counts.index]
        bars=ax.bar(counts.index.astype(int),counts.values,color=colors,edgecolor=INK,linewidth=.6)
        ax.axvline(cutoff+.5,color=INK,linestyle="--",linewidth=1.5)
        clean_axis(ax,"Publication year","Number of studies")
        numeric_ticks(ax,n=6)
        train_idx=np.flatnonzero(counts.index.to_numpy()<=cutoff)[0]
        test_idx_bar=np.flatnonzero(counts.index.to_numpy()>cutoff)[0]
        ax.legend(
            [bars[train_idx],bars[test_idx_bar]],
            [f"Train ≤ {cutoff}","Time-out test"],
            loc="upper left",
            bbox_to_anchor=(0.02,0.98),
            fontsize=10.8,
            frameon=False,
        )
    def f8b(ax):
        valid=test_idx[~c[test_idx]]
        ax.scatter(y[valid]*100,temporal_pred[valid]*100,s=28,facecolors="none",edgecolors=CORAL,alpha=.6,linewidth=.8)
        ax.plot([50,100],[50,100],color=MUTED,linestyle="--",linewidth=1.3)
        clean_axis(ax,"Observed rejection (%)","Predicted rejection (%)")
        ax.set_xlim(49,101);ax.set_ylim(49,101);numeric_ticks(ax,n=5)
        add_metric_box(
            ax,
            [f"Train ≤ {cutoff}",f"Test {cutoff+1}–{int(df.ref_year.max())}",f"R² = {temporal_metrics['r2']:.2f}",f"RMSE = {temporal_metrics['rmse']*100:.1f} pp"],
            loc=(.04,.06),
            va="bottom",
        )
    def f8c(ax):
        order=["Row-random","Study-blocked","Publication-year holdout"]
        sub=comparisons.set_index("strategy").reindex(order)
        bars=ax.bar(order,sub.r2,color=[GREEN,CORAL,LAVENDER],edgecolor=INK,linewidth=.8)
        ax.axhline(0,color=MUTED,linestyle="--",linewidth=1.2)
        for b,v in zip(bars,sub.r2):
            ax.text(
                b.get_x()+b.get_width()/2,
                v + (.012 if v >= 0 else .018),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontweight="normal",
                fontsize=11.8,
            )
        clean_axis(ax,None,"R²")
        ax.set_xticklabels(["Random","Study","Temporal"],rotation=0,ha="center")
        ax.tick_params(axis="x", pad=5)
    def f8d(ax):
        sub=shift_df.head(10).sort_values("smd")
        ax.barh([COMPACT_FEATURE_LABELS[x] for x in sub.feature],sub.smd,color=np.where(sub.smd>=0,CORAL,BLUE),edgecolor=INK,linewidth=.7)
        ax.axvline(0,color=MUTED,linewidth=1.1)
        ax.axvline(.5,color=MUTED,linestyle="--",linewidth=1.0,zorder=0)
        clean_axis(ax,"Standardized mean shift (future − earlier)",None,tick_size=11.0,label_size=14.2)
        ax.xaxis.labelpad=9
        numeric_ticks(ax,y=False,n=5)
    def f8e(ax):
        pivot=pd.crosstab(df.chain_group,df.feed_group).reindex(index=["C≤3","C4–5","C6–7","C≥8"],columns=["Pure water","Synthetic","Real water","Unreported"],fill_value=0)
        sns.heatmap(pivot,annot=True,annot_kws={"fontsize": 11.5, "fontweight": "normal"},fmt="d",cmap="Blues",linewidths=.7,linecolor="white",cbar_kws={"label":"Rows"},ax=ax)
        ax.set_xlabel("Feed matrix",fontweight="normal",fontsize=15);ax.set_ylabel("PFAS chain group",fontweight="normal",fontsize=15)
        ax.tick_params(bottom=True,left=True,labelsize=11.2)
        ax.set_xticklabels(["Pure","Synth.","Real","Unrep."],rotation=0,ha="center")
        ax.tick_params(axis="x", pad=5)
        plt.setp(ax.get_yticklabels(),rotation=0)
        ax.xaxis.labelpad=9
        for lab in ax.get_xticklabels()+ax.get_yticklabels():lab.set_fontweight("normal")
    panels=[f8a,f8b,f8c,f8d,f8e]
    fig=plt.figure(figsize=(12.2,7.1));gs=fig.add_gridspec(2,3,hspace=.44,wspace=.42)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,:2]),fig.add_subplot(gs[1,2])]
    for drawer,ax in zip(panels,axes):drawer(ax)
    bottom_left=axes[3].get_position()
    bottom_right=axes[4].get_position()
    axes[4].set_position([bottom_right.x0,bottom_left.y0,bottom_right.width,bottom_left.height])
    save_publication_figure(fig,FIG_ROOT/"Fig08_Publication_year_transfer_test","Fig08_Publication_year_transfer_test",panels,[(6.5,5),(6.5,5),(6.5,5),(9,6),(7,5.5)], combined_axes=axes)


def write_captions_and_tables(
    raw,
    df,
    split_results,
    ladder_summary,
    ladder_folds,
    study_means,
    residual_icc,
    conformal_folds,
    interval,
    shap_summary,
    shap_fold,
    sign_pivot,
    assoc,
    overlap_df,
    overlap_summary,
    refutation_summary,
    support_audit,
    temporal,
):
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {"stage": "Raw workbook", "rows": len(raw), "studies": raw.ref_id.nunique(), "membranes": raw.mem.nunique(), "PFAS": raw.PFAS.nunique()},
            {"stage": "R >= 0.5", "rows": int(raw.Rejection.ge(.5).sum()), "studies": raw.loc[raw.Rejection.ge(.5),"ref_id"].nunique(), "membranes": raw.loc[raw.Rejection.ge(.5),"mem"].nunique(), "PFAS": raw.loc[raw.Rejection.ge(.5),"PFAS"].nunique()},
            {"stage": "R >= 0.5 and NF/RO", "rows": len(df), "studies": df.ref_id.nunique(), "membranes": df.mem.nunique(), "PFAS": df.PFAS.nunique()},
            {"stage": "Published analysis", "rows": 2353, "studies": 60, "membranes": 111, "PFAS": 44},
        ]
    )
    summary.to_csv(TABLE_ROOT/"dataset_flow.csv",index=False,encoding="utf-8-sig")
    split_results.summary.to_csv(TABLE_ROOT/"split_strategy_summary.csv",index=False,encoding="utf-8-sig")
    split_results.folds.to_csv(TABLE_ROOT/"split_strategy_fold_metrics.csv",index=False,encoding="utf-8-sig")
    split_results.overlap.to_csv(TABLE_ROOT/"split_overlap_audit.csv",index=False,encoding="utf-8-sig")
    ladder_summary.to_csv(TABLE_ROOT/"censoring_model_summary.csv",index=False,encoding="utf-8-sig")
    ladder_folds.to_csv(TABLE_ROOT/"censoring_model_fold_metrics.csv",index=False,encoding="utf-8-sig")
    study_means.to_csv(TABLE_ROOT/"study_mean_residuals.csv",index=False,encoding="utf-8-sig")
    conformal_folds.to_csv(TABLE_ROOT/"group_conformal_fold_metrics.csv",index=False,encoding="utf-8-sig")
    interval.to_csv(TABLE_ROOT/"group_conformal_intervals.csv",index=False,encoding="utf-8-sig")
    shap_summary.to_csv(TABLE_ROOT/"shap_stability_summary.csv",index=False,encoding="utf-8-sig")
    shap_fold.to_csv(TABLE_ROOT/"shap_fold_importance.csv",index=False,encoding="utf-8-sig")
    sign_pivot.to_csv(TABLE_ROOT/"shap_direction_stability.csv",encoding="utf-8-sig")
    assoc.to_csv(TABLE_ROOT/"within_study_adjusted_associations.csv",index=False,encoding="utf-8-sig")
    overlap_df.to_csv(TABLE_ROOT/"monovalent_overlap_diagnostic.csv",index=False,encoding="utf-8-sig")
    overlap_summary.to_csv(TABLE_ROOT/"monovalent_overlap_summary.csv",index=False,encoding="utf-8-sig")
    refutation_summary.to_csv(TABLE_ROOT/"monovalent_refutation_summary.csv",index=False,encoding="utf-8-sig")
    support_audit.to_csv(TABLE_ROOT/"causal_support_audit.csv",index=False,encoding="utf-8-sig")
    cutoff,test_idx,temporal_pred,temporal_metrics,shift_df,comparisons=temporal
    shift_df.to_csv(TABLE_ROOT/"publication_year_feature_shift.csv",index=False,encoding="utf-8-sig")
    comparisons.to_csv(TABLE_ROOT/"publication_year_validation_metrics.csv",index=False,encoding="utf-8-sig")

    split_idx=split_results.summary.set_index("strategy")
    cluster_overlap = (
        split_results.overlap.loc[
            split_results.overlap["strategy"].eq("Cluster-stratified")
        ]
        .mean(numeric_only=True)
    )
    time_r2=temporal_metrics["r2"]
    text=f"""# Figure captions

## Fig. 1 | Dataset provenance, hierarchy, group-size imbalance, and missingness

The analysis view contains {len(df):,} PFAS-level records from {df.ref_id.nunique()} studies, {df.mem.nunique()} membranes, and {df.PFAS.nunique()} PFAS. Strict experimental-condition grouping yields {strict_condition_key(df).nunique():,} conditions. Panel (a) quantifies records retained at each screening stage and shows the unresolved 71-row difference from the published 2353-row analysis. Panel (b) ranks studies by their number of strict experimental conditions; panels (c-f) show PFAS multiplicity within otherwise identical conditions, the strongly imbalanced number of records contributed by each study, membrane, and PFAS, key coverage counts, and predictor missingness.

## Fig. 2 | Familiar-row accuracy and unseen-study transfer for the same model

Panel (a) quantifies the study and row allocation fixed before configuration selection: 48 development studies and 13 untouched test studies, with development rows divided into fitting/tuning and familiar-study test sets. Panels (b) and (c) show observed and predicted rejection for exact rows in the familiar-study and untouched-study tests; dashed lines denote identity. Panel (d) pairs logit-scale R², rejection-scale R², and C-index for the same fitted model, with Δ defined as the unseen-study minus familiar-row value. Panel (e) gives empirical cumulative distributions of exact-row absolute errors; annotations report RMSE and MAE in percentage points. Panel (f) reports the percentage of test rows sharing a study, membrane, or PFAS with the fitting set. The 13 unseen studies were excluded before configuration selection and fitting, and the same fitted censor-aware AFT model generated both sets of predictions.

## Fig. 3 | Censoring treatment and study heterogeneity

Three matched models compare treating censored lower bounds as exact observations, excluding censored observations, and fitting an AFT likelihood. Performance is evaluated with study-blocked folds. Residual study clustering is summarized by a descriptive residual ICC of approximately {residual_icc:.2f}; this is evidence of study heterogeneity, not proof that a random effect improves prediction for a new study. In Fig. 3d, residuals are observed minus 10-fold study-blocked out-of-fold AFT predictions, restricted to uncensored rows, averaged within study, converted to rejection percentage points, and ranked by absolute mean residual; the 14 largest study-level biases are shown. Negative values denote overprediction and positive values underprediction. This is a descriptive failure-map diagnostic, not external validation.

## Fig. 4 | Group-calibrated uncertainty under study shift

Prediction intervals use study-bootstrap ensemble dispersion as a heteroscedastic scale and a separate study-group calibration set. Coverage is evaluated on exact observations in held-out study folds; censored rows are assessed only for compatibility with their observed lower bound. These intervals are an empirical grouped-calibration diagnostic, not a distribution-free guarantee under arbitrary study shift.

## Fig. 5 | Applicability domain and reject option

Applicability is measured with fold-specific k-nearest-neighbour distance after training-fold median imputation and standardization. The distance-to-error panel uses a symmetric-log x-axis so that the main data cloud and rare extreme distances remain visible together. Panels connect domain distance to error, show the accuracy-coverage trade-off when distant predictions are rejected, compare all exact observations with prespecified transfer-relevant categories, summarize error distributions at observed missingness levels, and quantify domain-exclusion rates by water matrix. Linear Williams leverage is not used as the primary tree-model domain measure.

## Fig. 6 | Failure map and split-stable model associations

The failure heatmap reports study-blocked absolute error across PFAS chain and water-matrix strata; an em dash marks a stratum with no available observations. XGBoost contribution values are recomputed across five study-blocked folds. Mean absolute contribution, rank stability, and direction stability (Spearman ρ between each feature and its SHAP contribution) are shown together. ALE curves are centered model-response associations with between-fold bands; they are not causal effects or mechanistic proof.

## Fig. 7 | Restricted within-study associations and scope limits

The exposure-support audit retains monovalent cations for the primary adjusted analysis (203 independent conditions across 34 reporting studies, including 22 studies with within-study exposure variation; between-study variance share 45%). DOC and divalent cations remain exploratory, and trivalent cations are excluded because only 24 independent conditions from 6 studies are available and the between-study variance share rounds to 100%. Panel (a) compares reporting studies with the subset showing within-study exposure variation. Panel (b) diagnoses common support for non-zero versus zero monovalent-cation exposure; the primary exposure model remains continuous. Primary estimates use unpenalised study-fixed-effect OLS with cluster-robust and wild-cluster inference, while panel (c) retains ridge penalty-8 estimates as a sensitivity analysis. Panel (d) reports approximate continuous-outcome E-values for the point estimates and confidence-limit values. Panel (e) reports placebo-treatment, random-common-cause, and 80% study-subset refutations for the penalised sensitivity model. The supported claim is limited to adjusted within-study associations under the stated assumptions.

## Fig. 8 | Publication-year transfer test

Publication year is read directly from the source field `ref_year` in workbook sheet `PFAS_literature_dataset_2b`; Fig. 8a counts unique studies (`ref_id`), not PFAS-level rows, within each year. The model is trained on studies published through {cutoff} and evaluated on {cutoff+1}–{int(df.ref_year.max())} studies. The time-out R² is {time_r2:.2f} and RMSE is {temporal_metrics['rmse']*100:.1f} rejection percentage points. A dashed guide at standardized mean shift = 0.5 flags a practically substantial future-versus-earlier shift for visual screening. Feature-shift bars and the chain-length × water-matrix support matrix indicate where a future external experiment should be prioritized. This is a time-out literature test, not an independent laboratory validation.

## Global claim boundary

The generated figures support a reliability-audit paper: apparent performance depends on evaluation design; uncertainty and applicability limits can be quantified; and some learned associations are stable across study-blocked folds. Universal transfer, experimental mechanism confirmation, and causal intervention remain outside the supported claim.
"""
    (FIG_ROOT/"Figure_captions.md").write_text(text,encoding="utf-8")


def create_contact_sheet():
    active = [
        ("Fig01_Data_provenance_and_hierarchy", "Fig01_Data_provenance_and_hierarchy.png"),
        ("Fig02_Disjoint_familiar_vs_unseen_study_transfer", "Fig02_Disjoint_familiar_vs_unseen_study_transfer.png"),
        ("Fig03_Censoring_and_study_heterogeneity", "Fig03_Censoring_and_study_heterogeneity.png"),
        ("Fig04_Group_calibrated_uncertainty", "Fig04_Group_calibrated_uncertainty.png"),
        ("Fig05_Applicability_domain_and_reject_option", "Fig05_Applicability_domain_and_reject_option.png"),
        ("Fig06_Failure_map_and_stable_associations", "Fig06_Failure_map_and_stable_associations.png"),
        ("Fig07_Within_study_adjusted_associations", "Fig07_Within_study_adjusted_associations.png"),
        ("Fig08_Publication_year_transfer_test", "Fig08_Publication_year_transfer_test.png"),
    ]
    paths=[FIG_ROOT/folder/name for folder,name in active if (FIG_ROOT/folder/name).exists()]
    if not paths:return
    qa_root=TMP_ROOT/"figure_qa"
    qa_root.mkdir(parents=True,exist_ok=True)
    ncols=2
    nrows=int(np.ceil(len(paths)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(16,4.8*nrows),squeeze=False)
    for ax,path in zip(axes.flat,paths):
        ax.imshow(plt.imread(path));ax.axis("off");ax.set_title(path.parent.name,fontsize=10,fontweight="normal")
    for ax in axes.flat[len(paths):]:ax.axis("off")
    TMP_ROOT.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(qa_root/"combined_figures.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)

    sub_sheet_root=qa_root/"subfigure_contact_sheets"
    sub_sheet_root.mkdir(parents=True,exist_ok=True)
    for folder_name,_ in active:
        folder=FIG_ROOT/folder_name
        sub_paths=sorted((folder/"subfigures").glob("*.png"))
        if not sub_paths:
            continue
        ncols=3 if len(sub_paths)>4 else 2
        nrows=int(np.ceil(len(sub_paths)/ncols))
        sub_fig,sub_axes=plt.subplots(nrows,ncols,figsize=(5.2*ncols,4.4*nrows),squeeze=False)
        for sub_ax,sub_path in zip(sub_axes.flat,sub_paths):
            sub_ax.imshow(plt.imread(sub_path))
            sub_ax.axis("off")
            sub_ax.set_title(sub_path.stem,fontsize=10,fontweight="normal")
        for sub_ax in sub_axes.flat[len(sub_paths):]:
            sub_ax.axis("off")
        sub_fig.tight_layout()
        sub_fig.savefig(
            sub_sheet_root/f"{folder.name}_subfigures.png",
            dpi=170,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(sub_fig)


def main():
    raw,df=load_data()
    X=feature_matrix(df)
    print(f"Analysis rows={len(df)}, studies={df.ref_id.nunique()}, membranes={df.mem.nunique()}, PFAS={df.PFAS.nunique()}")
    split_results=run_split_audit(df,X)
    print(split_results.summary.to_string(index=False))
    ladder_summary,ladder_folds,ladder_preds,study_means,residual_icc=model_ladder(df,X)
    interval,conformal_folds=grouped_conformal(df,X)
    shap_summary,shap_fold,sign_pivot,ale_curves=shap_stability_and_ale(df,X)
    (
        assoc,
        boot_store,
        placebo_store,
        study_slopes,
        overlap_df,
        overlap_summary,
        refutation_summary,
        support_audit,
    )=within_study_associations(df)
    temporal=temporal_validation(df,X,split_results)
    draw_figures(raw,df,split_results,ladder_summary,ladder_folds,ladder_preds,study_means,residual_icc,interval,conformal_folds,shap_summary,shap_fold,sign_pivot,ale_curves,assoc,boot_store,placebo_store,study_slopes,overlap_df,overlap_summary,refutation_summary,support_audit,temporal)
    write_captions_and_tables(raw,df,split_results,ladder_summary,ladder_folds,study_means,residual_icc,conformal_folds,interval,shap_summary,shap_fold,sign_pivot,assoc,overlap_df,overlap_summary,refutation_summary,support_audit,temporal)
    create_contact_sheet()
    print("Generated 8 figures, independent subfigures, tables, captions, and QA contact sheet.")


if __name__=="__main__":
    main()
