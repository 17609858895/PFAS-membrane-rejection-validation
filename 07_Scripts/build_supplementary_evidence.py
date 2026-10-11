from __future__ import annotations

import json
import math
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "07_Scripts"
STYLE_ROOT = SCRIPT_ROOT / "figure_style"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(STYLE_ROOT))

import generate_pfas_paper_figures as core  # noqa: E402
from ml_publication_style import (  # noqa: E402
    configure_publication_style,
    numeric_ticks,
    style_axis,
)
from scientific_palettes import STYLE_COLORS, get_palette  # noqa: E402


SOURCE_XLSX = ROOT / "05_Source_Data" / "es6c02287_si_002.xlsx"
TABLE_ROOT = ROOT / "04_Tables"
SUPP_TABLE_ROOT = TABLE_ROOT / "Supplementary"
FIG_ROOT = ROOT / "03_Figures" / "Supplementary"
CACHE_PATH = ROOT / "99_Working_Files" / "crossref_study_doi_cache.json"

CORAL, BLUE, GREEN, TEAL, PURPLE, PINK, GOLD, LAVENDER, GRAY = get_palette(
    "nature_new", n=9
)
INK = STYLE_COLORS["ink"]
MUTED = STYLE_COLORS["muted"]
LIGHT = STYLE_COLORS["light"]


PREDICTOR_META = {
    "pwp_LMH/bar": ("Pure-water permeance", "L m−2 h−1 bar−1", "None"),
    "ON": ("Membrane surface oxygen:nitrogen ratio", "dimensionless", "None"),
    "pH": ("Feed pH", "pH units", "None"),
    "volume_Å3": ("PFAS van der Waals volume", "Å3", "None"),
    "dipole_debye": ("PFAS dipole moment", "D", "None"),
    "pKa": ("PFAS acid dissociation constant", "dimensionless", "None"),
    "temp_ºC": ("Operating temperature", "°C", "None"),
    "flux_LMH": ("Permeate flux", "L m−2 h−1", "log1p(max(x, 0))"),
    "pfas_feed_conc_µg/L": ("Initial concentration of the focal PFAS", "µg L−1", "log1p(max(x, 0))"),
    "mon_cat_mM": ("Total monovalent-cation concentration", "mM", "log1p(max(x, 0))"),
    "di_cat_mM": ("Total divalent-cation concentration", "mM", "log1p(max(x, 0))"),
    "tri_cat_mM": ("Total trivalent-cation concentration", "mM", "log1p(max(x, 0))"),
    "doc_mg/L": ("Dissolved organic carbon", "mg L−1", "log1p(max(x, 0))"),
}


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    SUPP_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SUPP_TABLE_ROOT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="PFAS_literature_dataset_2b")
    _, analysis = core.load_data()
    return raw, analysis


def evidence_flow(raw: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    threshold = raw.loc[pd.to_numeric(raw["Rejection"], errors="coerce") >= 0.5].copy()
    nfro = threshold.loc[threshold["mem_class"].astype(str).str.upper().isin(["NF", "RO"])].copy()
    flow = pd.DataFrame(
        [
            {
                "step": 0,
                "stage": "Public source workbook",
                "rule": "All machine-readable rows in es6c02287_si_002.xlsx",
                "rows": len(raw),
                "studies": raw["ref_id"].nunique(dropna=True),
                "rows_removed_at_step": 0,
                "studies_removed_at_step": 0,
            },
            {
                "step": 1,
                "stage": "Medium-to-high rejection",
                "rule": "Retain Rejection ≥ 0.50",
                "rows": len(threshold),
                "studies": threshold["ref_id"].nunique(dropna=True),
                "rows_removed_at_step": len(raw) - len(threshold),
                "studies_removed_at_step": raw["ref_id"].nunique(dropna=True)
                - threshold["ref_id"].nunique(dropna=True),
            },
            {
                "step": 2,
                "stage": "NF/RO analysis set",
                "rule": "Retain membrane class NF or RO",
                "rows": len(nfro),
                "studies": nfro["ref_id"].nunique(dropna=True),
                "rows_removed_at_step": len(threshold) - len(nfro),
                "studies_removed_at_step": threshold["ref_id"].nunique(dropna=True)
                - nfro["ref_id"].nunique(dropna=True),
            },
            {
                "step": 3,
                "stage": "Published analysis reported in reference [9]",
                "rule": "Published count; the public release does not archive the exact analysis snapshot or row-level exclusion IDs",
                "rows": 2353,
                "studies": 60,
                "rows_removed_at_step": len(nfro) - 2353,
                "studies_removed_at_step": nfro["ref_id"].nunique(dropna=True) - 60,
            },
        ]
    )
    flow["status"] = [
        "reproduced",
        "reproduced",
        "reproduced",
        "not reproducible from the public files; source-version boundary",
    ]
    assert len(analysis) == len(nfro) == 2424
    return flow


def fold_metrics() -> pd.DataFrame:
    base = pd.read_csv(TABLE_ROOT / "split_strategy_fold_metrics.csv")
    year = pd.read_csv(TABLE_ROOT / "publication_year_validation_metrics.csv")
    year_row = year.loc[year["strategy"].eq("Publication-year holdout")].iloc[0]
    row = {
        "strategy": "Publication-year holdout",
        "fold": "single holdout",
        "r2": float(year_row["r2"]),
        "logit_r2": float(year_row["logit_r2"]),
        "rmse": float(year_row["rmse"]),
        "mae": float(year_row["mae"]),
        "c_index": float(year_row["c_index"]),
        "bound_violation": float(year_row.get("bound_violation", np.nan)),
        "n": int(year_row["n"]),
        "n_exact": int(year_row["n_exact"]),
        "train_r2": np.nan,
        "train_test_gap": np.nan,
    }
    return pd.concat([base, pd.DataFrame([row])], ignore_index=True)


def predictor_table(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order, feature in enumerate(core.FEATURES, 1):
        label, unit, transform = PREDICTOR_META[feature]
        missing = int(analysis[feature].isna().sum())
        rows.append(
            {
                "predictor_no": order,
                "source_column": feature,
                "definition": label,
                "unit": unit,
                "missing_n": missing,
                "missing_percent": missing / len(analysis) * 100,
                "model_transform": transform,
                "model_missing_value_handling": "Native missing-value routing in XGBoost; no global imputation",
                "diagnostic_only_preprocessing": "Fold-specific median imputation and standardisation for distance/linear diagnostics only",
            }
        )
    return pd.DataFrame(rows)


def normalize_title(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", value).strip()


def load_crossref_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_crossref_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_doi(title: str, year: int | float, cache: dict[str, dict]) -> dict:
    key = normalize_title(title)
    if key in cache:
        return cache[key]
    url = "https://api.crossref.org/works?query.title=" + quote(str(title)) + "&rows=5&select=DOI,title,published,issued"
    req = Request(url, headers={"User-Agent": "PFAS-SI-audit/1.0 (mailto:research@example.org)"})
    result = {"doi": "", "matched_title": "", "title_similarity": np.nan, "doi_status": "unresolved"}
    try:
        with urlopen(req, timeout=25) as response:
            items = json.loads(response.read().decode("utf-8"))["message"]["items"]
        ranked = []
        for item in items:
            candidate = (item.get("title") or [""])[0]
            similarity = SequenceMatcher(None, key, normalize_title(candidate)).ratio()
            date_parts = item.get("published", item.get("issued", {})).get("date-parts", [[None]])
            candidate_year = date_parts[0][0] if date_parts and date_parts[0] else None
            year_ok = candidate_year is None or pd.isna(year) or abs(int(candidate_year) - int(year)) <= 2
            ranked.append((similarity + (0.03 if year_ok else -0.08), similarity, candidate, item.get("DOI", ""), candidate_year))
        if ranked:
            _, similarity, candidate, doi, candidate_year = max(ranked, key=lambda x: x[0])
            result.update(
                {
                    "doi": doi if similarity >= 0.88 else "",
                    "matched_title": candidate,
                    "title_similarity": similarity,
                    "matched_year": candidate_year,
                    "doi_status": "accepted title match" if similarity >= 0.88 else "manual verification required",
                }
            )
    except Exception as exc:  # network errors remain explicit in the output table
        result["doi_status"] = f"Crossref lookup failed: {type(exc).__name__}"
    cache[key] = result
    save_crossref_cache(cache)
    time.sleep(0.08)
    return result


def study_inventory(analysis: pd.DataFrame) -> pd.DataFrame:
    cache = load_crossref_cache()
    rows = []
    for ref_id, part in analysis.groupby("ref_id", sort=True, dropna=False):
        title = str(part["ref_title"].dropna().iloc[0]) if part["ref_title"].notna().any() else ""
        year = int(part["ref_year"].dropna().iloc[0]) if part["ref_year"].notna().any() else np.nan
        doi = resolve_doi(title, year, cache)
        water = "; ".join(sorted(part["feed_type"].dropna().astype(str).unique()))
        rows.append(
            {
                "study_id": ref_id,
                "publication_year": year,
                "title": title,
                "doi": doi.get("doi", ""),
                "doi_match_status": doi.get("doi_status", ""),
                "crossref_title_similarity": doi.get("title_similarity", np.nan),
                "rows_in_analysis": len(part),
                "unique_membranes": part["mem"].nunique(dropna=True),
                "unique_pfas": part["PFAS"].nunique(dropna=True),
                "contains_censored_rows": bool(part["is_censored"].any()),
                "censored_rows": int(part["is_censored"].sum()),
                "water_matrices": water,
            }
        )
    frame = pd.DataFrame(rows)
    assert len(frame) == 61
    return frame


def duplicate_distribution(analysis: pd.DataFrame, x_full: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    scaled = StandardScaler().fit_transform(imputer.fit_transform(x_full))
    clusters = core.KMeans(n_clusters=10, random_state=42, n_init=20).fit_predict(scaled)
    y = analysis["Rejection"].to_numpy(float)
    censored = analysis["is_censored"].to_numpy(bool)
    summaries, details = [], []
    for strategy, stratify in [("Cluster-stratified", clusters), ("Row-random", None)]:
        for rep in range(12):
            train, test = train_test_split(
                np.arange(len(analysis)), test_size=0.15, random_state=110 + rep, stratify=stratify
            )
            split_imputer = SimpleImputer(strategy="median")
            x_train = split_imputer.fit_transform(x_full[train])
            x_test = split_imputer.transform(x_full[test])
            train_map: dict[tuple[float, ...], list[int]] = {}
            for local, values in enumerate(x_train):
                train_map.setdefault(tuple(np.round(values, 12)), []).append(train[local])
            n_duplicate = n_same = n_same_censor = 0
            for local, values in enumerate(x_test):
                matches = train_map.get(tuple(np.round(values, 12)), [])
                if not matches:
                    continue
                idx = test[local]
                diffs = np.abs(y[matches] - y[idx])
                exact = np.isclose(diffs, 0, rtol=0, atol=1e-12)
                same_censor = exact & (censored[matches] == censored[idx])
                n_duplicate += 1
                n_same += int(exact.any())
                n_same_censor += int(same_censor.any())
                details.append(
                    {
                        "strategy": strategy,
                        "repeat": rep + 1,
                        "test_row_index": int(idx),
                        "training_matches": len(matches),
                        "minimum_absolute_rejection_difference_pp": float(diffs.min() * 100),
                        "any_identical_response": bool(exact.any()),
                        "any_identical_response_and_censoring": bool(same_censor.any()),
                    }
                )
            summaries.append(
                {
                    "strategy": strategy,
                    "repeat": rep + 1,
                    "test_rows": len(test),
                    "duplicate_feature_rows": n_duplicate,
                    "duplicate_feature_fraction": n_duplicate / len(test),
                    "identical_response_fraction_among_duplicates": n_same / n_duplicate if n_duplicate else np.nan,
                    "identical_response_and_censoring_fraction_among_duplicates": n_same_censor / n_duplicate if n_duplicate else np.nan,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(details)


def bootstrap_sensitivity(analysis: pd.DataFrame, x_full: np.ndarray) -> pd.DataFrame:
    y = analysis["Rejection"].to_numpy(float)
    censored = analysis["is_censored"].to_numpy(bool)
    groups = analysis["ref_id"].astype(str).to_numpy()
    rng = np.random.default_rng(20260717)
    rows = []
    model_counts = [8, 16, 32, 50]
    outer = GroupKFold(n_splits=10)
    for fold, (train_outer, test_idx) in enumerate(outer.split(x_full, y, groups), 1):
        train_groups = np.unique(groups[train_outer])
        rng.shuffle(train_groups)
        n_cal = max(4, int(np.ceil(len(train_groups) * 0.22)))
        cal_groups = set(train_groups[:n_cal])
        cal_idx = train_outer[np.array([g in cal_groups for g in groups[train_outer]])]
        fit_idx = train_outer[np.array([g not in cal_groups for g in groups[train_outer]])]
        unique_fit_groups = np.unique(groups[fit_idx])
        cal_predictions, test_predictions = [], []
        for b in range(max(model_counts)):
            sampled = rng.choice(unique_fit_groups, size=len(unique_fit_groups), replace=True)
            boot_idx = np.concatenate([fit_idx[groups[fit_idx] == g] for g in sampled])
            model = core.fit_aft(
                x_full[boot_idx],
                analysis.loc[boot_idx, "odds_R"].to_numpy(),
                analysis.loc[boot_idx, "is_censored"].to_numpy(),
                seed=9000 + fold * 100 + b,
                rounds=105,
            )
            cal_predictions.append(core.predict_aft(model, x_full[cal_idx]))
            test_predictions.append(core.predict_aft(model, x_full[test_idx]))
        cal_predictions = np.asarray(cal_predictions)
        test_predictions = np.asarray(test_predictions)
        exact_cal = ~censored[cal_idx]
        exact_test = ~censored[test_idx]
        for count in model_counts:
            cal = cal_predictions[:count]
            test = test_predictions[:count]
            cal_mean = cal.mean(axis=0)
            test_mean = test.mean(axis=0)
            cal_scale = np.maximum(cal.std(axis=0, ddof=1), 0.008)
            test_scale = np.maximum(test.std(axis=0, ddof=1), 0.008)
            scores = np.abs(y[cal_idx][exact_cal] - cal_mean[exact_cal]) / cal_scale[exact_cal]
            rmse = float(np.sqrt(np.mean((y[test_idx][exact_test] - test_mean[exact_test]) ** 2)))
            mae = float(np.mean(np.abs(y[test_idx][exact_test] - test_mean[exact_test])))
            for nominal in (0.80, 0.90, 0.95):
                q = np.quantile(scores, min(1, math.ceil((len(scores) + 1) * nominal) / len(scores)))
                lo = np.clip(test_mean - q * test_scale, 0.5, 1)
                hi = np.clip(test_mean + q * test_scale, 0.5, 1)
                coverage = np.mean((y[test_idx][exact_test] >= lo[exact_test]) & (y[test_idx][exact_test] <= hi[exact_test]))
                rows.append(
                    {
                        "fold": fold,
                        "bootstrap_models": count,
                        "nominal_coverage": nominal,
                        "coverage_exact": coverage,
                        "mean_interval_width_pp": float(np.mean((hi - lo) * 100)),
                        "rmse_pp": rmse * 100,
                        "mae_pp": mae * 100,
                        "calibration_studies": len(cal_groups),
                        "test_studies": len(np.unique(groups[test_idx])),
                        "calibration_exact_rows": int(exact_cal.sum()),
                        "test_exact_rows": int(exact_test.sum()),
                    }
                )
    return pd.DataFrame(rows)


def draw_duplicate_figure(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    configure_publication_style()
    order = ["Cluster-stratified", "Row-random"]
    colors = [BLUE, CORAL]

    def f_a(ax):
        for i, (name, color) in enumerate(zip(order, colors)):
            vals = summary.loc[summary.strategy == name, "duplicate_feature_fraction"].to_numpy() * 100
            jitter = np.linspace(-0.09, 0.09, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=42, color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
            ax.plot([i - 0.17, i + 0.17], [vals.mean(), vals.mean()], color=INK, lw=2.0)
        ax.set_xticks(range(2), ["Cluster\nstratified", "Row random"])
        style_axis(ax, ylabel="Duplicate test rows (%)", tick_size=11.5, label_size=14.0)
        ax.set_ylim(0, max(22, summary.duplicate_feature_fraction.max() * 115))
        numeric_ticks(ax, x=False, n=5)

    def f_b(ax):
        for name, color in zip(order, colors):
            vals = np.sort(detail.loc[detail.strategy == name, "minimum_absolute_rejection_difference_pp"].to_numpy())
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", lw=2.2, color=color, label=name)
        style_axis(ax, xlabel="Minimum response difference (pp)", ylabel="ECDF", tick_size=11.5, label_size=14.0)
        numeric_ticks(ax, n=5)
        ax.legend(loc="lower right", fontsize=10.0)

    def f_c(ax):
        x = np.arange(2)
        means = [summary.loc[summary.strategy == name, "identical_response_and_censoring_fraction_among_duplicates"].mean() * 100 for name in order]
        sds = [summary.loc[summary.strategy == name, "identical_response_and_censoring_fraction_among_duplicates"].std() * 100 for name in order]
        ax.bar(x, means, yerr=sds, color=colors, width=0.62, capsize=4, edgecolor=INK, linewidth=0.8)
        ax.set_xticks(x, ["Cluster\nstratified", "Row random"])
        style_axis(ax, ylabel="Exact response and censor match (%)", tick_size=11.5, label_size=14.0)
        ax.set_ylim(0, max(16, max(np.asarray(means) + np.asarray(sds)) * 1.2))
        numeric_ticks(ax, x=False, n=5)

    panels = [f_a, f_b, f_c]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.42, bottom=0.18)
    core.save_publication_figure(
        fig,
        FIG_ROOT / "FigS01_Duplicate_Audit",
        "FigS01_duplicate_feature_response_audit",
        panels,
        [(5.8, 4.8)] * 3,
        source_script=Path(__file__),
        combined_axes=axes,
    )


def draw_clipping_figure(frame: pd.DataFrame) -> None:
    configure_publication_style()
    labels = frame["upper_clip"].astype(str).tolist()
    x = np.arange(len(frame))

    def f_a(ax):
        ax.plot(x, frame["rows_changed"], color=BLUE, marker="o", lw=2.2, ms=7)
        ax.set_xticks(x, labels)
        style_axis(ax, xlabel="Upper clipping constant", ylabel="Rows changed", tick_size=11.5, label_size=14.0)
        ax.set_ylim(-0.05, 1.0)
        ax.text(0.5, 0.84, "No target values changed", transform=ax.transAxes, ha="center", color=MUTED, fontsize=10.5)
        numeric_ticks(ax, x=False, n=4)

    def f_b(ax):
        values = frame["maximum_odds_after_clip"].to_numpy(float)
        ax.plot(x, values, color=CORAL, marker="o", lw=2.2, ms=7)
        ax.set_xticks(x, labels)
        style_axis(ax, xlabel="Upper clipping constant", ylabel="Maximum rejection odds", tick_size=11.5, label_size=14.0)
        ax.set_ylim(0, max(values) * 1.12)
        ax.text(0.5, 0.84, f"Maximum odds = {values[0]:,.0f}", transform=ax.transAxes, ha="center", color=MUTED, fontsize=10.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        numeric_ticks(ax, x=False, n=5)

    panels = [f_a, f_b]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.40, bottom=0.19)
    core.save_publication_figure(
        fig,
        FIG_ROOT / "FigS02_Clipping_Sensitivity",
        "FigS02_clipping_constant_sensitivity",
        panels,
        [(5.8, 4.8)] * 2,
        source_script=Path(__file__),
        combined_axes=axes,
    )


def draw_temporal_figure(frame: pd.DataFrame) -> None:
    configure_publication_style()
    specs = [("r2", "temporal_r2", "R²", False), ("rmse", "temporal_rmse", "RMSE (pp)", True)]

    def panel(metric, actual_col, xlabel, to_pp):
        def draw(ax):
            vals = frame[metric].to_numpy() * (100 if to_pp else 1)
            actual = float(frame[actual_col].iloc[0]) * (100 if to_pp else 1)
            ax.hist(vals, bins=12, color=LIGHT, edgecolor=BLUE, linewidth=1.2)
            ax.axvline(actual, color=CORAL, lw=2.5)
            style_axis(ax, xlabel=f"Size-matched control {xlabel}", ylabel="Control splits", tick_size=11.5, label_size=14.0)
            ax.plot([], [], color=CORAL, lw=2.5, label="Actual temporal holdout")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=10.0)
            numeric_ticks(ax, n=5)
        return draw

    panels = [panel(*spec) for spec in specs]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.38, bottom=0.18)
    core.save_publication_figure(
        fig,
        FIG_ROOT / "FigS03_Temporal_Control",
        "FigS03_size_matched_temporal_control",
        panels,
        [(5.8, 4.8)] * 2,
        source_script=Path(__file__),
        combined_axes=axes,
    )


def draw_ridge_figure(ridge: pd.DataFrame, ols: pd.DataFrame) -> None:
    configure_publication_style()

    def f_a(ax):
        labels = ols["label"].astype(str).tolist()
        y = np.arange(len(ols))
        effect = ols["effect_pp_at_median"].to_numpy()
        low = ols["wild_cluster_ci_low_pp"].to_numpy()
        high = ols["wild_cluster_ci_high_pp"].to_numpy()
        ax.errorbar(effect, y, xerr=[effect - low, high - effect], fmt="o", color=BLUE, ecolor=BLUE, capsize=4, ms=7)
        ax.axvline(0, color=MUTED, lw=1.2, ls="--")
        ax.set_yticks(y, labels)
        style_axis(ax, xlabel="Adjusted association (pp / within-study SD)", tick_size=11.5, label_size=14.0)
        numeric_ticks(ax, y=False, n=5)

    def f_b(ax):
        for exposure, color in zip(ridge["treatment"].unique(), [BLUE, CORAL, TEAL, GOLD]):
            part = ridge.loc[ridge.treatment == exposure]
            ax.plot(part["penalty"], part["effect_pp_at_median"], marker="o", lw=2, color=color, label=part["label"].iloc[0])
        ax.axhline(0, color=MUTED, lw=1.1, ls="--")
        style_axis(ax, xlabel="Ridge penalty α", ylabel="Adjusted association (pp)", tick_size=11.5, label_size=14.0)
        ax.set_xticks(sorted(ridge["penalty"].unique()))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=9.5,
            frameon=False,
            borderaxespad=0.0,
        )
        numeric_ticks(ax, x=False, n=5)

    panels = [f_a, f_b]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.42, left=0.13, bottom=0.18)
    core.save_publication_figure(
        fig,
        FIG_ROOT / "FigS04_Ridge_OLS",
        "FigS04_ridge_and_cluster_robust_sensitivity",
        panels,
        [(6.2, 4.8)] * 2,
        source_script=Path(__file__),
        combined_axes=axes,
    )


def draw_bootstrap_figure(frame: pd.DataFrame) -> None:
    configure_publication_style()
    grouped = frame.groupby(["bootstrap_models", "nominal_coverage"], as_index=False).agg(
        mean_width=("mean_interval_width_pp", "mean"),
        sd_width=("mean_interval_width_pp", "std"),
        coverage=("coverage_exact", "mean"),
        sd_coverage=("coverage_exact", "std"),
        rmse=("rmse_pp", "mean"),
        sd_rmse=("rmse_pp", "std"),
    )
    colors = (TEAL, BLUE, PURPLE)

    def f_a(ax):
        for nominal, color in zip((0.80, 0.90, 0.95), colors):
            part = grouped.loc[grouped.nominal_coverage == nominal]
            ax.errorbar(part.bootstrap_models, part.mean_width, yerr=part.sd_width, marker="o", color=color, lw=2, capsize=3, label=f"{int(nominal*100)}%")
        style_axis(ax, xlabel="Bootstrap models per fold", ylabel="Mean interval width (pp)", tick_size=11.5, label_size=14.0)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=9.5,
            title="Nominal",
            frameon=False,
            borderaxespad=0.0,
        )
        ax.set_xticks([8, 16, 32, 50])

    def f_b(ax):
        for nominal, color in zip((0.80, 0.90, 0.95), colors):
            part = grouped.loc[grouped.nominal_coverage == nominal]
            ax.errorbar(part.bootstrap_models, part.coverage * 100, yerr=part.sd_coverage * 100, marker="o", color=color, lw=2, capsize=3)
            ax.axhline(nominal * 100, color=color, lw=0.9, ls="--", alpha=0.55)
        style_axis(ax, xlabel="Bootstrap models per fold", ylabel="Exact-row coverage (%)", tick_size=11.5, label_size=14.0)
        ax.set_xticks([8, 16, 32, 50])

    def f_c(ax):
        point = grouped.loc[grouped.nominal_coverage == 0.90]
        ax.errorbar(point.bootstrap_models, point.rmse, yerr=point.sd_rmse, marker="o", color=CORAL, lw=2.2, capsize=3)
        style_axis(ax, xlabel="Bootstrap models per fold", ylabel="Point RMSE (pp)", tick_size=11.5, label_size=14.0)
        ax.set_xticks([8, 16, 32, 50])

    panels = [f_a, f_b, f_c]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    for drawer, ax in zip(panels, axes):
        drawer(ax)
    fig.subplots_adjust(wspace=0.42, bottom=0.18)
    core.save_publication_figure(
        fig,
        FIG_ROOT / "FigS05_Bootstrap_Sensitivity",
        "FigS05_bootstrap_model_count_sensitivity",
        panels,
        [(5.8, 4.8)] * 3,
        source_script=Path(__file__),
        combined_axes=axes,
    )


def create_supplementary_contact_sheet() -> Path:
    """Assemble a QA-only overview without changing submission figures."""
    figure_paths = [
        FIG_ROOT / "FigS01_Duplicate_Audit" / "FigS01_duplicate_feature_response_audit.png",
        FIG_ROOT / "FigS02_Clipping_Sensitivity" / "FigS02_clipping_constant_sensitivity.png",
        FIG_ROOT / "FigS03_Temporal_Control" / "FigS03_size_matched_temporal_control.png",
        FIG_ROOT / "FigS04_Ridge_OLS" / "FigS04_ridge_and_cluster_robust_sensitivity.png",
        FIG_ROOT / "FigS05_Bootstrap_Sensitivity" / "FigS05_bootstrap_model_count_sensitivity.png",
    ]
    qa_root = ROOT / "99_Working_Files" / "figure_qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    out_path = qa_root / "supplementary_figures.png"

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for ax, path in zip(axes.ravel(), figure_paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.parent.name, fontsize=12, fontweight="normal", pad=8)
        ax.axis("off")
    for ax in axes.ravel()[len(figure_paths) :]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.08, hspace=0.16)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def figures_only() -> None:
    """Redraw SI figures from the persisted numerical tables only."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    duplicate_summary = pd.read_csv(SUPP_TABLE_ROOT / "Fig_S1_duplicate_audit_summary.csv")
    duplicate_detail = pd.read_csv(SUPP_TABLE_ROOT / "Fig_S1_duplicate_audit_row_values.csv")
    clipping = pd.read_csv(SUPP_TABLE_ROOT / "Fig_S2_clipping_sensitivity_values.csv")
    temporal = pd.read_csv(SUPP_TABLE_ROOT / "Fig_S3_size_matched_temporal_values.csv")
    ridge = pd.read_csv(SUPP_TABLE_ROOT / "Table_S5_ridge_sensitivity.csv")
    ols = pd.read_csv(SUPP_TABLE_ROOT / "Table_S6_cluster_robust_ols.csv")
    bootstrap = pd.read_csv(SUPP_TABLE_ROOT / "Fig_S5_bootstrap_model_count_sensitivity.csv")

    draw_duplicate_figure(duplicate_summary, duplicate_detail)
    draw_clipping_figure(clipping)
    draw_temporal_figure(temporal)
    draw_ridge_figure(ridge, ols)
    draw_bootstrap_figure(bootstrap)
    qa_path = create_supplementary_contact_sheet()
    print(f"Redrew five supplementary figures from saved numerical tables: {FIG_ROOT}")
    print(f"Supplementary QA contact sheet: {qa_path}")


def main() -> None:
    SUPP_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    raw, analysis = load_sets()
    x_full = core.feature_matrix(analysis)

    save_csv(evidence_flow(raw, analysis), "Table_S0_evidence_flow.csv")
    save_csv(fold_metrics(), "Table_S1_fold_level_metrics.csv")
    save_csv(predictor_table(analysis), "Table_S2_predictor_dictionary.csv")
    save_csv(study_inventory(analysis), "Table_S3_study_inventory.csv")
    baselines = pd.read_csv(TABLE_ROOT / "reviewer_m3_study_blocked_baselines.csv")
    save_csv(baselines, "Table_S4_baseline_models.csv")

    duplicate_summary, duplicate_detail = duplicate_distribution(analysis, x_full)
    save_csv(duplicate_summary, "Fig_S1_duplicate_audit_summary.csv")
    save_csv(duplicate_detail, "Fig_S1_duplicate_audit_row_values.csv")
    clipping = pd.read_csv(TABLE_ROOT / "reviewer_m5_clipping_sensitivity.csv")
    save_csv(clipping, "Fig_S2_clipping_sensitivity_values.csv")
    temporal = pd.read_csv(TABLE_ROOT / "reviewer_m4_size_matched_temporal_control.csv")
    save_csv(temporal, "Fig_S3_size_matched_temporal_values.csv")
    ridge = pd.read_csv(TABLE_ROOT / "reviewer_m8_ridge_sensitivity.csv")
    ols = pd.read_csv(TABLE_ROOT / "reviewer_m8_within_study_ols.csv")
    save_csv(ridge, "Table_S5_ridge_sensitivity.csv")
    save_csv(ols, "Table_S6_cluster_robust_ols.csv")

    bootstrap_path = SUPP_TABLE_ROOT / "Fig_S5_bootstrap_model_count_sensitivity.csv"
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
    else:
        bootstrap = bootstrap_sensitivity(analysis, x_full)
        save_csv(bootstrap, bootstrap_path.name)

    draw_duplicate_figure(duplicate_summary, duplicate_detail)
    draw_clipping_figure(clipping)
    draw_temporal_figure(temporal)
    draw_ridge_figure(ridge, ols)
    draw_bootstrap_figure(bootstrap)

    manifest = pd.DataFrame(
        [
            {"supplementary_item": "Text S1 / Fig. 1a", "data_file": "Table_S0_evidence_flow.csv"},
            {"supplementary_item": "Table S1", "data_file": "Table_S1_fold_level_metrics.csv"},
            {"supplementary_item": "Table S2", "data_file": "Table_S2_predictor_dictionary.csv"},
            {"supplementary_item": "Table S3", "data_file": "Table_S3_study_inventory.csv"},
            {"supplementary_item": "Table S4", "data_file": "Table_S4_baseline_models.csv"},
            {"supplementary_item": "Fig. S1", "data_file": "Fig_S1_duplicate_audit_summary.csv; Fig_S1_duplicate_audit_row_values.csv"},
            {"supplementary_item": "Fig. S2", "data_file": "Fig_S2_clipping_sensitivity_values.csv"},
            {"supplementary_item": "Fig. S3", "data_file": "Fig_S3_size_matched_temporal_values.csv"},
            {"supplementary_item": "Fig. S4 / Tables S5-S6", "data_file": "Table_S5_ridge_sensitivity.csv; Table_S6_cluster_robust_ols.csv"},
            {"supplementary_item": "Fig. S5", "data_file": "Fig_S5_bootstrap_model_count_sensitivity.csv"},
        ]
    )
    save_csv(manifest, "Supplementary_data_manifest.csv")
    print(f"Analysis rows={len(analysis)}, studies={analysis.ref_id.nunique()}")
    print(f"Supplementary tables: {SUPP_TABLE_ROOT}")
    print(f"Supplementary figures: {FIG_ROOT}")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures_only()
    else:
        main()
