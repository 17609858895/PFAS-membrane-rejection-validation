"""
Refined Nature-style figures for the biogas ML paper.

The visual grammar follows the local reference templates in 参考文献:
- model comparison + TOPSIS ranking,
- Wilcoxon pairwise significance maps,
- predicted-vs-observed scatter with a side error panel,
- SHAP, PDP, ALE and interaction views.

Figure titles and panel explanations are intentionally kept out of the images
and written to figures/图片说明.md.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from scientific_palettes import PLOT_COLORS, get_palette


warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_PATH = ROOT / "merged_train_df.csv"
TARGET = "y_biogas_STP"
RANDOM_STATE = 42

REACTORS = ["RI-FLEX", "R2-FLEX", "R3-FIXED DOME", "R4-FIXED DOME"]
REACTOR_COLORS = dict(zip(REACTORS, get_palette("bar_blue_green_5", n=4)))
TYPE_COLORS = {"FLEX": PLOT_COLORS["green"], "FIXED DOME": PLOT_COLORS["coral"]}
MODEL_COLORS = get_palette("pcie_vivid_9", n=9)

RENAME = {
    TARGET: "Biogas_STP",
    "biogas_lag1": "Biogas_lag1",
    "biogas_lag2": "Biogas_lag2",
    "biogas_roll3": "Biogas_roll3",
    "biogas_roll7": "Biogas_roll7",
    "biogas_delta1": "Biogas_delta1",
    "days_since_prev": "Days_gap",
    "manure_fed_kg": "Manure",
    "water_kg": "Water",
    "air_temp_in_situ": "T_local",
    "daily_mean_air_temp": "T_mean",
    "daily_max_air_temp": "T_max",
    "daily_min_air_temp": "T_min",
    "daily_solar": "Solar",
    "daily_precip": "Precip",
    "daily_atm_p": "Atm_P",
    "daily_vpd": "VPD",
    "daily_wind": "Wind",
    "daily_temp_range": "T_range",
    "reactor_id_enc": "Reactor_ID",
    "reactor_type_enc": "Reactor_type",
}


def rn(name: str) -> str:
    return RENAME.get(name, name)


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.labelsize": 15.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.7,
            "xtick.major.size": 7.2,
            "ytick.major.size": 7.2,
            "xtick.major.width": 1.55,
            "ytick.major.width": 1.55,
            "xtick.minor.size": 4.2,
            "ytick.minor.size": 4.2,
            "xtick.minor.width": 1.15,
            "ytick.minor.width": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(PLOT_COLORS["ink"])
        ax.spines[spine].set_linewidth(1.7)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", bottom=True, top=False, colors=PLOT_COLORS["ink"],
                   width=1.55, length=7.2, direction="out", labelsize=12.5)
    ax.tick_params(axis="y", which="major", left=True, right=False, colors=PLOT_COLORS["ink"],
                   width=1.55, length=7.2, direction="out", labelsize=12.5)
    ax.tick_params(axis="both", which="minor", colors=PLOT_COLORS["ink"], width=1.15, length=4.2,
                   direction="out")
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold", labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", labelpad=7)


def numeric_ticks(ax, x: bool = True, y: bool = True, n: int = 5) -> None:
    """Keep numeric axes readable; use 3-6 major ticks unless the domain differs."""
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))


def save(fig: plt.Figure, folder_name: str, stem: str) -> None:
    folder = BASE / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    pdf_path = folder / f"{stem}.pdf"
    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    except PermissionError:
        alt_pdf = folder / f"{stem}_refined.pdf"
        fig.savefig(alt_pdf, bbox_inches="tight", facecolor="white")
        print(f"  NOTE {pdf_path.name} is locked; wrote {alt_pdf.name} instead")
    export_subfigures(fig, folder, stem)
    plt.close(fig)
    print(f"  OK {folder_name}/{stem}.png and .pdf")


def export_subfigures(fig: plt.Figure, folder: Path, stem: str) -> None:
    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letters = list("abcdefghijklmnopqrstuvwxyz")
    exported = []
    for ax in fig.axes:
        if not ax.get_visible() or str(ax.get_label()).startswith("<colorbar"):
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        bbox = bbox.expanded(1.06, 1.06).transformed(fig.dpi_scale_trans.inverted())
        suffix = letters[len(exported)] if len(exported) < len(letters) else f"p{len(exported) + 1}"
        fig.savefig(subdir / f"{stem}_{suffix}.png", dpi=600, bbox_inches=bbox, facecolor="white")
        fig.savefig(subdir / f"{stem}_{suffix}.pdf", bbox_inches=bbox, facecolor="white")
        exported.append(suffix)
    readme = subdir / "README.md"
    readme.write_text(
        "# Subfigures\n\n"
        "Each panel was extracted from the final combined figure export.\n\n"
        + "\n".join(f"- {s}: `{stem}_{s}.png`, `{stem}_{s}.pdf`" for s in exported)
        + "\n",
        encoding="utf-8",
    )


def load_and_prepare():
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df_raw[TARGET].dropna()
    q1, q3 = y.quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)

    df = df_raw[df_raw[TARGET].notna() & (df_raw[TARGET] >= 0) & (df_raw[TARGET] <= upper)].copy()
    df = df.sort_values(["reactor_id", "date"]).reset_index(drop=True)
    grouped = df.groupby("reactor_id", group_keys=False)
    df["biogas_lag1"] = grouped[TARGET].shift(1)
    df["biogas_lag2"] = grouped[TARGET].shift(2)
    df["biogas_roll3"] = grouped[TARGET].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["biogas_roll7"] = grouped[TARGET].apply(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    df["biogas_delta1"] = df["biogas_lag1"] - df["biogas_lag2"]
    df["days_since_prev"] = grouped["date"].diff().dt.days
    df["reactor_type_plot"] = df["reactor_type"].astype(str).str.replace("_", " ", regex=False)
    df = df.sort_values(["date", "reactor_id"]).reset_index(drop=True)

    hist_f = ["biogas_lag1", "biogas_lag2", "biogas_roll3", "biogas_roll7", "biogas_delta1", "days_since_prev"]
    oper_f = ["manure_fed_kg", "water_kg", "air_temp_in_situ"]
    weat_f = [
        "daily_mean_air_temp",
        "daily_max_air_temp",
        "daily_min_air_temp",
        "daily_solar",
        "daily_precip",
        "daily_atm_p",
        "daily_vpd",
        "daily_wind",
        "daily_temp_range",
    ]
    hist_f = [c for c in hist_f if c in df.columns]
    oper_f = [c for c in oper_f if c in df.columns]
    weat_f = [c for c in weat_f if c in df.columns]
    cat_f = ["reactor_id", "reactor_type"]
    all_f = hist_f + oper_f + weat_f + cat_f

    dates = np.array(sorted(df["date"].unique()))
    test_cut = int(len(dates) * 0.8)
    cal_cut = int(len(dates) * 0.7)
    train_dates = set(dates[:test_cut])
    calib_dates = set(dates[cal_cut:test_cut])
    test_dates = set(dates[test_cut:])
    train = df[df["date"].isin(train_dates)].copy()
    calib = df[df["date"].isin(calib_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()
    return df_raw, df, upper, train, calib, test, hist_f, oper_f, weat_f, cat_f, all_f


def build_pipe(estimator, numeric_features, categorical_features):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("mdl", estimator)])


def model_library():
    return {
        "Ridge": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=18, weights="uniform"),
        "SVR": SVR(C=10, epsilon=30, gamma="scale"),
        "RF": RandomForestRegressor(
            n_estimators=360, max_depth=8, min_samples_leaf=8, max_features=0.75,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=360, max_depth=6, min_samples_leaf=8, max_features=0.75,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=120, learning_rate=0.035, max_depth=2, subsample=0.70,
            colsample_bytree=0.70, min_child_weight=12, reg_lambda=12, reg_alpha=1,
            random_state=RANDOM_STATE, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.035, max_depth=2, num_leaves=5,
            min_child_samples=25, subsample=0.70, colsample_bytree=0.70,
            reg_lambda=12, random_state=RANDOM_STATE, verbose=-1
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=130, learning_rate=0.035, depth=2, l2_leaf_reg=15,
            random_strength=1.5, random_state=RANDOM_STATE, verbose=0
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=0.002, learning_rate_init=0.003,
            early_stopping=True, max_iter=1500, random_state=RANDOM_STATE
        ),
    }


def metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "NSE": nse}


def entropy_topsis(frame):
    benefit, cost = ["R2", "NSE"], ["RMSE", "MAE", "MAPE"]
    x = frame[benefit + cost].astype(float).copy()
    norm = x / np.sqrt((x**2).sum(axis=0))
    p = norm / (norm.sum(axis=0) + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(frame) + 1e-12)
    weights = (1 - entropy) / (1 - entropy).sum()
    v = norm.mul(weights, axis=1)
    ideal = pd.Series({c: v[c].max() if c in benefit else v[c].min() for c in v.columns})
    nadir = pd.Series({c: v[c].min() if c in benefit else v[c].max() for c in v.columns})
    d_pos = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - nadir) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-12)


def evaluate_models(train, test, numeric_features, categorical_features, all_features):
    models = model_library()
    train_dates = np.array(sorted(train["date"].unique()))
    tscv = TimeSeriesSplit(n_splits=5)
    fitted, preds_test, preds_train, rows, fold_rows = {}, {}, {}, [], []

    for name, estimator in models.items():
        print(f"  {name}")
        for fold, (ti, vi) in enumerate(tscv.split(train_dates), start=1):
            fold_train = train[train["date"].isin(set(train_dates[ti]))]
            fold_val = train[train["date"].isin(set(train_dates[vi]))]
            pipe = build_pipe(clone(estimator), numeric_features, categorical_features)
            pipe.fit(fold_train[all_features], fold_train[TARGET])
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_R2": r2_score(fold_train[TARGET], pipe.predict(fold_train[all_features])),
                    "val_R2": r2_score(fold_val[TARGET], pipe.predict(fold_val[all_features])),
                    "val_RMSE": math.sqrt(mean_squared_error(fold_val[TARGET], pipe.predict(fold_val[all_features]))),
                }
            )

        pipe = build_pipe(clone(estimator), numeric_features, categorical_features)
        pipe.fit(train[all_features], train[TARGET])
        fitted[name] = pipe
        preds_train[name] = pipe.predict(train[all_features])
        preds_test[name] = pipe.predict(test[all_features])
        row = {"model": name}
        row.update(metrics(test[TARGET].to_numpy(), preds_test[name]))
        rows.append(row)

    performance = pd.DataFrame(rows).set_index("model")
    performance["TOPSIS"] = entropy_topsis(performance)
    performance = performance.sort_values("TOPSIS", ascending=False)
    return performance, pd.DataFrame(fold_rows), fitted, preds_train, preds_test


def select_robust_primary_model(performance, fold_df, preds_train, train):
    rows = []
    for model in performance.index:
        train_r2 = r2_score(train[TARGET], preds_train[model])
        test_r2 = performance.loc[model, "R2"]
        cv_vals = fold_df.loc[fold_df["model"] == model, "val_R2"]
        rows.append(
            {
                "model": model,
                "Train_R2": train_r2,
                "CV_R2_mean": cv_vals.mean(),
                "CV_R2_min": cv_vals.min(),
                "Test_R2": test_r2,
                "Gap": train_r2 - test_r2,
                "RMSE": performance.loc[model, "RMSE"],
                "TOPSIS": performance.loc[model, "TOPSIS"],
            }
        )
    diagnostics = pd.DataFrame(rows).set_index("model")
    best_rmse = diagnostics["RMSE"].min()
    top = diagnostics[
        (diagnostics["RMSE"] <= best_rmse * 1.01)
        & (diagnostics["Test_R2"] >= diagnostics["Test_R2"].max() - 0.02)
    ].copy()
    stable = top[top["CV_R2_min"] >= 0]
    if len(stable):
        selected = stable.sort_values(["CV_R2_mean", "Gap"], ascending=[False, True]).index[0]
    else:
        selected = top.sort_values(["Gap", "CV_R2_mean"], ascending=[True, False]).index[0]
    diagnostics["Robust_primary"] = diagnostics.index == selected
    return selected, diagnostics


def fig1_data_audit(df_raw, df, upper):
    folder, stem = "Fig01_data_audit", "Fig01_data_audit"
    fig = plt.figure(figsize=(15.8, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    waterfall = pd.DataFrame(
        {
            "stage": ["Raw join", "Valid target", "Non-negative", "3IQR clean"],
            "records": [
                len(df_raw),
                int(df_raw[TARGET].notna().sum()),
                int(((df_raw[TARGET].fillna(-1) >= 0) & df_raw[TARGET].notna()).sum()),
                len(df),
            ],
        }
    )
    x = np.arange(len(waterfall))
    ax.plot(x, waterfall["records"], color=PLOT_COLORS["navy"], lw=2.8, marker="o", ms=10,
            mfc="white", mec=PLOT_COLORS["navy"], mew=2.2)
    ax.fill_between(x, waterfall["records"], waterfall["records"].min() * 0.96,
                    color=PLOT_COLORS["blue"], alpha=0.12, linewidth=0)
    for xi, value in zip(x, waterfall["records"]):
        ax.text(xi, value + 8, f"{value}", ha="center", va="bottom",
                fontsize=17, fontweight="bold", color=PLOT_COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(waterfall["stage"], rotation=18, ha="right")
    style_axis(ax, xlabel="", ylabel="Records")
    numeric_ticks(ax, x=False, y=True, n=5)

    ax = fig.add_subplot(gs[0, 1])
    model_features = [TARGET, "manure_fed_kg", "water_kg", "air_temp_in_situ",
                      "daily_mean_air_temp", "daily_solar", "daily_precip",
                      "daily_vpd", "daily_wind"]
    availability = pd.Series(
        {
            rn(c): 100 - df_raw[c].isna().mean() * 100
            for c in model_features
            if c in df_raw.columns
        }
    ).sort_values()
    colors = [PLOT_COLORS["green"] if v >= 95 else PLOT_COLORS["blue"] if v >= 80 else PLOT_COLORS["coral"] for v in availability.values]
    ax.barh(availability.index, availability.values, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_xlim(0, 105)
    for yi, value in enumerate(availability.values):
        ax.text(value + 1.5, yi, f"{value:.0f}%", va="center", fontsize=15, fontweight="bold")
    style_axis(ax, xlabel="Available observations (%)", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)

    ax = fig.add_subplot(gs[1, 0])
    feature_groups = {
        "Target": [TARGET],
        "Operation": ["manure_fed_kg", "water_kg", "air_temp_in_situ"],
        "Weather": ["daily_mean_air_temp", "daily_solar", "daily_precip", "daily_vpd", "daily_wind"],
    }
    group_rows = []
    for group_name, cols in feature_groups.items():
        cols = [c for c in cols if c in df_raw.columns]
        group_rows.append(
            {
                "Feature group": group_name,
                "Available": 100 - df_raw[cols].isna().mean().mean() * 100,
                "Variables": len(cols),
            }
        )
    groups = pd.DataFrame(group_rows)
    ax.barh(groups["Feature group"], groups["Available"], color=[PLOT_COLORS["navy"], PLOT_COLORS["green"], PLOT_COLORS["coral"]],
            edgecolor="white", linewidth=1.0)
    for yi, row in groups.iterrows():
        ax.text(min(row["Available"] - 1.2, 98.6), yi, f"{row['Available']:.1f}%  ({int(row['Variables'])} vars)",
                va="center", ha="right", fontsize=15, fontweight="bold", color="white")
    ax.set_xlim(0, 102)
    style_axis(ax, xlabel="Mean availability in model variables (%)", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)

    ax = fig.add_subplot(gs[1, 1])
    comp = pd.DataFrame(
        [
            {"Reactor": rid, "Raw": len(df_raw[df_raw["reactor_id"] == rid]),
             "Valid": int(df_raw[df_raw["reactor_id"] == rid][TARGET].notna().sum()),
             "Clean": int((df["reactor_id"] == rid).sum())}
            for rid in REACTORS
        ]
    )
    x = np.arange(len(REACTORS))
    width = 0.18
    for j, (col, color) in enumerate(zip(["Raw", "Valid", "Clean"], [PLOT_COLORS["lavender"], PLOT_COLORS["blue"], PLOT_COLORS["green"]])):
        ax.bar(x + (j - 1) * width, comp[col], width=width, color=color, edgecolor="white", linewidth=1.0, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" ", "\n") for r in REACTORS])
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_axis(ax, ylabel="Records")
    numeric_ticks(ax, x=False, y=True, n=5)
    save(fig, folder, stem)


def fig2_target_distribution(df_raw, df, upper):
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.6))
    fig.subplots_adjust(hspace=0.27, wspace=0.24)

    ax = axes[0, 0]
    raw_pos = df_raw.loc[df_raw[TARGET] >= 0, TARGET].dropna()
    clean_y = df[TARGET].dropna()
    cap = max(2400, clean_y.quantile(0.995) * 1.1)
    ax.hist(raw_pos, bins=60, histtype="step", color=PLOT_COLORS["coral"], lw=2.0, label="Raw positive")
    ax.hist(clean_y, bins=42, color=PLOT_COLORS["green"], alpha=0.74, edgecolor="white", linewidth=0.8, label="Clean")
    ax.axvline(upper, color=PLOT_COLORS["ink"], lw=1.5, ls="--")
    ax.set_xlim(-50, cap)
    ax.legend(frameon=False)
    style_axis(ax, xlabel="Daily biogas at STP (mL)", ylabel="Frequency")
    numeric_ticks(ax, x=True, y=True, n=5)

    ax = axes[0, 1]
    plot = df.copy()
    plot["reactor_id"] = pd.Categorical(plot["reactor_id"], REACTORS)
    sns.violinplot(data=plot, x="reactor_id", y=TARGET, order=REACTORS, palette=REACTOR_COLORS,
                   inner=None, linewidth=0.8, cut=0, ax=ax)
    sns.boxplot(data=plot, x="reactor_id", y=TARGET, order=REACTORS, width=0.18,
                showcaps=True, boxprops={"facecolor": "white", "edgecolor": PLOT_COLORS["ink"], "linewidth": 1.2},
                medianprops={"color": PLOT_COLORS["red"], "linewidth": 1.7},
                whiskerprops={"color": PLOT_COLORS["ink"], "linewidth": 1.1},
                capprops={"color": PLOT_COLORS["ink"], "linewidth": 1.1},
                showfliers=False, ax=ax)
    ax.set_xticklabels([r.replace(" ", "\n") for r in REACTORS])
    style_axis(ax, xlabel="", ylabel="Daily biogas at STP (mL)")
    numeric_ticks(ax, x=False, y=True, n=5)

    ax = axes[1, 0]
    for rid in REACTORS:
        raw = df_raw[df_raw["reactor_id"] == rid].sort_values("date")
        clean = df[df["reactor_id"] == rid].sort_values("date")
        ax.scatter(raw["date"], raw[TARGET], s=18, facecolors="none", edgecolors=REACTOR_COLORS[rid], alpha=0.28, linewidths=1.0)
        ax.plot(clean["date"], clean[TARGET], lw=1.9, color=REACTOR_COLORS[rid], alpha=0.9, label=rid)
    ax.axhline(0, color=PLOT_COLORS["ink"], lw=1.0, ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(frameon=False, ncol=2)
    style_axis(ax, xlabel="Date", ylabel="Daily biogas at STP (mL)")
    numeric_ticks(ax, x=False, y=True, n=5)

    ax = axes[1, 1]
    for rid in REACTORS:
        sub = df[df["reactor_id"] == rid]
        ax.scatter(sub["daily_mean_air_temp"], sub[TARGET], s=50, facecolors="none",
                   edgecolors=REACTOR_COLORS[rid], alpha=0.75, linewidths=1.35, label=rid)
    xv = df["daily_mean_air_temp"].dropna()
    yv = df.loc[xv.index, TARGET]
    coef = np.polyfit(xv, yv, 1)
    line_x = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(line_x, np.polyval(coef, line_x), color=PLOT_COLORS["ink"], lw=2.0)
    style_axis(ax, xlabel="T_mean (deg C)", ylabel="Daily biogas at STP (mL)")
    numeric_ticks(ax, x=True, y=True, n=5)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              handletextpad=0.6, columnspacing=1.2)
    save(fig, "Fig02_target_distribution", "Fig02_target_distribution")


def correlation_pvalues(data: pd.DataFrame, method: str) -> pd.DataFrame:
    pvals = pd.DataFrame(np.nan, index=data.columns, columns=data.columns, dtype=float)
    for i, c1 in enumerate(data.columns):
        for j, c2 in enumerate(data.columns):
            pair = data[[c1, c2]].dropna()
            if i == j:
                pvals.loc[c1, c2] = 0.0
            elif len(pair) >= 4 and pair[c1].nunique() > 1 and pair[c2].nunique() > 1:
                if method == "spearman":
                    _, p = stats.spearmanr(pair[c1], pair[c2])
                else:
                    _, p = stats.pearsonr(pair[c1], pair[c2])
                pvals.loc[c1, c2] = p
    return pvals


def star_label(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def draw_grouped_corr_circle(ax, corr: pd.DataFrame, pvals: pd.DataFrame, group_map: dict[str, str], cmap, cbar_label: str):
    labels = list(corr.columns)
    n = len(labels)
    group_colors = {
        "Target": PLOT_COLORS["navy"],
        "History": PLOT_COLORS["blue"],
        "Operation": PLOT_COLORS["green"],
        "Weather": PLOT_COLORS["coral"],
    }
    norm = plt.Normalize(-1, 1)
    ax.set_facecolor("white")

    for i, ylab in enumerate(labels):
        for j, xlab in enumerate(labels):
            value = corr.loc[ylab, xlab]
            if not np.isfinite(value):
                continue
            radius = 0.38 * math.sqrt(abs(value))
            circle = patches.Circle((j, i), radius=radius, facecolor=cmap(norm(value)),
                                    edgecolor=PLOT_COLORS["ink"], linewidth=0.35, alpha=0.92)
            ax.add_patch(circle)
            sig = star_label(pvals.loc[ylab, xlab])
            if sig and i != j:
                ax.text(j, i, sig, ha="center", va="center", fontsize=12.5,
                        fontweight="bold", color=PLOT_COLORS["ink"])

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -1.05)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=12.5, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=5.5, width=1.25, colors=PLOT_COLORS["ink"])
    for spine in ax.spines.values():
        spine.set_color("#D7DCE2")
        spine.set_linewidth(1.0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#EEF1F4", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    positions = pd.Series(group_map)
    start = 0
    while start < n:
        group = positions.iloc[start]
        end = start
        while end + 1 < n and positions.iloc[end + 1] == group:
            end += 1
        color = group_colors.get(group, PLOT_COLORS["muted"])
        ax.add_patch(patches.Rectangle((start - 0.5, -0.92), end - start + 1, 0.18,
                                       facecolor=color, edgecolor="none", clip_on=False, alpha=0.92))
        ax.add_patch(patches.Rectangle((-0.92, start - 0.5), 0.18, end - start + 1,
                                       facecolor=color, edgecolor="none", clip_on=False, alpha=0.92))
        ax.text((start + end) / 2, -1.02, group, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=color, clip_on=False)
        start = end + 1

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontweight="bold", fontsize=13.5)
    cbar.ax.tick_params(labelsize=12, width=1.2, length=5)


def draw_diagonal_group_bracket(ax, start: int, end: int, label: str, n: int, offset: float, fontsize: float = 12.0):
    color = PLOT_COLORS["ink"]
    x0 = start - 0.5
    x1 = end + 0.5
    y0 = start - 1.08 - offset
    y1 = end - 0.08 - offset
    ax.plot([x0, x1], [y0, y1], color=color, lw=1.35, clip_on=False)
    ax.plot([x0, x0 + 0.28], [y0, y0 - 0.20], color=color, lw=1.35, clip_on=False)
    ax.plot([x1 - 0.22, x1], [y1 + 0.16, y1], color=color, lw=1.35, clip_on=False)
    ax.text((x0 + x1) / 2 + 0.35, (y0 + y1) / 2 - 0.18, label,
            rotation=-48, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", fontstyle="italic", color=color, clip_on=False)


def draw_lower_triangle_corr_heatmap(ax, corr: pd.DataFrame, pvals: pd.DataFrame, groups: list[str],
                                     cmap, label: str, show_cbar: bool = False):
    labels = list(corr.columns)
    n = len(labels)
    norm = plt.Normalize(-1, 1)
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            if j > i:
                continue
            value = corr.loc[row_label, col_label]
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=cmap(norm(value)), edgecolor="white", linewidth=0.95)
            ax.add_patch(rect)
            text_color = "white" if abs(value) > 0.68 else PLOT_COLORS["ink"]
            ax.text(j, i - 0.07, f"{value:.2f}", ha="center", va="center",
                    fontsize=8.9, fontweight="bold", color=text_color)
            sig = star_label(pvals.loc[row_label, col_label])
            if sig and i != j:
                ax.text(j, i + 0.25, sig, ha="center", va="center",
                        fontsize=7.8, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, n + 2.55)
    ax.set_ylim(n - 0.5, -1.05)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=10.2, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10.2, fontweight="bold")
    ax.tick_params(axis="both", which="major", length=7.0, width=1.55, colors=PLOT_COLORS["ink"], direction="out")
    for spine in ax.spines.values():
        spine.set_visible(False)

    group_names = pd.Series(groups, index=labels)
    start = 0
    bracket_labels = {
        "Target": "Output",
        "History": "Historical biogas",
        "Operation": "Operational conditions",
        "Weather": "Weather context",
    }
    bi = 0
    while start < n:
        group = group_names.iloc[start]
        end = start
        while end + 1 < n and group_names.iloc[end + 1] == group:
            end += 1
        if group != "Target":
            draw_diagonal_group_bracket(ax, start, end, bracket_labels[group], n,
                                        offset=0.18 + bi * 0.10, fontsize=10.6)
        start = end + 1
        bi += 1

    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = ax.inset_axes([0.62, 0.53, 0.31, 0.04], transform=ax.transAxes)
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=10.0, width=1.15, length=4.5)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontweight("bold")
        cbar.set_label(label, fontsize=10.2, fontweight="bold", labelpad=4)


def fig3_correlation(df, hist_f, oper_f, weat_f):
    hist_keep = [c for c in ["biogas_lag1", "biogas_lag2", "biogas_roll3", "biogas_roll7", "biogas_delta1"] if c in hist_f]
    weather_keep = [c for c in ["daily_mean_air_temp", "daily_solar", "daily_precip", "daily_vpd", "daily_temp_range"] if c in weat_f]
    cols = [TARGET] + hist_keep + oper_f + weather_keep
    cols = [c for c in cols if c in df.columns]
    raw = df[cols].copy()
    raw.columns = [rn(c) for c in raw.columns]
    sub = raw.dropna()
    groups = (
        ["Target"]
        + ["History"] * len(hist_keep)
        + ["Operation"] * len(oper_f)
        + ["Weather"] * len(weather_keep)
    )
    labels = list(raw.columns)
    n = len(labels)
    corr = sub.corr("pearson").reindex(index=labels, columns=labels)
    pvals = correlation_pvalues(raw, "pearson").reindex(index=labels, columns=labels)

    group_map = dict(zip(raw.columns, groups))

    fig = plt.figure(figsize=(19.2, 13.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.35, 0.82], hspace=0.42, wspace=0.16)
    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    draw_lower_triangle_corr_heatmap(fig.add_subplot(gs[0, 0]), corr, pvals, groups, cmap,
                                     "Pearson correlation coefficients (r)", show_cbar=True)
    spearman_mat = sub.corr("spearman").reindex(index=labels, columns=labels)
    draw_lower_triangle_corr_heatmap(fig.add_subplot(gs[0, 1]), spearman_mat,
                                     correlation_pvalues(raw, "spearman").reindex(index=labels, columns=labels),
                                     groups, cmap, "Spearman correlation coefficients (rho)", show_cbar=True)

    target_label = rn(TARGET)
    features = [c for c in raw.columns if c != target_label]
    mi_data = raw[[target_label] + features].dropna()
    mi = mutual_info_regression(mi_data[features].values, mi_data[target_label].values, random_state=RANDOM_STATE)
    assoc = pd.DataFrame({
        "Feature": features,
        "MI": mi,
        "Spearman": spearman_mat[target_label].drop(target_label).reindex(features).values,
        "Group": [group_map[f] for f in features],
    }).sort_values("MI", ascending=True)

    ax = fig.add_subplot(gs[1, 0])
    bar_colors = [PLOT_COLORS["blue"] if g == "History" else PLOT_COLORS["green"] if g == "Operation" else PLOT_COLORS["coral"]
                  for g in assoc["Group"]]
    ax.barh(assoc["Feature"], assoc["MI"], color=bar_colors, edgecolor="white", linewidth=1.0)
    for yi, row in enumerate(assoc.itertuples()):
        ax.scatter(row.MI, yi, s=95 + 520 * min(abs(row.Spearman), 1.0),
                   facecolors="white", edgecolors=PLOT_COLORS["ink"], linewidths=1.2, zorder=3)
    style_axis(ax, xlabel="Mutual information with biogas", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)
    ax.tick_params(axis="both", which="major", labelsize=12.5, length=7.2, width=1.55)
    ax.xaxis.label.set_size(15.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12.5)
        label.set_fontweight("bold")

    ax = fig.add_subplot(gs[1, 1])
    group_summary = assoc.groupby("Group").agg(
        Mean_abs_spearman=("Spearman", lambda s: np.nanmean(np.abs(s))),
        Mean_MI=("MI", "mean"),
        Features=("Feature", "count"),
    ).reindex(["History", "Operation", "Weather"]).dropna()
    xpos = np.arange(len(group_summary))
    ax.scatter(xpos, group_summary["Mean_abs_spearman"],
               s=850 * (group_summary["Mean_MI"] / (group_summary["Mean_MI"].max() + 1e-12) + 0.25),
               facecolors="white",
               edgecolors=[PLOT_COLORS["blue"], PLOT_COLORS["green"], PLOT_COLORS["coral"]][: len(group_summary)],
               linewidths=3.0)
    ax.plot(xpos, group_summary["Mean_abs_spearman"], color=PLOT_COLORS["muted"], lw=1.4, alpha=0.7)
    for xi, row in enumerate(group_summary.itertuples()):
        ax.text(xi, row.Mean_abs_spearman + 0.025, f"n={int(row.Features)}", ha="center",
                fontsize=14.5, fontweight="bold", color=PLOT_COLORS["ink"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(group_summary.index, fontweight="bold")
    ax.set_ylim(0, min(1.0, group_summary["Mean_abs_spearman"].max() + 0.16))
    style_axis(ax, xlabel="", ylabel="Mean |Spearman rho| with biogas")
    numeric_ticks(ax, x=False, y=True, n=5)
    ax.tick_params(axis="both", which="major", labelsize=12.5, length=7.2, width=1.55)
    ax.yaxis.label.set_size(15.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12.5)
        label.set_fontweight("bold")

    save(fig, "Fig03_correlation", "Fig03_correlation")


def fig4_model_comparison(performance, fold_df, preds_test, test):
    models = performance.index.tolist()
    fig = plt.figure(figsize=(15.4, 10.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0], polar=True)
    radar_metrics = ["R2", "NSE", "RMSE", "MAE", "MAPE"]
    scaled = performance[radar_metrics].copy()
    for c in ["RMSE", "MAE", "MAPE"]:
        scaled[c] = 1 - (scaled[c] - scaled[c].min()) / (scaled[c].max() - scaled[c].min() + 1e-12)
    for c in ["R2", "NSE"]:
        scaled[c] = (scaled[c] - scaled[c].min()) / (scaled[c].max() - scaled[c].min() + 1e-12)
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["R²", "NSE", "1-RMSE", "1-MAE", "1-MAPE"], fontweight="bold", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=11, fontweight="bold")
    ax.grid(True, color="#DADDE2", lw=1.0, alpha=0.65)
    for i, model in enumerate(models):
        vals = scaled.loc[model].tolist() + [scaled.loc[model].tolist()[0]]
        ax.plot(angles, vals, color=MODEL_COLORS[i], lw=2.0, alpha=0.92, label=model)
        ax.fill(angles, vals, color=MODEL_COLORS[i], alpha=0.045)
    ax.legend(loc="center left", bbox_to_anchor=(1.10, 0.50), frameon=False, fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    long = fold_df.melt(id_vars=["model", "fold"], value_vars=["train_R2", "val_R2"], var_name="split", value_name="R2")
    sns.boxplot(data=long, x="model", y="R2", hue="split", palette=[PLOT_COLORS["blue"], PLOT_COLORS["coral"]],
                showfliers=False, linewidth=1.2, ax=ax)
    sns.stripplot(data=long, x="model", y="R2", hue="split", dodge=True, palette=[PLOT_COLORS["blue"], PLOT_COLORS["coral"]],
                  size=3.5, alpha=0.65, linewidth=0, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Train", "Validation"], frameon=False, ncol=2, loc="lower left")
    ax.set_xticklabels(models, rotation=32, ha="right")
    style_axis(ax, xlabel="", ylabel="R²")
    numeric_ticks(ax, x=False, y=True, n=5)

    ax = fig.add_subplot(gs[1, 0])
    err = pd.DataFrame({m: np.abs(test[TARGET].values - preds_test[m]) for m in model_library()})
    pvals = pd.DataFrame(np.ones((len(models), len(models))), index=models, columns=models)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                pvals.loc[m1, m2] = np.nan
            elif i < j:
                try:
                    p = stats.wilcoxon(err[m1], err[m2], zero_method="wilcox").pvalue
                except Exception:
                    p = 1.0
                pvals.loc[m1, m2] = p
                pvals.loc[m2, m1] = p
    sns.heatmap(-np.log10(pvals), cmap=sns.light_palette(PLOT_COLORS["navy"], as_cmap=True), linewidths=0,
                cbar_kws={"label": "-log10(p)", "shrink": 0.72}, ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_axis(ax)

    ax = fig.add_subplot(gs[1, 1])
    rank = performance.sort_values("TOPSIS", ascending=True)
    y = np.arange(len(rank))
    ax.hlines(y, 0, rank["TOPSIS"], color="#D6DDE2", lw=4)
    ax.scatter(rank["TOPSIS"], y, s=180, c=[MODEL_COLORS[models.index(m)] for m in rank.index],
               edgecolors="white", linewidths=1.4, zorder=3)
    for yi, value in zip(y, rank["TOPSIS"]):
        ax.text(value + 0.014, yi, f"{value:.4f}", va="center", fontsize=14, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(rank.index)
    ax.set_xlim(0, min(1.05, rank["TOPSIS"].max() + 0.12))
    style_axis(ax, xlabel="Entropy-TOPSIS score", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)
    save(fig, "Fig04_model_comparison", "Fig04_model_comparison")


def add_joint_prediction_panel(fig, spec, model, y_train, p_train, y_test, p_test, color):
    pair = spec.subgridspec(1, 2, width_ratios=[4.1, 1.05], wspace=0.06)
    ax = fig.add_subplot(pair[0, 0])
    ax_err = fig.add_subplot(pair[0, 1], sharey=ax)
    all_y = np.concatenate([y_train, y_test, p_train, p_test])
    lo = max(0, np.nanmin(all_y) * 0.96)
    hi = np.nanmax(all_y) * 1.08
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color=PLOT_COLORS["blue"], alpha=0.08, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=PLOT_COLORS["muted"], lw=1.5, ls="--")
    ax.scatter(y_train, p_train, s=36, marker="^", color=color, alpha=0.35, edgecolors=PLOT_COLORS["ink"], linewidths=0.4)
    ax.scatter(y_test, p_test, s=46, facecolors="none", edgecolors=color, alpha=0.82, linewidths=1.45)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    train_r2 = r2_score(y_train, p_train)
    test_r2 = r2_score(y_test, p_test)
    rmse = math.sqrt(mean_squared_error(y_test, p_test))
    ax.text(0.06, 0.94, f"{model}\nR² train={train_r2:.3f}\nR² test={test_r2:.3f}\nRMSE={rmse:.0f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=12.2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B9C2D0", lw=1.2, alpha=0.92))
    residual = y_test - p_test
    ax_err.scatter(residual, p_test, s=36, facecolors="none", edgecolors=color, alpha=0.72, linewidths=1.3)
    ax_err.axvline(0, color=PLOT_COLORS["muted"], lw=1.5, ls="--")
    ax_err.set_xlim(np.nanpercentile(residual, 1) * 1.25, np.nanpercentile(residual, 99) * 1.25)
    ax_err.set_xlabel("Error", fontweight="bold", labelpad=6)
    style_axis(ax_err)
    numeric_ticks(ax_err, x=True, y=False, n=3)
    ax_err.tick_params(axis="x", labelsize=10.5)
    ax_err.tick_params(axis="y", left=False, labelleft=False)
    ax_err.spines["left"].set_visible(False)
    style_axis(ax, xlabel="Observed biogas (mL)", ylabel="Predicted biogas (mL)")
    numeric_ticks(ax, x=True, y=True, n=5)


def fig5_predicted_observed(performance, preds_train, preds_test, train, test):
    models = performance.index.tolist()
    fig = plt.figure(figsize=(18.4, 15.6))
    outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.24, wspace=0.24)
    for spec, model, color in zip(outer, models, MODEL_COLORS):
        add_joint_prediction_panel(
            fig, spec, model,
            train[TARGET].values, preds_train[model],
            test[TARGET].values, preds_test[model],
            color,
        )
    save(fig, "Fig05_pred_vs_obs", "Fig05_pred_vs_obs")


def fig6_temporal_validation(test, calib, preds_test, fitted, best_model, all_f):
    best_pipe = fitted[best_model]
    q90 = np.quantile(np.abs(calib[TARGET].values - best_pipe.predict(calib[all_f])), 0.90)
    q95 = np.quantile(np.abs(calib[TARGET].values - best_pipe.predict(calib[all_f])), 0.95)
    fig, axes = plt.subplots(4, 1, figsize=(14.2, 9.2), sharex=False)
    fig.subplots_adjust(hspace=0.20)
    panel = test.copy()
    panel["pred"] = preds_test[best_model]
    for ax, rid in zip(axes, REACTORS):
        sub = panel[panel["reactor_id"] == rid].sort_values("date")
        ax.fill_between(sub["date"], sub["pred"] - q95, sub["pred"] + q95, color=PLOT_COLORS["lavender"], alpha=0.16, linewidth=0)
        ax.fill_between(sub["date"], sub["pred"] - q90, sub["pred"] + q90, color=PLOT_COLORS["blue"], alpha=0.18, linewidth=0)
        ax.plot(sub["date"], sub[TARGET], color=REACTOR_COLORS[rid], lw=2.0, marker="o", ms=5, mfc="white", mew=1.2)
        ax.plot(sub["date"], sub["pred"], color=PLOT_COLORS["ink"], lw=1.7, marker="s", ms=4, mfc="white", mew=1.0)
        ax.text(0.01, 0.86, rid, transform=ax.transAxes, fontsize=12, fontweight="bold", color=REACTOR_COLORS[rid])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        style_axis(ax, ylabel="Biogas (mL)")
    axes[-1].set_xlabel("Held-out date", fontweight="bold")
    save(fig, "Fig06_temporal_validation", "Fig06_temporal_validation")


def fig7_residual_reliability(train, test, all_f, fitted, preds_test, best_model):
    y_test = test[TARGET].values
    p_test = preds_test[best_model]
    residual = y_test - p_test
    panel = test.copy()
    panel["pred"] = p_test
    panel["residual"] = residual

    rng = np.random.default_rng(RANDOM_STATE)
    rand_r2 = []
    for _ in range(60):
        shuffled = rng.permutation(train[TARGET].values)
        pipe = clone(fitted[best_model])
        pipe.fit(train[all_f], shuffled)
        rand_r2.append(r2_score(y_test, pipe.predict(test[all_f])))
    rand_r2 = np.array(rand_r2)

    best_pipe = fitted[best_model]
    pre = best_pipe.named_steps["pre"]
    xtr = pre.transform(train[all_f])
    xte = pre.transform(test[all_f])
    hinv = np.linalg.pinv(xtr.T @ xtr)
    leverage = np.einsum("ij,jk,ik->i", xte, hinv, xte)
    h_star = 3 * xtr.shape[1] / xtr.shape[0]
    std_res = residual / (residual.std() + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    fig.subplots_adjust(hspace=0.32, wspace=0.28, bottom=0.16)

    ax = axes[0, 0]
    sns.violinplot(data=panel, x="reactor_id", y="residual", order=REACTORS, palette=REACTOR_COLORS,
                   inner=None, linewidth=0.8, cut=0, ax=ax)
    sns.stripplot(data=panel, x="reactor_id", y="residual", order=REACTORS, color=PLOT_COLORS["ink"],
                  size=3.5, alpha=0.35, ax=ax)
    ax.axhline(0, color=PLOT_COLORS["ink"], lw=1.4, ls="--")
    ax.set_xticklabels([r.replace(" ", "\n") for r in REACTORS])
    style_axis(ax, xlabel="", ylabel="Residual (mL)")

    ax = axes[0, 1]
    mean = (y_test + p_test) / 2
    diff = residual
    md, sd = diff.mean(), diff.std()
    for rid in REACTORS:
        mask = test["reactor_id"].values == rid
        ax.scatter(mean[mask], diff[mask], s=44, facecolors="none", edgecolors=REACTOR_COLORS[rid],
                   linewidths=1.3, alpha=0.78, label=rid)
    ax.axhline(md, color=PLOT_COLORS["ink"], lw=1.5)
    ax.axhline(md + 1.96 * sd, color=PLOT_COLORS["red"], lw=1.2, ls="--")
    ax.axhline(md - 1.96 * sd, color=PLOT_COLORS["red"], lw=1.2, ls="--")
    style_axis(ax, xlabel="Mean of observed and predicted (mL)", ylabel="Difference (mL)")

    ax = axes[1, 0]
    ax.hist(rand_r2, bins=22, color=PLOT_COLORS["blue"], alpha=0.45, edgecolor="white")
    ax.axvline(r2_score(y_test, p_test), color=PLOT_COLORS["red"], lw=2.2)
    style_axis(ax, xlabel="R² after Y-randomisation", ylabel="Frequency")

    ax = axes[1, 1]
    for rid in REACTORS:
        mask = test["reactor_id"].values == rid
        ax.scatter(leverage[mask], std_res[mask], s=45, facecolors="none",
                   edgecolors=REACTOR_COLORS[rid], linewidths=1.25, alpha=0.78, label=rid)
    ax.axhline(3, color=PLOT_COLORS["red"], lw=1.2, ls="--")
    ax.axhline(-3, color=PLOT_COLORS["red"], lw=1.2, ls="--")
    ax.axvline(h_star, color=PLOT_COLORS["coral"], lw=1.3, ls=":")
    style_axis(ax, xlabel="Leverage h", ylabel="Standardised residual")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), handletextpad=0.6, columnspacing=1.4)
    save(fig, "Fig07_residual_reliability", "Fig07_residual_reliability")


def compute_shap_model(df, train, test, hist_f, oper_f, weat_f):
    df2 = df.copy()
    df2["reactor_id_enc"] = LabelEncoder().fit_transform(df2["reactor_id"])
    df2["reactor_type_enc"] = LabelEncoder().fit_transform(df2["reactor_type"])
    feats = hist_f + oper_f + weat_f + ["reactor_id_enc", "reactor_type_enc"]
    tr = df2[df2["date"].isin(set(train["date"].unique()))].copy()
    te = df2[df2["date"].isin(set(test["date"].unique()))].copy()
    xtr = tr[feats].fillna(tr[feats].median())
    xte = te[feats].fillna(xtr.median())
    model = cb.CatBoostRegressor(
        iterations=130, learning_rate=0.035, depth=2, l2_leaf_reg=15,
        random_strength=1.5, random_state=RANDOM_STATE, verbose=0
    )
    model.fit(xtr, tr[TARGET])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(xte)
    return model, feats, tr, te, xtr, xte, shap_values


def fig8_shap_importance(model, feats, te, xte, shap_values):
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
    top14 = mean_abs.head(14).index.tolist()
    fig = plt.figure(figsize=(18.2, 7.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.56, width_ratios=[1.32, 1.42, 1.22])

    ax = fig.add_subplot(gs[0, 0])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    for j, feat in enumerate(top14):
        idx = feats.index(feat)
        sv = shap_values[:, idx]
        xv = xte[feat].to_numpy()
        norm = (xv - np.nanmin(xv)) / (np.nanmax(xv) - np.nanmin(xv) + 1e-12)
        y = len(top14) - 1 - j + np.random.default_rng(j).uniform(-0.22, 0.22, len(sv))
        ax.scatter(sv, y, c=plt.cm.coolwarm(norm), s=18, alpha=0.65, linewidths=0)
    ax.axvline(0, color=PLOT_COLORS["ink"], lw=1.2, ls="--")
    ax.set_yticks(range(len(top14)))
    ax.set_yticklabels([rn(f) for f in reversed(top14)])
    style_axis(ax, xlabel="SHAP value (mL)", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)
    cax = ax.inset_axes([0.16, -0.22, 0.60, 0.045])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Feature value", fontweight="bold", fontsize=10.5, labelpad=3)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(width=1.1, length=4, labelsize=10)

    ax = fig.add_subplot(gs[0, 1])
    rtype = te["reactor_type"].astype(str).str.replace("_", " ", regex=False)
    top12 = mean_abs.head(12).index.tolist()
    bidx = [feats.index(f) for f in top12]
    flex = np.abs(shap_values[rtype.values == "FLEX"])[:, bidx].mean(axis=0)
    fixed = np.abs(shap_values[rtype.values == "FIXED DOME"])[:, bidx].mean(axis=0)
    ypos = np.arange(len(top12))
    ax.barh(ypos + 0.18, flex[::-1], height=0.34, color=PLOT_COLORS["green"], label="FLEX")
    ax.barh(ypos - 0.18, fixed[::-1], height=0.34, color=PLOT_COLORS["coral"], label="FIXED DOME")
    ax.set_yticks(ypos)
    ax.set_yticklabels([rn(f) for f in reversed(top12)])
    style_axis(ax, xlabel="Mean absolute SHAP (mL)", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              handletextpad=0.6, columnspacing=1.2)

    ax = fig.add_subplot(gs[0, 2])
    result = permutation_importance(model, xte, te[TARGET], scoring="neg_root_mean_squared_error",
                                    n_repeats=14, random_state=RANDOM_STATE, n_jobs=-1)
    pi = pd.Series(result.importances_mean, index=feats).sort_values(ascending=False).head(14)
    pi_err = pd.Series(result.importances_std, index=feats)
    colors = [PLOT_COLORS["navy"] if v >= 0 else PLOT_COLORS["coral"] for v in pi.values[::-1]]
    ax.barh(np.arange(len(pi)), pi.values[::-1], xerr=pi_err[pi.index].values[::-1],
            color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(np.arange(len(pi)))
    ax.set_yticklabels([rn(f) for f in reversed(pi.index)])
    style_axis(ax, xlabel="Permutation RMSE increase", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)
    save(fig, "Fig08_SHAP_importance", "Fig08_SHAP_importance")


def pdp_values(model, xbase, feat, n=36):
    vals = xbase[feat].dropna()
    grid = np.linspace(vals.quantile(0.05), vals.quantile(0.95), n)
    y = []
    for value in grid:
        tmp = xbase.copy()
        tmp[feat] = value
        y.append(model.predict(tmp).mean())
    return grid, np.array(y)


def ale_values(model, xbase, feat, bins=16):
    vals = xbase[feat].dropna().to_numpy()
    edges = np.unique(np.percentile(vals, np.linspace(0, 100, bins + 1)))
    mids = 0.5 * (edges[:-1] + edges[1:])
    effects = np.zeros_like(mids)
    ses = np.zeros_like(mids)
    for i in range(len(mids)):
        mask = (xbase[feat] >= edges[i]) & (xbase[feat] < edges[i + 1])
        if mask.sum() < 2:
            continue
        low = xbase.loc[mask].copy()
        high = xbase.loc[mask].copy()
        low[feat] = edges[i]
        high[feat] = edges[i + 1]
        deltas = model.predict(high) - model.predict(low)
        effects[i] = deltas.mean()
        ses[i] = deltas.std(ddof=1) / np.sqrt(len(deltas))
    centered = np.cumsum(effects)
    centered -= centered.mean()
    band = 1.96 * np.sqrt(np.cumsum(ses**2))
    return mids, centered, centered - band, centered + band


def fig9_pdp_ale_interaction(model, feats, xte, shap_values):
    top6 = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False).head(6).index.tolist()
    fig = plt.figure(figsize=(19.2, 15.9))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.62, wspace=0.50,
                           width_ratios=[1.05, 1.05, 1.05, 1.25])
    for i, feat in enumerate(top6):
        row = i // 2
        col = (i % 2) * 2
        ax = fig.add_subplot(gs[row, col])
        gx, gy = pdp_values(model, xte, feat)
        ax.plot(gx, gy, color=PLOT_COLORS["blue"], lw=2.0)
        ax.fill_between(gx, gy - gy.std() * 0.18, gy + gy.std() * 0.18, color=PLOT_COLORS["blue"], alpha=0.14, linewidth=0)
        style_axis(ax, xlabel=rn(feat), ylabel="PDP")
        numeric_ticks(ax, x=True, y=True, n=5)
        ax = fig.add_subplot(gs[row, col + 1])
        mx, my, low, high = ale_values(model, xte, feat)
        ax.plot(mx, my, color=PLOT_COLORS["coral"], lw=2.0)
        ax.fill_between(mx, low, high, color=PLOT_COLORS["coral"], alpha=0.16, linewidth=0)
        ax.axhline(0, color=PLOT_COLORS["ink"], lw=1.0, ls="--")
        style_axis(ax, xlabel=rn(feat), ylabel="ALE")
        numeric_ticks(ax, x=True, y=True, n=5)

    f1 = "biogas_lag1" if "biogas_lag1" in feats else top6[0]
    f2 = "daily_mean_air_temp" if "daily_mean_air_temp" in feats else top6[1]
    ax = fig.add_subplot(gs[3, 0])
    idx1 = feats.index(f1)
    sc = ax.scatter(xte[f1], shap_values[:, idx1], c=xte[f2], cmap="coolwarm", s=42,
                    facecolors="none", linewidths=1.0, alpha=0.80)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.62, pad=0.055)
    cbar.set_label(rn(f2), fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.1, length=4)
    style_axis(ax, xlabel=rn(f1), ylabel="SHAP")
    numeric_ticks(ax, x=True, y=True, n=5)

    ax = fig.add_subplot(gs[3, 1])
    gx = np.linspace(xte[f1].quantile(0.05), xte[f1].quantile(0.95), 24)
    gy = np.linspace(xte[f2].quantile(0.05), xte[f2].quantile(0.95), 24)
    z = np.zeros((len(gy), len(gx)))
    base = xte.copy()
    for i, yy in enumerate(gy):
        for j, xx in enumerate(gx):
            tmp = base.copy()
            tmp[f1] = xx
            tmp[f2] = yy
            z[i, j] = model.predict(tmp).mean()
    im = ax.imshow(z, origin="lower", aspect="auto", cmap=sns.light_palette(PLOT_COLORS["green"], as_cmap=True),
                   extent=[gx.min(), gx.max(), gy.min(), gy.max()])
    cbar = plt.colorbar(im, ax=ax, shrink=0.62, pad=0.055)
    cbar.set_label("Predicted", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.1, length=4)
    style_axis(ax, xlabel=rn(f1), ylabel=rn(f2))
    numeric_ticks(ax, x=True, y=True, n=5)

    ax = fig.add_subplot(gs[3, 2])
    idx2 = feats.index(f2)
    vals = pd.DataFrame({"x": xte[f1].values, "y": xte[f2].values, "shap": shap_values[:, idx1] + shap_values[:, idx2]})
    vals["xb"] = pd.qcut(vals["x"], 9, duplicates="drop")
    vals["yb"] = pd.qcut(vals["y"], 9, duplicates="drop")
    heat = vals.pivot_table(index="yb", columns="xb", values="shap", aggfunc="mean")
    sns.heatmap(heat, cmap=sns.diverging_palette(220, 18, as_cmap=True), center=0, linewidths=0,
                cbar=False, ax=ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    style_axis(ax, xlabel=rn(f1), ylabel=rn(f2))
    cax = ax.inset_axes([0.20, 1.14, 0.58, 0.040])
    sm = plt.cm.ScalarMappable(cmap=sns.diverging_palette(220, 18, as_cmap=True),
                               norm=plt.Normalize(np.nanmin(heat.values), np.nanmax(heat.values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Mean SHAP", fontsize=9.5, fontweight="bold", labelpad=2)
    cbar.ax.tick_params(labelsize=8.5, width=1.0, length=3.2)

    ax = fig.add_subplot(gs[3, 3])
    top_contrib = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=True).tail(10)
    ax.barh([rn(f) for f in top_contrib.index], top_contrib.values, color=PLOT_COLORS["teal"], edgecolor="white")
    style_axis(ax, xlabel="Mean absolute SHAP", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)
    ax.tick_params(axis="y", labelsize=11.5)
    save(fig, "Fig09_PDP_ALE_interaction", "Fig09_PDP_ALE_interaction")


def fig10_ablation(train, test, hist_f, oper_f, weat_f, cat_f, all_f, feats, te, shap_values):
    sets = {
        "Operation": oper_f + cat_f,
        "Weather": weat_f + cat_f,
        "History": hist_f + cat_f,
        "Full": all_f,
    }
    rows = []
    for set_name, features in sets.items():
        numeric = [f for f in features if f not in cat_f]
        cats = [f for f in features if f in cat_f]
        for model_name, estimator in model_library().items():
            pipe = build_pipe(clone(estimator), numeric, cats)
            pipe.fit(train[features], train[TARGET])
            pred = pipe.predict(test[features])
            row = {"Feature set": set_name, "Model": model_name}
            row.update(metrics(test[TARGET].values, pred))
            rows.append(row)
    abl = pd.DataFrame(rows)

    fig = plt.figure(figsize=(15.8, 11.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.54, wspace=0.30)
    order = list(model_library().keys())
    set_order = ["Operation", "Weather", "History", "Full"]
    set_colors = {"Operation": PLOT_COLORS["green"], "Weather": PLOT_COLORS["coral"], "History": PLOT_COLORS["blue"], "Full": PLOT_COLORS["navy"]}

    for ax, metric in zip([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], ["R2", "RMSE"]):
        for si, sname in enumerate(set_order):
            sub = abl[abl["Feature set"] == sname].set_index("Model").loc[order]
            ax.scatter(np.arange(len(order)) + (si - 1.5) * 0.17, sub[metric],
                       s=105, facecolors="white", edgecolors=set_colors[sname], linewidths=2.2, label=sname)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=32, ha="right")
        style_axis(ax, xlabel="", ylabel="R²" if metric == "R2" else metric)
        numeric_ticks(ax, x=False, y=True, n=5)
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.26),
                  handletextpad=0.5, columnspacing=1.0)

    ax = fig.add_subplot(gs[1, 0])
    group_idx = {
        "History": [feats.index(f) for f in hist_f if f in feats],
        "Operation": [feats.index(f) for f in oper_f if f in feats],
        "Weather": [feats.index(f) for f in weat_f if f in feats],
        "Reactor": [feats.index(f) for f in ["reactor_id_enc", "reactor_type_enc"] if f in feats],
    }
    sg = pd.DataFrame({k: np.abs(shap_values[:, idx]).sum(axis=1) if idx else np.zeros(len(te)) for k, idx in group_idx.items()})
    sg["reactor_type"] = te["reactor_type"].astype(str).str.replace("_", " ", regex=False).values
    gp = sg.groupby("reactor_type")[["History", "Operation", "Weather", "Reactor"]].mean()
    gp = gp.div(gp.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(gp))
    for name, color in zip(["History", "Operation", "Weather", "Reactor"], [PLOT_COLORS["blue"], PLOT_COLORS["green"], PLOT_COLORS["coral"], PLOT_COLORS["lavender"]]):
        ax.bar(gp.index, gp[name], bottom=bottom, width=0.34, color=color, edgecolor="white", linewidth=1.0, label=name)
        bottom += gp[name].values
    style_axis(ax, xlabel="", ylabel="SHAP contribution (%)")
    numeric_ticks(ax, x=False, y=True, n=5)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              handletextpad=0.5, columnspacing=0.9)

    ax = fig.add_subplot(gs[1, 1])
    full = abl[abl["Feature set"] == "Full"].set_index("Model")["RMSE"]
    y = np.arange(len(order))
    for si, sname in enumerate(["Operation", "Weather", "History"]):
        sub = abl[abl["Feature set"] == sname].set_index("Model").loc[order]
        delta = (sub["RMSE"] - full.loc[order]) / full.loc[order] * 100
        ax.scatter(np.full(len(order), si), y, s=np.clip(np.abs(delta) * 9, 35, 420),
                   facecolors="white", edgecolors=set_colors[sname], linewidths=1.8)
        for yi, value in zip(y, delta):
            ax.text(si, yi, f"{value:+.0f}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Operation", "Weather", "History"])
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    style_axis(ax, xlabel="Feature set removed from full context", ylabel="")
    save(fig, "Fig10_feature_ablation", "Fig10_feature_ablation")


def write_caption_md(best_model: str, performance: pd.DataFrame, diagnostics: pd.DataFrame | None = None) -> None:
    perf_display = performance.round(3).rename(columns={"R2": "R²"})
    topsis_model = performance.index[0]
    if diagnostics is not None:
        diag_display = diagnostics[["Train_R2", "CV_R2_mean", "CV_R2_min", "Test_R2", "Gap", "RMSE"]].round(3)
        diag_text = diag_display.rename(
            columns={
                "Train_R2": "Train R²",
                "CV_R2_mean": "CV mean R²",
                "CV_R2_min": "CV min R²",
                "Test_R2": "Test R²",
            }
        ).to_string()
    else:
        diag_text = "Not generated."
    text = f"""# 图片说明

本轮图片已按 Nature 风格统一重画：Arial 字体、600 dpi 高清导出、小清新配色、无图内主标题、无 `(a)` 这类子图标签说明。图片标题、子图说明和参考代码来源统一写在这里。

## 参考文献代码使用情况

- 预测-实测图参考：`参考文献/期刊配图：一套代码搞定回归模型性能对比可视化（附代码）` 和 `参考文献/期刊复现：随机森林RF结合相关性热图、预测效果图与特征重要性图实现模型性能评估与特征解读`，吸收了 1:1 线、训练/测试点区分和误差分布面板的结构。
- 模型比较参考：`参考文献/回归模型：基于熵权法与TOPSIS的多子模型DT、RF、XGB、LGBM、SVM等训练与权重赋值优化融合`，用于 Fig 4 的综合排序逻辑。
- Wilcoxon 参考：`参考文献/期刊复现：利用Wilcoxon符号秩检验比较机器学习模型性能差异确定最优模型配图`，用于 Fig 4 的成对显著性热图。
- 相关性参考：`参考文献/期刊配图：从线性到非线性三种相关性分析方法配图形式在特征工程中的系统比较（附代码）` 和 `参考文献/期刊配图：特征分类清晰可见带星号显著性标注的分组相关系数热图`，用于 Fig 3 的 Pearson、Spearman、mutual information、分组色条和显著性星号。
- SHAP/PDP/ALE 参考：`参考文献/期刊复现：基于熵权‑TOPSIS的集成机器学习SHAP‑PDP‑ALE全景解释框架` 和 `参考文献/期刊复现：基于SHAP-Sobol-Monte Carlo的模型可解释性分析框架实现及其扩展`，用于 Fig 8 和 Fig 9。

## 主图说明

**Fig. 1. Data audit and usable modelling records.**  
展示原始合并表、有效目标、非负目标和 3IQR 清洗后的记录变化；同时展示建模变量可用率、建模变量组完整度和各反应器记录数。`CH4_pct`、`y_CH4_STP`、`y_CH4_per_gVS`、pH、TAN、VFA 等 100% 或接近 100% 缺失/无信号变量不再放进主图，因为它们不参与主模型；这些变量更适合在 Methods 的变量筛选或 Limitations 中说明。

**Fig. 2. Target distribution and outlier treatment.**  
展示清洗前后目标分布、四个反应器的目标分布差异、原始与清洗后时序轨迹，以及 `T_mean` 与沼气产量的关系。

**Fig. 3. Correlation and nonlinear association screening.**  
保留 4 个子图，其中 Pearson 和 Spearman 热图改为参考图式样：下三角色块、格内相关系数、显著性星号和右上斜向特征组括号；下方继续展示目标变量 mutual information 排序，以及历史/运行/气象三类特征与目标变量的平均关联强度。变量命名统一为 `T_mean`、`T_max`、`T_min` 等简洁格式。

**Fig. 4. Nine-model comparison and statistical model selection.**  
模型扩展为 9 种：Ridge、KNN、SVR、RF、ExtraTrees、XGBoost、LightGBM、CatBoost、MLP。树模型与 boosting 模型均采用偏保守参数以降低训练-测试差距。包含多指标雷达、训练/验证 R² 分布、Wilcoxon 成对检验和熵权 TOPSIS 排序。TOPSIS 分数是相对综合得分；接近 1 的分数不代表模型完美，而代表相对排序的归一化上界。

**Fig. 5. Predicted versus observed biogas with side-error panels.**  
按你给的参考图重做为 3 x 3 矩阵：主面板为空心圆/三角形预测-实测散点，右侧嵌入误差分布小面板，框内标注训练 R²、测试 R² 和测试 RMSE，R² 保留 3 位小数以避免过度四舍五入。

**Fig. 6. Temporal validation and conformal uncertainty.**  
展示稳健主模型 `{best_model}` 在四个反应器测试期的时序预测，并保留 90% 和 95% conformal prediction 区间。该图用于检验时间外推表现和不确定性，不应解读为高精度预测。

**Fig. 7. Residual and reliability diagnostics.**  
展示残差分布、Bland-Altman 一致性、Y 随机化测试和 Williams 应用域。

**Fig. 8. Global feature importance.**  
展示 SHAP beeswarm、按 FLEX/FIXED DOME 分组的 SHAP bar，以及 permutation importance。

**Fig. 9. Nonlinear effects and interactions.**  
已扩展为 Top-6 特征的 PDP 与 ALE 分开显示，并补充 SHAP dependence、二维 PDP、SHAP interaction heatmap 和 Top feature contribution。ALE 曲线加入了基于局部差分标准误传播的近似 95% 置信带，用于提示小样本下非线性效应的不确定性；这些结果表示模型学习到的关联，不作为因果效应。

**Fig. 10. Feature-set ablation.**  
展示运行、气象、历史和全集特征在 9 模型下的 R²/RMSE 表现，外加特征组 SHAP 贡献和相对 Full 模型的 RMSE 退化气泡图。

## 当前模型结果

测试集 TOPSIS 最高模型为：`{topsis_model}`。考虑训练-测试差距、TimeSeriesSplit 验证稳定性和测试集 RMSE 后，稳健主模型选为：`{best_model}`。

```text
{perf_display.to_string()}
```

## 过拟合与稳健模型筛选

```text
{diag_text}
```

说明：`Train R²` 与 `Test R²` 差距较大时，说明模型仍存在过拟合或时序分布漂移风险。主文采用稳健主模型而不是单纯采用测试集 TOPSIS 第一名，以降低审稿人对过拟合的质疑。
"""
    (BASE / "图片说明.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    print("Loading data...")
    df_raw, df, upper, train, calib, test, hist_f, oper_f, weat_f, cat_f, all_f = load_and_prepare()
    numeric_f = hist_f + oper_f + weat_f
    print("Training 9 models...")
    performance, fold_df, fitted, preds_train, preds_test = evaluate_models(train, test, numeric_f, cat_f, all_f)
    best_model, diagnostics = select_robust_primary_model(performance, fold_df, preds_train, train)
    print(performance[["R2", "RMSE", "TOPSIS"]].round(3))
    print(diagnostics[["Train_R2", "CV_R2_mean", "CV_R2_min", "Test_R2", "Gap", "RMSE"]].round(3))
    print(f"Robust primary model: {best_model}")

    print("Drawing Fig01-Fig10...")
    fig1_data_audit(df_raw, df, upper)
    fig2_target_distribution(df_raw, df, upper)
    fig3_correlation(df, hist_f, oper_f, weat_f)
    fig4_model_comparison(performance, fold_df, preds_test, test)
    fig5_predicted_observed(performance, preds_train, preds_test, train, test)
    fig6_temporal_validation(test, calib, preds_test, fitted, best_model, all_f)
    fig7_residual_reliability(train, test, all_f, fitted, preds_test, best_model)
    shap_model, feats2, tr2, te2, xtr, xte, shap_values = compute_shap_model(df, train, test, hist_f, oper_f, weat_f)
    fig8_shap_importance(shap_model, feats2, te2, xte, shap_values)
    fig9_pdp_ale_interaction(shap_model, feats2, xte, shap_values)
    fig10_ablation(train, test, hist_f, oper_f, weat_f, cat_f, all_f, feats2, te2, shap_values)
    write_caption_md(best_model, performance, diagnostics)
    print("All refined main figures generated.")


if __name__ == "__main__":
    main()
