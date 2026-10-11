"""Refined supplementary figures with the same Nature-style visual system."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from ml_main_figure_templates import (  # noqa: E402
    MODEL_COLORS,
    TARGET,
    configure_style,
    build_pipe,
    load_and_prepare,
    model_library,
    numeric_ticks,
    save,
    style_axis,
)
from scientific_palettes import PLOT_COLORS  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_rmse(pipe, train, all_features, dates, splitter):
    scores = []
    for ti, vi in splitter.split(dates):
        tr_dates = set(dates[ti])
        va_dates = set(dates[vi])
        tr = train[train["date"].isin(tr_dates)]
        va = train[train["date"].isin(va_dates)]
        pipe.fit(tr[all_features], tr[TARGET])
        scores.append(math.sqrt(mean_squared_error(va[TARGET], pipe.predict(va[all_features]))))
    return float(np.mean(scores))


def fig_s1_learning_curves(train, numeric_f, cat_f, all_f):
    models = model_library()
    dates = np.array(sorted(train["date"].unique()))
    fractions = [0.24, 0.38, 0.52, 0.66, 0.82, 1.0]
    fig, axes = plt.subplots(3, 3, figsize=(15.6, 12.4))
    fig.subplots_adjust(hspace=0.32, wspace=0.28, top=0.93)

    for ax, (name, estimator), color in zip(axes.flat, models.items(), MODEL_COLORS):
        rows = []
        for frac in fractions:
            n_dates = max(24, int(len(dates) * frac))
            sub_dates = dates[:n_dates]
            splitter = TimeSeriesSplit(n_splits=4)
            tr_scores, va_scores = [], []
            for ti, vi in splitter.split(sub_dates):
                tr = train[train["date"].isin(set(sub_dates[ti]))]
                va = train[train["date"].isin(set(sub_dates[vi]))]
                pipe = build_pipe(clone(estimator), numeric_f, cat_f)
                pipe.fit(tr[all_f], tr[TARGET])
                tr_scores.append(r2_score(tr[TARGET], pipe.predict(tr[all_f])))
                va_scores.append(r2_score(va[TARGET], pipe.predict(va[all_f])))
            rows.append({"n": len(train[train["date"].isin(set(sub_dates))]), "train": np.mean(tr_scores), "val": np.mean(va_scores)})
        lc = pd.DataFrame(rows)
        ax.plot(lc["n"], lc["train"], color=color, lw=2.0, marker="o", ms=5, mfc="white", mew=1.2,
                label="Training")
        ax.plot(lc["n"], lc["val"], color=color, lw=2.0, marker="s", ms=5, mfc="white", mew=1.2, ls="--",
                label="Validation")
        ax.fill_between(lc["n"], lc["train"], lc["val"], color=color, alpha=0.10, linewidth=0)
        ax.text(0.06, 0.86, name, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=color,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78})
        ax.set_ylim(-0.35, 1.08)
        ax.axhline(0, color="#D8DDE3", lw=1.0)
        style_axis(ax, xlabel="Training samples", ylabel="R²")
        numeric_ticks(ax, x=True, y=True, n=5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), handlelength=2.8, columnspacing=2.0)
    save(fig, "FigS1_learning_curves", "FigS1_learning_curves")


def fig_s2_bayesian(train, all_f, numeric_f, cat_f):
    import lightgbm as lgb
    import xgboost as xgb

    dates = np.array(sorted(train["date"].unique()))
    splitter = TimeSeriesSplit(n_splits=4)
    studies = {}

    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        }
        return cv_rmse(build_pipe(lgb.LGBMRegressor(**params, random_state=42, verbose=-1), numeric_f, cat_f), train, all_f, dates, splitter)

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        }
        return cv_rmse(build_pipe(xgb.XGBRegressor(**params, random_state=42, verbosity=0), numeric_f, cat_f), train, all_f, dates, splitter)

    for name, objective in [("LightGBM", objective_lgb), ("XGBoost", objective_xgb)]:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        studies[name] = study

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.6))
    fig.subplots_adjust(hspace=0.32, wspace=0.30)
    colors = {"LightGBM": PLOT_COLORS["green"], "XGBoost": PLOT_COLORS["coral"]}
    for col, (name, study) in enumerate(studies.items()):
        values = np.array([t.value for t in study.trials])
        best = np.minimum.accumulate(values)
        trials = np.arange(1, len(values) + 1)
        ax = axes[0, col]
        ax.scatter(trials, values, s=42, facecolors="none", edgecolors=colors[name], linewidths=1.3, alpha=0.82)
        ax.plot(trials, best, color=PLOT_COLORS["ink"], lw=2.0)
        ax.axhline(study.best_value, color=PLOT_COLORS["red"], lw=1.4, ls="--")
        ax.text(0.04, 0.88, name, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=colors[name],
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78})
        style_axis(ax, xlabel="Trial", ylabel="CV RMSE")
        numeric_ticks(ax, x=True, y=True, n=5)

        ax = axes[1, col]
        try:
            imp = optuna.importance.get_param_importances(study)
            ser = pd.Series(imp).sort_values()
            ax.barh(list(ser.index), ser.values, color=colors[name], alpha=0.82, edgecolor="white")
        except Exception:
            ax.text(0.5, 0.5, "Importance unavailable", transform=ax.transAxes, ha="center", va="center", fontweight="bold")
        style_axis(ax, xlabel="Hyperparameter importance", ylabel="")
        numeric_ticks(ax, x=True, y=False, n=5)
    save(fig, "FigS2_bayesian_optimisation", "FigS2_bayesian_optimisation")


def residualise(x_ctrl, y, model=None):
    if model is None:
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_ctrl)
    model.fit(x_scaled, y)
    return y - model.predict(x_scaled)


def fig_s3_causal(df, hist_f, oper_f, weat_f):
    treatments = {
        "T_mean": "daily_mean_air_temp",
        "Solar": "daily_solar",
        "Precip": "daily_precip",
        "VPD": "daily_vpd",
        "Wind": "daily_wind",
    }
    controls = [f for f in hist_f + oper_f if f in df.columns]
    dml = df.dropna(subset=controls + list(treatments.values()) + [TARGET]).copy()
    dml["reactor_type_plot"] = dml["reactor_type"].astype(str).str.replace("_", " ", regex=False)

    x_ctrl = dml[controls].fillna(dml[controls].median())
    y_res = residualise(x_ctrl, dml[TARGET].values)
    rows = []
    for label, col in treatments.items():
        t_res = residualise(x_ctrl, dml[col].values)
        slope, _, _, pval, se = stats.linregress(t_res, y_res)
        rows.append({"Treatment": label, "ATE": slope, "CI": 1.96 * se, "p": pval})
    ate = pd.DataFrame(rows).sort_values("ATE")

    cate_rows = []
    for rtype in ["FLEX", "FIXED DOME"]:
        sub = dml[dml["reactor_type_plot"] == rtype]
        x_sub = sub[controls].fillna(sub[controls].median())
        y_sub = residualise(x_sub, sub[TARGET].values)
        for label, col in treatments.items():
            t_sub = residualise(x_sub, sub[col].values)
            slope, _, _, pval, se = stats.linregress(t_sub, y_sub)
            cate_rows.append({"Reactor": rtype, "Treatment": label, "CATE": slope, "CI": 1.96 * se, "p": pval})
    cate = pd.DataFrame(cate_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.2))
    fig.subplots_adjust(wspace=0.36)

    ax = axes[0]
    ypos = np.arange(len(ate))
    colors = [PLOT_COLORS["red"] if p < 0.05 else PLOT_COLORS["blue"] for p in ate["p"]]
    ax.barh(ypos, ate["ATE"], xerr=ate["CI"], color=colors, alpha=0.72, edgecolor="white")
    ax.axvline(0, color=PLOT_COLORS["ink"], lw=1.3, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ate["Treatment"])
    style_axis(ax, xlabel="ATE (mL per unit)", ylabel="")
    numeric_ticks(ax, x=True, y=False, n=5)

    ax = axes[1]
    x = np.arange(len(treatments))
    for offset, (rtype, color) in zip([-0.18, 0.18], [("FLEX", PLOT_COLORS["green"]), ("FIXED DOME", PLOT_COLORS["coral"])]):
        sub = cate[cate["Reactor"] == rtype].set_index("Treatment").loc[list(treatments)]
        ax.errorbar(x + offset, sub["CATE"], yerr=sub["CI"], fmt="o", ms=7, mfc="white", mec=color,
                    mew=1.7, ecolor=color, elinewidth=1.2, capsize=3, label=rtype)
    ax.axhline(0, color=PLOT_COLORS["ink"], lw=1.2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(list(treatments), rotation=30, ha="right")
    ax.legend(frameon=False)
    style_axis(ax, xlabel="", ylabel="CATE (mL per unit)")
    numeric_ticks(ax, x=False, y=True, n=5)

    ax = axes[2]
    ax.scatter(ate["ATE"], -np.log10(np.clip(ate["p"], 1e-10, None)), s=92,
               facecolors="white", edgecolors=PLOT_COLORS["navy"], linewidths=1.7)
    ax.axvline(0, color=PLOT_COLORS["ink"], lw=1.2, ls="--")
    ax.axhline(-np.log10(0.05), color=PLOT_COLORS["red"], lw=1.2, ls=":")
    for _, row in ate.iterrows():
        ax.text(row["ATE"], -np.log10(max(row["p"], 1e-10)), f" {row['Treatment']}", fontsize=10, fontweight="bold")
    style_axis(ax, xlabel="ATE (mL per unit)", ylabel="-log10(p)")
    numeric_ticks(ax, x=True, y=True, n=5)
    save(fig, "FigS3_residualized_weather", "FigS3_residualized_weather")


def main():
    configure_style()
    df_raw, df, upper, train, calib, test, hist_f, oper_f, weat_f, cat_f, all_f = load_and_prepare()
    numeric_f = hist_f + oper_f + weat_f
    fig_s1_learning_curves(train, numeric_f, cat_f, all_f)
    fig_s2_bayesian(train, all_f, numeric_f, cat_f)
    fig_s3_causal(df, hist_f, oper_f, weat_f)
    caption_path = BASE / "图片说明.md"
    caption = caption_path.read_text(encoding="utf-8")
    caption = caption.split("\n## 附录图说明")[0].rstrip()
    caption += (
        "\n\n## 附录图说明\n\n"
        "**Fig. S1. Learning curves.** 实线表示训练集 R²，虚线表示时间序列交叉验证 R²。验证 R² 出现负值说明该折模型劣于仅用验证集均值的基线，是小样本时序外推不稳定的表现，而不是代码错误。\n\n"
        "**Fig. S2. Bayesian optimisation.** LightGBM 与 XGBoost 的 Optuna TPE 搜索轨迹和超参数重要性。\n\n"
        "**Fig. S3. Residualized weather association.** 用残差化回归评估气象变量与沼气产量的条件关联，仅作为敏感性补充，不作因果结论。\n"
    )
    caption_path.write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    main()
