from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm, t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
TABLE_ROOT = ROOT / "04_Tables"
sys.path.insert(0, str(SCRIPT_PATH.parent))

import generate_pfas_paper_figures as core  # noqa: E402


def save_csv(frame: pd.DataFrame, name: str) -> None:
    path = TABLE_ROOT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {path.relative_to(ROOT)} ({len(frame)} rows)")


def pooled_and_fold_summary(
    name: str,
    y: np.ndarray,
    censored: np.ndarray,
    predictions: np.ndarray,
    fold_rows: list[dict[str, float]],
) -> dict[str, float | str]:
    pooled = core.evaluation(y, predictions, censored)
    fold = pd.DataFrame(fold_rows)
    row: dict[str, float | str] = {"model": name, **pooled}
    for metric in ["r2", "logit_r2", "rmse", "mae", "c_index", "bound_violation"]:
        row[f"fold_mean_{metric}"] = float(fold[metric].mean())
        row[f"fold_sd_{metric}"] = float(fold[metric].std(ddof=1))
    return row


def claim_matched_metric_table() -> None:
    summary = pd.read_csv(TABLE_ROOT / "split_strategy_summary.csv")
    temporal = pd.read_csv(TABLE_ROOT / "publication_year_validation_metrics.csv")
    temporal = temporal.loc[temporal["strategy"].eq("Publication-year holdout")]
    combined = pd.concat([summary, temporal], ignore_index=True)
    combined["rmse_pp"] = combined["rmse"] * 100
    combined["mae_pp"] = combined["mae"] * 100
    keep = [
        "strategy",
        "r2",
        "logit_r2",
        "rmse_pp",
        "mae_pp",
        "c_index",
        "bound_violation",
        "n",
        "n_exact",
    ]
    save_csv(combined[keep], "reviewer_m1_claim_matched_metrics.csv")


def clipping_audit(df: pd.DataFrame) -> None:
    rows = []
    original = df["Rejection"].to_numpy(float)
    for upper in [0.9999, 0.99999, 0.999999]:
        transformed = np.clip(original, 0.500001, upper)
        odds = transformed / (1 - transformed)
        rows.append(
            {
                "upper_clip": upper,
                "rows_changed": int(np.sum(np.abs(transformed - original) > 0)),
                "maximum_rejection_after_clip": float(transformed.max()),
                "maximum_odds_after_clip": float(odds.max()),
                "target_identical_to_0.999999": bool(
                    np.array_equal(
                        transformed,
                        np.clip(original, 0.500001, 0.999999),
                    )
                ),
            }
        )
    save_csv(pd.DataFrame(rows), "reviewer_m5_clipping_sensitivity.csv")


def logistic_normal_mean_from_rejection(
    median_rejection: np.ndarray, sigma: float = 1.2, nodes: int = 40
) -> np.ndarray:
    x, w = hermgauss(nodes)
    median_rejection = np.clip(np.asarray(median_rejection, float), 1e-9, 1 - 1e-9)
    mu = np.log(median_rejection / (1 - median_rejection))
    draws = mu[:, None] + math.sqrt(2) * sigma * x[None, :]
    logistic_draws = 1 / (1 + np.exp(-np.clip(draws, -40, 40)))
    return logistic_draws @ w / math.sqrt(math.pi)


def study_blocked_baselines_and_retransformation(
    df: pd.DataFrame, x_full: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    x_two = df[["volume_Å3", "pwp_LMH/bar"]].to_numpy(float)
    x_empirical = df[["logP", "volume_Å3"]].to_numpy(float)
    matrices = {
        "13-feature censor-aware AFT": x_full,
        "2-feature AFT (PFAS volume + water permeance)": x_two,
        "Empirical AFT (logP + PFAS volume)": x_empirical,
    }
    predictions = {name: np.full(len(df), np.nan) for name in matrices}
    predictions["Training-set exact-row mean"] = np.full(len(df), np.nan)
    mean_rejection_prediction = np.full(len(df), np.nan)
    p90_rejection_prediction = np.full(len(df), np.nan)
    fold_rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in predictions
    }

    splitter = GroupKFold(n_splits=10)
    for fold, (train, test) in enumerate(splitter.split(x_full, y, groups), 1):
        for name, matrix in matrices.items():
            model = core.fit_aft(
                matrix[train],
                df.loc[train, "odds_R"].to_numpy(),
                censored[train],
                seed=700 + fold,
                rounds=145,
            )
            pred = core.predict_aft(model, matrix[test])
            predictions[name][test] = pred
            fold_rows[name].append(
                {"fold": fold, **core.evaluation(y[test], pred, censored[test])}
            )
            if name == "13-feature censor-aware AFT":
                mean_rejection_prediction[test] = logistic_normal_mean_from_rejection(pred)
                median_log_odds = np.log(np.clip(pred, 1e-9, 1 - 1e-9) / (1 - np.clip(pred, 1e-9, 1 - 1e-9)))
                p90_rejection_prediction[test] = 1 / (
                    1 + np.exp(-(median_log_odds + 1.2 * norm.ppf(0.90)))
                )

        exact_train = train[~censored[train]]
        pred_mean = np.repeat(y[exact_train].mean(), len(test))
        predictions["Training-set exact-row mean"][test] = pred_mean
        fold_rows["Training-set exact-row mean"].append(
            {"fold": fold, **core.evaluation(y[test], pred_mean, censored[test])}
        )

    summary = pd.DataFrame(
        [
            pooled_and_fold_summary(name, y, censored, pred, fold_rows[name])
            for name, pred in predictions.items()
        ]
    )
    summary["rmse_pp"] = summary["rmse"] * 100
    summary["mae_pp"] = summary["mae"] * 100
    save_csv(summary, "reviewer_m3_study_blocked_baselines.csv")

    median_pred = predictions["13-feature censor-aware AFT"]
    retransformation_rows = []
    for label, pred in [
        ("AFT P50 (conditional median on rejection scale)", median_pred),
        ("AFT logistic-normal mean on rejection scale", mean_rejection_prediction),
        ("AFT P90 on rejection scale", p90_rejection_prediction),
    ]:
        metrics = core.evaluation(y, pred, censored)
        cens_idx = censored
        shortfall = np.maximum(y[cens_idx] - pred[cens_idx], 0)
        violated = shortfall > 0
        retransformation_rows.append(
            {
                "prediction_summary": label,
                **metrics,
                "rmse_pp": metrics["rmse"] * 100,
                "mae_pp": metrics["mae"] * 100,
                "mean_shortfall_among_violations_pp": (
                    float(shortfall[violated].mean() * 100) if violated.any() else 0.0
                ),
                "mean_shortfall_all_censored_pp": float(shortfall.mean() * 100),
                "median_shortfall_among_violations_pp": (
                    float(np.median(shortfall[violated]) * 100) if violated.any() else 0.0
                ),
            }
        )
    save_csv(
        pd.DataFrame(retransformation_rows),
        "reviewer_m6_m7_retransformation_and_bounds.csv",
    )
    predictions["AFT logistic-normal mean on rejection scale"] = mean_rejection_prediction
    predictions["AFT P90 on rejection scale"] = p90_rejection_prediction
    return summary, predictions


def size_matched_temporal_control(
    df: pd.DataFrame, x_full: np.ndarray, repeats: int = 40
) -> None:
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    years = df["ref_year"].to_numpy(float)
    temporal_train = np.flatnonzero(years <= 2023)
    temporal_test = np.flatnonzero(years > 2023)
    n_train_rows = len(temporal_train)
    n_train_studies = len(np.unique(groups[temporal_train]))
    n_test_studies = len(np.unique(groups[temporal_test]))
    temporal_model = core.fit_aft(
        x_full[temporal_train],
        df.loc[temporal_train, "odds_R"].to_numpy(),
        censored[temporal_train],
        seed=1600,
        rounds=180,
    )
    temporal_pred = core.predict_aft(temporal_model, x_full[temporal_test])
    temporal_metrics = core.evaluation(
        y[temporal_test], temporal_pred, censored[temporal_test]
    )

    rng = np.random.default_rng(20260814)
    unique_groups = np.unique(groups)
    records = []
    attempts = 0
    while len(records) < repeats and attempts < repeats * 200:
        attempts += 1
        selected_groups = rng.choice(
            unique_groups, size=n_train_studies, replace=False
        )
        selected_mask = np.isin(groups, selected_groups)
        available = np.flatnonzero(selected_mask)
        if len(available) < n_train_rows:
            continue
        mandatory = np.array(
            [rng.choice(np.flatnonzero(groups == group)) for group in selected_groups]
        )
        remaining_pool = np.setdiff1d(available, mandatory, assume_unique=False)
        extra = rng.choice(
            remaining_pool,
            size=n_train_rows - len(mandatory),
            replace=False,
        )
        train = np.concatenate([mandatory, extra])
        test = np.flatnonzero(~selected_mask)
        if len(np.unique(groups[test])) != n_test_studies:
            continue
        model = core.fit_aft(
            x_full[train],
            df.loc[train, "odds_R"].to_numpy(),
            censored[train],
            seed=3600 + len(records),
            rounds=180,
        )
        pred = core.predict_aft(model, x_full[test])
        records.append(
            {
                "repeat": len(records) + 1,
                "train_rows": len(train),
                "train_studies": len(np.unique(groups[train])),
                "test_rows": len(test),
                "test_studies": len(np.unique(groups[test])),
                **core.evaluation(y[test], pred, censored[test]),
            }
        )
    if len(records) != repeats:
        raise RuntimeError(f"Only generated {len(records)} of {repeats} matched controls")
    frame = pd.DataFrame(records)
    frame["temporal_r2"] = temporal_metrics["r2"]
    frame["temporal_rmse"] = temporal_metrics["rmse"]
    frame["temporal_c_index"] = temporal_metrics["c_index"]
    frame["temporal_r2_percentile_among_controls"] = float(
        (np.sum(frame["r2"] <= temporal_metrics["r2"]) + 1) / (len(frame) + 1)
    )
    frame["temporal_rmse_upper_tail_p"] = float(
        (np.sum(frame["rmse"] >= temporal_metrics["rmse"]) + 1) / (len(frame) + 1)
    )
    save_csv(frame, "reviewer_m4_size_matched_temporal_control.csv")


def within_study_ols(df: pd.DataFrame, bootstrap_repeats: int = 2000) -> None:
    rng = np.random.default_rng(20260814)
    treatments = ["mon_cat_mM", "di_cat_mM", "doc_mg/L"]
    labels = {
        "mon_cat_mM": "Monovalent cations",
        "di_cat_mM": "Divalent cations",
        "doc_mg/L": "DOC",
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
    median_logit = float(base["logit_R"].median())
    result_rows = []
    ridge_rows = []

    for treatment in treatments:
        work = base[["ref_id", "logit_R", treatment] + covariates].copy()
        work = work.dropna(subset=[treatment])
        work[treatment] = np.log1p(
            pd.to_numeric(work[treatment], errors="coerce").clip(lower=0)
        )
        for col in ["flux_LMH", "pfas_feed_conc_µg/L"]:
            work[col] = np.log1p(
                pd.to_numeric(work[col], errors="coerce").clip(lower=0)
            )
        variable = work.groupby("ref_id")[treatment].nunique(dropna=True)
        variable = variable[variable > 1].index
        work = work.loc[work["ref_id"].isin(variable)].copy()
        work[covariates] = work[covariates].fillna(
            work[covariates].median(numeric_only=True)
        )
        model_columns = [treatment] + covariates
        demeaned = work[model_columns + ["logit_R"]] - work.groupby("ref_id")[
            model_columns + ["logit_R"]
        ].transform("mean")
        scaler = StandardScaler()
        x = scaler.fit_transform(demeaned[model_columns].to_numpy(float))
        y = demeaned["logit_R"].to_numpy(float)
        groups = work["ref_id"].astype(str).to_numpy()

        fitted = sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
            use_t=True,
        )
        beta = float(fitted.params[0])
        cluster_se = float(fitted.bse[0])
        crit = float(t.ppf(0.975, max(1, len(np.unique(groups)) - 1)))
        cluster_low = beta - crit * cluster_se
        cluster_high = beta + crit * cluster_se

        unique_groups, group_index = np.unique(groups, return_inverse=True)
        signs = rng.choice(
            [-1.0, 1.0], size=(len(unique_groups), bootstrap_repeats)
        )
        resid_draws = fitted.resid[:, None] * signs[group_index]
        projection = np.linalg.pinv(x.T @ x) @ x.T
        boot_beta = beta + (projection @ resid_draws)[0]
        wild_low, wild_high = np.quantile(boot_beta, [0.025, 0.975])

        def pp(effect: float) -> float:
            return float(
                (
                    core.logistic(median_logit + effect)
                    - core.logistic(median_logit)
                )
                * 100
            )

        result_rows.append(
            {
                "treatment": treatment,
                "label": labels[treatment],
                "rows": len(work),
                "clusters": len(unique_groups),
                "effect_logit_per_within_sd": beta,
                "effect_pp_at_median": pp(beta),
                "cluster_robust_se": cluster_se,
                "cluster_robust_ci_low_pp": pp(cluster_low),
                "cluster_robust_ci_high_pp": pp(cluster_high),
                "wild_cluster_ci_low_pp": pp(float(wild_low)),
                "wild_cluster_ci_high_pp": pp(float(wild_high)),
                "wild_cluster_replicates": bootstrap_repeats,
            }
        )
        for alpha in [0, 1, 4, 8, 16]:
            ridge = Ridge(alpha=alpha, fit_intercept=False).fit(x, y)
            effect = float(ridge.coef_[0])
            ridge_rows.append(
                {
                    "treatment": treatment,
                    "label": labels[treatment],
                    "penalty": alpha,
                    "effect_logit_per_within_sd": effect,
                    "effect_pp_at_median": pp(effect),
                    "rows": len(work),
                    "clusters": len(unique_groups),
                }
            )
    save_csv(pd.DataFrame(result_rows), "reviewer_m8_within_study_ols.csv")
    save_csv(pd.DataFrame(ridge_rows), "reviewer_m8_ridge_sensitivity.csv")


def duplicate_response_audit(df: pd.DataFrame, x_full: np.ndarray) -> None:
    imputer = SimpleImputer(strategy="median")
    scaled = StandardScaler().fit_transform(imputer.fit_transform(x_full))
    clusters = core.KMeans(n_clusters=10, random_state=42, n_init=20).fit_predict(scaled)
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    rows = []
    for strategy, stratify in [
        ("Cluster-stratified", clusters),
        ("Row-random", None),
    ]:
        for rep in range(12):
            train, test = train_test_split(
                np.arange(len(df)),
                test_size=0.15,
                random_state=110 + rep,
                stratify=stratify,
            )
            split_imputer = SimpleImputer(strategy="median")
            x_train = split_imputer.fit_transform(x_full[train])
            x_test = split_imputer.transform(x_full[test])
            train_map: dict[tuple[float, ...], list[int]] = {}
            for local_idx, values in enumerate(x_train):
                key = tuple(np.round(values, 12))
                train_map.setdefault(key, []).append(train[local_idx])
            duplicate = 0
            same_response = 0
            same_response_and_censoring = 0
            for local_idx, values in enumerate(x_test):
                key = tuple(np.round(values, 12))
                matches = train_map.get(key, [])
                if not matches:
                    continue
                duplicate += 1
                test_index = test[local_idx]
                exact_response_match = np.isclose(
                    y[matches], y[test_index], rtol=0, atol=1e-12
                )
                if exact_response_match.any():
                    same_response += 1
                if np.any(
                    exact_response_match
                    & (censored[matches] == censored[test_index])
                ):
                    same_response_and_censoring += 1
            rows.append(
                {
                    "strategy": strategy,
                    "repeat": rep + 1,
                    "test_rows": len(test),
                    "duplicate_feature_rows": duplicate,
                    "duplicate_feature_fraction": duplicate / len(test),
                    "duplicate_with_identical_response": same_response,
                    "identical_response_fraction_among_duplicates": (
                        same_response / duplicate if duplicate else np.nan
                    ),
                    "duplicate_with_identical_response_and_censoring": same_response_and_censoring,
                    "identical_response_and_censoring_fraction_among_duplicates": (
                        same_response_and_censoring / duplicate if duplicate else np.nan
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, "reviewer_m10_duplicate_response_audit.csv")
    summary = (
        frame.groupby("strategy", as_index=False)
        .agg(
            repeats=("repeat", "size"),
            mean_duplicate_feature_fraction=("duplicate_feature_fraction", "mean"),
            sd_duplicate_feature_fraction=("duplicate_feature_fraction", "std"),
            mean_identical_response_fraction_among_duplicates=(
                "identical_response_fraction_among_duplicates",
                "mean",
            ),
            mean_identical_response_and_censoring_fraction_among_duplicates=(
                "identical_response_and_censoring_fraction_among_duplicates",
                "mean",
            ),
        )
    )
    save_csv(summary, "reviewer_m10_duplicate_response_summary.csv")


def conformal_unit_audit(df: pd.DataFrame, x_full: np.ndarray) -> None:
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    rng = np.random.default_rng(20260717)
    rows = []
    outer = GroupKFold(n_splits=10)
    for fold, (train_outer, test_idx) in enumerate(outer.split(x_full, y, groups), 1):
        train_groups = np.unique(groups[train_outer])
        rng.shuffle(train_groups)
        n_cal = max(4, int(np.ceil(len(train_groups) * 0.22)))
        cal_groups = set(train_groups[:n_cal])
        cal_idx = train_outer[
            np.array([group in cal_groups for group in groups[train_outer]])
        ]
        rows.append(
            {
                "fold": fold,
                "outer_train_studies": len(train_groups),
                "calibration_studies": len(cal_groups),
                "calibration_rows": len(cal_idx),
                "calibration_exact_rows": int((~censored[cal_idx]).sum()),
                "test_studies": len(np.unique(groups[test_idx])),
                "test_rows": len(test_idx),
                "current_quantile_unit": "exact calibration rows",
                "exchangeable_claim_unit": "studies",
                "bootstrap_models_current_figure": 8,
            }
        )
    save_csv(pd.DataFrame(rows), "reviewer_m9_conformal_unit_audit.csv")


def main() -> None:
    _, df = core.load_data()
    x_full = core.feature_matrix(df)
    claim_matched_metric_table()
    clipping_audit(df)
    study_blocked_baselines_and_retransformation(df, x_full)
    size_matched_temporal_control(df, x_full)
    within_study_ols(df)
    duplicate_response_audit(df, x_full)
    conformal_unit_audit(df, x_full)


if __name__ == "__main__":
    main()
