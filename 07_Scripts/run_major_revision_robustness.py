from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
TABLE_ROOT = ROOT / "04_Tables" / "Revision"
WORK_ROOT = ROOT / "99_Working_Files" / "major_revision"
SEED = 20260831
REPEATED_SPLITS = 50
PAIRED_BOOTSTRAPS = 5000
WEBB_REPLICATES = 9999

sys.path.insert(0, str(SCRIPT_PATH.parent))
import audit_internal_high_accuracy_claim as audit  # noqa: E402
import generate_pfas_paper_figures as core  # noqa: E402


CONSERVATIVE = next(item for item in audit.CANDIDATES if item["name"] == "conservative")
HIGH_CAPACITY = next(item for item in audit.CANDIDATES if item["name"] == "d6_c1")


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    path = TABLE_ROOT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {path.relative_to(ROOT)} ({len(frame)} rows)", flush=True)
    return path


def save_json(payload: object, name: str) -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)
    return path


def extended_metrics(df: pd.DataFrame, idx: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = df.loc[idx, "Rejection"].to_numpy(float)
    censored = df.loc[idx, "is_censored"].to_numpy(bool)
    exact = ~censored
    if exact.sum() >= 2:
        y_exact = y[exact]
        p_exact = np.asarray(pred, float)[exact]
        sd = float(np.std(y_exact, ddof=1))
        rmse = float(mean_squared_error(y_exact, p_exact) ** 0.5)
        mae = float(mean_absolute_error(y_exact, p_exact))
        r2 = float(r2_score(y_exact, p_exact))
        ly = np.log(np.clip(y_exact, 1e-6, 1 - 1e-6) / (1 - np.clip(y_exact, 1e-6, 1 - 1e-6)))
        lp = np.log(np.clip(p_exact, 1e-6, 1 - 1e-6) / (1 - np.clip(p_exact, 1e-6, 1 - 1e-6)))
        logit_r2 = float(r2_score(ly, lp))
        nrmse = float(rmse / sd) if sd > 0 else np.nan
    else:
        sd = rmse = mae = r2 = logit_r2 = nrmse = np.nan
    bound_violation = (
        float(np.mean(np.asarray(pred)[censored] < y[censored])) if censored.any() else np.nan
    )
    return {
        "r2": r2,
        "logit_r2": logit_r2,
        "rmse": rmse,
        "rmse_pp": rmse * 100,
        "mae": mae,
        "mae_pp": mae * 100,
        "exact_outcome_sd": sd,
        "exact_outcome_sd_pp": sd * 100,
        "nrmse_by_test_sd": nrmse,
        "c_index": core.censored_concordance(y, pred, censored),
        "bound_violation": bound_violation,
        "n": int(len(idx)),
        "n_exact": int(exact.sum()),
        "n_censored": int(censored.sum()),
        "studies": int(df.loc[idx, "ref_id"].astype(str).nunique()),
    }


def within_study_familiar_split(
    indices: np.ndarray, groups: np.ndarray, seed: int, fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Hold rows out within development studies while retaining every study in training."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for group in np.unique(groups[indices]):
        rows = indices[groups[indices] == group].copy()
        rng.shuffle(rows)
        if len(rows) < 2:
            train_parts.append(rows)
            continue
        n_test = min(len(rows) - 1, max(1, int(round(len(rows) * fraction))))
        test_parts.append(rows[:n_test])
        train_parts.append(rows[n_test:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))


def tune_grouped(
    x: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    candidates: list[dict],
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    groups = df["ref_id"].astype(str).to_numpy()
    rows: list[dict] = []
    splitter = GroupKFold(n_splits=min(3, np.unique(groups[train_idx]).size))
    for candidate_no, cfg in enumerate(candidates, 1):
        fold_scores: list[float] = []
        for fold, (tr_local, va_local) in enumerate(
            splitter.split(x[train_idx], groups=groups[train_idx]), 1
        ):
            tr = train_idx[tr_local]
            va = train_idx[va_local]
            model = audit.fit("aft", x, df, tr, cfg, seed + candidate_no * 100 + fold)
            pred = audit.predict("aft", model, x[va])
            fold_scores.append(float(extended_metrics(df, va, pred)["logit_r2"]))
        rows.append(
            {
                "candidate": cfg["name"],
                "mean_grouped_inner_logit_r2": float(np.nanmean(fold_scores)),
                "sd_grouped_inner_logit_r2": float(np.nanstd(fold_scores, ddof=1)),
                "fold_scores": ";".join(f"{v:.8g}" for v in fold_scores),
                **cfg,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        "mean_grouped_inner_logit_r2", ascending=False
    )
    winner_name = str(frame.iloc[0]["candidate"])
    winner = next(item for item in candidates if item["name"] == winner_name)
    return frame, winner


def composition_row(df: pd.DataFrame, idx: np.ndarray, label: str) -> dict:
    part = df.loc[idx]
    exact = part.loc[~part["is_censored"], "Rejection"].astype(float)
    row: dict[str, object] = {
        "evaluation_regime": label,
        "rows": len(part),
        "studies": part["ref_id"].astype(str).nunique(),
        "exact_rows": len(exact),
        "censored_share": float(part["is_censored"].mean()),
        "exact_rejection_mean": float(exact.mean()),
        "exact_rejection_sd": float(exact.std(ddof=1)),
        "exact_rejection_q25": float(exact.quantile(0.25)),
        "exact_rejection_median": float(exact.median()),
        "exact_rejection_q75": float(exact.quantile(0.75)),
        "unique_membranes": part["mem"].nunique(dropna=True),
        "unique_pfas": part["PFAS"].nunique(dropna=True),
        "mean_missing_predictor_fraction": float(part[core.FEATURES].isna().mean(axis=1).mean()),
    }
    feed = part["feed_group"].fillna("Unreported").value_counts(normalize=True)
    for name in ["Pure water", "Synthetic", "Real water", "Unreported"]:
        row[f"feed_share_{name.lower().replace(' ', '_')}"] = float(feed.get(name, 0.0))
    return row


def fixed_split_architecture_audit(df: pd.DataFrame, x: np.ndarray) -> dict:
    groups = df["ref_id"].astype(str).to_numpy()
    all_idx = np.arange(len(df))
    development, unseen = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=audit.SEED).split(
            x, groups=groups
        )
    )
    # Preserve the exact published familiar/unseen partition for the requested
    # 2 x 2 architecture audit; only the inner tuning splitter is changed.
    familiar_train, familiar_test = train_test_split(
        development,
        test_size=0.15,
        random_state=audit.SEED + 20,
        shuffle=True,
    )

    tuning, selected = tune_grouped(x, df, familiar_train, audit.CANDIDATES, SEED + 1000)
    save_csv(tuning, "Table_R3_01_grouped_inner_cv_12_candidates.csv")

    rows: list[dict] = []
    architectures = [CONSERVATIVE, HIGH_CAPACITY]
    if selected["name"] not in {item["name"] for item in architectures}:
        architectures.append(selected)
    for cfg in architectures:
        model = audit.fit("aft", x, df, familiar_train, cfg, audit.SEED + 900)
        for regime, idx in [("familiar_rows", familiar_test), ("unseen_studies", unseen)]:
            pred = audit.predict("aft", model, x[idx])
            rows.append(
                {
                    "architecture": cfg["name"],
                    "evaluation_regime": regime,
                    "same_fitted_model": True,
                    **extended_metrics(df, idx, pred),
                }
            )
    refit = audit.fit("aft", x, df, development, selected, audit.SEED + 901)
    refit_pred = audit.predict("aft", refit, x[unseen])
    rows.append(
        {
            "architecture": selected["name"],
            "evaluation_regime": "unseen_studies_after_frozen_spec_refit",
            "same_fitted_model": False,
            **extended_metrics(df, unseen, refit_pred),
        }
    )
    save_csv(pd.DataFrame(rows), "Table_R3_02_architecture_by_generalization_2x2.csv")
    save_csv(
        pd.DataFrame(
            [
                composition_row(df, familiar_test, "familiar_rows"),
                composition_row(df, unseen, "unseen_studies"),
            ]
        ),
        "Table_R3_03_familiar_unseen_composition.csv",
    )

    allocation = []
    sets = {
        "development_familiar_train": familiar_train,
        "development_familiar_test": familiar_test,
        "unseen_study_test": unseen,
    }
    for regime, idx in sets.items():
        allocation.extend(
            {
                "row_index": int(i),
                "study": str(groups[i]),
                "regime": regime,
                "seed": SEED,
            }
            for i in idx
        )
    save_csv(pd.DataFrame(allocation), "Table_R3_04_fixed_split_membership.csv")
    return {
        "selected": selected,
        "development": development,
        "unseen": unseen,
        "familiar_train": familiar_train,
        "familiar_test": familiar_test,
    }


def repeated_study_partitions(df: pd.DataFrame, x: np.ndarray) -> None:
    groups = df["ref_id"].astype(str).to_numpy()
    metrics_rows: list[dict] = []
    selection_rows: list[dict] = []
    membership_rows: list[dict] = []
    candidates = [CONSERVATIVE, HIGH_CAPACITY]
    for repeat in range(REPEATED_SPLITS):
        repeat_seed = SEED + 10000 + repeat
        development, unseen = next(
            GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=repeat_seed).split(
                x, groups=groups
            )
        )
        familiar_train, familiar_test = within_study_familiar_split(
            development, groups, repeat_seed + 1
        )
        tuning, selected = tune_grouped(
            x, df, familiar_train, candidates, repeat_seed + 10
        )
        for _, row in tuning.iterrows():
            selection_rows.append(
                {
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "candidate": row["candidate"],
                    "mean_grouped_inner_logit_r2": row["mean_grouped_inner_logit_r2"],
                    "selected": row["candidate"] == selected["name"],
                }
            )
        model = audit.fit("aft", x, df, familiar_train, selected, repeat_seed + 20)
        for regime, idx in [("familiar_rows", familiar_test), ("unseen_studies", unseen)]:
            pred = audit.predict("aft", model, x[idx])
            metrics_rows.append(
                {
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "selected_architecture": selected["name"],
                    "evaluation_regime": regime,
                    **extended_metrics(df, idx, pred),
                }
            )
        for study in np.unique(groups):
            if study in set(groups[unseen]):
                regime = "unseen_study_test"
            else:
                regime = "development"
            membership_rows.append(
                {
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "study": study,
                    "regime": regime,
                }
            )
        print(
            f"repeated study partitions {repeat + 1}/{REPEATED_SPLITS}: {selected['name']}",
            flush=True,
        )
    metrics = pd.DataFrame(metrics_rows)
    save_csv(metrics, "Table_R3_05_repeated_50_study_partition_metrics.csv")
    save_csv(pd.DataFrame(selection_rows), "Table_R3_06_repeated_50_inner_selection.csv")
    save_csv(pd.DataFrame(membership_rows), "Table_R3_07_repeated_50_study_membership.csv")
    summary = (
        metrics.groupby("evaluation_regime", as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            median_r2=("r2", "median"),
            q025_r2=("r2", lambda s: s.quantile(0.025)),
            q975_r2=("r2", lambda s: s.quantile(0.975)),
            median_rmse_pp=("rmse_pp", "median"),
            q025_rmse_pp=("rmse_pp", lambda s: s.quantile(0.025)),
            q975_rmse_pp=("rmse_pp", lambda s: s.quantile(0.975)),
            median_nrmse=("nrmse_by_test_sd", "median"),
            median_c_index=("c_index", "median"),
        )
    )
    save_csv(summary, "Table_R3_08_repeated_50_summary.csv")


def grouped_predictions(
    df: pd.DataFrame, x: np.ndarray, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    y = df["Rejection"].to_numpy(float)
    groups = df["ref_id"].astype(str).to_numpy()
    censored = df["is_censored"].to_numpy(bool)
    pred = np.full(len(df), np.nan)
    baseline = np.full(len(df), np.nan)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=10).split(x, y, groups), 1):
        model = audit.fit("aft", x, df, train, cfg, seed + fold)
        pred[test] = audit.predict("aft", model, x[test])
        exact_train = train[~censored[train]]
        baseline[test] = float(y[exact_train].mean())
    return pred, baseline


def threshold_sensitivity(raw: pd.DataFrame, main: pd.DataFrame, x_main: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nfro = raw.loc[raw["mem_class"].astype(str).str.upper().isin(["NF", "RO"])].copy()
    nfro = nfro.reset_index(drop=True)
    r = nfro["Rejection"].clip(1e-6, 0.999999)
    nfro["logit_R"] = np.log(r / (1 - r))
    nfro["odds_R"] = r / (1 - r)
    x_all = core.feature_matrix(nfro)

    rows: list[dict] = []
    # The manuscript's primary study-grouped comparison used 145 boosting rounds.
    # Freeze that setting for the threshold and operational-baseline sensitivities.
    primary_cfg = {**CONSERVATIVE, "rounds": 145}
    # Reuse the declared fold seeds (700 + fold) from the reviewer-check script.
    main_pred, baseline_pred = grouped_predictions(main, x_main, primary_cfg, 700)
    all_pred, _ = grouped_predictions(nfro, x_all, primary_cfg, 700)
    rows.append(
        {
            "analysis_set": "Primary R >= 0.5",
            "minimum_rejection": float(main["Rejection"].min()),
            **extended_metrics(main, np.arange(len(main)), main_pred),
        }
    )
    rows.append(
        {
            "analysis_set": "Sensitivity without R >= 0.5",
            "minimum_rejection": float(nfro["Rejection"].min()),
            **extended_metrics(nfro, np.arange(len(nfro)), all_pred),
        }
    )
    save_csv(pd.DataFrame(rows), "Table_R3_09_no_R50_threshold_sensitivity.csv")
    return main_pred, baseline_pred


def pair_concordance(y1, p1, c1, y2, p2, c2) -> tuple[float, float]:
    yi = np.asarray(y1)[:, None]
    yj = np.asarray(y2)[None, :]
    pi = np.asarray(p1)[:, None]
    pj = np.asarray(p2)[None, :]
    ci = np.asarray(c1)[:, None]
    cj = np.asarray(c2)[None, :]
    exact = (~ci) & (~cj) & (np.abs(yi - yj) > 1e-12)
    i_lower = ci & (~cj) & (yi > yj)
    j_lower = (~ci) & cj & (yj > yi)
    comparable = exact | i_lower | j_lower
    direction = np.sign(yi - yj)
    pred_direction = np.sign(pi - pj)
    numerator = np.sum((direction == pred_direction) & comparable) + 0.5 * np.sum(
        (pred_direction == 0) & comparable
    )
    return float(numerator), float(np.sum(comparable))


def concordance_quadratic(
    df: pd.DataFrame, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    labels = sorted(np.unique(groups).tolist())
    numerator = np.zeros((len(labels), len(labels)))
    denominator = np.zeros_like(numerator)
    for i, left in enumerate(labels):
        li = np.flatnonzero(groups == left)
        if len(li) > 1:
            a, b = pair_concordance(
                y[li[:-1]], prediction[li[:-1]], censored[li[:-1]],
                y[li[1:]], prediction[li[1:]], censored[li[1:]],
            )
            # The rectangular call above is not an i<j mask; recompute the diagonal exactly.
            a = b = 0.0
            for row_pos in range(len(li) - 1):
                aa, bb = pair_concordance(
                    y[li[row_pos : row_pos + 1]],
                    prediction[li[row_pos : row_pos + 1]],
                    censored[li[row_pos : row_pos + 1]],
                    y[li[row_pos + 1 :]],
                    prediction[li[row_pos + 1 :]],
                    censored[li[row_pos + 1 :]],
                )
                a += aa
                b += bb
            numerator[i, i] = a
            denominator[i, i] = b
        for j in range(i + 1, len(labels)):
            right = labels[j]
            rj = np.flatnonzero(groups == right)
            a, b = pair_concordance(
                y[li], prediction[li], censored[li], y[rj], prediction[rj], censored[rj]
            )
            numerator[i, j] = numerator[j, i] = a / 2
            denominator[i, j] = denominator[j, i] = b / 2
    return numerator, denominator, labels


def paired_study_bootstrap(
    df: pd.DataFrame, model_pred: np.ndarray, baseline_pred: np.ndarray
) -> None:
    y = df["Rejection"].to_numpy(float)
    exact = ~df["is_censored"].to_numpy(bool)
    group_values = df["ref_id"].astype(str).to_numpy()
    labels = sorted(np.unique(group_values).tolist())
    group_to_no = {group: number for number, group in enumerate(labels)}
    group_no = np.array([group_to_no[group] for group in group_values])
    rng = np.random.default_rng(SEED + 40000)
    samples = rng.integers(0, len(labels), size=(PAIRED_BOOTSTRAPS, len(labels)))
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=len(labels)), 1, samples
    ).astype(float)
    weights = counts[:, group_no]
    weights[:, ~exact] = 0
    n_exact = weights.sum(axis=1)
    weighted_mean = (weights @ y) / n_exact
    sst = np.sum(weights * (y[None, :] - weighted_mean[:, None]) ** 2, axis=1)

    def scalar_metrics(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = prediction - y
        sse = weights @ (error**2)
        sae = weights @ np.abs(error)
        return 1 - sse / sst, np.sqrt(sse / n_exact), sae / n_exact

    model_r2, model_rmse, model_mae = scalar_metrics(model_pred)
    base_r2, base_rmse, base_mae = scalar_metrics(baseline_pred)

    model_num, model_den, model_labels = concordance_quadratic(df, model_pred)
    base_num, base_den, base_labels = concordance_quadratic(df, baseline_pred)
    assert model_labels == labels == base_labels
    # Primary ranking estimand: equal-study mean within-study concordance.
    # Cross-study pairs can create spurious ranking for fold-specific constants.
    comparable = np.diag(model_den) > 0
    model_by_study = np.diag(model_num)[comparable] / np.diag(model_den)[comparable]
    base_by_study = np.diag(base_num)[comparable] / np.diag(base_den)[comparable]
    assert np.allclose(base_by_study, 0.5)
    scored_counts = counts[:, comparable]
    model_c = (scored_counts @ model_by_study) / scored_counts.sum(axis=1)
    base_c = (scored_counts @ base_by_study) / scored_counts.sum(axis=1)

    deltas = {
        "R2 model minus baseline": model_r2 - base_r2,
        "RMSE (pp) model minus baseline": 100 * (model_rmse - base_rmse),
        "MAE (pp) model minus baseline": 100 * (model_mae - base_mae),
        "C-index within-study model minus baseline": model_c - base_c,
    }
    observed_model = extended_metrics(df, np.arange(len(df)), model_pred)
    observed_base = extended_metrics(df, np.arange(len(df)), baseline_pred)
    observed = {
        "R2 model minus baseline": observed_model["r2"] - observed_base["r2"],
        "RMSE (pp) model minus baseline": 100 * (observed_model["rmse"] - observed_base["rmse"]),
        "MAE (pp) model minus baseline": 100 * (observed_model["mae"] - observed_base["mae"]),
        "C-index within-study model minus baseline": model_by_study.mean() - base_by_study.mean(),
    }
    rows = []
    for metric, values in deltas.items():
        improvement = values > 0 if metric.startswith(("R2", "C-index")) else values < 0
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        rows.append(
            {
                "contrast": metric,
                "observed_difference": observed[metric],
                "bootstrap_q025": low,
                "bootstrap_median": median,
                "bootstrap_q975": high,
                "fraction_bootstrap_draws_improving": float(np.mean(improvement)),
                "bootstrap_replicates": PAIRED_BOOTSTRAPS,
                "resampling_unit": "study",
            }
        )
    save_csv(pd.DataFrame(rows), "Table_R3_10_paired_study_bootstrap_model_vs_mean.csv")
    save_csv(pd.DataFrame(deltas), "Table_R3_10b_paired_study_bootstrap_draws.csv")
    save_csv(pd.DataFrame(counts, columns=labels), "Table_R3_10c_bootstrap_study_multiplicities.csv")


def prepare_within_work(df: pd.DataFrame, treatment: str, exact_only: bool) -> pd.DataFrame:
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
    base = df.loc[~df["is_censored"]].copy() if exact_only else df.copy()
    work = base[["ref_id", "logit_R", treatment] + covariates].copy()
    work = work.dropna(subset=[treatment])
    work[treatment] = np.log1p(pd.to_numeric(work[treatment], errors="coerce").clip(lower=0))
    for col in ["flux_LMH", "pfas_feed_conc_µg/L"]:
        work[col] = np.log1p(pd.to_numeric(work[col], errors="coerce").clip(lower=0))
    variable = work.groupby("ref_id")[treatment].nunique(dropna=True)
    variable = variable[variable > 1].index
    return work.loc[work["ref_id"].isin(variable)].copy()


def fit_within_ols(
    work: pd.DataFrame, treatment: str, rng: np.random.Generator, webb: bool = True
) -> dict[str, float | int]:
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
    work = work.copy()
    work[covariates] = work[covariates].fillna(work[covariates].median(numeric_only=True))
    model_columns = [treatment] + covariates
    demeaned = work[model_columns + ["logit_R"]] - work.groupby("ref_id")[
        model_columns + ["logit_R"]
    ].transform("mean")
    x = StandardScaler().fit_transform(demeaned[model_columns].to_numpy(float))
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
    result: dict[str, float | int] = {
        "rows": len(work),
        "clusters": len(np.unique(groups)),
        "effect_logit_per_within_sd": beta,
        "cluster_robust_se": cluster_se,
        "cluster_ci_low": beta - crit * cluster_se,
        "cluster_ci_high": beta + crit * cluster_se,
    }
    if webb:
        unique_groups, group_index = np.unique(groups, return_inverse=True)
        webb_values = np.array(
            [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)]
        )
        weights = rng.choice(webb_values, size=(len(unique_groups), WEBB_REPLICATES))
        residual_draws = fitted.resid[:, None] * weights[group_index]
        projection = np.linalg.pinv(x.T @ x) @ x.T
        bootstrap_beta = beta + (projection @ residual_draws)[0]
        low, high = np.quantile(bootstrap_beta, [0.025, 0.975])
        result.update(
            {
                "webb_ci_low": float(low),
                "webb_ci_high": float(high),
                "webb_two_sided_p": float(
                    2
                    * min(
                        (np.sum(bootstrap_beta <= 0) + 1) / (len(bootstrap_beta) + 1),
                        (np.sum(bootstrap_beta >= 0) + 1) / (len(bootstrap_beta) + 1),
                    )
                ),
                "webb_replicates": WEBB_REPLICATES,
            }
        )
    return result


def eligibility_webb_and_common_support(df: pd.DataFrame) -> None:
    treatments = ["mon_cat_mM", "di_cat_mM", "doc_mg/L"]
    labels = {
        "mon_cat_mM": "Monovalent cations",
        "di_cat_mM": "Divalent cations",
        "doc_mg/L": "DOC",
    }
    eligibility_rows: list[dict] = []
    webb_rows: list[dict] = []
    rng = np.random.default_rng(SEED + 50000)
    for treatment in treatments:
        all_work = prepare_within_work(df, treatment, exact_only=False)
        exact_work = prepare_within_work(df, treatment, exact_only=True)
        all_studies = set(all_work["ref_id"].astype(str).unique())
        exact_studies = set(exact_work["ref_id"].astype(str).unique())
        dropped = sorted(all_studies - exact_studies)
        reason = ""
        if dropped:
            reason = (
                "After restriction to exact outcomes, the remaining exact rows in the dropped "
                "study/studies have no within-study variation in the exposure."
            )
        eligibility_rows.append(
            {
                "exposure": labels[treatment],
                "all_outcome_varying_studies": len(all_studies),
                "exact_outcome_varying_studies": len(exact_studies),
                "exact_outcome_ols_rows": len(exact_work),
                "dropped_studies": "; ".join(dropped),
                "reason": reason,
            }
        )
        webb_rows.append(
            {
                "exposure": labels[treatment],
                "analysis": "all eligible exact rows",
                **fit_within_ols(exact_work, treatment, rng, webb=True),
            }
        )

    overlap = pd.read_csv(ROOT / "04_Tables" / "monovalent_overlap_diagnostic.csv")
    mon_work = prepare_within_work(df, "mon_cat_mM", exact_only=True)
    aligned = (
        len(overlap) == len(mon_work)
        and overlap["ref_id"].astype(str).tolist() == mon_work["ref_id"].astype(str).tolist()
        and np.allclose(
            overlap["mon_cat_mM"].to_numpy(float),
            np.expm1(mon_work["mon_cat_mM"].to_numpy(float)),
            equal_nan=True,
        )
    )
    if not aligned:
        raise RuntimeError("Stored common-support rows do not align with the exact-row OLS work set")
    in_support = overlap["in_common_support"].astype(bool).to_numpy()
    subset = mon_work.loc[in_support].copy()
    webb_rows.append(
        {
            "exposure": "Monovalent cations",
            "analysis": "common-support subset sensitivity",
            **fit_within_ols(subset, "mon_cat_mM", rng, webb=True),
        }
    )
    summary = pd.read_csv(ROOT / "04_Tables" / "monovalent_overlap_summary.csv").iloc[0]
    common = pd.DataFrame(
        [
            {
                "primary_ols_rows": int(summary["rows"]),
                "primary_ols_studies": int(summary["studies"]),
                "common_support_retained_rows": int(in_support.sum()),
                "common_support_excluded_rows": int((~in_support).sum()),
                "common_support_retained_fraction": float(in_support.mean()),
                "common_support_low": float(summary["common_support_low"]),
                "common_support_high": float(summary["common_support_high"]),
                "interpretation": "The 878-row OLS is primary; common support is a diagnostic and an 823-row sensitivity subset.",
            }
        ]
    )
    save_csv(pd.DataFrame(eligibility_rows), "Table_R1_02_exposure_eligibility_flow.csv")
    save_csv(pd.DataFrame(webb_rows), "Table_R3_11_webb_wild_cluster_and_common_support.csv")
    save_csv(common, "Table_R1_03_common_support_denominators.csv")


def finite_group_quantile(scores: np.ndarray, nominal: float) -> tuple[float, int]:
    ordered = np.sort(np.asarray(scores, float))
    rank = int(math.ceil((len(ordered) + 1) * nominal))
    if rank > len(ordered):
        return np.inf, rank
    return float(ordered[rank - 1]), rank


def study_unit_conformal(df: pd.DataFrame, x: np.ndarray) -> None:
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED + 60000)
    fold_rows: list[dict] = []
    study_rows: list[dict] = []
    for fold, (outer_train, test) in enumerate(
        GroupKFold(n_splits=10).split(x, y, groups), 1
    ):
        train_groups = np.unique(groups[outer_train])
        shuffled = train_groups.copy()
        rng.shuffle(shuffled)
        n_cal = max(20, int(math.ceil(len(shuffled) * 0.40)))
        n_cal = min(n_cal, len(shuffled) - 3)
        cal_groups = set(shuffled[:n_cal])
        cal = outer_train[np.isin(groups[outer_train], list(cal_groups))]
        fit = outer_train[~np.isin(groups[outer_train], list(cal_groups))]
        unique_fit = np.unique(groups[fit])
        cal_predictions: list[np.ndarray] = []
        test_predictions: list[np.ndarray] = []
        for bootstrap in range(12):
            sampled = rng.choice(unique_fit, size=len(unique_fit), replace=True)
            boot_idx = np.concatenate([fit[groups[fit] == group] for group in sampled])
            model = audit.fit(
                "aft", x, df, boot_idx, CONSERVATIVE, SEED + 61000 + fold * 100 + bootstrap
            )
            cal_predictions.append(audit.predict("aft", model, x[cal]))
            test_predictions.append(audit.predict("aft", model, x[test]))
        cal_predictions_array = np.asarray(cal_predictions)
        test_predictions_array = np.asarray(test_predictions)
        cal_mean = cal_predictions_array.mean(axis=0)
        test_mean = test_predictions_array.mean(axis=0)
        cal_scale = np.maximum(cal_predictions_array.std(axis=0, ddof=1), 0.008)
        test_scale = np.maximum(test_predictions_array.std(axis=0, ddof=1), 0.008)

        study_scores = []
        for group in sorted(cal_groups):
            local = np.flatnonzero(groups[cal] == group)
            local = local[~censored[cal][local]]
            if len(local):
                scores = np.abs(y[cal][local] - cal_mean[local]) / cal_scale[local]
                study_scores.append(float(np.max(scores)))
        scores_array = np.asarray(study_scores)
        for nominal in (0.80, 0.90, 0.95):
            q, rank = finite_group_quantile(scores_array, nominal)
            lo = np.clip(test_mean - q * test_scale, 0.0, 1.0)
            hi = np.clip(test_mean + q * test_scale, 0.0, 1.0)
            exact_test = ~censored[test]
            row_coverage = float(
                np.mean((y[test][exact_test] >= lo[exact_test]) & (y[test][exact_test] <= hi[exact_test]))
            )
            censored_compatibility = (
                float(np.mean(hi[censored[test]] >= y[test][censored[test]]))
                if censored[test].any()
                else np.nan
            )
            study_all_covered = []
            for group in np.unique(groups[test]):
                local = np.flatnonzero(groups[test] == group)
                exact_local = local[~censored[test][local]]
                if len(exact_local):
                    covered = bool(
                        np.all(
                            (y[test][exact_local] >= lo[exact_local])
                            & (y[test][exact_local] <= hi[exact_local])
                        )
                    )
                    study_all_covered.append(covered)
                    study_rows.append(
                        {
                            "fold": fold,
                            "nominal": nominal,
                            "study": group,
                            "exact_rows": len(exact_local),
                            "all_exact_rows_covered": covered,
                        }
                    )
            fold_rows.append(
                {
                    "fold": fold,
                    "nominal": nominal,
                    "outer_test_studies": len(np.unique(groups[test])),
                    "calibration_studies_selected": len(cal_groups),
                    "calibration_studies_with_exact_scores": len(scores_array),
                    "calibration_score_unit": "maximum standardized exact-row residual per study",
                    "order_statistic_rank": rank,
                    "quantile": q,
                    "row_exact_coverage": row_coverage,
                    "study_all_exact_rows_coverage": float(np.mean(study_all_covered)),
                    "censored_lower_bound_compatibility": censored_compatibility,
                    "mean_interval_width_pp": float(np.mean((hi - lo) * 100)),
                    "scale_floor": 0.008,
                    "bootstrap_models": 12,
                }
            )
        print(f"study-unit conformal fold {fold}/10", flush=True)
    save_csv(pd.DataFrame(fold_rows), "Table_R3_12_study_unit_conformal_folds.csv")
    save_csv(pd.DataFrame(study_rows), "Table_R3_13_study_unit_conformal_study_coverage.csv")


def balance_record(df: pd.DataFrame, idx: np.ndarray, design: str, repeat: int, partition: str) -> dict:
    part = df.loc[idx]
    exact = part.loc[~part["is_censored"], "Rejection"].astype(float)
    feed = part["feed_group"].fillna("Unreported").value_counts(normalize=True)
    return {
        "design": design,
        "repeat": repeat,
        "partition": partition,
        "rows": len(part),
        "studies": part["ref_id"].astype(str).nunique(),
        "unique_membranes": part["mem"].nunique(dropna=True),
        "unique_pfas": part["PFAS"].nunique(dropna=True),
        "censored_share": float(part["is_censored"].mean()),
        "mean_missing_predictor_fraction": float(part[core.FEATURES].isna().mean(axis=1).mean()),
        "exact_outcome_mean": float(exact.mean()),
        "exact_outcome_sd": float(exact.std(ddof=1)),
        "pure_water_share": float(feed.get("Pure water", 0.0)),
        "synthetic_share": float(feed.get("Synthetic", 0.0)),
        "real_water_share": float(feed.get("Real water", 0.0)),
        "unreported_feed_share": float(feed.get("Unreported", 0.0)),
    }


def temporal_control_disclosure(df: pd.DataFrame) -> None:
    groups = df["ref_id"].astype(str).to_numpy()
    years = df["ref_year"].to_numpy(float)
    temporal_train = np.flatnonzero(years <= 2023)
    temporal_test = np.flatnonzero(years > 2023)
    train_rows = len(temporal_train)
    train_studies = len(np.unique(groups[temporal_train]))
    test_studies = len(np.unique(groups[temporal_test]))
    rows = [
        balance_record(df, temporal_train, "publication_year_holdout", 0, "train"),
        balance_record(df, temporal_test, "publication_year_holdout", 0, "test"),
    ]
    membership = []
    for partition, idx in [("train", temporal_train), ("test", temporal_test)]:
        membership.extend(
            {
                "design": "publication_year_holdout",
                "repeat": 0,
                "partition": partition,
                "study": study,
            }
            for study in sorted(np.unique(groups[idx]).tolist())
        )

    rng = np.random.default_rng(20260814)
    unique_groups = np.unique(groups)
    generated = 0
    attempts = 0
    while generated < 40 and attempts < 8000:
        attempts += 1
        selected_groups = rng.choice(unique_groups, size=train_studies, replace=False)
        selected_mask = np.isin(groups, selected_groups)
        available = np.flatnonzero(selected_mask)
        if len(available) < train_rows:
            continue
        mandatory = np.array(
            [rng.choice(np.flatnonzero(groups == group)) for group in selected_groups]
        )
        remaining = np.setdiff1d(available, mandatory, assume_unique=False)
        extra = rng.choice(remaining, size=train_rows - len(mandatory), replace=False)
        train = np.concatenate([mandatory, extra])
        test = np.flatnonzero(~selected_mask)
        if len(np.unique(groups[test])) != test_studies:
            continue
        generated += 1
        rows.extend(
            [
                balance_record(df, train, "size_matched_group_control", generated, "train"),
                balance_record(df, test, "size_matched_group_control", generated, "test"),
            ]
        )
        for partition, idx in [("train", train), ("test", test)]:
            membership.extend(
                {
                    "design": "size_matched_group_control",
                    "repeat": generated,
                    "partition": partition,
                    "study": study,
                }
                for study in sorted(np.unique(groups[idx]).tolist())
            )
    if generated != 40:
        raise RuntimeError(f"Could reproduce only {generated}/40 matched temporal controls")
    save_csv(pd.DataFrame(rows), "Table_R2_06_temporal_control_balance_disclosure.csv")
    save_csv(pd.DataFrame(membership), "Table_R2_07_temporal_control_study_membership.csv")
    save_json(
        {
            "matching_variables": ["training row count", "training study count", "held-out study count"],
            "not_matched": [
                "PFAS identity/count",
                "membrane identity/count",
                "water-matrix composition",
                "censoring share",
                "predictor missingness",
                "outcome distribution",
            ],
            "temporal_train_rows": train_rows,
            "temporal_train_studies": train_studies,
            "temporal_test_studies": test_studies,
            "control_repeats": generated,
            "seed": 20260814,
        },
        "temporal_control_matching_definition.json",
    )


def applicability_domain_disclosure(df: pd.DataFrame, x: np.ndarray) -> None:
    groups = df["ref_id"].astype(str).to_numpy()
    exact = ~df["is_censored"].to_numpy(bool)
    distance = np.full(len(df), np.nan)
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x[train])
        x_test = imputer.transform(x[test])
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        k = min(8, len(train))
        distance[test] = (
            NearestNeighbors(n_neighbors=k)
            .fit(x_train)
            .kneighbors(x_test, return_distance=True)[0]
            .mean(axis=1)
        )
    median_distance = float(np.nanmedian(distance))
    relative = distance / median_distance
    threshold = float(np.quantile(relative[exact], 0.90))
    save_csv(
        pd.DataFrame(
            {
                "row_index": np.arange(len(df)),
                "study": groups,
                "is_exact": exact,
                "mean_k8_euclidean_distance": distance,
                "relative_distance": relative,
                "outside_exact_p90_domain": relative > threshold,
            }
        ),
        "Table_R2_04_applicability_domain_row_distances.csv",
    )
    save_json(
        {
            "feature_columns": core.FEATURES,
            "preprocessing": "Median imputation fitted on each leave-one-study-out training fold, followed by StandardScaler fitted on the same training fold.",
            "distance": "For each held-out row, arithmetic mean Euclidean distance to the k=8 nearest training rows in the fold-standardized 13-feature space.",
            "relative_distance": "Raw mean kNN distance divided by the pooled median raw distance across all held-out rows.",
            "threshold": "P90 of relative distance among exact-outcome held-out rows, pooled across leave-one-study-out folds.",
            "median_raw_distance": median_distance,
            "exact_p90_relative_threshold": threshold,
            "outside_rule": "relative_distance > exact_p90_relative_threshold",
        },
        "applicability_domain_definition.json",
    )


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    raw, df = core.load_data()
    x = core.feature_matrix(df)
    print(f"primary analysis: rows={len(df)}, studies={df.ref_id.nunique()}", flush=True)

    fixed = fixed_split_architecture_audit(df, x)
    repeated_study_partitions(df, x)
    model_pred, baseline_pred = threshold_sensitivity(raw, df, x)
    paired_study_bootstrap(df, model_pred, baseline_pred)
    eligibility_webb_and_common_support(df)
    study_unit_conformal(df, x)
    temporal_control_disclosure(df)
    applicability_domain_disclosure(df, x)

    manifest = {
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "seed": SEED,
        "primary_rows": len(df),
        "primary_studies": int(df.ref_id.nunique()),
        "selected_fixed_split_candidate": fixed["selected"]["name"],
        "repeated_study_partitions": REPEATED_SPLITS,
        "paired_study_bootstraps": PAIRED_BOOTSTRAPS,
        "webb_six_point_replicates": WEBB_REPLICATES,
        "notes": [
            "The one-off full search compares all 12 declared candidates under grouped inner cross-validation.",
            "The 50 repeated study partitions prespecify a two-candidate choice (conservative versus d6_c1) to bound computational multiplicity.",
            "No source file outside the isolated revision package is written by this script.",
        ],
    }
    save_json(manifest, "major_revision_analysis_manifest.json")
    print("major revision robustness analysis complete", flush=True)


if __name__ == "__main__":
    main()
