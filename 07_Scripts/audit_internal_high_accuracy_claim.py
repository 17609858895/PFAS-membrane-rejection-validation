from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold, train_test_split

import generate_pfas_paper_figures as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "99_Working_Files" / "high_accuracy_claim_audit"
SEED = 20260816


CANDIDATES = [
    dict(name="conservative", depth=3, child=14, eta=0.045, rounds=130, lam=7.0, alpha=0.15, sub=0.82, col=0.82, scale=1.2),
    dict(name="d3_c1", depth=3, child=1, eta=0.035, rounds=420, lam=2.0, alpha=0.0, sub=0.90, col=0.90, scale=1.0),
    dict(name="d4_c1", depth=4, child=1, eta=0.030, rounds=520, lam=2.0, alpha=0.0, sub=0.90, col=0.90, scale=1.0),
    dict(name="d4_c3", depth=4, child=3, eta=0.050, rounds=360, lam=3.0, alpha=0.0, sub=0.90, col=0.90, scale=0.8),
    dict(name="d5_c1", depth=5, child=1, eta=0.025, rounds=650, lam=2.0, alpha=0.0, sub=0.90, col=0.90, scale=0.8),
    dict(name="d5_c3", depth=5, child=3, eta=0.045, rounds=420, lam=3.0, alpha=0.0, sub=0.90, col=0.90, scale=1.0),
    dict(name="d6_c1", depth=6, child=1, eta=0.025, rounds=720, lam=2.0, alpha=0.0, sub=0.90, col=0.90, scale=0.8),
    dict(name="d6_c3", depth=6, child=3, eta=0.040, rounds=480, lam=3.0, alpha=0.0, sub=0.95, col=0.95, scale=1.0),
    dict(name="d7_c1", depth=7, child=1, eta=0.020, rounds=820, lam=2.0, alpha=0.0, sub=0.95, col=0.95, scale=0.8),
    dict(name="d4_regular", depth=4, child=5, eta=0.035, rounds=500, lam=7.0, alpha=0.1, sub=0.85, col=0.85, scale=1.2),
    dict(name="d5_regular", depth=5, child=5, eta=0.030, rounds=600, lam=7.0, alpha=0.1, sub=0.85, col=0.85, scale=1.2),
    dict(name="full_sample", depth=5, child=1, eta=0.030, rounds=600, lam=1.0, alpha=0.0, sub=1.0, col=1.0, scale=0.8),
]


def aft_fit(x, odds, censored, cfg, seed):
    matrix = xgb.DMatrix(x, missing=np.nan)
    lower = np.asarray(odds, dtype=float)
    upper = np.where(np.asarray(censored, dtype=bool), np.inf, lower)
    matrix.set_float_info("label_lower_bound", lower)
    matrix.set_float_info("label_upper_bound", upper)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": cfg["scale"],
        "tree_method": "hist",
        "max_depth": cfg["depth"],
        "min_child_weight": cfg["child"],
        "eta": cfg["eta"],
        "subsample": cfg["sub"],
        "colsample_bytree": cfg["col"],
        "lambda": cfg["lam"],
        "alpha": cfg["alpha"],
        "seed": seed,
        "nthread": 4,
    }
    return xgb.train(params, matrix, num_boost_round=cfg["rounds"], verbose_eval=False)


def conventional_fit(x, logit_y, cfg, seed):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=cfg["rounds"],
        max_depth=cfg["depth"],
        min_child_weight=cfg["child"],
        learning_rate=cfg["eta"],
        subsample=cfg["sub"],
        colsample_bytree=cfg["col"],
        reg_lambda=cfg["lam"],
        reg_alpha=cfg["alpha"],
        random_state=seed,
        n_jobs=4,
    )
    model.fit(x, logit_y)
    return model


def predict(route, model, x):
    if route == "aft":
        odds = np.clip(model.predict(xgb.DMatrix(x, missing=np.nan)), 1e-8, 1e8)
        return odds / (1.0 + odds)
    logit = model.predict(x)
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))


def fit(route, x, df, idx, cfg, seed):
    if route == "aft":
        return aft_fit(
            x[idx],
            df.loc[idx, "odds_R"].to_numpy(),
            df.loc[idx, "is_censored"].to_numpy(),
            cfg,
            seed,
        )
    return conventional_fit(x[idx], df.loc[idx, "logit_R"].to_numpy(), cfg, seed)


def tune(route, x, df, train_idx):
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    inner = KFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    records = []
    for number, cfg in enumerate(CANDIDATES, 1):
        fold_scores = []
        for fold, (inner_train, inner_val) in enumerate(inner.split(train_idx), 1):
            tr = train_idx[inner_train]
            va = train_idx[inner_val]
            model = fit(route, x, df, tr, cfg, SEED + number * 10 + fold)
            pred = predict(route, model, x[va])
            metrics = core.evaluation(y[va], pred, censored[va])
            fold_scores.append(metrics["logit_r2"])
        records.append(
            {
                "route": route,
                "candidate": cfg["name"],
                "mean_inner_logit_r2": float(np.mean(fold_scores)),
                "sd_inner_logit_r2": float(np.std(fold_scores, ddof=1)),
                **cfg,
            }
        )
        print(
            f"{route} {number:02d}/{len(CANDIDATES)} {cfg['name']}: "
            f"inner logit R2={np.mean(fold_scores):.3f}",
            flush=True,
        )
    frame = pd.DataFrame(records).sort_values("mean_inner_logit_r2", ascending=False)
    return frame, frame.iloc[0].to_dict()


def evaluate_frozen(route, x, df, conventional_train, conventional_test, cfg):
    y = df["Rejection"].to_numpy(float)
    censored = df["is_censored"].to_numpy(bool)
    model = fit(route, x, df, conventional_train, cfg, SEED + 500)
    test_pred = predict(route, model, x[conventional_test])
    conventional = core.evaluation(
        y[conventional_test], test_pred, censored[conventional_test]
    )

    groups = df["ref_id"].astype(str).to_numpy()
    group_pred = np.full(len(df), np.nan)
    fold_rows = []
    for fold, (tr, te) in enumerate(
        GroupKFold(n_splits=10).split(x, y, groups), 1
    ):
        fold_model = fit(route, x, df, tr, cfg, SEED + 600 + fold)
        pred = predict(route, fold_model, x[te])
        group_pred[te] = pred
        fold_rows.append(
            {"route": route, "fold": fold, **core.evaluation(y[te], pred, censored[te])}
        )
    grouped = core.evaluation(y, group_pred, censored)
    return conventional, grouped, pd.DataFrame(fold_rows), test_pred, group_pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, df = core.load_data()
    x = core.feature_matrix(df)
    all_idx = np.arange(len(df))
    conventional_train, conventional_test = train_test_split(
        all_idx,
        test_size=0.15,
        random_state=SEED,
        shuffle=True,
    )

    summary = []
    all_tuning = []
    all_folds = []
    for route in ("aft", "conventional_lower_bound_as_exact"):
        route_key = "aft" if route == "aft" else "conventional"
        tuning, best = tune(route_key, x, df, conventional_train)
        tuning["reported_route"] = route
        all_tuning.append(tuning)
        conventional, grouped, folds, _, _ = evaluate_frozen(
            route_key,
            x,
            df,
            conventional_train,
            conventional_test,
            best,
        )
        folds["reported_route"] = route
        all_folds.append(folds)
        summary.append(
            {
                "route": route,
                "best_candidate": best["candidate"],
                "inner_cv_logit_r2": best["mean_inner_logit_r2"],
                **{f"conventional_{k}": v for k, v in conventional.items()},
                **{f"study_blocked_{k}": v for k, v in grouped.items()},
            }
        )
        print(
            f"FROZEN {route}: conventional logit R2={conventional['logit_r2']:.3f}, "
            f"study-blocked logit R2={grouped['logit_r2']:.3f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(summary)
    pd.concat(all_tuning, ignore_index=True).to_csv(
        OUT / "parameter_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_folds, ignore_index=True).to_csv(
        OUT / "study_blocked_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary_frame.to_csv(OUT / "claim_audit_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "claim_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
