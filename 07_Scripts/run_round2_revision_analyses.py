"""Second-round reviewer analyses for the PFAS membrane-rejection validation study.

Every result is written to ``04_Tables/Round2`` (or ``$PFAS_OUTPUT_ROOT/04_Tables/Round2``)
and carries a dataset label:

* ``P2424`` - primary analysis set: NF/RO membranes, rejection >= 0.50 (2,424 rows, 61 studies).
* ``A2353`` - author-rule sensitivity set: rejection >= 0.50 after the source author's stated
  join omissions (2,353 rows, 60 studies, includes 26 UF-labelled rows).
* ``U2487`` - unfiltered NF/RO set without the rejection threshold (2,487 rows, 61 studies).

Run ``python 07_Scripts/run_round2_revision_analyses.py`` from the repository root; use
``--only`` to run selected sections.  All splits, seeds and hyperparameters are declared here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, stats
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
OUTPUT_ROOT = Path(os.environ.get("PFAS_OUTPUT_ROOT", ROOT))
OUT = OUTPUT_ROOT / "04_Tables" / "Round2"
FIG_OUT = OUTPUT_ROOT / "03_Figures" / "Round2"

sys.path.insert(0, str(SCRIPT_PATH.parent))
import audit_internal_high_accuracy_claim as audit  # noqa: E402
import generate_pfas_paper_figures as core  # noqa: E402

SEED = 20261025
CONSERVATIVE = next(item for item in audit.CANDIDATES if item["name"] == "conservative")
AUTHOR_OMITTED_MEMBRANES = ["NF270-PEI_Jeong", "NF270-PAA_Jeong"]
AUTHOR_OMITTED_REFERENCES = ["Ma J., 2022", "Ma et al, 2024 (2)"]
N_JOBS = int(os.environ.get("PFAS_N_JOBS", "8"))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_csv(frame: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {path} ({len(frame)} rows)", flush=True)
    return path


def save_json(payload: object, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Data sets with persistent labels and stable source-row identifiers
# ---------------------------------------------------------------------------

def row_hash(row: pd.Series) -> str:
    text = "|".join("" if pd.isna(v) else str(v) for v in row.to_numpy())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def prepare(frame: pd.DataFrame, low_clip: float) -> pd.DataFrame:
    """Mirror ``core.load_data`` derived columns for any eligible subset."""
    out = frame.reset_index(drop=True).copy()
    r = out["Rejection"].clip(low_clip, 0.999999)
    out["logit_R"] = np.log(r / (1 - r))
    out["odds_R"] = r / (1 - r)
    out["feed_group"] = (
        out["feed_type"].fillna("Unreported")
        .replace({"Raw": "Real water", "Lab": "Synthetic", "Pure water": "Pure water"})
    )
    out["chain_group"] = pd.cut(
        out["carbons"], bins=[-np.inf, 3, 5, 7, np.inf], labels=["C≤3", "C4–5", "C6–7", "C≥8"]
    )
    return out


def load_datasets() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if core.JSON_PATH.exists():
        raise RuntimeError("Remove the cached JSON workbook copy; Round-2 analyses read the source xlsx only.")
    raw, primary = core.load_data()
    raw = raw.copy()
    source_cols = [c for c in raw.columns if c != "is_censored"]
    raw["source_row"] = np.arange(len(raw)) + 2  # Excel row number (header is row 1)
    raw["row_uid"] = [f"W{r:04d}-{row_hash(raw.loc[i, source_cols])}" for i, r in zip(raw.index, raw["source_row"])]
    nfro = raw["mem_class"].astype(str).str.upper().isin(["NF", "RO"])
    r50 = raw["Rejection"].ge(0.5)
    author_drop = raw["mem"].isin(AUTHOR_OMITTED_MEMBRANES) | raw["ref_id"].isin(AUTHOR_OMITTED_REFERENCES)
    raw["in_P2424"] = r50 & nfro
    raw["in_A2353"] = r50 & ~author_drop
    raw["in_U2487"] = nfro
    p = prepare(raw.loc[raw["in_P2424"]], 0.500001)
    # Alignment check against the published loader.
    assert len(p) == len(primary) == 2424
    assert np.allclose(p["logit_R"].to_numpy(), primary["logit_R"].to_numpy())
    assert (p["ref_id"].astype(str).to_numpy() == primary["ref_id"].astype(str).to_numpy()).all()
    a = prepare(raw.loc[raw["in_A2353"]], 0.500001)
    u = prepare(raw.loc[raw["in_U2487"]], 1e-6)
    assert len(a) == 2353 and len(u) == 2487
    return raw, {"P2424": p, "A2353": a, "U2487": u}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def within_study_c(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, int]:
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    g = df["ref_id"].astype(str).to_numpy()
    values = []
    for study in np.unique(g):
        idx = np.flatnonzero(g == study)
        if len(idx) < 2:
            continue
        ci = core.censored_concordance(y[idx], pred[idx], c[idx])
        if np.isfinite(ci):
            values.append(ci)
    return (float(np.mean(values)) if values else np.nan), len(values)


def metrics(df: pd.DataFrame, pred: np.ndarray, idx: np.ndarray | None = None, ranking: bool = True) -> dict:
    if idx is None:
        idx = np.arange(len(df))
    part = df.iloc[idx]
    pred = np.asarray(pred, float)
    if len(pred) != len(idx):  # full-length prediction vector supplied with a subset index
        pred = pred[idx]
    y = part["Rejection"].to_numpy(float)
    c = part["is_censored"].to_numpy(bool)
    e = ~c
    out: dict[str, float | int] = {"n": int(len(idx)), "n_exact": int(e.sum()), "n_censored": int(c.sum()),
                                   "studies": int(part["ref_id"].nunique())}
    if e.sum() >= 2:
        ye, pe = y[e], pred[e]
        sse = float(np.sum((ye - pe) ** 2))
        sst = float(np.sum((ye - ye.mean()) ** 2))
        sd = float(np.std(ye, ddof=1))
        rmse = math.sqrt(sse / len(ye))
        ly, lp = logit(ye), logit(pe)
        out.update(
            r2=1 - sse / sst if sst > 0 else np.nan,
            logit_r2=1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2),
            rmse_pp=100 * rmse,
            mae_pp=100 * float(np.mean(np.abs(ye - pe))),
            exact_sd_pp=100 * sd,
            nrmse=rmse / sd if sd > 0 else np.nan,
            mean_bias_pp=100 * float(np.mean(pe - ye)),
        )
        if np.std(pe) > 1e-12:
            slope, intercept = np.polyfit(pe, ye, 1)
            out.update(calibration_slope=float(slope), calibration_intercept=float(intercept))
        else:
            out.update(calibration_slope=np.nan, calibration_intercept=np.nan)
    out["pooled_c"] = core.censored_concordance(y, pred, c)
    out["lower_bound_violation"] = float(np.mean(pred[c] < y[c])) if c.any() else np.nan
    if ranking:
        wc, ns = within_study_c(part, pred)
        out["within_study_c"] = wc
        out["scored_studies"] = ns
    return out


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def fit_predict_aft(df: pd.DataFrame, x: np.ndarray, train: np.ndarray, test: np.ndarray, cfg: dict, seed: int,
                    weight: np.ndarray | None = None, margin: bool = False):
    import xgboost as xgb

    dtrain = xgb.DMatrix(x[train], missing=np.nan)
    lower = df["odds_R"].to_numpy(float)[train]
    upper = np.where(df["is_censored"].to_numpy(bool)[train], np.inf, lower)
    dtrain.set_float_info("label_lower_bound", lower)
    dtrain.set_float_info("label_upper_bound", upper)
    if weight is not None:
        dtrain.set_weight(weight[train])
    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": cfg["scale"], "tree_method": "hist", "max_depth": cfg["depth"],
        "min_child_weight": cfg["child"], "eta": cfg["eta"], "subsample": cfg["sub"],
        "colsample_bytree": cfg["col"], "lambda": cfg["lam"], "alpha": cfg["alpha"], "seed": int(seed),
        "nthread": 4,
    }
    model = xgb.train(params, dtrain, num_boost_round=int(cfg["rounds"]), verbose_eval=False)
    odds = np.clip(model.predict(xgb.DMatrix(x[test], missing=np.nan)), 1e-8, 1e8)
    if margin:
        return np.log(odds)
    return odds / (1 + odds)


def tobit_fit_predict(df: pd.DataFrame, cols: list[str], train: np.ndarray, test: np.ndarray,
                      log_cols: set[str] | None = None, ridge: float = 1e-3) -> np.ndarray:
    """Right-censored normal (Tobit-type) regression on the logit scale.

    Exact rows contribute log phi((y - xb)/s) - log s; censored rows contribute
    log(1 - Phi((L - xb)/s)).  Median imputation and standardisation use the
    training fold only.
    """
    log_cols = log_cols or set()
    xf = df[cols].astype(float).copy()
    for col in log_cols:
        xf[col] = np.log(xf[col].clip(lower=1e-9))
    imp = SimpleImputer(strategy="median").fit(xf.iloc[train])
    sc = StandardScaler().fit(imp.transform(xf.iloc[train]))
    xtr = sc.transform(imp.transform(xf.iloc[train]))
    xte = sc.transform(imp.transform(xf.iloc[test]))
    xtr = np.column_stack([np.ones(len(xtr)), xtr])
    xte = np.column_stack([np.ones(len(xte)), xte])
    yv = df["logit_R"].to_numpy(float)[train]
    cens = df["is_censored"].to_numpy(bool)[train]

    def nll(theta):
        beta, log_s = theta[:-1], theta[-1]
        s = math.exp(log_s)
        z = (yv - xtr @ beta) / s
        ll = np.where(cens, stats.norm.logsf(z), stats.norm.logpdf(z) - log_s)
        return -ll.sum() + ridge * np.sum(beta[1:] ** 2)

    beta0 = np.linalg.lstsq(xtr, yv, rcond=None)[0]
    theta0 = np.concatenate([beta0, [math.log(np.std(yv - xtr @ beta0) + 1e-3)]])
    res = optimize.minimize(nll, theta0, method="L-BFGS-B")
    return core.logistic(xte @ res.x[:-1])


def shrunk_group_mean(df: pd.DataFrame, col: str, train: np.ndarray, test: np.ndarray, k: float = 5.0) -> np.ndarray:
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    exact_train = train[~c[train]]
    overall = float(y[exact_train].mean())
    keys = df[col].astype(str).to_numpy()
    stats_ = pd.DataFrame({"k": keys[exact_train], "y": y[exact_train]}).groupby("k")["y"].agg(["sum", "count"])
    lookup = ((stats_["sum"] + k * overall) / (stats_["count"] + k)).to_dict()
    return np.array([lookup.get(key, overall) for key in keys[test]])


def kmeans_clusters(x: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(x))
    return KMeans(n_clusters=10, random_state=42, n_init=20).fit_predict(scaled)


# ---------------------------------------------------------------------------
# Section A: dataset registry and row-level eligibility log
# ---------------------------------------------------------------------------

def section_registry(raw: pd.DataFrame, sets: dict[str, pd.DataFrame]) -> None:
    inventory = pd.read_csv(OUTPUT_ROOT / "04_Tables" / "Supplementary" / "Table_S3_study_inventory.csv")
    doi = dict(zip(inventory["study_id"], inventory["doi"].str.replace(r"\.s001$", "", regex=True)))
    r50 = raw["Rejection"].ge(0.5)
    nfro = raw["mem_class"].astype(str).str.upper().isin(["NF", "RO"])
    mem_drop = raw["mem"].isin(AUTHOR_OMITTED_MEMBRANES)
    ref_drop = raw["ref_id"].isin(AUTHOR_OMITTED_REFERENCES)
    reasons = []
    for i in raw.index:
        codes = []
        if not r50[i]:
            codes.append("R_LT_0.50")
        if not nfro[i]:
            cls = raw.at[i, "mem_class"]
            codes.append("CLASS_UF" if str(cls).upper() == "UF" else "CLASS_MISSING")
        if mem_drop[i]:
            codes.append("AUTHOR_JOIN_MEMBRANE_ID")
        if ref_drop[i]:
            codes.append("AUTHOR_JOIN_REFERENCE_ID")
        reasons.append(";".join(codes) if codes else "ELIGIBLE_ALL")
    log = pd.DataFrame({
        "source_worksheet": "PFAS_literature_dataset_2b",
        "source_row": raw["source_row"],
        "row_uid": raw["row_uid"],
        "ref_id": raw["ref_id"],
        "doi": raw["ref_id"].map(doi),
        "ref_year": raw["ref_year"],
        "membrane_id": raw["mem"],
        "mem_class": raw["mem_class"],
        "PFAS": raw["PFAS"],
        "Rejection": raw["Rejection"],
        "LOQ_flag": raw["LOQ"],
        "feed_type": raw["feed_type"],
        "reason_codes": reasons,
        "in_P2424_primary": raw["in_P2424"],
        "in_A2353_author_rule": raw["in_A2353"],
        "in_U2487_unfiltered": raw["in_U2487"],
        "in_shared_P2424_A2353": raw["in_P2424"] & raw["in_A2353"],
    })
    save_csv(log, "R2_A01_row_level_eligibility_log.csv")

    def block(mask, label, rule, role):
        part = raw.loc[mask]
        cens = part["LOQ"].astype(str).str.lower().eq("right")
        return {
            "label": label, "eligibility_rule": rule, "role": role, "rows": int(mask.sum()),
            "studies": part["ref_id"].nunique(), "membranes": part["mem"].nunique(), "pfas": part["PFAS"].nunique(),
            "censored_rows": int(cens.sum()), "censored_pct": round(100 * float(cens.mean()), 2),
            "exact_rows": int((~cens).sum()),
        }

    reg = pd.DataFrame([
        block(pd.Series(True, index=raw.index), "W2517", "All rows of worksheet PFAS_literature_dataset_2b", "Source workbook"),
        block(r50, "R2450", "Rejection >= 0.50", "Intermediate"),
        block(raw["in_P2424"], "P2424", "Rejection >= 0.50 and membrane class NF or RO", "Primary analysis set"),
        block(raw["in_A2353"], "A2353", "Rejection >= 0.50; omit membrane IDs NF270-PEI_Jeong, NF270-PAA_Jeong and reference IDs 'Ma J., 2022', 'Ma et al, 2024 (2)'", "Author-rule sensitivity"),
        block(raw["in_P2424"] & raw["in_A2353"], "S2327", "Intersection of P2424 and A2353", "Overlap description only"),
        block(raw["in_U2487"], "U2487", "Membrane class NF or RO; no rejection threshold", "Threshold sensitivity"),
    ])
    save_csv(reg, "R2_A02_dataset_registry.csv")

    stages = []
    studies_all = set(raw["ref_id"])
    for label, mask in [("R2450", r50), ("P2424", raw["in_P2424"]), ("A2353", raw["in_A2353"]), ("U2487", raw["in_U2487"])]:
        kept = set(raw.loc[mask, "ref_id"])
        for study in sorted(studies_all - kept):
            part = raw.loc[raw["ref_id"] == study]
            stages.append({"dataset": label, "study_absent": study, "rows_in_workbook": len(part),
                           "classes_in_workbook": ";".join(sorted(part["mem_class"].astype(str).unique())),
                           "rows_with_R_ge_0.50": int(part["Rejection"].ge(0.5).sum())})
    save_csv(pd.DataFrame(stages), "R2_A03_study_count_changes.csv")


# ---------------------------------------------------------------------------
# Section B: complete fold tables (leave-one-study-out and 10-fold study-blocked)
# ---------------------------------------------------------------------------

def per_group_table(df, pred, base, groups, label_col, design, extra=None):
    rows = []
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    for gval in pd.unique(groups):
        idx = np.flatnonzero(groups == gval)
        e = idx[~c[idx]]
        m = metrics(df, pred, idx, ranking=False)
        b = metrics(df, base, idx, ranking=False)
        rec = {
            "dataset": "P2424", "design": design, label_col: gval,
            "studies_in_fold": ";".join(sorted(df.iloc[idx]["ref_id"].astype(str).unique())) if label_col == "fold" else gval,
            "test_rows": len(idx), "exact_rows": len(e), "censored_rows": int(c[idx].sum()),
            "exact_sd_pp": 100 * float(np.std(y[e], ddof=1)) if len(e) > 1 else np.nan,
            "mean_observed_exact": float(y[e].mean()) if len(e) else np.nan,
            "mean_model_prediction": float(pred[idx].mean()),
            "baseline_prediction": float(base[idx][0]),
            "model_r2": m.get("r2", np.nan), "model_logit_r2": m.get("logit_r2", np.nan),
            "model_rmse_pp": m.get("rmse_pp", np.nan), "model_mae_pp": m.get("mae_pp", np.nan),
            "model_c_index": m["pooled_c"], "model_lower_bound_violation": m["lower_bound_violation"],
            "baseline_r2": b.get("r2", np.nan), "baseline_rmse_pp": b.get("rmse_pp", np.nan),
            "baseline_mae_pp": b.get("mae_pp", np.nan),
        }
        if extra:
            rec.update(extra(idx))
        rows.append(rec)
    return pd.DataFrame(rows)


def loso_predictions(df, x, rounds=115, seed_base=500, cfg=None):
    """Published leave-one-study-out design (core.run_split_audit): seed 500 + fold, 115 rounds."""
    cfg = {**(cfg or CONSERVATIVE), "rounds": rounds}
    groups = df["ref_id"].fillna("<NA>").astype(str).to_numpy()
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    pred = np.full(len(df), np.nan)
    base = np.full(len(df), np.nan)
    fold_of = np.zeros(len(df), int)
    for fold_no, g in enumerate(pd.unique(groups), 1):
        te = np.flatnonzero(groups == g)
        tr = np.flatnonzero(groups != g)
        pred[te] = fit_predict_aft(df, x, tr, te, cfg, seed_base + fold_no)
        base[te] = y[tr[~c[tr]]].mean()
        fold_of[te] = fold_no
    return pred, base, fold_of


def tenfold_predictions(df, x, rounds=145, seed_base=700, cfg=None, weight=None):
    """Published 10-fold study-blocked design (Table 2): GroupKFold(10), seed 700 + fold, 145 rounds."""
    cfg = {**(cfg or CONSERVATIVE), "rounds": rounds}
    groups = df["ref_id"].astype(str).to_numpy()
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    pred = np.full(len(df), np.nan)
    base = np.full(len(df), np.nan)
    fold_of = np.zeros(len(df), int)
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=10).split(x, y, groups), 1):
        pred[te] = fit_predict_aft(df, x, tr, te, cfg, seed_base + fold, weight=weight)
        base[te] = y[tr[~c[tr]]].mean()
        fold_of[te] = fold
    return pred, base, fold_of


def section_fold_tables(sets) -> dict:
    df = sets["P2424"]
    x = core.feature_matrix(df)
    groups = df["ref_id"].fillna("<NA>").astype(str).to_numpy()
    loso, loso_base, loso_fold = loso_predictions(df, x)
    t = per_group_table(df, loso, loso_base, groups, "study", "Leave-one-study-out (61 fits; conservative AFT, 115 rounds, seed 500+fold)")
    t.insert(2, "fold_index", [int(loso_fold[groups == s][0]) for s in t["study"]])
    save_csv(t, "R2_B01_loso_complete_fold_table.csv")
    tenfold, ten_base, ten_fold = tenfold_predictions(df, x)
    t10 = per_group_table(df, tenfold, ten_base, ten_fold, "fold", "10-fold study-blocked GroupKFold (conservative AFT, 145 rounds, seed 700+fold)")
    save_csv(t10, "R2_B02_tenfold_study_blocked_fold_table.csv")
    pooled = pd.DataFrame([
        {"dataset": "P2424", "design": "Leave-one-study-out", "predictor": "Conservative AFT", **metrics(df, loso)},
        {"dataset": "P2424", "design": "Leave-one-study-out", "predictor": "Training-fold exact-row mean", **metrics(df, loso_base)},
        {"dataset": "P2424", "design": "10-fold study-blocked", "predictor": "Conservative AFT", **metrics(df, tenfold)},
        {"dataset": "P2424", "design": "10-fold study-blocked", "predictor": "Training-fold exact-row mean", **metrics(df, ten_base)},
    ])
    pooled["sum_of_fold_exact_rows"] = [t["exact_rows"].sum()] * 2 + [t10["exact_rows"].sum()] * 2
    save_csv(pooled, "R2_B03_fold_table_pooled_reconciliation.csv")
    rows = pd.DataFrame({
        "row_uid": df["row_uid"], "study": df["ref_id"], "is_censored": df["is_censored"],
        "rejection": df["Rejection"], "loso_fold": loso_fold, "loso_prediction": loso,
        "loso_training_mean": loso_base, "tenfold_fold": ten_fold, "tenfold_prediction": tenfold,
        "tenfold_training_mean": ten_base,
    })
    save_csv(rows, "R2_B04_row_level_out_of_fold_predictions.csv")
    return {"loso": loso, "loso_base": loso_base, "tenfold": tenfold, "ten_base": ten_base, "ten_fold": ten_fold}


# ---------------------------------------------------------------------------
# Section C: fold membership and leakage audit
# ---------------------------------------------------------------------------

def design_splits(df: pd.DataFrame, x: np.ndarray) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    n = len(df)
    allidx = np.arange(n)
    y = df["Rejection"].to_numpy(float)
    groups = df["ref_id"].fillna("<NA>").astype(str).to_numpy()
    clusters = kmeans_clusters(x)
    designs: dict[str, list] = {}
    for name, strat in [("Row-random 85:15 x12 (published)", None), ("Cluster-stratified 85:15 x12 (published)", clusters)]:
        designs[name] = []
        for rep in range(12):
            tr, te = train_test_split(allidx, test_size=0.15, random_state=110 + rep, stratify=strat)
            designs[name].append((str(rep + 1), np.sort(tr), np.sort(te)))
    for name, col in [("Leave-one-study-out", "ref_id"), ("Leave-one-membrane-out", "mem"), ("Leave-one-PFAS-out", "PFAS")]:
        g = df[col].fillna("<NA>").astype(str).to_numpy()
        designs[name] = [(str(v), np.flatnonzero(g != v), np.flatnonzero(g == v)) for v in pd.unique(g)]
    designs["10-fold study-blocked GroupKFold"] = [
        (str(f), tr, te) for f, (tr, te) in enumerate(GroupKFold(10).split(x, y, groups), 1)]
    designs["5-fold study-blocked GroupKFold (SHAP/ALE)"] = [
        (str(f), tr, te) for f, (tr, te) in enumerate(GroupKFold(5).split(x, y, groups), 1)]
    years = df["ref_year"].to_numpy(float)
    designs["Publication-year holdout"] = [("2024-2025", np.flatnonzero(years <= 2023), np.flatnonzero(years > 2023))]
    dev, unseen = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=audit.SEED).split(x, groups=groups))
    ftr, fte = train_test_split(dev, test_size=0.15, random_state=audit.SEED + 20, shuffle=True)
    designs["Fixed holdout: familiar-study test"] = [("familiar", np.sort(ftr), np.sort(fte))]
    designs["Fixed holdout: untouched-study test"] = [("untouched", np.sort(ftr), np.sort(unseen))]
    designs["Harmonized 10-fold row-random KFold"] = [
        (str(f), tr, te) for f, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=SEED).split(x), 1)]
    designs["Harmonized 10-fold cluster-stratified"] = [
        (str(f), tr, te) for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=SEED).split(x, clusters), 1)]
    return designs


def section_leakage(sets) -> None:
    df = sets["P2424"]
    x = core.feature_matrix(df)
    cond = core.strict_condition_key(df).to_numpy()
    study = df["ref_id"].fillna("<NA>").astype(str).to_numpy()
    mem = df["mem"].fillna("<NA>").astype(str).to_numpy()
    pfas = df["PFAS"].fillna("<NA>").astype(str).to_numpy()
    fvec = pd.DataFrame(x).round(10).astype(str).agg("|".join, axis=1).to_numpy()
    designs = design_splits(df, x)
    fold_rows, member_rows = [], []
    for design, splits in designs.items():
        for fold, tr, te in splits:
            rec = {"dataset": "P2424", "design": design, "fold": fold, "train_rows": len(tr), "test_rows": len(te)}
            for lab, arr in [("study", study), ("membrane", mem), ("pfas", pfas), ("strict_condition", cond), ("feature_vector", fvec)]:
                seen = set(arr[tr])
                rec[f"test_rows_sharing_{lab}_pct"] = 100 * float(np.mean([v in seen for v in arr[te]]))
                uniq = np.unique(arr[te])
                rec[f"test_unique_{lab}"] = len(uniq)
                rec[f"test_unique_{lab}_in_train_pct"] = 100 * float(np.mean([v in seen for v in uniq]))
            fold_rows.append(rec)
            member_rows.extend({"design": design, "fold": fold, "row_uid": df["row_uid"].iat[i], "study": study[i],
                                "membrane": mem[i], "pfas": pfas[i]} for i in te)
    folds = pd.DataFrame(fold_rows)
    save_csv(folds, "R2_C01_fold_overlap_audit.csv")
    cond_ids = {k: f"C{i:04d}" for i, k in enumerate(pd.unique(cond), 1)}
    members = pd.DataFrame(member_rows)
    uid_to_cond = dict(zip(df["row_uid"], [cond_ids[k] for k in cond]))
    members["strict_condition_id"] = members["row_uid"].map(uid_to_cond)
    save_csv(members, "R2_C02_test_fold_membership_long.csv")
    value_cols = [c for c in folds.columns if c.endswith("_pct")]
    summary = folds.groupby("design", sort=False).agg(folds=("fold", "count"), **{c: (c, "mean") for c in value_cols}).reset_index()
    save_csv(summary, "R2_C03_fold_overlap_summary_by_design.csv")
    stages = pd.DataFrame([
        ["log1p transform of flux, feed concentration and ion/DOC predictors", "Row-wise deterministic", "Before splitting; uses no sample statistics"],
        ["Clipping of rejection to [0.500001, 0.999999] and logit/odds", "Row-wise deterministic", "Before splitting; uses no sample statistics"],
        ["Missing predictors in XGBoost", "Native sparsity-aware split routing", "Learned inside each training fold only"],
        ["Median imputation and standardisation (kNN distance, propensity, linear/Tobit models)", "SimpleImputer + StandardScaler", "Fitted on each training fold; applied to the paired test fold"],
        ["Hyperparameter selection, nested analyses (Round 2)", "Inner GroupKFold(3) by study or KFold(3)", "Inside each outer training fold only"],
        ["Hyperparameter selection, published conservative architecture", "Three-fold GroupKFold on the 1,679 fixed-split fitting rows", "Fixed-split development rows; reused for other designs (disclosed; see nested analysis)"],
        ["Applicability threshold (Round 2 out-of-fold rule)", "P90 of leave-own-study-out kNN distances among training rows", "Each outer training fold only"],
        ["Applicability threshold (published retrospective display)", "Pooled P90 of held-out relative distances", "Uses pooled held-out predictor distances (retrospective)"],
        ["K-means labels for cluster stratification", "KMeans(k=10) after median imputation and scaling", "Full predictor matrix, outcomes not used; reconstructs the source design"],
        ["Training-fold mean baseline", "Mean exact rejection of training rows", "Recomputed in every outer fold"],
    ], columns=["operation", "implementation", "data used / stage"])
    save_csv(stages, "R2_C04_preprocessing_stage_audit.csv")


# ---------------------------------------------------------------------------
# Section D: harmonized designs (identical hyperparameters and denominators)
# ---------------------------------------------------------------------------

def run_design(df, x, splits, cfg, seed_base):
    pred = np.full(len(df), np.nan)
    for k, (_, tr, te) in enumerate(splits, 1):
        pred[te] = fit_predict_aft(df, x, tr, te, cfg, seed_base + k)
    return pred


def section_harmonized(sets) -> dict:
    cfg = {**CONSERVATIVE, "rounds": 130}
    rows, per_group, store = [], [], {}
    for label in ["P2424", "U2487"]:
        df = sets[label]
        x = core.feature_matrix(df)
        y = df["Rejection"].to_numpy(float)
        groups = df["ref_id"].fillna("<NA>").astype(str).to_numpy()
        clusters = kmeans_clusters(x)
        years = df["ref_year"].to_numpy(float)
        designs = {
            "10-fold row-random": [(str(f), tr, te) for f, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=SEED).split(x), 1)],
            "10-fold cluster-stratified": [(str(f), tr, te) for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=SEED).split(x, clusters), 1)],
            "10-fold study-blocked": [(str(f), tr, te) for f, (tr, te) in enumerate(GroupKFold(10).split(x, y, groups), 1)],
        }
        for name, col in [("Leave-one-study-out", "ref_id"), ("Leave-one-membrane-out", "mem"), ("Leave-one-PFAS-out", "PFAS")]:
            g = df[col].fillna("<NA>").astype(str).to_numpy()
            designs[name] = [(str(v), np.flatnonzero(g != v), np.flatnonzero(g == v)) for v in pd.unique(g)]
        designs["Publication-year holdout"] = [("2024-2025", np.flatnonzero(years <= 2023), np.flatnonzero(years > 2023))]
        for name, splits in designs.items():
            pred = run_design(df, x, splits, cfg, SEED + 1000)
            store[(label, name)] = pred
            tested = np.flatnonzero(np.isfinite(pred))
            rows.append({"dataset": label, "design": name, "fits": len(splits), "rounds": 130,
                         "seed_rule": f"{SEED + 1000}+fit index", **metrics(df, pred[tested], tested)})
            if name in ("Leave-one-study-out", "Leave-one-membrane-out", "Leave-one-PFAS-out") and label == "P2424":
                col = {"Leave-one-study-out": "ref_id", "Leave-one-membrane-out": "mem", "Leave-one-PFAS-out": "PFAS"}[name]
                g = df[col].fillna("<NA>").astype(str).to_numpy()
                for gval, tr, te in splits:
                    m = metrics(df, pred, te, ranking=False)
                    per_group.append({"dataset": label, "design": name, "group": gval, "group_rows": len(te),
                                      "group_rows_in_full_data": int((g == gval).sum()),
                                      **{k: m.get(k, np.nan) for k in ["n_exact", "r2", "rmse_pp", "mae_pp", "nrmse", "exact_sd_pp"]}})
            print(f"harmonized {label} {name}: R2={rows[-1].get('r2', np.nan):.3f}", flush=True)
    save_csv(pd.DataFrame(rows), "R2_D01_harmonized_design_comparison.csv")
    pg = pd.DataFrame(per_group)
    save_csv(pg, "R2_D02_harmonized_leave_one_group_metrics.csv")
    return store


# ---------------------------------------------------------------------------
# Section E: nested tuning (12 candidates) with identical inner protocols
# ---------------------------------------------------------------------------

def _inner_score(df, x, train, cfg, inner, seed):
    groups = df["ref_id"].astype(str).to_numpy()
    if inner == "grouped":
        splitter = GroupKFold(n_splits=3).split(x[train], groups=groups[train])
    else:
        splitter = KFold(n_splits=3, shuffle=True, random_state=seed).split(train)
    scores = []
    for k, (a, b) in enumerate(splitter, 1):
        tr, va = train[a], train[b]
        p = fit_predict_aft(df, x, tr, va, cfg, seed + k)
        m = metrics(df, p, va, ranking=False)
        scores.append(m.get("logit_r2", np.nan))
    return float(np.nanmean(scores))


def _outer_fold(df, x, design, inner, fold, tr, te):
    seed = SEED + 5000 + 97 * fold
    scores = {cfg["name"]: _inner_score(df, x, tr, cfg, inner, seed + i * 11) for i, cfg in enumerate(audit.CANDIDATES)}
    best = max(scores, key=scores.get)
    cfg = next(c for c in audit.CANDIDATES if c["name"] == best)
    pred = fit_predict_aft(df, x, tr, te, cfg, seed + 999)
    return design, inner, fold, te, pred, best, scores


def section_nested(sets) -> dict:
    df = sets["P2424"]
    x = core.feature_matrix(df)
    y = df["Rejection"].to_numpy(float)
    groups = df["ref_id"].astype(str).to_numpy()
    clusters = kmeans_clusters(x)
    outer = {
        "10-fold row-random": list(KFold(10, shuffle=True, random_state=SEED).split(x)),
        "10-fold cluster-stratified": list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(x, clusters)),
        "10-fold study-blocked": list(GroupKFold(10).split(x, y, groups)),
    }
    jobs = [(d, inner, f, tr, te) for d, splits in outer.items() for inner in ("grouped", "row") for f, (tr, te) in enumerate(splits, 1)]
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_outer_fold)(df, x, *job) for job in jobs)
    print(f"nested tuning finished in {time.time() - t0:.0f} s", flush=True)
    preds, sel_rows = {}, []
    for design, inner, fold, te, pred, best, scores in results:
        preds.setdefault((design, inner), np.full(len(df), np.nan))[te] = pred
        sel_rows.append({"dataset": "P2424", "outer_design": design, "inner_protocol": inner, "fold": fold, "selected": best,
                         **{f"inner_logit_r2_{k}": v for k, v in scores.items()}})
    save_csv(pd.DataFrame(sel_rows), "R2_E01_nested_selection_by_fold.csv")
    rows = []
    for (design, inner), pred in preds.items():
        sel = pd.DataFrame(sel_rows)
        sel = sel[(sel.outer_design == design) & (sel.inner_protocol == inner)]["selected"].value_counts()
        rows.append({"dataset": "P2424", "outer_design": design,
                     "inner_protocol": "GroupKFold(3) by study" if inner == "grouped" else "KFold(3) row-random",
                     "selected_counts": "; ".join(f"{k}:{v}" for k, v in sel.items()), **metrics(df, pred)})
    save_csv(pd.DataFrame(rows), "R2_E02_nested_tuning_summary.csv")
    return preds


# ---------------------------------------------------------------------------
# Section F: expanded prediction-time baselines under 10-fold study blocking
# ---------------------------------------------------------------------------

def paired_bootstrap(df, preds: dict[str, np.ndarray], reference: str, draws: int = 5000, seed: int = 20260831 + 40000) -> pd.DataFrame:
    """Study-level paired bootstrap; identical draw matrix to Table S10 (run_major_revision_robustness.py)."""
    y = df["Rejection"].to_numpy(float)
    exact = ~df["is_censored"].to_numpy(bool)
    g = df["ref_id"].astype(str).to_numpy()
    labels = sorted(np.unique(g))
    gno = np.array([labels.index(v) for v in g])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(labels), size=(draws, len(labels)))
    counts = np.apply_along_axis(lambda row: np.bincount(row, minlength=len(labels)), 1, samples).astype(float)
    w = counts[:, gno]
    w[:, ~exact] = 0
    n = w.sum(1)
    mean = (w @ y) / n
    sst = np.sum(w * (y[None] - mean[:, None]) ** 2, axis=1)

    def scal(p):
        err = p - y
        sse = w @ (err ** 2)
        return 1 - sse / sst, np.sqrt(sse / n) * 100, (w @ np.abs(err)) / n * 100

    def study_c(p):
        out = np.full(len(labels), np.nan)
        c = df["is_censored"].to_numpy(bool)
        for i, s in enumerate(labels):
            idx = np.flatnonzero(g == s)
            if len(idx) > 1:
                out[i] = core.censored_concordance(y[idx], p[idx], c[idx])
        return out

    ref = scal(preds[reference])
    ref_c = study_c(preds[reference])
    rows = []
    for name, p in preds.items():
        if name == reference:
            continue
        cur = scal(p)
        cur_c = study_c(p)
        ok = np.isfinite(cur_c) & np.isfinite(ref_c)
        cc = counts[:, ok]
        dc = (cc @ (cur_c[ok] - ref_c[ok])) / cc.sum(1)
        for metric, d, better in [("R2", cur[0] - ref[0], lambda v: v > 0), ("RMSE (pp)", cur[1] - ref[1], lambda v: v < 0),
                                  ("MAE (pp)", cur[2] - ref[2], lambda v: v < 0), ("Within-study C", dc, lambda v: v > 0)]:
            lo, med, hi = np.quantile(d, [0.025, 0.5, 0.975])
            rows.append({"dataset": df.attrs.get("label", ""), "predictor": name, "reference": reference, "metric": metric,
                         "bootstrap_median_difference": med, "q025": lo, "q975": hi,
                         "fraction_draws_better": float(np.mean(better(d))), "draws": draws, "resampling_unit": "study; fitted predictions held fixed"})
    return pd.DataFrame(rows)


def section_baselines(sets, fold_store, nested_preds) -> dict:
    out_rows, boot_frames, all_preds = [], [], {}
    for label in ["P2424", "U2487"]:
        df = sets[label]
        df.attrs["label"] = label
        x = core.feature_matrix(df)
        y = df["Rejection"].to_numpy(float)
        c = df["is_censored"].to_numpy(bool)
        groups = df["ref_id"].astype(str).to_numpy()
        df = df.copy()
        df["is_RO"] = df["mem_class"].astype(str).str.upper().eq("RO").astype(float)
        df.attrs["label"] = label
        preds = {k: np.full(len(df), np.nan) for k in [
            "Conservative AFT, 13 features (145 rounds)", "Linear censored-normal (Tobit), 13 features",
            "Physical censored-normal: log volume + log permeance", "Physical censored-normal: + RO indicator",
            "2-feature AFT (PFAS volume + water permeance)", "Empirical AFT (logP + PFAS volume)",
            "PFAS-specific training mean (shrunk, k = 5)", "Membrane-specific training mean (shrunk, k = 5)",
            "Training-fold exact-row median", "Training-fold exact-row mean"]}
        x_two = df[["volume_Å3", "pwp_LMH/bar"]].to_numpy(float)
        x_emp = df[["logP", "volume_Å3"]].to_numpy(float)
        fold_mean = []
        for fold, (tr, te) in enumerate(GroupKFold(10).split(x, y, groups), 1):
            cfg = {**CONSERVATIVE, "rounds": 145}
            preds["Conservative AFT, 13 features (145 rounds)"][te] = fit_predict_aft(df, x, tr, te, cfg, 700 + fold)
            preds["2-feature AFT (PFAS volume + water permeance)"][te] = fit_predict_aft(df, x_two, tr, te, cfg, 700 + fold)
            preds["Empirical AFT (logP + PFAS volume)"][te] = fit_predict_aft(df, x_emp, tr, te, cfg, 700 + fold)
            preds["Linear censored-normal (Tobit), 13 features"][te] = tobit_fit_predict(df, core.FEATURES, tr, te)
            preds["Physical censored-normal: log volume + log permeance"][te] = tobit_fit_predict(
                df, ["volume_Å3", "pwp_LMH/bar"], tr, te, log_cols={"volume_Å3", "pwp_LMH/bar"})
            preds["Physical censored-normal: + RO indicator"][te] = tobit_fit_predict(
                df, ["volume_Å3", "pwp_LMH/bar", "is_RO"], tr, te, log_cols={"volume_Å3", "pwp_LMH/bar"})
            preds["PFAS-specific training mean (shrunk, k = 5)"][te] = shrunk_group_mean(df, "PFAS", tr, te)
            preds["Membrane-specific training mean (shrunk, k = 5)"][te] = shrunk_group_mean(df, "mem", tr, te)
            ex = tr[~c[tr]]
            preds["Training-fold exact-row median"][te] = float(np.median(y[ex]))
            preds["Training-fold exact-row mean"][te] = float(y[ex].mean())
            fold_mean.append({"dataset": label, "fold": fold, "training_rows": len(tr), "training_exact_rows": len(ex),
                              "training_mean_rejection": float(y[ex].mean()), "test_rows": len(te),
                              "test_studies": len(np.unique(groups[te]))})
        if label == "P2424":
            preds["Grouped-nested AFT (12 candidates; inner GroupKFold by study)"] = nested_preds[("10-fold study-blocked", "grouped")]
            assert np.allclose(preds["Conservative AFT, 13 features (145 rounds)"], fold_store["tenfold"])
        for name, p in preds.items():
            out_rows.append({"dataset": label, "predictor": name, **metrics(df, p)})
        boot_frames.append(paired_bootstrap(df, preds, "Training-fold exact-row mean"))
        save_csv(pd.DataFrame(fold_mean), f"R2_F03_training_mean_by_fold_{label}.csv")
        all_preds[label] = preds
    save_csv(pd.DataFrame(out_rows), "R2_F01_expanded_baselines_10fold_study_blocked.csv")
    save_csv(pd.concat(boot_frames, ignore_index=True), "R2_F02_paired_study_bootstrap_vs_mean.csv")
    return all_preds


# ---------------------------------------------------------------------------
# Section G: AFT likelihood diagnostics
# ---------------------------------------------------------------------------

def section_aft_diagnostics(sets) -> pd.DataFrame:
    df = sets["P2424"]
    x = core.feature_matrix(df)
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    ly = df["logit_R"].to_numpy(float)
    groups = df["ref_id"].astype(str).to_numpy()
    cfg = {**CONSERVATIVE, "rounds": 145}
    f = np.full(len(df), np.nan)
    sigma_star = np.full(len(df), np.nan)
    fold_rows = []
    for fold, (tr, te) in enumerate(GroupKFold(10).split(x, y, groups), 1):
        f[te] = fit_predict_aft(df, x, tr, te, cfg, 700 + fold, margin=True)
        # Training-only recalibration of the scale: inner GroupKFold(3) out-of-fold margins.
        inner = np.full(len(tr), np.nan)
        for k, (a, b) in enumerate(GroupKFold(3).split(x[tr], groups=groups[tr]), 1):
            inner[b] = fit_predict_aft(df, x, tr[a], tr[b], cfg, 8000 + 10 * fold + k, margin=True)
        res_y, res_c = ly[tr] - inner, c[tr]

        def nll(log_s):
            s = math.exp(log_s)
            z = res_y / s
            return -np.sum(np.where(res_c, stats.norm.logsf(z), stats.norm.logpdf(z) - log_s))

        s_hat = math.exp(optimize.minimize_scalar(nll, bounds=(-4, 3), method="bounded").x)
        sigma_star[te] = s_hat
        fold_rows.append({"fold": fold, "training_rows": len(tr), "inner_oof_sigma_mle": s_hat,
                          "inner_oof_exact_residual_sd": float(np.std(res_y[~res_c], ddof=1))})
    save_csv(pd.DataFrame(fold_rows), "R2_G02_fold_scale_recalibration.csv")
    rng = np.random.default_rng(SEED + 123)
    rows = []
    row_out = pd.DataFrame({"row_uid": df["row_uid"], "is_censored": c, "logit_observed_or_bound": ly, "margin_f": f,
                            "sigma_star_fold": sigma_star})
    for label, sig in [("Nominal AFT scale (sigma = 1.2)", np.full(len(df), 1.2)), ("Fold-recalibrated scale (inner OOF MLE)", sigma_star)]:
        z = (ly - f) / sig
        u_low = stats.norm.cdf(z)
        u = np.where(c, rng.uniform(u_low, 1.0), u_low)
        rqr = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        cs = -np.log(np.clip(1 - u_low, 1e-300, 1))  # Cox-Snell (censored rows give lower bounds)
        key = "nominal" if label.startswith("Nominal") else "recal"
        row_out[f"rqr_{key}"] = rqr
        row_out[f"cox_snell_{key}"] = cs
        rec = {"scale": label, "median_sigma": float(np.nanmedian(sig)), "rqr_mean": float(np.mean(rqr)), "rqr_sd": float(np.std(rqr, ddof=1)),
               "rqr_skew": float(stats.skew(rqr)), "rqr_shapiro_like_ks_p": float(stats.kstest(rqr, "norm").pvalue),
               "exact_z_sd": float(np.std(z[~c], ddof=1)), "exact_z_mean": float(np.mean(z[~c])),
               "censored_mean_prob_compatible": float(np.mean(1 - stats.norm.cdf(z[c]))),
               "censored_share_prob_compatible_lt_0.5": float(np.mean((1 - stats.norm.cdf(z[c])) < 0.5))}
        for nominal in (0.80, 0.90, 0.95):
            q = stats.norm.ppf(0.5 + nominal / 2)
            lo, hi = core.logistic(f - q * sig), core.logistic(f + q * sig)
            rec[f"exact_coverage_{int(nominal * 100)}"] = float(np.mean((y[~c] >= lo[~c]) & (y[~c] <= hi[~c])))
            rec[f"censored_compatibility_{int(nominal * 100)}"] = float(np.mean(hi[c] >= y[c]))
            rec[f"mean_width_pp_{int(nominal * 100)}"] = float(np.mean(hi - lo) * 100)
            rec[f"mean_width_pp_clipped_0.5_1_{int(nominal * 100)}"] = float(np.mean(np.clip(hi, .5, 1) - np.clip(lo, .5, 1)) * 100)
        # Cox-Snell: Kaplan-Meier cumulative hazard versus identity.
        order = np.argsort(cs)
        cs_s, ev = cs[order], (~c)[order]
        at_risk = len(cs_s) - np.arange(len(cs_s))
        H = np.cumsum(ev / at_risk)
        sel = (cs_s > 0.05) & (cs_s < np.quantile(cs_s, 0.95))
        rec["cox_snell_KM_hazard_slope"] = float(np.polyfit(cs_s[sel], H[sel], 1)[0])
        rows.append(rec)
    out = pd.DataFrame(rows)
    save_csv(out, "R2_G01_aft_likelihood_diagnostics.csv")
    save_csv(row_out, "R2_G03_aft_row_residuals.csv")
    return row_out


# ---------------------------------------------------------------------------
# Section H: censoring-mechanism simulation
# ---------------------------------------------------------------------------

def section_censoring_simulation(sets) -> None:
    base = sets["P2424"]
    base = base.loc[~base["is_censored"]].reset_index(drop=True)
    true_r = base["Rejection"].to_numpy(float)
    feed = base["pfas_feed_conc_µg/L"].to_numpy(float)
    groups = base["ref_id"].astype(str).to_numpy()
    x = core.feature_matrix(base)
    rows, detail = [], []
    for target in (0.05, 0.127, 0.25):
        def frac(loq):
            b = 1 - loq / feed
            return np.nanmean(np.where(np.isfinite(b), true_r > b, False))
        lo_, hi_ = 1e-8, 1e4
        for _ in range(200):
            mid = math.sqrt(lo_ * hi_)
            if frac(mid) > target:
                hi_ = mid
            else:
                lo_ = mid
        loq = math.sqrt(lo_ * hi_)
        bound = np.clip(1 - loq / feed, 0.5, 0.9999)
        cens = np.isfinite(bound) & (true_r > bound)
        sim = base.copy()
        sim["Rejection"] = np.where(cens, bound, true_r)
        sim["is_censored"] = cens
        r = sim["Rejection"].clip(0.500001, 0.999999)
        sim["odds_R"] = r / (1 - r)
        sim["logit_R"] = np.log(r / (1 - r))
        preds = {m: np.full(len(sim), np.nan) for m in ["AFT (bounds as lower bounds)", "Bounds treated as exact", "Censored rows excluded"]}
        for fold, (tr, te) in enumerate(GroupKFold(10).split(x, true_r, groups), 1):
            cfg = {**CONSERVATIVE, "rounds": 145}
            preds["AFT (bounds as lower bounds)"][te] = fit_predict_aft(sim, x, tr, te, cfg, 700 + fold)
            m1 = core.fit_naive(x[tr], sim["Rejection"].to_numpy()[tr], 700 + fold, 145)
            preds["Bounds treated as exact"][te] = np.clip(m1.predict(x[te]), 0.5, 1.0)
            keep = tr[~cens[tr]]
            m2 = core.fit_naive(x[keep], sim["Rejection"].to_numpy()[keep], 700 + fold, 145)
            preds["Censored rows excluded"][te] = np.clip(m2.predict(x[te]), 0.5, 1.0)
        for method, p in preds.items():
            err_true = p - true_r
            err_rep = p - sim["Rejection"].to_numpy()
            rec = {"target_censoring": target, "realised_censoring": float(cens.mean()), "global_LOQ_ug_per_L": loq,
                   "method": method,
                   "rmse_vs_true_all_pp": 100 * math.sqrt(np.mean(err_true ** 2)),
                   "bias_vs_true_all_pp": 100 * float(np.mean(err_true)),
                   "rmse_vs_true_censored_pp": 100 * math.sqrt(np.mean(err_true[cens] ** 2)),
                   "bias_vs_true_censored_pp": 100 * float(np.mean(err_true[cens])),
                   "rmse_vs_true_uncensored_pp": 100 * math.sqrt(np.mean(err_true[~cens] ** 2)),
                   "r2_vs_true_all": 1 - np.sum(err_true ** 2) / np.sum((true_r - true_r.mean()) ** 2),
                   "naive_rmse_vs_reported_all_pp": 100 * math.sqrt(np.mean(err_rep ** 2)),
                   "exact_only_rmse_vs_reported_pp": 100 * math.sqrt(np.mean(err_rep[~cens] ** 2)),
                   "lower_bound_violation": float(np.mean(p[cens] < sim["Rejection"].to_numpy()[cens])),
                   "n_rows": len(sim), "n_censored": int(cens.sum())}
            rows.append(rec)
            print(f"censoring sim {target:.3f} {method}: bias(cens)={rec['bias_vs_true_censored_pp']:.2f}", flush=True)
    save_csv(pd.DataFrame(rows), "R2_H01_censoring_mechanism_simulation.csv")


# ---------------------------------------------------------------------------
# Section I: out-of-fold applicability-domain screening
# ---------------------------------------------------------------------------

def section_applicability(sets, fold_store) -> None:
    df = sets["P2424"]
    x = core.feature_matrix(df)
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].fillna("<NA>").astype(str).to_numpy()
    pred = fold_store["loso"]
    dist = np.full(len(df), np.nan)
    thr = np.full(len(df), np.nan)
    fold_rows = []
    for fold_no, g in enumerate(pd.unique(groups), 1):
        te = np.flatnonzero(groups == g)
        tr = np.flatnonzero(groups != g)
        imp = SimpleImputer(strategy="median").fit(x[tr])
        sc = StandardScaler().fit(imp.transform(x[tr]))
        xtr = sc.transform(imp.transform(x[tr]))
        xte = sc.transform(imp.transform(x[te]))
        dist[te] = NearestNeighbors(n_neighbors=8).fit(xtr).kneighbors(xte)[0].mean(1)
        ref = []
        gtr = groups[tr]
        for s in np.unique(gtr):
            inside = gtr == s
            nn = NearestNeighbors(n_neighbors=8).fit(xtr[~inside])
            d = nn.kneighbors(xtr[inside])[0].mean(1)
            ref.append(d[~c[tr][inside]])
        ref = np.concatenate(ref)
        t_fold = float(np.quantile(ref, 0.90))
        thr[te] = t_fold
        fold_rows.append({"fold_index": fold_no, "held_out_study": g, "training_rows": len(tr), "threshold_P90_training_only": t_fold,
                          "test_rows": len(te), "test_rows_outside": int(np.sum(dist[te] > t_fold))})
    save_csv(pd.DataFrame(fold_rows), "R2_I01_fold_specific_thresholds.csv")
    outside = dist > thr
    err = np.abs(pred - y)
    rows = []
    e = ~c
    for cut in (0.10, 0.20):
        large = err > cut
        tp = int(np.sum(outside & large & e)); fn = int(np.sum(~outside & large & e))
        fp = int(np.sum(outside & ~large & e)); tn = int(np.sum(~outside & ~large & e))
        rows.append({"dataset": "P2424", "design": "Leave-one-study-out (115 rounds)", "large_error_threshold_pp": 100 * cut,
                     "exact_rows": int(e.sum()), "flagged_large": tp, "flagged_small": fp, "retained_large": fn, "retained_small": tn,
                     "sensitivity": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                     "flag_rate_exact": float(np.mean(outside[e])),
                     "rmse_all_pp": 100 * math.sqrt(np.mean((pred[e] - y[e]) ** 2)),
                     "rmse_retained_pp": 100 * math.sqrt(np.mean((pred[e & ~outside] - y[e & ~outside]) ** 2)),
                     "rmse_flagged_pp": 100 * math.sqrt(np.mean((pred[e & outside] - y[e & outside]) ** 2)) if np.any(e & outside) else np.nan,
                     "censored_rows": int(c.sum()), "censored_flagged": int(np.sum(outside & c)),
                     "lb_violation_retained": float(np.mean(pred[c & ~outside] < y[c & ~outside])),
                     "lb_violation_flagged": float(np.mean(pred[c & outside] < y[c & outside])) if np.any(c & outside) else np.nan})
    save_csv(pd.DataFrame(rows), "R2_I02_out_of_fold_screening_contingency.csv")
    # Reject-option curve with training-only distance ranks: drop rows in decreasing order of dist/threshold.
    ratio = dist / thr
    order = np.argsort(-ratio[e])
    ye, pe = y[e][order], pred[e][order]
    curve = []
    for frac in np.arange(0.0, 0.51, 0.05):
        k = int(round(frac * len(ye)))
        curve.append({"withheld_fraction": round(frac, 2), "retained_exact_rows": len(ye) - k,
                      "rmse_retained_pp": 100 * math.sqrt(np.mean((pe[k:] - ye[k:]) ** 2))})
    save_csv(pd.DataFrame(curve), "R2_I03_out_of_fold_reject_curve.csv")
    save_csv(pd.DataFrame({"row_uid": df["row_uid"], "study": groups, "is_censored": c, "knn8_distance": dist,
                           "fold_threshold": thr, "outside_training_only_rule": outside, "abs_error": err}),
             "R2_I04_row_distances_out_of_fold.csv")


# ---------------------------------------------------------------------------
# Section J: temporal split margins versus size-matched controls
# ---------------------------------------------------------------------------

def section_temporal(sets) -> None:
    df = sets["P2424"]
    groups = df["ref_id"].astype(str).to_numpy()
    mem = df["mem"].fillna("<NA>").astype(str).to_numpy()
    pfas = df["PFAS"].fillna("<NA>").astype(str).to_numpy()
    cens = df["is_censored"].to_numpy(bool)
    feed = df["feed_group"].astype(str).to_numpy()
    years = df["ref_year"].to_numpy(float)
    ttrain, ttest = np.flatnonzero(years <= 2023), np.flatnonzero(years > 2023)

    def margins(tr, te, design, rep):
        return {"design": design, "repeat": rep, "train_rows": len(tr), "train_studies": len(np.unique(groups[tr])),
                "test_rows": len(te), "test_studies": len(np.unique(groups[te])),
                "test_rows_membrane_seen_pct": 100 * np.mean(np.isin(mem[te], mem[tr])),
                "test_rows_pfas_seen_pct": 100 * np.mean(np.isin(pfas[te], pfas[tr])),
                "test_unique_membranes": len(np.unique(mem[te])),
                "test_unique_membranes_seen_pct": 100 * np.mean(np.isin(np.unique(mem[te]), mem[tr])),
                "test_unique_pfas": len(np.unique(pfas[te])),
                "test_unique_pfas_seen_pct": 100 * np.mean(np.isin(np.unique(pfas[te]), pfas[tr])),
                "test_censored_pct": 100 * cens[te].mean(), "test_real_water_pct": 100 * np.mean(feed[te] == "Real water"),
                "test_exact_sd_pp": 100 * np.std(df["Rejection"].to_numpy()[te][~cens[te]], ddof=1)}

    rows = [margins(ttrain, ttest, "Publication-year holdout", 0)]
    rng = np.random.default_rng(20260814)  # identical generator to run_reviewer_requested_checks.py
    unique_groups = np.unique(groups)
    n_rows, n_st, n_te = len(ttrain), len(np.unique(groups[ttrain])), len(np.unique(groups[ttest]))
    made = attempts = 0
    while made < 40 and attempts < 8000:
        attempts += 1
        sel = rng.choice(unique_groups, size=n_st, replace=False)
        mask = np.isin(groups, sel)
        avail = np.flatnonzero(mask)
        if len(avail) < n_rows:
            continue
        mand = np.array([rng.choice(np.flatnonzero(groups == g)) for g in sel])
        rest = np.setdiff1d(avail, mand, assume_unique=False)
        extra = rng.choice(rest, size=n_rows - len(mand), replace=False)
        tr = np.concatenate([mand, extra])
        te = np.flatnonzero(~mask)
        if len(np.unique(groups[te])) != n_te:
            continue
        made += 1
        rows.append(margins(tr, te, "Size-matched study-grouped control (refitted, 180 rounds)", made))
    out = pd.DataFrame(rows)
    ctrl = pd.read_csv(OUTPUT_ROOT / "04_Tables" / "reviewer_m4_size_matched_temporal_control.csv")
    out = out.merge(ctrl[["repeat", "r2", "rmse"]].rename(columns={"r2": "control_r2", "rmse": "control_rmse"}), on="repeat", how="left")
    save_csv(out, "R2_J01_temporal_split_margins.csv")


# ---------------------------------------------------------------------------
# Section K/L: common-support rule and descriptor correlations / identifiability
# ---------------------------------------------------------------------------

def section_support_and_descriptors(sets) -> None:
    ov = pd.read_csv(OUTPUT_ROOT / "04_Tables" / "monovalent_overlap_diagnostic.csv")
    ov["in_common_support"] = ov["in_common_support"].astype(bool)
    by_study = ov.groupby("ref_id").agg(rows=("in_common_support", "size"), excluded=("in_common_support", lambda s: int((~s).sum())),
                                        nonzero_rows=("exposed", "sum")).reset_index()
    by_study = by_study.loc[by_study["excluded"] > 0]
    save_csv(by_study, "R2_K01_binary_overlap_excluded_rows_by_study.csv")
    df = sets["P2424"]
    x = pd.DataFrame(core.feature_matrix(df), columns=[core.FEATURE_LABELS[f] for f in core.FEATURES])
    rho = x.corr(method="spearman")
    pairs = [{"feature_1": a, "feature_2": b, "spearman_rho": rho.loc[a, b]} for i, a in enumerate(rho.index) for b in rho.columns[i + 1:]]
    pairs = pd.DataFrame(pairs).sort_values("spearman_rho", key=np.abs, ascending=False)
    save_csv(pairs, "R2_L01_predictor_spearman_pairs.csv")
    mol = df[["PFAS", "volume_Å3", "dipole_debye", "pKa"]].drop_duplicates()
    memb = df[["mem", "pwp_LMH/bar", "ON"]].astype(str).drop_duplicates()
    study_of_pair = df.assign(pair=df["pwp_LMH/bar"].astype(str) + "|" + df["ON"].astype(str)).groupby("pair")["ref_id"].nunique()
    ident = pd.DataFrame([
        {"quantity": "PFAS identities", "value": df["PFAS"].nunique()},
        {"quantity": "Distinct (volume, dipole, pKa) descriptor triples", "value": len(mol[["volume_Å3", "dipole_debye", "pKa"]].drop_duplicates())},
        {"quantity": "Membrane identities", "value": df["mem"].nunique()},
        {"quantity": "Distinct (permeance, O:N) pairs", "value": study_of_pair.size},
        {"quantity": "Share of rows whose (permeance, O:N) pair occurs in one study only (%)",
         "value": 100 * float(df.assign(pair=df["pwp_LMH/bar"].astype(str) + "|" + df["ON"].astype(str))["pair"].map(study_of_pair).eq(1).mean())},
    ])
    save_csv(ident, "R2_L02_descriptor_identifiability.csv")


# ---------------------------------------------------------------------------
# Section M: high-frequency entity sensitivity and equal-study weighting
# ---------------------------------------------------------------------------

def section_frequency(sets, fold_store) -> None:
    df = sets["P2424"]
    x = core.feature_matrix(df)
    y = df["Rejection"].to_numpy(float)
    groups = df["ref_id"].astype(str).to_numpy()
    n_study = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(float)
    w = (1.0 / n_study)
    w = w / w.mean()
    weighted, _, _ = tenfold_predictions(df, x, weight=w)
    rows = []
    for label, p in [("Unweighted (published)", fold_store["tenfold"]), ("Equal-study weights (1/n_study)", weighted)]:
        m = metrics(df, p)
        # Equal-study weighted RMSE (each study contributes equally).
        e = ~df["is_censored"].to_numpy(bool)
        se = pd.DataFrame({"g": groups[e], "se": (p[e] - y[e]) ** 2}).groupby("g")["se"].mean()
        m["equal_study_rmse_pp"] = 100 * math.sqrt(se.mean())
        rows.append({"dataset": "P2424", "design": "10-fold study-blocked (145 rounds)", "weighting": label, **m})
    save_csv(pd.DataFrame(rows), "R2_M01_equal_study_weighting.csv")

    cfg = {**CONSERVATIVE, "rounds": 130}
    rng = np.random.default_rng(SEED + 555)
    recs = []
    for etype, col in [("Study", "ref_id"), ("Membrane", "mem"), ("PFAS", "PFAS")]:
        keys = df[col].fillna("<NA>").astype(str).to_numpy()
        counts = pd.Series(keys).value_counts()
        target = 0.20 * len(df)

        def evaluate(selected, kind, rep):
            te = np.flatnonzero(np.isin(keys, selected))
            tr = np.flatnonzero(~np.isin(keys, selected))
            p = fit_predict_aft(df, x, tr, te, cfg, SEED + 600 + rep)
            m = metrics(df, p, te, ranking=False)
            recs.append({"entity_type": etype, "holdout": kind, "repeat": rep, "entities_held_out": len(selected),
                         "held_out_rows": len(te), **{k: m.get(k, np.nan) for k in ["n_exact", "r2", "rmse_pp", "nrmse", "exact_sd_pp", "pooled_c"]},
                         "entities": ";".join(selected) if kind.startswith("Top") else ""})

        top, tot = [], 0
        for k_, v in counts.items():
            if tot >= target:
                break
            top.append(k_); tot += v
        evaluate(top, "Top-frequency entities (>=20% rows)", 0)
        for rep in range(1, 21):
            perm = rng.permutation(counts.index.to_numpy())
            sel, tot = [], 0
            for k_ in perm:
                if tot >= target:
                    break
                sel.append(k_); tot += counts[k_]
            evaluate(sel, "Random entities (>=20% rows)", rep)
        print(f"frequency holdout {etype} done", flush=True)
    save_csv(pd.DataFrame(recs), "R2_M02_high_frequency_entity_holdout.csv")
    loso = pd.read_csv(OUT / "R2_B01_loso_complete_fold_table.csv")
    pg = pd.read_csv(OUT / "R2_D02_harmonized_leave_one_group_metrics.csv") if (OUT / "R2_D02_harmonized_leave_one_group_metrics.csv").exists() else None
    cors = []
    sub = loso.loc[loso["exact_rows"] >= 5]
    for metric in ["model_rmse_pp", "model_mae_pp"]:
        r = stats.spearmanr(sub["test_rows"], sub[metric])
        cors.append({"entity_type": "Study (LOSO, 115 rounds)", "metric": metric, "groups": len(sub), "spearman_rho_vs_rows": r.statistic, "p_value": r.pvalue})
    if pg is not None:
        for design, lab in [("Leave-one-membrane-out", "Membrane (harmonized)"), ("Leave-one-PFAS-out", "PFAS (harmonized)")]:
            s = pg.loc[(pg["design"] == design) & (pg["n_exact"] >= 5)]
            for metric in ["rmse_pp", "mae_pp"]:
                r = stats.spearmanr(s["group_rows"], s[metric])
                cors.append({"entity_type": lab, "metric": metric, "groups": len(s), "spearman_rho_vs_rows": r.statistic, "p_value": r.pvalue})
    save_csv(pd.DataFrame(cors), "R2_M03_error_vs_entity_frequency.csv")


# ---------------------------------------------------------------------------
# Section N: author-rule (A2353) companion results for the new analyses
# ---------------------------------------------------------------------------

def section_author_rule(sets) -> None:
    df = sets["A2353"]
    x = core.feature_matrix(df)
    y = df["Rejection"].to_numpy(float)
    c = df["is_censored"].to_numpy(bool)
    groups = df["ref_id"].astype(str).to_numpy()
    cfg = {**CONSERVATIVE, "rounds": 130}
    rows = []
    for name, splits in [
        ("10-fold row-random", list(KFold(10, shuffle=True, random_state=SEED).split(x))),
        ("10-fold study-blocked", list(GroupKFold(10).split(x, y, groups))),
    ]:
        pred = np.full(len(df), np.nan); base = np.full(len(df), np.nan)
        for k, (tr, te) in enumerate(splits, 1):
            pred[te] = fit_predict_aft(df, x, tr, te, cfg, SEED + 1000 + k)
            base[te] = y[tr[~c[tr]]].mean()
        rows.append({"dataset": "A2353", "design": name, "predictor": "Conservative AFT (130 rounds)", **metrics(df, pred)})
        rows.append({"dataset": "A2353", "design": name, "predictor": "Training-fold exact-row mean", **metrics(df, base)})
    save_csv(pd.DataFrame(rows), "R2_N01_author_rule_harmonized.csv")


SECTIONS = ["registry", "folds", "leakage", "harmonized", "nested", "baselines", "aft", "censoring", "applicability",
            "temporal", "support", "frequency", "author"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=SECTIONS)
    args = parser.parse_args()
    run = set(args.only or SECTIONS)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw, sets = load_datasets()
    timings = {}

    def timed(name, fn, *a):
        t = time.time()
        result = fn(*a)
        timings[name] = round(time.time() - t, 1)
        return result

    if "registry" in run:
        timed("registry", section_registry, raw, sets)
    fold_store = timed("folds", section_fold_tables, sets) if run & {"folds", "baselines", "applicability", "frequency"} else None
    if "leakage" in run:
        timed("leakage", section_leakage, sets)
    if "harmonized" in run:
        timed("harmonized", section_harmonized, sets)
    nested = timed("nested", section_nested, sets) if run & {"nested", "baselines"} else None
    if "baselines" in run:
        timed("baselines", section_baselines, sets, fold_store, nested)
    if "aft" in run:
        timed("aft", section_aft_diagnostics, sets)
    if "censoring" in run:
        timed("censoring", section_censoring_simulation, sets)
    if "applicability" in run:
        timed("applicability", section_applicability, sets, fold_store)
    if "temporal" in run:
        timed("temporal", section_temporal, sets)
    if "support" in run:
        timed("support", section_support_and_descriptors, sets)
    if "frequency" in run:
        timed("frequency", section_frequency, sets, fold_store)
    if "author" in run:
        timed("author", section_author_rule, sets)
    save_json({"script": "07_Scripts/run_round2_revision_analyses.py", "seed": SEED, "sections_run": sorted(run),
               "section_runtime_seconds": timings, "total_runtime_seconds": round(time.time() - t0, 1),
               "datasets": {k: len(v) for k, v in sets.items()}}, f"R2_run_manifest_{'_'.join(sorted(run)) if args.only else 'all'}.json")


if __name__ == "__main__":
    main()
