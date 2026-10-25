"""Author-rule (A2353) membership sensitivity reported in Table S16.

Applies the source author's stated join omissions to the public workbook, keeps the
original 13-study untouched test, shared-row allocations and 10 study-fold assignments,
and refits the two prespecified architectures without model selection.
Outputs: 04_Tables/AuthorRule (or $PFAS_OUTPUT_ROOT/04_Tables/AuthorRule).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PFAS_OUTPUT_ROOT", ROOT)) / "04_Tables" / "AuthorRule"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(OUT.parents[1] / "99_Working_Files" / "matplotlib_cache")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT / "07_Scripts"))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
import sklearn
import xgboost
import audit_internal_high_accuracy_claim as audit
import generate_pfas_paper_figures as core


MEMBRANE_IDS = {"NF270-PEI_Jeong", "NF270-PAA_Jeong"}
REFERENCE_IDS = {"Ma J., 2022", "Ma et al, 2024 (2)"}
FANG_FOLD = 1  # Declared before fitting; the new study was absent from primary NF/RO set.
start = time.time()


def write_csv(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def derived_frame(raw, mask):
    df = raw.loc[mask].copy().reset_index(drop=True)
    rejection = df["Rejection"].clip(0.500001, 0.999999)
    df["logit_R"] = np.log(rejection / (1.0 - rejection))
    df["odds_R"] = rejection / (1.0 - rejection)
    df["feed_group"] = df["feed_type"].fillna("Unreported").replace(
        {"Raw": "Real water", "Lab": "Synthetic", "Pure water": "Pure water"}
    )
    return df


def metrics(df, idx, pred):
    part = df.iloc[idx]
    y = part["Rejection"].to_numpy(float)
    censored = part["is_censored"].to_numpy(bool)
    summary = core.evaluation(y, pred, censored)
    by_study = []
    study_array = part["ref_id"].astype(str).to_numpy()
    for study in np.unique(study_array):
        mask = study_array == study
        value = core.censored_concordance(y[mask], pred[mask], censored[mask])
        if np.isfinite(value):
            by_study.append(value)
    summary["rmse_pp"] = summary["rmse"] * 100
    summary["mae_pp"] = summary["mae"] * 100
    summary["exact_outcome_sd_pp"] = float(np.std(y[~censored], ddof=1) * 100)
    summary["nrmse_by_exact_test_sd"] = summary["rmse_pp"] / summary["exact_outcome_sd_pp"]
    summary["studies"] = part["ref_id"].nunique()
    summary["n_censored"] = int(censored.sum())
    summary["c_index_equal_study_within_study"] = float(np.mean(by_study))
    summary["scorable_studies_for_within_study_C"] = len(by_study)
    return summary


# Force the actual workbook, not a possibly older JSON export.
core.JSON_PATH = OUT / "not_a_source.json"
raw, _ = core.load_data()
raw["source_excel_row"] = raw.index + 2
threshold = raw["Rejection"].ge(0.5)
nfro = raw["mem_class"].astype(str).str.upper().isin(["NF", "RO"])
email_drop = raw["mem"].isin(MEMBRANE_IDS) | raw["ref_id"].isin(REFERENCE_IDS)
primary = derived_frame(raw, threshold & nfro)
author_rule = derived_frame(raw, threshold & ~email_drop)
shared = derived_frame(raw, threshold & nfro & ~email_drop)
assert len(raw) == 2517 and len(primary) == 2424 and len(author_rule) == 2353 and len(shared) == 2327
assert author_rule["ref_id"].nunique() == 60 and len(author_rule.loc[~author_rule["mem_class"].isin(["NF", "RO"])]) == 26

membership = raw[["source_excel_row", "ref_id", "ref_title", "mem", "mem_class", "Rejection", "PFAS", "LOQ"]].copy()
membership["passes_rejection_050"] = threshold
membership["current_primary_2424"] = threshold & nfro
membership["email_named_membrane"] = raw["mem"].isin(MEMBRANE_IDS)
membership["email_named_reference"] = raw["ref_id"].isin(REFERENCE_IDS)
membership["author_rule_2353"] = threshold & ~email_drop
membership["shared_NFRO_2327"] = threshold & nfro & ~email_drop
write_csv(membership, "source_membership_audit.csv")
flow = []
for name, frame in [("Public workbook", raw), ("R >= 0.50", raw.loc[threshold]), ("Current primary NF/RO", primary), ("Author-email-rule reconstruction", author_rule), ("NF/RO intersection", shared)]:
    flow.append({"analysis_set": name, "rows": len(frame), "studies": frame["ref_id"].nunique(), "membranes": frame["mem"].nunique(), "PFAS": frame["PFAS"].nunique(), "censored": int(frame["is_censored"].sum()), "source_labelled_UF_rows": int(frame["mem_class"].eq("UF").sum())})
write_csv(pd.DataFrame(flow), "source_membership_flow.csv")

# Reproduce the original fixed outer holdout and row-random development partition.
x_primary = core.feature_matrix(primary)
groups_primary = primary["ref_id"].astype(str).to_numpy()
development, unseen = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=audit.SEED).split(x_primary, groups=groups_primary))
familiar_train, familiar_test = train_test_split(development, test_size=0.15, random_state=audit.SEED + 20, shuffle=True)
primary_regime = {}
for label, idx in [("development_familiar_train", familiar_train), ("development_familiar_test", familiar_test), ("unseen_study_test", unseen)]:
    primary_regime.update({int(primary.iloc[i]["source_excel_row"]): label for i in idx})
unseen_studies = set(primary.iloc[unseen]["ref_id"].astype(str))
primary["frozen_regime"] = primary["source_excel_row"].map(primary_regime)
author_rule["frozen_regime"] = author_rule["source_excel_row"].map(primary_regime)

# Shared rows retain their exact roles. Added rows inherit their study's outer role.
new_unseen = author_rule["frozen_regime"].isna() & author_rule["ref_id"].isin(unseen_studies)
author_rule.loc[new_unseen, "frozen_regime"] = "unseen_study_test"
new_development_idx = author_rule.index[author_rule["frozen_regime"].isna()].to_numpy()
new_train, new_test = train_test_split(new_development_idx, test_size=0.15, random_state=audit.SEED + 20, shuffle=True)
author_rule.loc[new_train, "frozen_regime"] = "development_familiar_train"
author_rule.loc[new_test, "frozen_regime"] = "development_familiar_test"
assert author_rule["frozen_regime"].notna().all()

# Retain the original study-to-fold assignments, including all shared rows.
study_fold = {}
for fold, (_, te) in enumerate(GroupKFold(n_splits=10).split(x_primary, groups=groups_primary), 1):
    study_fold.update({str(study): fold for study in primary.iloc[te]["ref_id"].unique()})
for study in author_rule["ref_id"].unique():
    if str(study) not in study_fold:
        assert str(study) == "Fang et al, 2024"
        study_fold[str(study)] = FANG_FOLD
for frame in [primary, author_rule]:
    frame["frozen_study_fold"] = frame["ref_id"].astype(str).map(study_fold).astype(int)

configs = {name: next(dict(c) for c in audit.CANDIDATES if c["name"] == name) for name in ["conservative", "d6_c1"]}
summary_rows, fold_rows, prediction_rows, allocation_rows = [], [], [], []
for set_name, frame in [("Primary NFRO 2424", primary), ("Author-rule 2353 (includes 26 source-labelled UF)", author_rule)]:
    x = core.feature_matrix(frame)
    allocation_rows.append(frame[["source_excel_row", "ref_id", "mem_class", "frozen_regime", "frozen_study_fold"]].assign(analysis_set=set_name))
    # XGBoost row subsampling depends on input order. Preserve the original
    # train_test_split order, not merely the set of original training rows.
    if frame is primary:
        tr = familiar_train.copy()
    else:
        lookup = dict(zip(frame["source_excel_row"].astype(int), frame.index))
        tr = np.asarray(
            [lookup[int(primary.iloc[i]["source_excel_row"])]
             for i in familiar_train
             if int(primary.iloc[i]["source_excel_row"]) in lookup]
            + new_train.tolist(), dtype=int
        )
    for name, cfg in configs.items():
        model = audit.fit("aft", x, frame, tr, cfg, audit.SEED + 900)
        for regime in ["development_familiar_test", "unseen_study_test"]:
            te = frame.index[frame["frozen_regime"].eq(regime)].to_numpy()
            pred = audit.predict("aft", model, x[te])
            record = {"analysis_set": set_name, "design": "Fixed familiar/unseen same fitted model", "architecture": name, "evaluation": regime, "fit_rows": len(tr), "fit_studies": frame.iloc[tr]["ref_id"].nunique(), **metrics(frame, te, pred)}
            summary_rows.append(record)
            prediction_rows.append(pd.DataFrame({"analysis_set": set_name, "design": "Fixed familiar/unseen", "architecture": name, "evaluation": regime, "source_excel_row": frame.iloc[te]["source_excel_row"].to_numpy(), "prediction": pred}))
            print(set_name, name, regime, "R2", record["r2"], "logitR2", record["logit_r2"], flush=True)
    oof = np.full(len(frame), np.nan)
    baseline = np.full(len(frame), np.nan)
    cfg = {**configs["conservative"], "rounds": 145}
    for fold in range(1, 11):
        te = frame.index[frame["frozen_study_fold"].eq(fold)].to_numpy()
        tr = frame.index[frame["frozen_study_fold"].ne(fold)].to_numpy()
        model = audit.fit("aft", x, frame, tr, cfg, 700 + fold)
        oof[te] = audit.predict("aft", model, x[te])
        exact_tr = tr[~frame.iloc[tr]["is_censored"].to_numpy(bool)]
        baseline[te] = frame.iloc[exact_tr]["Rejection"].mean()
        for name, pred in [("conservative_145", oof[te]), ("training-fold exact-row mean", baseline[te])]:
            fold_rows.append({"analysis_set": set_name, "fold": fold, "architecture": name, "fit_rows": len(tr), **metrics(frame, te, pred)})
    assert np.isfinite(oof).all() and np.isfinite(baseline).all()
    for name, pred in [("conservative_145", oof), ("training-fold exact-row mean", baseline)]:
        record = {"analysis_set": set_name, "design": "10-fold frozen study assignments", "architecture": name, "evaluation": "Pooled out-of-fold", "fit_rows": None, "fit_studies": None, **metrics(frame, np.arange(len(frame)), pred)}
        summary_rows.append(record)
        prediction_rows.append(pd.DataFrame({"analysis_set": set_name, "design": "10-fold frozen study assignments", "architecture": name, "evaluation": "Pooled out-of-fold", "source_excel_row": frame["source_excel_row"].to_numpy(), "prediction": pred}))
        print(set_name, name, "10fold pooled R2", record["r2"], "RMSEpp", record["rmse_pp"], flush=True)
    write_csv(pd.DataFrame(summary_rows), "sensitivity_summary.csv")

write_csv(pd.DataFrame(summary_rows), "sensitivity_summary.csv")
write_csv(pd.DataFrame(fold_rows), "sensitivity_fold_metrics.csv")
write_csv(pd.concat(prediction_rows, ignore_index=True), "sensitivity_predictions.csv")
write_csv(pd.concat(allocation_rows, ignore_index=True), "sensitivity_split_assignments.csv")

metadata = {
    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
    "workbook_sha256": hashlib.sha256(core.WORKBOOK_PATH.read_bytes()).hexdigest(),
    "author_rules_mapped_to_exact_workbook_values": {"mem": sorted(MEMBRANE_IDS), "ref_id": sorted(REFERENCE_IDS)},
    "features": core.FEATURES,
    "same_model_configurations": configs,
    "same_model_fit_seed": audit.SEED + 900,
    "grouped_config": {**configs["conservative"], "rounds": 145},
    "grouped_fit_seed": "700 + fold, folds numbered 1 to 10",
    "original_holdout_study_ids": sorted(unseen_studies),
    "author_rule_holdout_study_ids": sorted(set(author_rule.loc[author_rule["frozen_regime"].eq("unseen_study_test"), "ref_id"])),
    "shared_rows_keep_original_regimes_and_folds": True,
    "shared_training_rows_keep_original_train_test_split_order": True,
    "new_rows_inherit_outer_study_identity": True,
    "new_source_study_Fang_role": "development; study fold 1 declared before fitting",
    "new_development_rows_allocation": "train_test_split(test_size=0.15, random_state=20260836, shuffle=True) applied only to newly added development rows",
    "new_development_rows": int(len(new_development_idx)),
    "new_development_train_rows": int(len(new_train)),
    "new_development_familiar_test_rows": int(len(new_test)),
    "new_unseen_rows": int(new_unseen.sum()),
    "metrics": "R2, RMSE, MAE on exact rows; RMSE/MAE additionally reported in percentage points. Logit metrics use existing core.evaluation clipping [0.500001,0.999999]. C-index pooled censored-compatible; separate equal-study within-study C-index also supplied.",
    "no_hyperparameter_selection_performed": True,
    "not_an_exact_original_snapshot_or_fitted_model_reproduction": True,
    "author_rule_view_contains_26_source_labelled_UF_rows": True,
    "runtime_seconds": round(time.time() - start, 2),
}
(OUT / "sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE", metadata["runtime_seconds"], "seconds", flush=True)
