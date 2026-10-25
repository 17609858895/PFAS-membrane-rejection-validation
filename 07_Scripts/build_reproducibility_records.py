"""Reproducibility records: data dictionary, environment lock, provenance, hashes, verification.

Writes (under ``$PFAS_OUTPUT_ROOT``, default the repository root):

* ``requirements-lock.txt`` - exact versions of every imported distribution and its dependencies.
* ``04_Tables/Round2/R2_P01_data_dictionary.csv`` - predictors and identifiers: source, unit,
  transformation, missingness, imputation rule and pipeline stage.
* ``04_Tables/Round2/R2_P02_display_item_provenance.csv`` - dataset label, splitter, folds, seeds,
  features, loss, hyperparameter source, post-processing and result files for each table/figure.
* ``reproducibility_manifest.csv`` - every generated file with its generating script and SHA-256.
* ``verification_against_committed.csv`` (with ``--compare-to``) - regenerated versus committed tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("PFAS_OUTPUT_ROOT", ROOT))
sys.path.insert(0, str(ROOT / "07_Scripts"))

DISTRIBUTIONS = [
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "statsmodels", "openpyxl", "matplotlib", "seaborn",
    "joblib", "threadpoolctl", "patsy", "packaging", "python-dateutil", "pytz", "tzdata", "six", "et-xmlfile",
    "contourpy", "cycler", "fonttools", "kiwisolver", "pillow", "pyparsing",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment_lock() -> None:
    lines = [f"# Python {platform.python_version()} ({platform.platform()})",
             "# Exact versions used for the manuscript analyses; install with: python -m pip install -r requirements-lock.txt"]
    for dist in DISTRIBUTIONS:
        try:
            lines.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            lines.append(f"# {dist}: not installed")
    (OUTPUT_ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_dictionary() -> pd.DataFrame:
    import generate_pfas_paper_figures as core

    _, df = core.load_data()
    rows = []
    sources = {
        "pwp_LMH/bar": ("Membrane property reported in the original study; compiled by the source authors", "L m-2 h-1 bar-1"),
        "ON": ("Membrane surface O:N ratio reported in the original study (XPS); compiled by the source authors", "dimensionless"),
        "pH": ("Feed condition reported in the original study", "pH units"),
        "volume_Å3": ("PFAS descriptor supplied in the source workbook; computed by the source authors with Chemicalize (ChemAxon Marvin), independent of rejection outcomes", "Å3"),
        "dipole_debye": ("PFAS descriptor supplied in the source workbook; computed by the source authors with Chemicalize (ChemAxon Marvin)", "D"),
        "pKa": ("PFAS descriptor supplied in the source workbook; computed by the source authors with Chemicalize (ChemAxon Marvin)", "dimensionless"),
        "temp_ºC": ("Operating condition reported in the original study (only when controlled)", "°C"),
        "flux_LMH": ("Operating condition reported in the original study", "L m-2 h-1"),
        "pfas_feed_conc_µg/L": ("Feed condition reported in the original study", "µg L-1"),
        "mon_cat_mM": ("Feed composition reported in the original study, summed by the source authors by ion valency", "mM"),
        "di_cat_mM": ("Feed composition reported in the original study, summed by the source authors by ion valency", "mM"),
        "tri_cat_mM": ("Feed composition reported in the original study, summed by the source authors by ion valency", "mM"),
        "doc_mg/L": ("Feed composition reported in the original study; DOC of model foulants converted by the source authors", "mg L-1"),
    }
    for i, col in enumerate(core.FEATURES, 1):
        miss = df[col].isna()
        rows.append({
            "role": "Predictor", "no": i, "source_column": col, "label": core.FEATURE_LABELS[col],
            "source": sources[col][0], "unit": sources[col][1],
            "transformation": "log1p(max(x, 0))" if col in core.LOG_FEATURES else "none",
            "missing_n_P2424": int(miss.sum()), "missing_pct_P2424": round(100 * float(miss.mean()), 1),
            "missing_studies_P2424": int(df.loc[miss, "ref_id"].nunique()),
            "imputation_xgboost": "none (native missing-value routing learned in each training fold)",
            "imputation_other_models": "median fitted on the training fold (kNN distance, propensity, Tobit/linear models)",
            "pipeline_stage": "transformation is row-wise and precedes splitting; no statistic is estimated on the full data",
            "created_or_modified_by_this_study": "no (value taken unchanged from workbook)",
        })
    others = [
        ("Outcome", "Rejection", "Observed rejection (fraction); exact when LOQ = 1, lower bound when LOQ = 'right'", "fraction",
         "clipped to [0.500001, 0.999999] (P2424, A2353) or [1e-6, 0.999999] (U2487); logit and odds derived row-wise"),
        ("Censoring flag", "LOQ", "'right' = right-censored lower bound; 1 = exact. No left- or interval-censored codes occur.", "-", "is_censored = (LOQ == 'right')"),
        ("Group", "ref_id", "Publication identifier (study); primary validation group", "-", "string"),
        ("Group", "mem", "Membrane identifier", "-", "string"),
        ("Group", "PFAS", "PFAS identifier", "-", "string"),
        ("Eligibility", "mem_class", "Membrane class (NF, RO, UF, missing)", "-", "NF/RO rule"),
        ("Temporal split", "ref_year", "Publication year", "year", "<= 2023 training; 2024-2025 test"),
        ("Stratification", "feed_type", "Feed matrix (pure water, synthetic 'Lab', real 'Raw')", "-", "relabelled for display"),
        ("Stratification", "carbons", "Number of backbone carbons", "-", "binned C<=3, C4-5, C6-7, C>=8 for display"),
    ]
    for role, col, src, unit, tr in others:
        rows.append({"role": role, "source_column": col, "label": col, "source": src, "unit": unit, "transformation": tr,
                     "missing_n_P2424": int(df[col].isna().sum()), "pipeline_stage": "row-wise before splitting",
                     "created_or_modified_by_this_study": "derived columns only"})
    out = pd.DataFrame(rows)
    path = OUTPUT_ROOT / "04_Tables" / "Round2" / "R2_P01_data_dictionary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


PROVENANCE = [
    # item, dataset, splitter / design, folds or fits, seeds, features, loss & hyperparameter source, post-processing, script, result files
    ("Figure 1", "-", "Schematic workflow (not generated from data)", "-", "-", "-", "-", "-", "drawn by the authors", "03_Figures/PFAS_claim_matched_validation_workflow_*"),
    ("Figure 2", "W2517 / P2424 / A2353", "Descriptive (no model)", "-", "-", "-", "-", "counts on all rows", "generate_pfas_paper_figures.py", "03_Figures/Fig01_*; 04_Tables/dataset_flow.csv"),
    ("Figure 3; Table S6; Table S14", "P2424", "GroupShuffleSplit(test 0.20) -> 13 untouched studies; familiar rows train_test_split(0.15)", "1 fixed split", "split 20260816; familiar 20260836; selection KFold(3) 20260817; fit 20261716",
     "13 predictors", "normal AFT; d6_c1 chosen among 12 candidates by row-random 3-fold CV (Fig. 3); conservative and d6_c1 both shown in Table S6", "P50 back-transform; exact-row metrics",
     "audit_disjoint_high_accuracy_claim.py; generate_disjoint_transfer_fig2.py; run_major_revision_robustness.py", "99_Working_Files/high_accuracy_claim_audit/disjoint_claim_audit.json; 04_Tables/Revision/Table_R3_02*, R3_03*, R3_04*"),
    ("Figure 4", "P2424", "10-fold GroupKFold by study (deterministic)", "10", "700 + fold", "13 predictors",
     "normal AFT (conservative, 145 rounds) vs squared-error XGBoost (lower bounds as exact; censored excluded); fixed a priori", "P50, logistic-normal mean, P90",
     "generate_pfas_paper_figures.py; run_reviewer_requested_checks.py", "04_Tables/censoring_model_*.csv; study_mean_residuals.csv; reviewer_m6_m7_retransformation_and_bounds.csv"),
    ("Figure 5; Table S11", "P2424", "10-fold GroupKFold outer; calibration studies inside outer training fold", "10",
     "rng 20260717 (row scores); 20260831+60000 (study scores)", "13 predictors", "conservative AFT, 105 rounds x 8 bootstrap models (row) or 130 x 12 (study)",
     "clip 0.50-1.00 (row) or 0-1 and 0.50-1.00 (study); empirical coverage", "generate_pfas_paper_figures.py; run_major_revision_robustness.py",
     "04_Tables/group_conformal_*.csv; Revision/Table_R3_12*, R3_13*"),
    ("Figure 6", "P2424", "Leave-one-study-out", "61", "500 + fold", "13 predictors (kNN: median-imputed, standardised in training fold)",
     "conservative AFT, 115 rounds", "pooled retrospective P90 threshold (display); training-only fold thresholds in Table S24",
     "generate_pfas_paper_figures.py; run_major_revision_robustness.py; run_round2_revision_analyses.py", "Revision/Table_R2_04*; Round2/R2_I0*"),
    ("Figure 7a", "P2424", "Leave-one-study-out; exact rows aggregated over folds (n = 2,117)", "61", "500 + fold", "13 predictors", "conservative AFT, 115 rounds", "MAE by chain group x matrix, rejection scale", "generate_pfas_paper_figures.py", "03_Figures/Fig06_*"),
    ("Figure 7b-e", "P2424", "5-fold GroupKFold by study (deterministic)", "5", "1200 + fold", "13 predictors", "conservative AFT, 150 rounds",
     "SHAP on the margin (log odds) of test rows; ALE on rejection scale in training rows, 15 quantile bins", "generate_pfas_paper_figures.py", "04_Tables/shap_*.csv"),
    ("Figure 8; Tables S12, S15", "P2424 exact rows with complete exposure (878/939/766 rows)", "Study-demeaned regression; cross-fitted propensity GroupKFold(5)", "-",
     "404 (core); 20260814 (Rademacher); 20260831+50000 (Webb)", "exposure + 8 covariates", "OLS (primary); ridge 0-16 (sensitivity)", "back-transform at median logit",
     "generate_pfas_paper_figures.py; run_reviewer_requested_checks.py; run_major_revision_robustness.py", "within_study_adjusted_associations.csv; monovalent_*; Revision/Table_R1_02*, R3_11*"),
    ("Figure 9; Fig. S2; Table S26", "P2424", "Publication year <= 2023 vs 2024-2025; 40 size-matched study-grouped controls", "1 + 40", "1600; controls rng 20260814, fit 3600 + k",
     "13 predictors", "conservative AFT, 180 rounds", "exact-row metrics; add-one tail probabilities", "generate_pfas_paper_figures.py; run_reviewer_requested_checks.py",
     "publication_year_*.csv; reviewer_m4_size_matched_temporal_control.csv; Round2/R2_J01*"),
    ("Table 1 (upper block)", "P2424", "Row-random and cluster-stratified 85:15 x 12 (row-averaged); LOSO; LOMO; LOPO; temporal", "12/12/61/112/37/1",
     "110 + repeat; 500 + fold; 1600", "13 predictors", "conservative AFT; 130/130/115/115/115/180 rounds", "mean prediction per tested row; exact-row metrics",
     "generate_pfas_paper_figures.py; repair_reviewer_consistency_20260905.py", "split_strategy_summary.csv; Consistency_20260905/Table_C01*"),
    ("Table 1 (lower blocks); Table S20", "P2424, U2487", "KFold(10, shuffle) / StratifiedKFold(10) on k-means labels / GroupKFold(10) / LOSO / LOMO / LOPO / temporal", "10/10/10/61/112/37/1",
     "splits 20261025; fits 20262025 + fit index", "13 predictors", "conservative AFT, 130 rounds for every design", "exact-row metrics; same 2,117 exact rows except temporal",
     "run_round2_revision_analyses.py", "Round2/R2_D01*, R2_D02*"),
    ("Table 2; Tables S10, S21", "P2424, U2487", "10-fold GroupKFold by study (deterministic)", "10", "700 + fold; nested 20261025 + 5000 + 97*fold", "13 predictors or stated subset",
     "conservative AFT 145 rounds; grouped-nested selection among 12 candidates; Tobit models by maximum likelihood", "within-study C averaged over 59 studies; paired study bootstrap 5,000 draws (fixed predictions)",
     "run_reviewer_requested_checks.py; repair_reviewer_consistency_20260905.py; run_round2_revision_analyses.py", "reviewer_m3_*; Consistency_20260905/Table_C06*, C10*; Round2/R2_F0*, R2_E0*"),
    ("Tables S2, S3", "P2424", "Descriptive (no model)", "-", "-", "-", "-", "counts; frozen Crossref DOI lookup", "build_supplementary_evidence.py; build_reproducibility_records.py",
     "Supplementary/Table_S3_study_inventory.csv; Round2/R2_P01_data_dictionary.csv"),
    ("Table S1", "W2517, R2450, P2424, A2353, S2327, U2487", "Eligibility rules", "-", "-", "-", "-", "row-level reason codes", "run_round2_revision_analyses.py", "Round2/R2_A01*, R2_A02*, R2_A03*"),
    ("Table S4; Tables S17-S18", "P2424", "Published designs; complete LOSO and 10-fold tables", "-", "as in Table 1 and Table 2", "13 predictors", "conservative AFT", "per-fold exact-row metrics and training-fold mean",
     "build_supplementary_evidence.py; run_round2_revision_analyses.py", "Supplementary/Table_S1_fold_level_metrics.csv; Round2/R2_B0*"),
    ("Tables S5, S7, S8", "P2424", "Inner GroupKFold(3) on fixed fitting rows; 50 GroupShuffleSplit partitions", "12 candidates; 50", "20260831 (+1000; +10000 + repeat)", "13 predictors", "normal AFT candidates", "empirical partition ranges",
     "run_major_revision_robustness.py; repair_reviewer_consistency_20260905.py", "Revision/Table_R3_01*, R3_05*-R3_08*; Consistency_20260905/Table_C03*, C04*"),
    ("Table S9", "P2424, U2487", "10-fold GroupKFold", "10", "700 + fold", "13 predictors", "conservative AFT, 145 rounds", "exact-row metrics", "run_major_revision_robustness.py", "Revision/Table_R3_09*"),
    ("Table S16", "A2353", "Original fixed split and study-fold assignments retained", "1; 10", "fit 20261716; 700 + fold; added rows 20260836", "13 predictors", "conservative and d6_c1 without retuning", "exact-row metrics",
     "run_author_rule_sensitivity.py", "AuthorRule/sensitivity_summary.csv"),
    ("Table S19", "P2424", "All designs above", "-", "-", "-", "-", "share of test rows/entities seen in training", "run_round2_revision_analyses.py", "Round2/R2_C0*"),
    ("Table S22; Fig. S5", "P2424", "10-fold GroupKFold; inner GroupKFold(3) for scale recalibration", "10 (x3)", "700 + fold; 8000 + 10*fold + k", "13 predictors", "conservative AFT, 145 rounds", "randomized quantile and Cox-Snell residuals; parametric interval coverage",
     "run_round2_revision_analyses.py", "Round2/R2_G0*"),
    ("Table S23; Fig. S6", "P2424 exact rows (semi-synthetic censoring)", "10-fold GroupKFold", "10", "700 + fold", "13 predictors", "AFT vs squared-error XGBoost (145 rounds)", "errors against the known values", "run_round2_revision_analyses.py", "Round2/R2_H01*"),
    ("Table S24; Fig. S7", "P2424", "Leave-one-study-out; training-only thresholds", "61", "500 + fold", "13 predictors", "conservative AFT, 115 rounds", "contingency of flagged vs large-error rows", "run_round2_revision_analyses.py", "Round2/R2_I0*"),
    ("Table S25", "P2424", "10-fold GroupKFold (weights); entity holdouts", "10; 21 per entity type", "700 + fold; 20261025 + 600 + repeat", "13 predictors", "conservative AFT", "exact-row metrics", "run_round2_revision_analyses.py", "Round2/R2_M0*"),
]


def provenance_table() -> pd.DataFrame:
    cols = ["display_item", "dataset_label", "splitter_or_design", "folds_or_fits", "random_seeds", "features", "loss_and_hyperparameter_source",
            "post_processing", "generating_scripts", "result_files"]
    out = pd.DataFrame(PROVENANCE, columns=cols)
    path = OUTPUT_ROOT / "04_Tables" / "Round2" / "R2_P02_display_item_provenance.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def generating_script(rel: str) -> str:
    rules = [
        ("04_Tables/Round2/R2_P", "build_reproducibility_records.py"), ("04_Tables/Round2/", "run_round2_revision_analyses.py"),
        ("04_Tables/AuthorRule/", "run_author_rule_sensitivity.py"), ("04_Tables/Revision/", "run_major_revision_robustness.py"),
        ("04_Tables/Consistency_20260905/", "repair_reviewer_consistency_20260905.py"), ("04_Tables/Supplementary/", "build_supplementary_evidence.py"),
        ("04_Tables/reviewer_m", "run_reviewer_requested_checks.py"), ("04_Tables/", "generate_pfas_paper_figures.py"),
        ("03_Figures/Fig02_", "generate_disjoint_transfer_fig2.py"),
        ("03_Figures/Supplementary/FigS05_AFT", "draw_round2_supplementary_figures.py"),
        ("03_Figures/Supplementary/FigS06_Censoring", "draw_round2_supplementary_figures.py"),
        ("03_Figures/Supplementary/FigS07_Out", "draw_round2_supplementary_figures.py"),
        ("03_Figures/Supplementary/", "build_supplementary_evidence.py"),
        ("03_Figures/", "generate_pfas_paper_figures.py"), ("99_Working_Files/high_accuracy_claim_audit/", "audit_disjoint_high_accuracy_claim.py"),
        ("99_Working_Files/major_revision/", "run_major_revision_robustness.py"), ("requirements-lock.txt", "build_reproducibility_records.py"),
    ]
    for prefix, script in rules:
        if rel.startswith(prefix):
            return script
    return ""


def manifest() -> pd.DataFrame:
    log_path = OUTPUT_ROOT / "run_all_log.json"
    runtimes = {}
    if log_path.exists():
        for step in json.loads(log_path.read_text(encoding="utf-8"))["steps"]:
            runtimes[Path(step["script"]).name] = step["runtime_seconds"]
    rows = []
    for sub in ["03_Figures", "04_Tables", "99_Working_Files/high_accuracy_claim_audit", "99_Working_Files/major_revision"]:
        base = OUTPUT_ROOT / sub
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and "matplotlib_cache" not in p.parts):
            rel = path.relative_to(OUTPUT_ROOT).as_posix()
            script = generating_script(rel)
            rows.append({"file": rel, "bytes": path.stat().st_size, "sha256": sha256(path), "generating_script": script,
                         "script_runtime_seconds": runtimes.get(script, np.nan)})
    inputs = [ROOT / "05_Source_Data" / "es6c02287_si_002.xlsx", ROOT / "05_Source_Data" / "derived" / "crossref_study_doi_cache.json",
              ROOT / "candidate_configurations.json"] + sorted((ROOT / "07_Scripts").glob("*.py"))
    for path in inputs:
        rows.append({"file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path),
                     "generating_script": "input (unchanged)" if "05_Source_Data" in path.parts else "code/configuration"})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_ROOT / "reproducibility_manifest.csv", index=False, encoding="utf-8-sig")
    return out


def compare_tables(new_root: Path, old_root: Path) -> pd.DataFrame:
    rows = []
    pairs = [(p, old_root / p.relative_to(new_root)) for p in sorted((new_root / "04_Tables").rglob("*.csv"))]
    pairs += [(new_root / "99_Working_Files" / "major_revision" / n, old_root / "06_Reviewer_Revision_Results" / n)
              for n in ["applicability_domain_definition.json", "temporal_control_matching_definition.json"]]
    for new, old in pairs:
        rel = new.relative_to(new_root).as_posix()
        if not old.exists() or not new.exists():
            rows.append({"file": rel, "status": "no committed counterpart" if new.exists() else "not regenerated"})
            continue
        if sha256(new) == sha256(old):
            rows.append({"file": rel, "status": "byte-identical", "max_abs_numeric_difference": 0.0})
            continue
        if new.suffix == ".json":
            a, b = json.loads(new.read_text(encoding="utf-8")), json.loads(old.read_text(encoding="utf-8"))
            diffs = [abs(float(a[k]) - float(b[k])) for k in a if k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))]
            same = all(a[k] == b[k] for k in a if k in b and not isinstance(a[k], (int, float)))
            rows.append({"file": rel, "status": "numerically identical" if same and max(diffs or [0]) < 1e-9 else "differs",
                         "max_abs_numeric_difference": max(diffs or [0])})
            continue
        try:
            a, b = pd.read_csv(new), pd.read_csv(old)
        except Exception as exc:  # pragma: no cover
            rows.append({"file": rel, "status": f"unreadable: {exc}"})
            continue
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            rows.append({"file": rel, "status": "shape or columns differ", "new_shape": str(a.shape), "committed_shape": str(b.shape)})
            continue
        num = a.select_dtypes("number").columns
        diff = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if len(num) else 0.0
        nan_match = bool((a[num].isna().to_numpy() == b[num].isna().to_numpy()).all()) if len(num) else True
        txt = [c for c in a.columns if c not in num]
        txt_match = bool((a[txt].astype(str).to_numpy() == b[txt].astype(str).to_numpy()).all()) if txt else True
        status = "numerically identical" if diff < 1e-9 and nan_match and txt_match else ("within 1e-6" if diff < 1e-6 and nan_match and txt_match else "differs")
        rows.append({"file": rel, "status": status, "max_abs_numeric_difference": diff, "text_columns_identical": txt_match})
    out = pd.DataFrame(rows)
    out.to_csv(new_root / "verification_against_committed.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-to", default=None)
    args = parser.parse_args()
    environment_lock()
    data_dictionary()
    provenance_table()
    man = manifest()
    print(f"manifest: {len(man)} files")
    if args.compare_to and Path(args.compare_to).resolve() != OUTPUT_ROOT.resolve():
        ver = compare_tables(OUTPUT_ROOT, Path(args.compare_to).resolve())
        print(ver["status"].value_counts().to_string())
    meta = {"workbook_sha256": sha256(ROOT / "05_Source_Data" / "es6c02287_si_002.xlsx"), "python": platform.python_version(),
            "platform": platform.platform()}
    (OUTPUT_ROOT / "04_Tables" / "Round2" / "R2_P03_input_checksums.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
