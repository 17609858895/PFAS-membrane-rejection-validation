# PFAS membrane-rejection validation: data and code

Source data, analysis scripts, configurations, row-level logs and machine-readable results for the claim-matched validation of PFAS rejection by nanofiltration and reverse-osmosis membranes.

## Source data

`05_Source_Data/es6c02287_si_002.xlsx` is the unchanged Supporting Information workbook accompanying:

Tomsovic Y, Gu S, Doudrick K, Straub AP. *Mechanistic Insights into PFAS Rejection in Nanofiltration and Reverse Osmosis from Data-Driven Analysis*. Environmental Science & Technology. 2026;60:13711–13723. https://doi.org/10.1021/acs.est.6c02287

The source article and its Supporting Information are distributed under the Creative Commons Attribution 4.0 International licence. The workbook is redistributed here with attribution.

SHA-256 of the workbook: `ecd4349b334d6dfad4e3a076481e0345a59d71ed7d2523d0eb4a4082e8095a3a`

`05_Source_Data/derived/crossref_study_doi_cache.json` is a frozen Crossref title-to-DOI lookup used for the study inventory (Table S2); with it, no network access is needed.

## Analysis sets

The present study modifies the source dataset through its own eligibility rules. Every result is labelled by one of three analysis sets:

| Label | Rule | Rows | Studies |
|---|---|---|---|
| `P2424` (primary) | worksheet `PFAS_literature_dataset_2b`, rejection ≥ 0.50, membrane class NF or RO | 2,424 | 61 |
| `A2353` (author-rule sensitivity) | rejection ≥ 0.50, omitting membrane IDs NF270-PEI_Jeong, NF270-PAA_Jeong and reference IDs `Ma J., 2022`, `Ma et al, 2024 (2)` as stated by the source first author | 2,353 | 60 |
| `U2487` (threshold sensitivity) | membrane class NF or RO, no rejection threshold | 2,487 | 61 |

`04_Tables/Round2/R2_A01_row_level_eligibility_log.csv` lists all 2,517 workbook rows with Excel row number, a stable row identifier, study, DOI, membrane, PFAS, reason codes and membership of each set; `R2_A02_dataset_registry.csv` gives the counts. A2353 matches the published row and study counts, but the source modelling rows were not supplied, so its identity with the original modelling set cannot be verified.

## Environment

Python 3.12.3. Install the exact versions in an isolated environment:

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt` lists the direct dependencies only.

## Reproduction

Run everything from the repository root with one command:

```bash
python run_all.py
```

`run_all.py` executes the scripts below in this order and writes all outputs to `reproduced/<UTC timestamp>/` (or `--output-root DIR`), so committed results are never overwritten. Use `--steps` to run selected steps.

| Step | Script | Outputs |
|---|---|---|
| figures | `07_Scripts/generate_pfas_paper_figures.py` | main-text figure data, split, censoring, interval, SHAP and association tables |
| checks | `07_Scripts/run_reviewer_requested_checks.py` | baselines, retransformation, temporal controls, ridge/OLS sensitivity |
| supplementary | `07_Scripts/build_supplementary_evidence.py` | Supplementary tables and Figs S1–S4 |
| fixed_split | `07_Scripts/audit_disjoint_high_accuracy_claim.py` | fixed 13-study holdout (Fig. 3 numbers) |
| fig3 | `07_Scripts/generate_disjoint_transfer_fig2.py` | Fig. 3 |
| robustness | `07_Scripts/run_major_revision_robustness.py` | `04_Tables/Revision` |
| consistency | `07_Scripts/repair_reviewer_consistency_20260905.py` | `04_Tables/Consistency_20260905` |
| author_rule | `07_Scripts/run_author_rule_sensitivity.py` | `04_Tables/AuthorRule` (Table S16) |
| round2 | `07_Scripts/run_round2_revision_analyses.py` | `04_Tables/Round2` (Tables S17–S26, harmonised and nested designs, baselines, diagnostics) |
| round2_figures | `07_Scripts/draw_round2_supplementary_figures.py` | Figs S5–S7 |
| records | `07_Scripts/build_reproducibility_records.py` | lock file, data dictionary, provenance table, manifest, verification |

The final step writes `reproducibility_manifest.csv` (every output with its generating script, runtime and SHA-256 checksum, plus checksums of inputs and code) and `verification_against_committed.csv`. A clean regeneration on Windows (32 cores) took about 22 min and reproduced all 86 committed result tables underlying the previously reported analyses and Table S16 (84 byte-identical, 2 identical to numerical precision). PNG/PDF checksums can differ between runs because of embedded metadata; verification therefore uses the numerical tables.

### Floating-point reproducibility

Byte-for-byte identity of regenerated tables depends on last-digit floating-point behaviour of multithreaded linear algebra (Intel MKL in this environment) and is not guaranteed across machines or sessions; the reported values do not depend on it.

As a second check (11 November 2026), commit `a9d6aec51788d6d19fc3f9a7d1234d94167a2787` was cloned into an empty directory and `python run_all.py` was run with all 24 distributions matching `requirements-lock.txt`. All 11 steps completed in about 17 min. Of the 123 regenerated tables (the 86 tables above plus 37 tables in `04_Tables/Round2`):

- 99 were byte-identical to the committed files;
- 23 differed by less than 5e-6 in any value. `verification_against_committed.csv` labels differences below 1e-9 `numerically identical`, below 1e-6 `within 1e-6` and anything larger `differs`, so `R2_F01` and `R2_F02` are labelled `differs`; their maximum differences (1.5e-6 and 4.5e-6) come only from the censored-normal (Tobit) models fitted by numerical maximum likelihood;
- one auxiliary table, `04_Tables/Consistency_20260905/Table_C06_within_study_concordance.csv`, differed only for the `AFT logistic-normal mean on rejection scale` summary. Rows with identical P50 predictions are exact ties, but their logistic-normal means can differ in the last floating-point digit after Gauss-Hermite integration, which breaks the tie arbitrarily and changes the per-study concordance of that summary (mean within-study C 0.640 in this run versus 0.638 in the committed file).

The within-study C of the logistic-normal-mean summary is not reported in the manuscript or Supplementary Information. The P50 within-study C (0.639) and every reported value were unchanged at their reported precision. Later commits differ from `a9d6aec` only in this documentation section.

All random seeds are fixed in the scripts; `04_Tables/Round2/R2_P02_display_item_provenance.csv` (Table S27) gives the dataset, splitter, folds, seeds, loss, hyperparameter source, post-processing, script and result files for every table and figure.

## Contents

- `candidate_configurations.json` — the 12 candidate AFT architectures.
- `06_Reviewer_Revision_Results/` — applicability-distance and temporal-matching definitions and the analysis configuration.
- `04_Tables/` — precomputed machine-readable results (top level, `Supplementary/`, `Revision/`, `Consistency_20260905/`, `AuthorRule/`, `Round2/`).
- `04_Tables/Round2/R2_P01_data_dictionary.csv` — source, unit, transformation, missingness, imputation rule and pipeline stage of every predictor.
- `04_Tables/Round2/R2_C02_test_fold_membership_long.csv` — test-fold membership of every row for every design; `R2_B04_row_level_out_of_fold_predictions.csv` — leave-one-study-out and 10-fold out-of-fold predictions with training-fold baselines.
- `source_reconciliation/` — earlier row-level filter audit sent to the source authors.

The primary ranking comparison is the equal-weight mean of within-study C-index over 59 scorable studies: 0.639 for the model versus 0.500 for the constant baseline. Bootstrap intervals resample studies while holding fitted predictions fixed. Partition ranges describe variation across sampled partitions, not population confidence intervals.
