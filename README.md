# PFAS membrane-rejection validation: data and code

Source data, analysis scripts, model configurations and machine-readable results for the claim-matched validation of PFAS rejection by nanofiltration and reverse-osmosis membranes.

## Source data

`05_Source_Data/es6c02287_si_002.xlsx` is the unchanged Supporting Information workbook accompanying:

Tomsovic Y, Gu S, Doudrick K, Straub AP. *Mechanistic Insights into PFAS Rejection in Nanofiltration and Reverse Osmosis from Data-Driven Analysis*. Environmental Science & Technology. 2026;60:13711–13723. https://doi.org/10.1021/acs.est.6c02287

The source article and its Supporting Information are distributed under the Creative Commons Attribution 4.0 International licence. The workbook is redistributed here with attribution.

The analysis retains NF/RO records with rejection ≥0.50 from worksheet `PFAS_literature_dataset_2b`, yielding 2,424 records from 61 studies, 112 membranes and 37 PFAS. The source article reports 2,353 records from 60 studies. The exact membership responsible for the net difference of 71 records and one study remains unresolved. `source_reconciliation/` provides the row-level filtering audit and study counts.

## Environment

The analysis used Python 3.12.3. Install the pinned dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

## Reproduction

Run these commands from the repository root, in order:

```bash
python 07_Scripts/generate_pfas_paper_figures.py
python 07_Scripts/run_reviewer_requested_checks.py
python 07_Scripts/build_supplementary_evidence.py
python 07_Scripts/audit_disjoint_high_accuracy_claim.py
python 07_Scripts/generate_disjoint_transfer_fig2.py
python 07_Scripts/run_major_revision_robustness.py
python 07_Scripts/repair_reviewer_consistency_20260905.py
```

The scripts write figures to `03_Figures`, tables to `04_Tables`, and intermediate results to `99_Working_Files`. Precomputed machine-readable results are included in `04_Tables`, including the inputs required by the final consistency script.

## Configurations and analysis outputs

`candidate_configurations.json` records the 12 candidate architectures. The scripts specify the normal AFT loss, scale parameters, tree settings and fixed random seeds. `06_Reviewer_Revision_Results/` contains the applicability-distance and temporal-matching definitions and the analysis configuration.

`04_Tables/Revision/` contains the architecture comparison, 50 study-partition memberships and metrics, threshold sensitivity, paired study bootstrap, Webb-weight inference and study-level interval analysis. `04_Tables/Consistency_20260905/` contains the filtered analysis snapshot, row identifiers, baseline predictions, within-study concordance, bootstrap draws and interval outputs.

The primary ranking comparison is the equal-weight mean of within-study C-index over 59 scorable studies: 0.639105 for the model versus 0.500000 for the constant baseline. Pooled concordance is a separate descriptive measure. The bootstrap resamples studies while holding the fitted predictions fixed. The 50-partition analysis selects between two prespecified architectures; the fixed-split analysis evaluates all 12 candidates. Partition ranges describe variation across the sampled partitions.
