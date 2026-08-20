# PFAS membrane-rejection validation: source data and code

This repository contains only the machine-readable source workbook and the Python code used for the claim-matched validation of PFAS rejection by nanofiltration and reverse-osmosis membranes. Manuscripts, Supporting Information files, rendered figures, derived tables, reviewer files, and submission packages are intentionally excluded.

## Source data

`05_Source_Data/es6c02287_si_002.xlsx` is the machine-readable Supporting Information workbook accompanying:

Tomsovic Y, Gu S, Doudrick K, Straub AP. *Mechanistic Insights into PFAS Rejection in Nanofiltration and Reverse Osmosis from Data-Driven Analysis*. Environmental Science & Technology. 2026;60:13711–13723. https://doi.org/10.1021/acs.est.6c02287

The source article and its Supporting Information are distributed under the Creative Commons Attribution 4.0 International licence. The workbook is retained unchanged and redistributed here with attribution.

## Analysis view

The code applies two documented filters to the source worksheet `PFAS_literature_dataset_2b`:

1. retain rejection values greater than or equal to 0.50;
2. retain nanofiltration or reverse-osmosis records.

These filters yield 2,424 records from 61 studies, 112 membranes, and 37 PFAS. The source article reports 2,353 records from 60 studies; the 71-record difference is retained as an unresolved source-version discrepancy.

## Environment

Create an isolated Python environment and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The analysis was run with Python 3.12.3. Fixed seeds are defined inside the scripts.

## Reproduction order

Run the following commands from the repository root:

```bash
python 07_Scripts/generate_pfas_paper_figures.py
python 07_Scripts/run_reviewer_requested_checks.py
python 07_Scripts/build_supplementary_evidence.py
python 07_Scripts/audit_disjoint_high_accuracy_claim.py
python 07_Scripts/generate_disjoint_transfer_fig2.py
```

The scripts create `03_Figures`, `04_Tables`, and `99_Working_Files` locally. These generated outputs are ignored by Git and are not part of the public repository.

The clipping audit and exact ridge/OLS outputs remain reproducible in the code as optional diagnostics, although the corresponding redundant display items are not included in the final Supporting Information.

