"""Single entry point that regenerates every analysis output in a declared order.

Usage (from the repository root, after ``python -m pip install -r requirements-lock.txt``):

    python run_all.py                         # full run into ./reproduced/<UTC timestamp>
    python run_all.py --output-root my_run    # choose the output directory
    python run_all.py --steps round2 records  # run selected steps only

Outputs are written under ``--output-root`` (exported to the scripts as
``PFAS_OUTPUT_ROOT``), so curated files committed in ``03_Figures``/``04_Tables``
are never overwritten unless ``--output-root .`` is given explicitly.  The final
step writes ``reproducibility_manifest.csv`` (file, generating script, SHA-256) and
``verification_against_committed.csv`` (regenerated versus committed tables).
Inputs are read only from ``05_Source_Data``; no network access is required.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("figures", "07_Scripts/generate_pfas_paper_figures.py",
     "Main-text Figs 2 and 4-9 data, Table 1 (published designs), split/censoring/interval/SHAP/association tables"),
    ("checks", "07_Scripts/run_reviewer_requested_checks.py",
     "Prediction-time baselines (Table 2), retransformation, temporal controls, ridge/OLS sensitivity"),
    ("supplementary", "07_Scripts/build_supplementary_evidence.py",
     "Supplementary tables (evidence flow, study inventory, predictors, fold metrics) and Figs S1-S4"),
    ("fixed_split", "07_Scripts/audit_disjoint_high_accuracy_claim.py",
     "Fixed 13-study holdout with row-random configuration selection (Fig. 3 numbers)"),
    ("fig3", "07_Scripts/generate_disjoint_transfer_fig2.py", "Fig. 3 drawing"),
    ("robustness", "07_Scripts/run_major_revision_robustness.py",
     "Tables S5-S12, S14: 12-candidate grouped audit, 2x2, 50 partitions, threshold, bootstrap, Webb, study-unit intervals"),
    ("consistency", "07_Scripts/repair_reviewer_consistency_20260905.py",
     "Table 1 SD/nRMSE, within-study concordance, paired bootstrap, interval summaries"),
    ("author_rule", "07_Scripts/run_author_rule_sensitivity.py", "Author-rule A2353 sensitivity (Table S16)"),
    ("round2", "07_Scripts/run_round2_revision_analyses.py",
     "Second-round analyses (dataset registry, fold tables, leakage audit, harmonized/nested designs, baselines, diagnostics)"),
    ("round2_figures", "07_Scripts/draw_round2_supplementary_figures.py",
     "Figs S5-S7 (AFT diagnostics, censoring simulation, out-of-fold applicability) from saved tables"),
    ("records", "07_Scripts/build_reproducibility_records.py",
     "Data dictionary, environment lock, provenance table, SHA-256 manifest and verification"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--steps", nargs="*", choices=[s[0] for s in STEPS])
    parser.add_argument("--jobs", type=int, default=8, help="parallel workers for nested tuning")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output_root) if args.output_root else ROOT / "reproduced" / stamp
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PFAS_OUTPUT_ROOT": str(out), "PFAS_N_JOBS": str(args.jobs), "MPLBACKEND": "Agg",
           "PYTHONIOENCODING": "utf-8"}
    selected = [s for s in STEPS if not args.steps or s[0] in args.steps]
    log = []
    for key, script, purpose in selected:
        t0 = time.time()
        print(f"\n=== [{key}] {script}", flush=True)
        cmd = [sys.executable, script]
        if key == "records":
            cmd += ["--compare-to", str(ROOT)]
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        log.append({"step": key, "script": script, "purpose": purpose, "return_code": proc.returncode,
                    "runtime_seconds": round(time.time() - t0, 1)})
        (out / "run_all_log.json").write_text(json.dumps({"output_root": str(out), "python": sys.version, "steps": log}, indent=2),
                                              encoding="utf-8")
        if proc.returncode != 0:
            sys.exit(f"step {key} failed with return code {proc.returncode}")
    print(f"\nAll requested steps completed; outputs in {out}")


if __name__ == "__main__":
    main()
