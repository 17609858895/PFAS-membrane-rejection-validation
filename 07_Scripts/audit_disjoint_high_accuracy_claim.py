from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import audit_internal_high_accuracy_claim as audit
import generate_pfas_paper_figures as core


def metrics(df, idx, pred):
    return core.evaluation(
        df.loc[idx, "Rejection"].to_numpy(float),
        pred,
        df.loc[idx, "is_censored"].to_numpy(bool),
    )


def main():
    audit.OUT.mkdir(parents=True, exist_ok=True)
    _, df = core.load_data()
    x = core.feature_matrix(df)
    groups = df["ref_id"].astype(str).to_numpy()
    all_idx = np.arange(len(df))

    development_idx, unseen_study_idx = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=audit.SEED
        ).split(x, groups=groups)
    )
    familiar_train_idx, familiar_test_idx = train_test_split(
        development_idx,
        test_size=0.15,
        random_state=audit.SEED + 20,
        shuffle=True,
    )

    tuning, best = audit.tune("aft", x, df, familiar_train_idx)
    tuning.to_csv(
        audit.OUT / "disjoint_parameter_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Exact same fitted model: familiar rows versus studies excluded from every
    # fitting and tuning operation.
    same_model = audit.fit(
        "aft", x, df, familiar_train_idx, best, audit.SEED + 900
    )
    familiar_pred = audit.predict("aft", same_model, x[familiar_test_idx])
    unseen_same_model_pred = audit.predict("aft", same_model, x[unseen_study_idx])

    # Deployment-style refit with the frozen specification and all development
    # studies; the unseen-study test remains untouched.
    refit_model = audit.fit(
        "aft", x, df, development_idx, best, audit.SEED + 901
    )
    unseen_refit_pred = audit.predict("aft", refit_model, x[unseen_study_idx])

    result = {
        "best_candidate": best["candidate"],
        "inner_cv_logit_r2": best["mean_inner_logit_r2"],
        "development_studies": int(np.unique(groups[development_idx]).size),
        "unseen_test_studies": int(np.unique(groups[unseen_study_idx]).size),
        "familiar_train_rows": int(len(familiar_train_idx)),
        "familiar_test_rows": int(len(familiar_test_idx)),
        "unseen_test_rows": int(len(unseen_study_idx)),
        "same_fitted_model_familiar_test": metrics(
            df, familiar_test_idx, familiar_pred
        ),
        "same_fitted_model_unseen_studies": metrics(
            df, unseen_study_idx, unseen_same_model_pred
        ),
        "frozen_spec_refit_unseen_studies": metrics(
            df, unseen_study_idx, unseen_refit_pred
        ),
        "unseen_study_ids": sorted(np.unique(groups[unseen_study_idx]).tolist()),
    }
    (audit.OUT / "disjoint_claim_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "evaluation": name,
                **value,
            }
            for name, value in result.items()
            if isinstance(value, dict)
        ]
    ).to_csv(
        audit.OUT / "disjoint_claim_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
