"""
Typed entry points for the single-atom catalysis ML figure templates.

This module keeps the original adsorption-energy workflow available while
exposing figure-type names that are easier to search and reuse in new papers.
Edit the underlying functions in `sac_adsorption_energy_templates.py` when a
project needs dataset-specific changes.
"""

from __future__ import annotations

from sac_adsorption_energy_templates import (
    ale_curve,
    build_preprocessor,
    configure_style,
    counterfactual_effect,
    entropy_topsis,
    figure_1 as sac_workflow_overview,
    figure_2 as sac_dataset_distribution_correlation,
    figure_3 as sac_feature_selection_importance,
    figure_4 as sac_model_comparison_prediction_learning,
    figure_5 as sac_topsis_wilcoxon_error,
    figure_6 as sac_shap_pdp_ale_interpretability,
    figure_7 as sac_sensitivity_barrier_analysis,
    figure_8 as sac_transfer_periodic_heatmap,
    feature_columns,
    label_feature,
    label_model,
    make_features,
    make_models,
    metrics,
    pairwise_mi,
    pdp_curve,
    prepare_training,
    savefig,
    style_axes,
    top_numeric_features,
)


SAC_TEMPLATE_MAP = {
    "workflow": {
        "functions": ["sac_workflow_overview"],
        "source": "figure_1",
        "use_for": "paper workflow, dataset logic, train/external-test schematic",
    },
    "dataset_distribution": {
        "functions": ["sac_dataset_distribution_correlation"],
        "source": "figure_2",
        "use_for": "label availability, target distributions, grouped correlation overview",
    },
    "heatmap_correlation": {
        "functions": ["pairwise_mi", "sac_dataset_distribution_correlation"],
        "source": "figure_2",
        "use_for": "correlation heatmap, mutual information map, feature relationship overview",
    },
    "feature_selection": {
        "functions": ["top_numeric_features", "pairwise_mi", "sac_feature_selection_importance"],
        "source": "figure_3",
        "use_for": "permutation importance, feature consensus, clustering/dendrogram panels",
    },
    "model_comparison": {
        "functions": ["make_models", "metrics", "entropy_topsis", "sac_model_comparison_prediction_learning"],
        "source": "figure_4",
        "use_for": "model benchmark, train/test metrics, prediction panels, learning curve",
    },
    "prediction_measured": {
        "functions": ["metrics", "sac_model_comparison_prediction_learning"],
        "source": "figure_4",
        "use_for": "predicted-vs-observed scatter with side residual/error panel",
    },
    "topsis_wilcoxon": {
        "functions": ["entropy_topsis", "sac_topsis_wilcoxon_error"],
        "source": "figure_5",
        "use_for": "TOPSIS weights, Wilcoxon pairwise comparison, error distributions",
    },
    "shap_importance": {
        "functions": ["sac_shap_pdp_ale_interpretability"],
        "source": "figure_6",
        "use_for": "SHAP-style contribution summary and top-feature effects",
    },
    "pdp_ale": {
        "functions": ["pdp_curve", "ale_curve", "sac_shap_pdp_ale_interpretability"],
        "source": "figure_6",
        "use_for": "PDP, ALE, dependence plots, approximate uncertainty bands",
    },
    "sensitivity": {
        "functions": ["counterfactual_effect", "sac_sensitivity_barrier_analysis"],
        "source": "figure_7",
        "use_for": "counterfactual sensitivity and mechanism/barrier-style panels",
    },
    "transfer_periodic_heatmap": {
        "functions": ["sac_transfer_periodic_heatmap"],
        "source": "figure_8",
        "use_for": "external dopant transfer, domain shift, ranking, periodic-table heatmap",
    },
}


def get_sac_template_map() -> dict[str, dict[str, object]]:
    """Return the searchable single-atom-catalysis template map."""
    return SAC_TEMPLATE_MAP.copy()
