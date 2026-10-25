"""Prespecified v3 pilot controls (stages 1–2 software; training wave not launched).

Study weighting, budgets, tuning rules, and seeds are declared here so a later
smoke run cannot invent them after seeing results. Uncertainty safeguards are
the existing v2 layers; this module does not relax them.
"""

from __future__ import annotations

from typing import Any

PILOT_TASK = "predose_conditioned"
SELECTIVE_SHARING = "required_deferred"

STUDY_WEIGHTING = "study_balanced"
STUDY_WEIGHTING_RULE = (
    "Each STUDYID contributes equal total weight via study_balanced_weights; "
    "weights average to 1. Donor studies in a pooled train set receive the same "
    "per-study total as target studies. Validation stays target-only."
)

ROOT_SEED = 20261024
SEED_NAMESPACES = {
    "generate_compounds": ("compounds",),
    "generate_target_study": ("study", "{s}"),
    "generate_donor_study": ("donor", "{d}"),
    "target_splits": "run.seed",
    "spline_fit": "models.spline.seed or run.seed",
    "transformer_fit": "models.transformer.seed or run.seed",
    "evaluate_predose": "run.seed + 17",
}

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
SPLINE_BUDGET = {
    "n_bootstrap": 6,
    "oof_folds": 5,
    "n_time_knots": 3,
    "n_expo_knots": 2,
    "ridge_alphas": list(RIDGE_ALPHAS),
}
TRANSFORMER_BUDGET_SMOKE = {
    "n_members": 1,
    "max_epochs": 20,
    "patience": 6,
    "d_model": 32,
    "n_heads": 2,
    "n_layers": 1,
}
TUNING_RULES = {
    "spline": (
        "Select ridge alpha by validation study-balanced WRMSE on the frozen "
        "target-species validation studies. Do not use donor rows for early stopping."
    ),
    "transformer": (
        "Early-stop on validation RMSE. Covariance is the shared spline "
        "HierarchicalCovariance (required). Member 0 is the full-train mean; "
        "later members are study-bootstrap refits of the mean only."
    ),
}

UNCERTAINTY_SAFEGUARDS = {
    "covariance": "HierarchicalCovariance on study-level OOF residuals; covariance parameters are not bootstrapped",
    "parameter": "study-bootstrap of the mean only",
    "transformer_covariance": "shared with the spline; required; not a separately fitted residual",
    "no_fabricated_baseline_only": True,
    "no_design_recommendations": True,
}

DONOR_SCHEDULE = {
    "order": "STUDYID sort",
    "nested": True,
    "k_levels": [0, 1, 2, 3],
    "k0": "context_specific / zero-donor equivalence",
}

LABELS = (
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
)


def as_dict() -> dict[str, Any]:
    return {
        "task": PILOT_TASK,
        "selective_sharing": SELECTIVE_SHARING,
        "study_weighting": STUDY_WEIGHTING,
        "study_weighting_rule": STUDY_WEIGHTING_RULE,
        "root_seed": ROOT_SEED,
        "seed_namespaces": SEED_NAMESPACES,
        "spline_budget": SPLINE_BUDGET,
        "transformer_budget_smoke": TRANSFORMER_BUDGET_SMOKE,
        "tuning_rules": TUNING_RULES,
        "uncertainty_safeguards": UNCERTAINTY_SAFEGUARDS,
        "donor_schedule": DONOR_SCHEDULE,
        "labels": list(LABELS),
    }
