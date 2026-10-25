"""Named v3 sharing regimes and the selective-sharing contracts.

Context-specific and pooled remain the stages 1–2 regimes. Selective v1 shares
compound descriptors and planned-dose/regimen features, and keeps growth,
baseline, and residual scale species-specific. Selective regularized is an
added variant: regularized species-specific deviations in nonlinear growth and
treatment response. v1 is retained. The split is recorded on every selective
fit.
"""

from __future__ import annotations

from typing import Any

SHARING_REGIMES = ("context_specific", "pooled", "selective", "selective_regularized")
DONOR_REGIMES = ("pooled", "selective", "selective_regularized")
SELECTIVE_REGIMES = ("selective", "selective_regularized")

SELECTIVE_V1: dict[str, Any] = {
    "variant": "v1",
    "shared": ["compound", "regimen"],
    "species_specific": ["bio", "baseline", "species"],
    "residual": "target_species_only",
    "note": (
        "Shared: compound descriptors and planned-dose/regimen query features. "
        "Species-specific: growth/baseline token groups and residual scale. "
        "Not a Bayesian species random-effect variant."
    ),
}

DONOR_SCENARIOS: dict[str, dict[str, float]] = {
    "matched": {"gmax_scale": 1.0, "baseline_shift": 0.0, "effect_scale": 1.0},
    "growth_shift": {"gmax_scale": 1.6, "baseline_shift": 0.20, "effect_scale": 1.0},
    "treatment_shift": {"gmax_scale": 1.0, "baseline_shift": 0.0, "effect_scale": 2.0},
    "both": {"gmax_scale": 1.6, "baseline_shift": 0.20, "effect_scale": 2.0},
}

LABELS = (
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
)


def parse_sharing(value: str | None) -> str:
    v = (value or "context_specific").strip().lower()
    if v not in SHARING_REGIMES:
        raise ValueError(f"unknown sharing regime {value!r}; expected {SHARING_REGIMES}")
    return v


def uses_donors(sharing: str | None) -> bool:
    return parse_sharing(sharing) in DONOR_REGIMES


def is_selective(sharing: str | None) -> bool:
    return parse_sharing(sharing) in SELECTIVE_REGIMES


def resolve_donor_scenario(name: str | None) -> dict[str, float]:
    key = (name or "matched").strip().lower()
    if key not in DONOR_SCENARIOS:
        raise ValueError(f"unknown donor_scenario {name!r}; expected {tuple(DONOR_SCENARIOS)}")
    return dict(DONOR_SCENARIOS[key])


SELECTIVE_REGULARIZED: dict[str, Any] = {
    "variant": "regularized",
    "shared": ["compound", "regimen"],
    "species_specific": ["growth_deviation", "treatment_deviation"],
    "residual": "target_species_only",
    "regularization": {
        "growth": "ridge inflation on SPECIES × time-basis coefficients (lambda_growth)",
        "treatment": "ridge inflation on SPECIES × exposure/dose coefficients (lambda_treatment)",
        "transformer": "phase1 target fit; phase2 extra L2 of species/growth tokens toward phase1 and treatment-deviation head toward 0",
    },
    "note": (
        "Selective v1 is retained. This variant adds regularized species-specific "
        "deviations in nonlinear growth and treatment response. Deviations shrink "
        "toward the shared mean; they are not unpenalized species RE."
    ),
}


def selective_record(model: str, variant: str = "v1") -> dict[str, Any]:
    rec = dict(SELECTIVE_REGULARIZED if variant == "regularized" else SELECTIVE_V1)
    rec["model"] = model
    if variant == "regularized":
        if model == "transformer":
            rec["training"] = (
                "phase1: all token groups on target-species train animals; "
                "phase2: extra L2 of bio/baseline/species toward phase1 "
                "(growth deviations) and treatment-deviation head toward 0"
            )
        elif model == "spline":
            rec["training"] = (
                "joint ridge with SPECIES×time (growth) and SPECIES×exposure "
                "(treatment) deviations; inflated ridge on those columns; "
                "HierarchicalCovariance on target-species OOF residuals only"
            )
        elif model == "bayesian":
            rec["training"] = (
                "MAIN mean including SPECIES×growth and SPECIES×treatment columns; "
                "reliability gate applies; unreliable fits are not a successful benchmark"
            )
    elif model == "transformer":
        rec["training"] = (
            "phase1: all token groups on target-species train animals; "
            "phase2: freeze bio/baseline/species, train compound/regimen on pooled train"
        )
    elif model == "spline":
        rec["training"] = (
            "joint ridge mean with SPECIES_* time-basis interactions; "
            "HierarchicalCovariance fit on target-species OOF residuals only"
        )
    elif model == "bayesian":
        rec["training"] = (
            "MAIN mean with SPECIES_* fixed effects (not a new species RE); "
            "reliability gate applies; unreliable fits are not a successful benchmark"
        )
    return rec
