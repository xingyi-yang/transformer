"""Named v3 sharing regimes and the selective-sharing v1 contract.

Context-specific and pooled remain the stages 1–2 regimes. Selective v1 shares
compound descriptors and planned-dose/regimen features, and keeps growth,
baseline, and residual scale species-specific. The split is recorded on every
selective fit. Further selective variants are amendments, not implied.
"""

from __future__ import annotations

from typing import Any

SHARING_REGIMES = ("context_specific", "pooled", "selective")
DONOR_REGIMES = ("pooled", "selective")

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


def resolve_donor_scenario(name: str | None) -> dict[str, float]:
    key = (name or "matched").strip().lower()
    if key not in DONOR_SCENARIOS:
        raise ValueError(f"unknown donor_scenario {name!r}; expected {tuple(DONOR_SCENARIOS)}")
    return dict(DONOR_SCENARIOS[key])


def selective_record(model: str) -> dict[str, Any]:
    rec = dict(SELECTIVE_V1)
    rec["model"] = model
    if model == "transformer":
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
