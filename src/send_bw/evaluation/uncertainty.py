"""Document predictive variability vs incompletely estimated parameter uncertainty.

Covariance parameters hitting optimizer bounds are recorded. They do not make
the predictive layer a complete parameter-uncertainty statement.
"""

from __future__ import annotations

from typing import Any

COMPONENTS = {
    "predictive_variability": (
        "HierarchicalCovariance study, animal, and residual draws in sample_study. "
        "This is predictive variability of new-study trajectories under the fitted residual layer."
    ),
    "parameter_uncertainty": (
        "Spline: study-bootstrap of the mean only. Transformer: members are mean-only "
        "(n_members=1 has no bootstrap spread). Covariance parameters are not bootstrapped. "
        "Bayesian posterior draws, when reliable, are the only complete parameter+predictive "
        "statement; an unreliable Bayesian fit is not counted as a successful benchmark."
    ),
}


def inspect_covariance(cov: dict[str, Any] | None) -> dict[str, Any]:
    cov = cov or {}
    info = cov.get("fit_info") or {}
    at_bound = list(info.get("params_at_bound") or [])
    return {
        "layer": "HierarchicalCovariance",
        "params_at_bound": at_bound,
        "n_params_at_bound": len(at_bound),
        "boundary_flag": bool(at_bound),
        "converged": info.get("converged"),
        "nll": info.get("nll"),
        "n_studies": info.get("n_studies"),
        "n_animals": info.get("n_animals"),
        "n_obs": info.get("n_obs"),
        "covariance_parameters_bootstrapped": False,
        "components": dict(COMPONENTS),
        "note": (
            "A parameter at bound is an incompletely estimated covariance component. "
            "Do not treat predictive intervals as complete parameter-uncertainty intervals."
        ),
    }


def bayesian_gate(diagnostics: dict[str, Any] | None, status: str | None = None,
                  reason: str | None = None) -> dict[str, Any]:
    d = diagnostics or {}
    if status in ("not_run", "failed"):
        return {
            "status": status,
            "reliable": False,
            "counted_as_successful_benchmark": False,
            "rhat_max": None,
            "ess_bulk_min": None,
            "ess_tail_min": None,
            "divergences": None,
            "reason": reason or ("Bayesian inference failed" if status == "failed" else "Bayesian was not run"),
            "components": dict(COMPONENTS),
        }
    reliable = bool(d.get("reliable"))
    return {
        "status": "trained",
        "reliable": reliable,
        "counted_as_successful_benchmark": reliable,
        "rhat_max": d.get("rhat_max"),
        "ess_bulk_min": d.get("ess_bulk_min"),
        "ess_tail_min": d.get("ess_tail_min"),
        "divergences": d.get("divergences"),
        "reason": None if reliable else (reason or "Bayesian diagnostics failed the reliability gate"),
        "components": dict(COMPONENTS),
    }
