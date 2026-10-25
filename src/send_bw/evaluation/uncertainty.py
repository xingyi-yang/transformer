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


RELIABILITY_THRESHOLDS = {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}
COMPILE_MARKERS = (
    "compileerror",
    "python.h",
    "pyconfig.h",
    "failed to compile",
    "compilation failed",
    "importerror",
    "modulenotfounderror",
)


def looks_like_compile_or_backend(text: str | None) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in COMPILE_MARKERS)


def sampling_diagnosis(diagnostics: dict[str, Any] | None, *, compile_error: str | None = None,
                       status: str | None = None) -> dict[str, Any]:
    """Classify Bayesian failure as compile/backend vs sampling, without relaxing thresholds."""
    th = dict(RELIABILITY_THRESHOLDS)
    if looks_like_compile_or_backend(compile_error):
        return {
            "failure_class": "compile_or_backend",
            "reliable": False,
            "counted_as_successful_benchmark": False,
            "thresholds": th,
            "compile_error": compile_error or (diagnostics or {}).get("reason"),
            "note": "This is not a sampling-mixing diagnosis. Thresholds were not applied.",
        }
    if status == "failed":
        return {
            "failure_class": "unreliable",
            "reliable": False,
            "counted_as_successful_benchmark": False,
            "thresholds": th,
            "exception": compile_error or (diagnostics or {}).get("reason"),
            "note": (
                "Sampling raised before a diagnostic summary was written. "
                "This is not a compile/backend classification. Thresholds are unchanged."
            ),
        }
    if status == "not_run" or not diagnostics:
        return {
            "failure_class": "not_run",
            "reliable": False,
            "counted_as_successful_benchmark": False,
            "thresholds": th,
            "note": "Bayesian was not sampled.",
        }
    d = diagnostics
    families = d.get("family_diagnostics") or {}
    worst_rhat = d.get("worst_rhat_params") or {}
    worst_ess = d.get("worst_ess_params") or {}
    draws = int(d.get("draws") or 0)
    chains = int(d.get("chains") or 0)
    n_post = draws * chains
    reliable = bool(d.get("reliable"))
    if reliable:
        klass = "reliable"
    else:
        names = " ".join(list(worst_rhat) + list(worst_ess))
        if n_post and n_post < 2 * th["ess_bulk_min"] and float(d.get("ess_bulk_min") or 0) < th["ess_bulk_min"]:
            klass = "inadequate_budget"
        elif any(k.startswith(("ell", "sigma_p", "sd_s", "sd_a")) for k in list(worst_rhat) + list(worst_ess)):
            klass = "covariance_mixing"
        elif any(k.startswith(("theta", "beta")) for k in list(worst_rhat) + list(worst_ess)):
            klass = "mean_identifiability"
        elif names:
            klass = "mixing"
        else:
            klass = "unreliable"
    return {
        "failure_class": klass,
        "reliable": reliable,
        "counted_as_successful_benchmark": reliable,
        "thresholds": th,
        "rhat_max": d.get("rhat_max"),
        "ess_bulk_min": d.get("ess_bulk_min"),
        "ess_tail_min": d.get("ess_tail_min"),
        "divergences": d.get("divergences"),
        "draws": draws,
        "tune": d.get("tune"),
        "chains": chains,
        "n_posterior_draws": n_post,
        "dropped_nonidentifiable_columns": d.get("dropped_nonidentifiable_columns"),
        "family_diagnostics": families,
        "worst_rhat_params": worst_rhat,
        "worst_ess_params": worst_ess,
        "priors": d.get("priors"),
        "note": (
            "Reliability thresholds are unchanged (rhat<=1.01, ESS>=400, divergences==0). "
            "Compile/backend failures are classified separately from sampling."
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
