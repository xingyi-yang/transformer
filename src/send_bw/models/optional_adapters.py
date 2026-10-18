"""Optional models (TFT, Chronos-2, DeepAR): capability declarations and dependency/permission gates.

None of these is a required benchmark and none may be substituted under a required benchmark's
name. In this build the adapters only check prerequisites and report `not_run` with the reason;
no forecasting logic is emulated. Implementing them requires verifying the installed library APIs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OptionalModelInfo:
    name: str
    package: str
    task: str
    capabilities: list[str]
    limitations: list[str]
    requires_download: bool = False
    status: str = "not_run"
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


CATALOG = {
    "tft": OptionalModelInfo(
        "tft", "pytorch_forecasting", "secondary (early-history forecasting)",
        ["static covariates (sex, strain, compound descriptors)", "known future inputs (planned dose features)",
         "observed past targets (early history)"],
        ["requires an observed encoder window: no genuine baseline-only pre-study support",
         "not interchangeable with the custom baseline-conditioned transformer"]),
    "chronos2": OptionalModelInfo(
        "chronos2", "chronos", "secondary (early-history, zero-shot first; fine-tuning separate)",
        ["univariate/multivariate zero-shot probabilistic forecasting from observed history"],
        ["pretrained weights must be downloaded: explicit permission required",
         "covariate support must be verified against the installed version before use",
         "no baseline-only pre-study support"], requires_download=True),
    "deepar": OptionalModelInfo(
        "deepar", "gluonts", "secondary (early-history forecasting)",
        ["autoregressive recurrent probabilistic model with a likelihood head", "static and dynamic covariates"],
        ["needs observed history for conditioning", "irregular visit times require explicit handling"]),
}


def check_optional(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an honest status record for an optional model (never raises for missing deps)."""
    config = config or {}
    info = OptionalModelInfo(**asdict(CATALOG[name]))
    if not config.get("enabled", False):
        info.reason = "disabled in config"
        return asdict(info)
    if importlib.util.find_spec(info.package) is None:
        info.reason = f"optional dependency '{info.package}' not installed"
        return asdict(info)
    if info.requires_download and not config.get("download_permission", False):
        info.reason = "pretrained-weight download not authorized (set download_permission: true after approval)"
        return asdict(info)
    info.reason = ("dependency present but adapter not implemented in this build: the installed API must be "
                   "verified before any results are produced")
    return asdict(info)


def optional_status(models_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    opt = models_cfg.get("optional", {}) or {}
    return {n: check_optional(n, opt.get(n)) for n in CATALOG}
