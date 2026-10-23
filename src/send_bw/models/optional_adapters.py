"""Optional models (TFT, Chronos-2, DeepAR): capability declarations and dependency/permission gates.

None of these is a required benchmark and none may be substituted under a required benchmark's
name. Chronos-2 and DeepAR still only check prerequisites. TFT is a verified PyTorch
Forecasting adapter for the secondary early-history task; without the extra it stays
``not_run``. Implementing or running an optional model never registers it as
``transformer``.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any

BASELINE_ONLY_REASON = (
    "TFT requires an observed encoder window: baseline-only / pre-study prediction is not fabricated")
DESIGN_REASON = (
    "TFT cannot generate complete hypothetical studies and is not a design simulator")


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
         "not interchangeable with the custom baseline-conditioned transformer",
         "not used for design simulation or the production final analysis"]),
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


def _has(obj: Any, name: str) -> bool:
    return hasattr(obj, name)


def _sig_params(obj: Any) -> set[str]:
    try:
        return set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return set()


def verify_tft_api() -> dict[str, Any]:
    """Inspect the installed pytorch_forecasting / Lightning API before any fit.

    Returns ``ok=False`` with an explicit reason when the extra is missing or the
    symbols this adapter relies on are absent. Does not download anything.
    """
    missing: list[str] = []
    if importlib.util.find_spec("pytorch_forecasting") is None:
        return {"ok": False, "reason": "optional dependency 'pytorch_forecasting' not installed",
                "missing": ["pytorch_forecasting"]}
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as e:
        return {"ok": False, "reason": f"pytorch_forecasting import failed: {e}",
                "missing": ["pytorch_forecasting"]}

    for cls_name, cls, attrs in (
        ("TimeSeriesDataSet", TimeSeriesDataSet,
         ("from_dataset", "from_parameters", "get_parameters", "to_dataloader")),
        ("TemporalFusionTransformer", TemporalFusionTransformer, ("from_dataset", "predict")),
        ("QuantileLoss", QuantileLoss, ()),
    ):
        for attr in attrs:
            if not _has(cls, attr):
                missing.append(f"{cls_name}.{attr}")

    init_params = _sig_params(TimeSeriesDataSet.__init__)
    if "allow_missing_timesteps" not in init_params:
        missing.append("TimeSeriesDataSet.allow_missing_timesteps")

    from_ds = _sig_params(TimeSeriesDataSet.from_dataset)
    if not ({"predict", "predict_mode"} & from_ds):
        missing.append("TimeSeriesDataSet.from_dataset(predict|predict_mode)")

    q_params = _sig_params(QuantileLoss.__init__)
    if q_params and "quantiles" not in q_params and "quantile" not in q_params:
        missing.append("QuantileLoss(quantiles=...)")

    trainer = None
    if importlib.util.find_spec("lightning") is not None:
        trainer = "lightning.pytorch.Trainer"
    elif importlib.util.find_spec("pytorch_lightning") is not None:
        trainer = "pytorch_lightning.Trainer"
    else:
        missing.append("lightning.pytorch.Trainer")

    if missing:
        return {"ok": False,
                "reason": "installed pytorch_forecasting API mismatch: " + ", ".join(missing),
                "missing": missing, "trainer": trainer}
    return {
        "ok": True,
        "reason": ("verified TimeSeriesDataSet + TemporalFusionTransformer.from_dataset + "
                   "QuantileLoss + CPU Trainer"),
        "missing": [],
        "trainer": trainer,
        "allow_missing_timesteps": True,
        "from_dataset_params": sorted(from_ds),
    }


def check_optional(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an honest status record for an optional model (never raises for missing deps)."""
    config = config or {}
    info = OptionalModelInfo(**asdict(CATALOG[name]))
    if not config.get("enabled", False):
        info.reason = "disabled in config"
        return asdict(info)
    if name == "tft":
        verified = verify_tft_api()
        info.extra["verify"] = {k: verified[k] for k in ("ok", "reason", "missing", "trainer") if k in verified}
        if not verified["ok"]:
            info.reason = verified["reason"]
            return asdict(info)
        info.status = "adapter_ready"
        info.reason = ("pytorch_forecasting API verified; adapter is early-history only "
                       "(no baseline-only pre-study, not a design simulator, not the required transformer)")
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
