"""Development increment-moment match. Never uses an exposed test split."""

from __future__ import annotations

from typing import Any

import numpy as np

from send_bw.evaluation.forecast import _study_draws
from send_bw.evaluation.increment import compare_study
from send_bw.models.base import ModelData

FORBIDDEN_SPLITS = {"test", "test_compound_holdout"}


def match_increment_moment(model: Any, data: ModelData, *, split_name: str,
                           n_draws: int = 40, seed: int = 0,
                           target_ratio: float = 1.0) -> dict[str, Any]:
    """Scale the residual process so development increment ratio approaches 1.

    `split_name` must be a development split (train or validation). Test is refused.
    """
    if str(split_name).lower() in FORBIDDEN_SPLITS:
        raise ValueError("increment-moment matching must not use an exposed test split")
    if not hasattr(model, "cov"):
        raise TypeError("increment-moment matching requires a covariance layer")
    ratios = []
    for k, sid in enumerate(data.studies):
        _a, _t, zs, y = _study_draws(model, data, sid, n_draws, seed + k)
        cmp = compare_study(y, zs, np.isfinite(y))
        if cmp["matched_ratio_pred_over_obs"] is not None:
            ratios.append(float(cmp["matched_ratio_pred_over_obs"]))
    before = float(np.mean(ratios)) if ratios else None
    factor = 1.0
    if before not in (None, 0):
        factor = float(np.clip(target_ratio / before, 0.25, 2.0))
        model.cov.scale_process(factor)
    return {
        "split": split_name,
        "n_studies": int(len(data.studies)),
        "n_draws": int(n_draws),
        "n_ratios": int(len(ratios)),
        "ratio_before": before,
        "target_ratio": float(target_ratio),
        "scale_factor": factor,
        "used_test": False,
        "note": "Development increment moment only. Exposed test outcomes were not used.",
    }
