"""Sparse target-history encoder grid. Missing encoder rows are omitted, not filled.

W0 equals predose-conditioned (no post-baseline encoder). Animals with no encoder
under a later rule are counted in n_animals_without_encoder.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from send_bw.evaluation.forecast import evaluate_prestudy
from send_bw.models.base import ModelData, ResponseModel

ENCODER_RULES = ("W0", "W1", "Wlast14", "Wall14")


def encoder_history(obs: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, int]:
    """Return encoder rows and the number of animals in `obs` with an empty encoder."""
    rule = str(rule)
    animals = set(obs["USUBJID"].astype(str))
    post = obs.loc[~obs.get("IS_BASELINE", False) if "IS_BASELINE" in obs.columns else slice(None)]
    if "IS_BASELINE" not in obs.columns:
        post = obs[obs["DAY"] > 0]
    if rule == "W0":
        hist = obs.iloc[0:0][["USUBJID", "DAY", "LOGW"]].copy() if len(obs) else obs
        return hist, len(animals)
    if rule == "W1":
        first = post.sort_values(["USUBJID", "DAY"]).groupby("USUBJID", as_index=False).head(1)
        hist = first[["USUBJID", "DAY", "LOGW"]]
    elif rule == "Wlast14":
        win = post[post["DAY"] <= 14].sort_values(["USUBJID", "DAY"])
        hist = win.groupby("USUBJID", as_index=False).tail(1)[["USUBJID", "DAY", "LOGW"]]
    elif rule == "Wall14":
        hist = post[post["DAY"] <= 14][["USUBJID", "DAY", "LOGW"]]
    else:
        raise ValueError(f"unknown encoder rule {rule!r}; expected {ENCODER_RULES}")
    have = set(hist["USUBJID"].astype(str))
    return hist.reset_index(drop=True), int(len(animals - have))


def evaluate_sparse_grid(model: ResponseModel, data: ModelData, encoders: list[str] | None = None,
                         n_draws: int = 80, seed: int = 0, contrast_day: float = 28.0) -> dict[str, Any]:
    encoders = list(encoders or ENCODER_RULES)
    out: dict[str, Any] = {"task": "sparse_history", "model": model.name, "encoders": {},
                           "labels": [
                               "Synthetic diagnostic.",
                               "Not design qualification.",
                               "No real-data validation.",
                               "No operational recommendation.",
                           ]}
    for rule in encoders:
        if rule == "W0":
            r = evaluate_prestudy(model, data, n_draws=n_draws, seed=seed, contrast_day=contrast_day)
            block = {"rule": "W0", "equals": "predose_conditioned", "status": "ok",
                     "n_animals_without_encoder": 0,
                     "summary": r["summary"], "note": "predose only; no post-baseline encoder"}
            out["encoders"][rule] = block
            continue
        if not hasattr(model, "_condition"):
            out["encoders"][rule] = {
                "rule": rule, "status": "not_supported",
                "reason": "early-history conditioning not implemented for this model",
                "n_animals_without_encoder": None,
            }
            continue
        eh = dict(_evaluate_rule(model, data, rule))
        eh["rule"] = rule
        out["encoders"][rule] = eh
    return out


def _evaluate_rule(model: ResponseModel, data: ModelData, rule: str) -> dict[str, Any]:
    """Condition on the encoder rule's rows only; do not fill missing animals."""
    from send_bw.evaluation.forecast import _score_marginals, _summarize_which

    rows = []
    n_skipped = 0
    for sid in data.studies:
        a = data.animals[data.animals["STUDYID"] == sid].reset_index(drop=True)
        o = data.obs[data.obs["STUDYID"] == sid]
        hist, n_miss = encoder_history(o, rule)
        n_skipped += n_miss
        fut = o[~o.set_index(["USUBJID", "DAY"]).index.isin(hist.set_index(["USUBJID", "DAY"]).index)]
        fut = fut[fut["DAY"] > 0]
        if fut.empty:
            continue
        times = np_unique_days(fut)
        try:
            cond = model.predict_marginals(a, times, history=hist)
        except NotImplementedError:
            return {"task": "sparse_history", "model": model.name, "status": "not_supported",
                    "reason": "early-history conditioning not implemented for this model",
                    "n_animals_without_encoder": n_skipped}
        except Exception as e:  # noqa: BLE001
            return {"task": "sparse_history", "model": model.name, "status": "failed",
                    "reason": f"{type(e).__name__}: {e}", "n_animals_without_encoder": n_skipped}
        rows.append(_score_marginals(fut, cond, "conditioned"))
    if not rows:
        return {"task": "sparse_history", "status": "not_run",
                "reason": "no post-encoder observations to score",
                "n_animals_without_encoder": n_skipped}
    import pandas as pd

    df = pd.concat(rows, ignore_index=True)
    return {"task": "sparse_history", "status": "ok", "n_animals_without_encoder": n_skipped,
            "note": "encoder rows omitted, not filled", **_summarize_which(df)}


def np_unique_days(fut: pd.DataFrame):
    import numpy as np

    return np.array(sorted(fut["DAY"].unique()), float)
