"""Share-or-abstain policy: donor pickers and validation-only abstention.

Existing donor “selection” is the STUDYID-sort prefix already used by
``donor_study_order`` / ``compose_train_with_donors``. This module does not add
a learned donor ranker. Abstention compares target-only vs a selected-donor
candidate on target validation only.

Synthetic diagnostic. Not design qualification. No operational recommendation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from send_bw.data.splits import donor_study_order

PREFIX_RULE = "studyid_sort_prefix"
RANDOM_RULE = "rng_choice_without_replacement"
ALL_RULE = "all_eligible"
ZERO_RULE = "zero_donors"
EXPLICIT_RULE = "explicit_ids"

ABSTAIN_METRIC = "val_rmse_log"
ABSTAIN_MIN_IMPROVEMENT = 0.0
ABSTAIN_TIE = "target_only"
LOWER_IS_BETTER = True


def resolve_donors(
    eligible: list[str],
    *,
    k: int,
    rule: str = PREFIX_RULE,
    rng_seed: int | None = None,
    explicit: list[str] | None = None,
    allow_donors: bool = True,
) -> dict[str, Any]:
    """Pick k donors from the eligible bank. Never uses target test outcomes."""
    bank = donor_study_order(eligible)
    if not allow_donors or int(k) <= 0 and explicit is None and rule != ALL_RULE:
        return {
            "eligible": bank,
            "selected": [],
            "k": 0,
            "rule": ZERO_RULE,
            "rng_seed": None,
            "test_outcomes_used": False,
            "validation_outcomes_used": False,
        }
    if explicit is not None:
        wanted = [str(s) for s in explicit]
        missing = [s for s in wanted if s not in set(bank)]
        if missing:
            raise ValueError(f"requested donors not in eligible bank: {missing}")
        selected = list(dict.fromkeys(wanted))
        return {
            "eligible": bank,
            "selected": selected,
            "k": len(selected),
            "rule": EXPLICIT_RULE,
            "rng_seed": None,
            "test_outcomes_used": False,
            "validation_outcomes_used": False,
        }
    key = str(rule)
    if key in {"zero", ZERO_RULE}:
        selected: list[str] = []
        used = ZERO_RULE
    elif key in {"all", ALL_RULE}:
        selected = list(bank)
        used = ALL_RULE
    elif key in {"random", RANDOM_RULE}:
        if rng_seed is None:
            raise ValueError("random donor rule requires rng_seed")
        n = min(max(int(k), 0), len(bank))
        if n == 0:
            selected = []
        else:
            rng = np.random.default_rng(int(rng_seed))
            idx = rng.choice(len(bank), size=n, replace=False)
            selected = [bank[int(i)] for i in idx]
        used = RANDOM_RULE
    elif key in {"prefix", PREFIX_RULE, "selected"}:
        selected = bank[: max(int(k), 0)]
        used = PREFIX_RULE
    else:
        raise ValueError(f"unknown donor rule {rule!r}")
    return {
        "eligible": bank,
        "selected": selected,
        "k": len(selected),
        "rule": used,
        "rng_seed": int(rng_seed) if used == RANDOM_RULE else None,
        "test_outcomes_used": False,
        "validation_outcomes_used": False,
    }


def decide_abstention(
    target_only_val: float | None,
    candidate_val: float | None,
    *,
    metric: str = ABSTAIN_METRIC,
    min_improvement: float = ABSTAIN_MIN_IMPROVEMENT,
    lower_is_better: bool = LOWER_IS_BETTER,
    n_val_studies: int | None = None,
) -> dict[str, Any]:
    """Choose target-only vs sharing using target validation only.

    Default: target-only on missing scores, n_val_studies < 1, ties, or
    improvement ≤ ``min_improvement``. Threshold is not taken from test.
    """
    rec: dict[str, Any] = {
        "metric": metric,
        "min_improvement": float(min_improvement),
        "lower_is_better": bool(lower_is_better),
        "tie_rule": ABSTAIN_TIE,
        "target_only_val": target_only_val,
        "candidate_val": candidate_val,
        "n_val_studies": n_val_studies,
        "test_outcomes_used": False,
        "threshold_from_test": False,
    }
    if n_val_studies is not None and int(n_val_studies) < 1:
        rec.update({"decision": "target_only", "reason": "insufficient_validation_support",
                    "delta_candidate_minus_target": None})
        return rec
    if target_only_val is None or candidate_val is None:
        rec.update({"decision": "target_only", "reason": "insufficient_validation_support",
                    "delta_candidate_minus_target": None})
        return rec
    a = float(target_only_val)
    b = float(candidate_val)
    delta = b - a
    rec["delta_candidate_minus_target"] = delta
    if lower_is_better:
        if delta < -float(min_improvement):
            rec.update({"decision": "share", "reason": "validation_improvement"})
        elif delta == 0 or abs(delta) <= float(min_improvement):
            rec.update({"decision": "target_only", "reason": "tie_or_insufficient_improvement"})
        else:
            rec.update({"decision": "target_only", "reason": "candidate_worse"})
    else:
        if delta > float(min_improvement):
            rec.update({"decision": "share", "reason": "validation_improvement"})
        elif delta == 0 or abs(delta) <= float(min_improvement):
            rec.update({"decision": "target_only", "reason": "tie_or_insufficient_improvement"})
        else:
            rec.update({"decision": "target_only", "reason": "candidate_worse"})
    return rec


def val_score(block: dict[str, Any], metric: str = ABSTAIN_METRIC) -> float | None:
    if not block or block.get("status") != "ok":
        return None
    key = "rmse_log" if metric == ABSTAIN_METRIC else metric
    v = block.get(key)
    if isinstance(v, dict):
        v = v.get("study_balanced")
    return None if v is None else float(v)
