"""Inner target-training study-fold donor ranking.

One candidate per eligible donor. Rank by mean inner-fold RMSE on held-out
target TRAIN studies. Outer-validation share-or-abstain is separate
(``decide_abstention``). Test outcomes and generator scenario labels are
rejected inputs; they cannot change the ranking.

Synthetic diagnostic. Not design qualification. No operational recommendation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from send_bw.data.splits import donor_study_order
from send_bw.share_policy import decide_abstention, val_score

RANK_RULE = "inner_target_train_study_fold"
RANK_LABEL = (
    "inner target-training study-fold rank "
    "(not test-informed, not generator-scenario-informed)"
)
N_INNER_FOLDS = 2
K_DONORS = 1
RANK_METRIC = "inner_rmse_log"
LOWER_IS_BETTER = True
TIE_BREAK = "studyid_sort"


class DonorRankError(ValueError):
    pass


ScoreFn = Callable[[str, list[str], list[str]], float | None]


def inner_target_train_folds(train_ids: list[str], n_folds: int = N_INNER_FOLDS) -> list[dict[str, Any]]:
    """Deterministic study folds from target TRAIN STUDYIDs only.

    Studies are sorted, then assigned round-robin. No test, validation, or
    generator-scenario information is used.
    """
    ids = donor_study_order(train_ids)
    n = max(int(n_folds), 0)
    if n < 2 or len(ids) < n:
        return []
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, sid in enumerate(ids):
        buckets[i % n].append(sid)
    folds: list[dict[str, Any]] = []
    for i, held in enumerate(buckets):
        fit = [s for j, bucket in enumerate(buckets) if j != i for s in bucket]
        folds.append({
            "fold": i,
            "fit_target_ids": list(fit),
            "heldout_target_ids": list(held),
            "n_fit": len(fit),
            "n_heldout": len(held),
        })
    return folds


def _reject_leak(*, test_study_ids: list[str] | None, validation_study_ids: list[str] | None,
                 generator_scenario: Any, scenario_label: Any, test_outcomes: Any,
                 bank: Any) -> None:
    if generator_scenario is not None or scenario_label is not None or bank is not None:
        raise DonorRankError("generator scenario labels cannot influence ranking")
    if test_outcomes is not None:
        raise DonorRankError("test outcomes cannot influence ranking")
    if test_study_ids is None:
        test_study_ids = []
    if validation_study_ids is None:
        validation_study_ids = []
    if any(x is None for x in (test_study_ids, validation_study_ids)):
        raise DonorRankError("split id lists must be concrete")


def _assert_disjoint(name: str, used: set[str], forbidden: set[str]) -> None:
    leak = sorted(used & forbidden)
    if leak:
        raise DonorRankError(f"{name} overlap forbidden split IDs: {leak}")


def rank_one_donor_candidates(
    eligible_donors: list[str],
    target_train_ids: list[str],
    score_fn: ScoreFn,
    *,
    k: int = K_DONORS,
    n_folds: int = N_INNER_FOLDS,
    test_study_ids: list[str] | None = None,
    validation_study_ids: list[str] | None = None,
    generator_scenario: Any = None,
    scenario_label: Any = None,
    test_outcomes: Any = None,
    bank: Any = None,
) -> dict[str, Any]:
    """Rank one-donor candidates by mean inner-fold RMSE.

    ``score_fn(donor_id, fit_target_ids, heldout_target_ids) -> rmse | None``.
    The ranker never passes test IDs, test scores, or scenario labels to
    ``score_fn``. Missing fold scores skip that fold; a donor with no scores
    is ranked last. Ties break by STUDYID sort, not by test.
    """
    _reject_leak(test_study_ids=test_study_ids, validation_study_ids=validation_study_ids,
                 generator_scenario=generator_scenario, scenario_label=scenario_label,
                 test_outcomes=test_outcomes, bank=bank)
    test_set = {str(s) for s in (test_study_ids or [])}
    val_set = {str(s) for s in (validation_study_ids or [])}
    forbidden = test_set | val_set
    bank_ids = donor_study_order(eligible_donors)
    train_ids = donor_study_order(target_train_ids)
    _assert_disjoint("eligible donors", set(bank_ids), forbidden)
    _assert_disjoint("target train", set(train_ids), forbidden)
    if set(bank_ids) & set(train_ids):
        raise DonorRankError("donor IDs overlap target TRAIN studies")
    folds = inner_target_train_folds(train_ids, n_folds=n_folds)
    candidates: list[dict[str, Any]] = []
    for donor in bank_ids:
        fold_scores: list[dict[str, Any]] = []
        values: list[float] = []
        for fold in folds:
            fit_ids = list(fold["fit_target_ids"])
            held = list(fold["heldout_target_ids"])
            _assert_disjoint("inner fit", set(fit_ids), forbidden)
            _assert_disjoint("inner heldout", set(held), forbidden)
            raw = score_fn(donor, fit_ids, held)
            rec = {
                "fold": fold["fold"],
                "fit_target_ids": fit_ids,
                "heldout_target_ids": held,
                "score": None if raw is None else float(raw),
            }
            fold_scores.append(rec)
            if raw is not None:
                values.append(float(raw))
        mean = sum(values) / len(values) if values else None
        candidates.append({
            "donor_id": donor,
            "n_scored_folds": len(values),
            "n_folds": len(folds),
            "mean_inner_rmse": mean,
            "folds": fold_scores,
        })
    ranked = sorted(
        candidates,
        key=lambda c: (
            c["mean_inner_rmse"] is None,
            c["mean_inner_rmse"] if c["mean_inner_rmse"] is not None else 0.0,
            c["donor_id"],
        ),
    )
    selected = [c["donor_id"] for c in ranked
                if c["mean_inner_rmse"] is not None][: max(int(k), 0)]
    return {
        "rule": RANK_RULE,
        "rule_label": RANK_LABEL,
        "metric": RANK_METRIC,
        "lower_is_better": LOWER_IS_BETTER,
        "tie_break": TIE_BREAK,
        "k": len(selected),
        "n_folds": len(folds),
        "eligible": bank_ids,
        "target_train_ids": train_ids,
        "selected": selected,
        "candidates": ranked,
        "learned": True,
        "similarity_based": False,
        "validation_ranked": False,
        "inner_train_fold_ranked": True,
        "test_outcomes_used": False,
        "generator_scenario_used": False,
        "outer_validation_used": False,
        "insufficient_folds": len(folds) == 0,
    }


def share_or_abstain_after_rank(
    target_only_val: float | None,
    ranked_val: float | None,
    *,
    n_val_studies: int | None = None,
    test_outcomes: Any = None,
    generator_scenario: Any = None,
) -> dict[str, Any]:
    """Outer-validation abstention after ranking. Test/scenario still rejected."""
    if test_outcomes is not None:
        raise DonorRankError("test outcomes cannot influence abstention")
    if generator_scenario is not None:
        raise DonorRankError("generator scenario labels cannot influence abstention")
    dec = decide_abstention(target_only_val, ranked_val, n_val_studies=n_val_studies)
    dec["rank_rule"] = RANK_RULE
    dec["test_outcomes_used"] = False
    dec["generator_scenario_used"] = False
    return dec


def recording_score_fn(inner_rmse: dict[str, float]) -> tuple[ScoreFn, list[dict[str, Any]]]:
    """Test helper: returns scores by donor and records every call."""
    calls: list[dict[str, Any]] = []

    def _fn(donor_id: str, fit_ids: list[str], held_ids: list[str]) -> float | None:
        calls.append({
            "donor_id": donor_id,
            "fit_target_ids": list(fit_ids),
            "heldout_target_ids": list(held_ids),
        })
        if donor_id not in inner_rmse:
            return None
        return float(inner_rmse[donor_id])

    return _fn, calls


# Keep val_score import used by callers that score outer validation after rank.
__all__ = (
    "DonorRankError",
    "K_DONORS",
    "N_INNER_FOLDS",
    "RANK_LABEL",
    "RANK_METRIC",
    "RANK_RULE",
    "inner_target_train_folds",
    "rank_one_donor_candidates",
    "recording_score_fn",
    "share_or_abstain_after_rank",
    "val_score",
)
