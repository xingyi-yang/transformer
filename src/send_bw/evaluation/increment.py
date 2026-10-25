"""Matched increment diagnostic: identical statistics, hierarchical replication.

Observed studies and each simulated study draw use the same animal increment SD
and the same study-level mean of those SDs. Between-draw variability of the
study-level statistic is reported separately and is not mixed into the increment
SD used for the pred/obs ratio.

This module does not change evaluate_prestudy packs. Those packs average
per-animal increment SDs across draws that each carry a new study slope, then
pool animals across studies. That mismatched ratio is preserved as a published
artifact and is not rewritten here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Animal increment SD matches evaluate_prestudy: ndarray.std() default ddof=0.
ANIMAL_SD_DDOF = 0
# Between-draw / between-dataset spread of the study-level statistic.
STATISTIC_SD_DDOF = 1


def consecutive_increments(y: np.ndarray, observed_idx: np.ndarray) -> np.ndarray:
    """Diffs between consecutive observed visits (indices already time-ordered)."""
    idx = np.asarray(observed_idx)
    if idx.size < 2:
        return np.asarray([], float)
    return np.diff(np.asarray(y, float)[idx])


def animal_increment_sd(increments: np.ndarray) -> float | None:
    """SD of one animal's consecutive-visit increments."""
    inc = np.asarray(increments, float)
    if inc.size < 1:
        return None
    return float(np.std(inc, ddof=ANIMAL_SD_DDOF))


def study_increment_stat(animal_sds: list[float] | np.ndarray) -> float | None:
    """Study-level increment statistic: unweighted mean of animal increment SDs."""
    vals = [float(x) for x in np.asarray(animal_sds, float).ravel() if np.isfinite(x)]
    if not vals:
        return None
    return float(np.mean(vals))


def _animal_sds_from_matrix(y: np.ndarray, mask: np.ndarray) -> list[float]:
    sds: list[float] = []
    y = np.asarray(y, float)
    mask = np.asarray(mask, bool)
    for i in range(y.shape[0]):
        mi = np.where(mask[i])[0]
        if mi.size < 2:
            continue
        sd = animal_increment_sd(consecutive_increments(y[i], mi))
        if sd is not None:
            sds.append(sd)
    return sds


def observed_study_increment(y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """One realized observed study: animal SDs → study-level mean."""
    animal_sds = _animal_sds_from_matrix(y, mask)
    return {
        "animal_sds": animal_sds,
        "n_animals": int(len(animal_sds)),
        "study_stat": study_increment_stat(animal_sds),
    }


def simulated_study_increments(zs: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Each draw is one realized study. Same animal→study statistic as observed.

    Does not stack increments across draws before taking an SD. That would mix
    between-draw study-effect variability into the increment SD.
    """
    zs = np.asarray(zs, float)
    mask = np.asarray(mask, bool)
    if zs.ndim != 3:
        raise ValueError("zs must be (n_draws, n_animals, n_times)")
    draw_stats: list[float] = []
    n_animals_used: list[int] = []
    for d in range(zs.shape[0]):
        animal_sds = _animal_sds_from_matrix(zs[d], mask)
        stat = study_increment_stat(animal_sds)
        if stat is None:
            continue
        draw_stats.append(stat)
        n_animals_used.append(len(animal_sds))
    arr = np.asarray(draw_stats, float)
    return {
        "draw_study_stats": draw_stats,
        "n_draws": int(len(draw_stats)),
        "n_animals": int(n_animals_used[0]) if n_animals_used else 0,
        "expected_study_stat": float(arr.mean()) if arr.size else None,
        "between_draw_sd": float(arr.std(ddof=STATISTIC_SD_DDOF)) if arr.size > 1 else None,
    }


def mismatched_animal_pooled_pred_sd(zs: np.ndarray, mask: np.ndarray) -> float | None:
    """Published-pack style: mean over animals of (mean over draws of animal SD).

    Kept only to show why the pack ratio is not the matched study-level statistic.
    """
    zs = np.asarray(zs, float)
    mask = np.asarray(mask, bool)
    per_animal: list[float] = []
    for i in range(zs.shape[1]):
        mi = np.where(mask[i])[0]
        if mi.size < 2:
            continue
        sds = []
        for d in range(zs.shape[0]):
            sd = animal_increment_sd(consecutive_increments(zs[d, i], mi))
            if sd is not None:
                sds.append(sd)
        if sds:
            per_animal.append(float(np.mean(sds)))
    if not per_animal:
        return None
    return float(np.mean(per_animal))


def mixed_draw_increment_sd(zs: np.ndarray, mask: np.ndarray) -> float | None:
    """Incorrect: SD of increments stacked across draws (mixes study effects)."""
    zs = np.asarray(zs, float)
    mask = np.asarray(mask, bool)
    stacked: list[float] = []
    for i in range(zs.shape[1]):
        mi = np.where(mask[i])[0]
        if mi.size < 2:
            continue
        for d in range(zs.shape[0]):
            stacked.extend(consecutive_increments(zs[d, i], mi).tolist())
    if not stacked:
        return None
    return float(np.std(np.asarray(stacked, float), ddof=ANIMAL_SD_DDOF))


def compare_study(y: np.ndarray, zs: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Matched observed vs simulated increment statistics for one study."""
    obs = observed_study_increment(y, mask)
    sim = simulated_study_increments(zs, mask)
    obs_stat = obs["study_stat"]
    pred_e = sim["expected_study_stat"]
    ratio = None
    if obs_stat not in (None, 0) and pred_e is not None:
        ratio = float(pred_e / obs_stat)
    return {
        "observed": obs,
        "simulated": sim,
        "matched_ratio_pred_over_obs": ratio,
        "mismatched_animal_pooled_pred": mismatched_animal_pooled_pred_sd(zs, mask),
        "incorrect_mixed_draw_sd": mixed_draw_increment_sd(zs, mask),
        "note": (
            "expected_study_stat is E[same study-level mean of animal increment SDs] "
            "over draws. between_draw_sd is the SD of that study-level statistic. "
            "The matched ratio uses those two study-level numbers only."
        ),
    }


def hierarchical_mean(values: list[float], *, group_keys: list[Any]) -> dict[str, Any]:
    """Mean of group means. Outer groups are independent dataset replicates."""
    if not values:
        return {"n_groups": 0, "n_values": 0, "mean_of_group_means": None,
                "sd_of_group_means": None, "pooled_mean": None, "group_means": {}}
    by: dict[Any, list[float]] = {}
    for v, k in zip(values, group_keys, strict=True):
        if v is None or not np.isfinite(v):
            continue
        by.setdefault(k, []).append(float(v))
    means = {k: float(np.mean(vs)) for k, vs in by.items() if vs}
    arr = np.asarray(list(means.values()), float)
    pooled = [x for vs in by.values() for x in vs]
    return {
        "n_groups": int(len(means)),
        "n_values": int(len(pooled)),
        "mean_of_group_means": float(arr.mean()) if arr.size else None,
        "sd_of_group_means": float(arr.std(ddof=STATISTIC_SD_DDOF)) if arr.size > 1 else None,
        "pooled_mean": float(np.mean(pooled)) if pooled else None,
        "group_means": {str(k): v for k, v in means.items()},
        "note": (
            "n_groups is the outer replication count (datasets). "
            "pooled_mean treats nested studies/trains as iid and is not the dataset-level SE."
        ),
    }
