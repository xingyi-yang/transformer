"""Proper scores and calibration helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per column. samples: (S, n), y: (n,)."""
    S = samples.shape[0]
    xs = np.sort(samples, axis=0)
    t1 = np.mean(np.abs(samples - y[None]), axis=0)
    i = np.arange(1, S + 1)[:, None]
    t2 = np.sum((2 * i - S - 1) * xs, axis=0) / (S * S)
    return t1 - t2


def crps_gaussian(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def energy_score(samples: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_pairs: int = 200) -> float:
    """Multivariate energy score for one trajectory. samples: (S, d), y: (d,)."""
    S = samples.shape[0]
    t1 = np.mean(np.linalg.norm(samples - y[None], axis=1))
    i = rng.integers(0, S, max_pairs)
    j = rng.integers(0, S, max_pairs)
    t2 = np.mean(np.linalg.norm(samples[i] - samples[j], axis=1))
    return float(t1 - 0.5 * t2)


def interval_hits(samples: np.ndarray, y: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(samples, 0.5 - level / 2, axis=0)
    hi = np.quantile(samples, 0.5 + level / 2, axis=0)
    return (lo <= y) & (y <= hi), hi - lo


def pit(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(samples <= y[None], axis=0)
