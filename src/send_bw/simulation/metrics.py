"""Monte Carlo operating characteristics with uncertainty."""

from __future__ import annotations

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval; finite and valid at k = 0 and k = n."""
    if n == 0:
        return (0.0, 1.0)
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def mcse_prop(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n > 0 else float("nan")


def prop_summary(x: np.ndarray, level: float = 0.95) -> dict[str, float]:
    x = np.asarray(x, bool)
    n, k = len(x), int(x.sum())
    p = k / n if n else float("nan")
    lo, hi = clopper_pearson(k, n, level)
    return {"est": p, "k": k, "n": n, "mcse": mcse_prop(p, n), "ci_lo": lo, "ci_hi": hi}
