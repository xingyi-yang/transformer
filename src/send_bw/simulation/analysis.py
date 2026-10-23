"""Prespecified final analyses (identical for every response generator).

Estimand: Delta_g(T) = mu_g(T) - mu_0(T), the baseline-standardized active-control difference in
mean log weight at fixed horizon T (exp(Delta) is a geometric-mean ratio). Sex is a covariate;
a common treatment effect across sexes is assumed (equal allocation per sex).

endpoint_ancova: OLS of Z(T) on group, centered baseline and sex. Uses only Z(T); it cannot gain
power from intermediate visits.

rm_gls: implemented and requestable; not a production default after locked
A/B/C1 diagnostics (see docs/remediation/analysis_closeout_20261022/CLOSEOUT.md).
Repeated-measures GLS using all post-baseline scheduled visits with
  - time-specific intercepts, sex-by-time and baseline-by-time effects (saturated over visits),
  - group-by-time effects through a FIXED basis psi(t) = [t/T, (t/T)^2] (truncated to the number
    of visits), anchored at 0 at baseline, so Delta_g(T) = psi(T)' gamma_g = sum(gamma_g),
  - a working within-animal covariance V0 estimated from historical training data (plug-in,
    scaled by an estimated factor s^2), model-based or CR1 cluster-robust variance,
  - t reference distribution with df = N_animals - (3 + G * k).
Benefit from intermediate visits relies on the psi(t) shape assumption; bias/coverage and
false-positive behaviour must be verified by simulation for every candidate design.

mmrm_visit_contrast (Candidate C1, experimental, opt-in; not a default production analysis):
  saturated treatment-by-visit means with an explicit endpoint contrast gamma_{g,m},
  historical plug-in V0 weights, and animal-level CR1 sandwich SEs. Finite-sample
  multiplier and df choices are experimental (not claimed conservative). Not generally
  valid under prior-outcome-dependent MAR when V0 is misspecified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class AnalysisResult:
    est: np.ndarray  # (G,)
    se: np.ndarray
    df: float
    p: np.ndarray
    failed: bool
    reason: str = ""
    sec_est: np.ndarray | None = None  # secondary fixed weighted-average contrast (GLS only)
    sec_se: np.ndarray | None = None


def _fail(G: int, reason: str) -> AnalysisResult:
    nan = np.full(G, np.nan)
    return AnalysisResult(nan, nan.copy(), np.nan, np.ones(G), True, reason)


def secondary_basis(query_times: np.ndarray, weights: np.ndarray, T: float, k: int) -> np.ndarray:
    """c such that c' gamma_g = sum_j a_j [mu_g(t_j) - mu_0(t_j)] under the GLS group basis."""
    q = np.asarray(query_times, float) / T
    return (np.asarray(weights, float)[:, None] * np.column_stack([q, q**2])[:, :k]).sum(0)


def secondary_visit_weights(query_times: np.ndarray, weights: np.ndarray, schedule: np.ndarray,
                            atol: float = 1e-9) -> np.ndarray | None:
    """Map a secondary contrast onto scheduled visits, or None if any query time is absent.

    Sparse schedules do not interpolate missing query days.
    """
    times = np.asarray(schedule, float)
    w_sched = np.zeros(len(times))
    for t, a in zip(np.asarray(query_times, float), np.asarray(weights, float), strict=True):
        j = int(np.argmin(np.abs(times - t)))
        if abs(times[j] - t) > atol:
            return None
        w_sched[j] += float(a)
    return w_sched


def _between_design(zb: np.ndarray, sex: np.ndarray, group: np.ndarray, G: int) -> np.ndarray:
    zb_c = zb - zb.mean()
    D = np.zeros((len(zb), G))
    for g in range(1, G + 1):
        D[:, g - 1] = group == g
    return np.column_stack([np.ones(len(zb)), zb_c, (sex == 1).astype(float), D])


def endpoint_ancova(zb: np.ndarray, sex: np.ndarray, group: np.ndarray, zT: np.ndarray, G: int) -> AnalysisResult:
    ok = np.isfinite(zT)
    X = _between_design(zb[ok], sex[ok], group[ok], G)
    y = zT[ok]
    n, p = X.shape
    if n - p < 2:
        return _fail(G, "insufficient_df")
    for g in range(G + 1):
        if not np.any(group[ok] == g):
            return _fail(G, f"empty_group_{g}")
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return _fail(G, "singular")
    beta = XtX_inv @ X.T @ y
    r = y - X @ beta
    s2 = r @ r / (n - p)
    se = np.sqrt(np.diag(XtX_inv)[3:] * s2)
    est = beta[3:]
    tstat = est / se
    df = n - p
    pval = 2 * stats.t.sf(np.abs(tstat), df)
    return AnalysisResult(est, se, float(df), pval, False)


def rm_gls(zb: np.ndarray, sex: np.ndarray, group: np.ndarray, Z: np.ndarray, times: np.ndarray, T: float,
           V0: np.ndarray, G: int, variance: str = "model",
           secondary: tuple[np.ndarray, np.ndarray] | None = None) -> AnalysisResult:
    """Z: (N, m) post-baseline outcomes at `times` (NaN = missing). V0: (m, m) working covariance."""
    N, m = Z.shape
    if times[-1] != T:
        raise ValueError("every schedule must include the analysis horizon T as its last visit")
    k = min(2, m)
    psi = np.column_stack([times / T, (times / T) ** 2])[:, :k]
    zb_c = zb - zb.mean()
    sexm = (sex == 1).astype(float)
    p_time = 3 * m
    p = p_time + G * k
    obs = np.isfinite(Z)
    if not obs[:, -1].any():
        return _fail(G, "no_endpoint_data")
    for g in range(G + 1):
        if not np.any((group == g) & obs.any(1)):
            return _fail(G, f"empty_group_{g}")
    # build the full (N, m, p) design once; rows of unobserved visits are dropped per pattern
    Xf = np.zeros((N, m, p))
    ar = np.arange(m)
    Xf[:, ar, ar] = 1.0
    Xf[:, ar, m + ar] = sexm[:, None]
    Xf[:, ar, 2 * m + ar] = zb_c[:, None]
    for g in range(1, G + 1):
        sel = group == g
        Xf[np.ix_(sel, ar, np.arange(p_time + (g - 1) * k, p_time + g * k))] = psi[None]
    XtVX = np.zeros((p, p))
    XtVy = np.zeros(p)
    groups_by_pattern: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    any_obs = obs.any(1)
    keys = np.packbits(obs, axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    for pid in np.unique(inv):
        rows = np.where((inv == pid) & any_obs)[0]
        if not len(rows):
            continue
        o = obs[rows[0]]
        try:
            Vi = np.linalg.inv(V0[np.ix_(o, o)])
        except np.linalg.LinAlgError:
            return _fail(G, "singular_working_covariance")
        Xp = Xf[rows][:, o, :]
        yp = Z[rows][:, o]
        XtVX += np.einsum("aqi,qr,arj->ij", Xp, Vi, Xp)
        XtVy += np.einsum("aqi,qr,ar->i", Xp, Vi, yp)
        groups_by_pattern.append((rows, Xp, Vi, yp))
    n_animals = int(any_obs.sum())
    n_obs = int(obs.sum())
    df = n_animals - (3 + G * k)
    if df < 2 or n_obs - p < 2:
        return _fail(G, "insufficient_df")
    try:
        bread = np.linalg.inv(XtVX)
    except np.linalg.LinAlgError:
        return _fail(G, "singular")
    if not np.all(np.isfinite(bread)) or np.linalg.cond(XtVX) > 1e12:
        return _fail(G, "ill_conditioned")
    beta = bread @ XtVy
    C = np.zeros((G, p))
    for g in range(G):
        C[g, p_time + g * k:p_time + (g + 1) * k] = psi[-1]
    if variance == "model":
        q = 0.0
        for _, Xp, Vi, yp in groups_by_pattern:
            r = yp - Xp @ beta
            q += float(np.einsum("aq,qr,ar->", r, Vi, r))
        cov_b = q / (n_obs - p) * bread
    elif variance == "cluster_robust":
        meat = np.zeros((p, p))
        for _, Xp, Vi, yp in groups_by_pattern:
            r = yp - Xp @ beta
            u = np.einsum("aqi,qr,ar->ai", Xp, Vi, r)
            meat += u.T @ u
        cov_b = bread @ meat @ bread * n_animals / max(n_animals - (3 + G * k), 1)
    else:
        raise ValueError(variance)
    est = C @ beta
    se = np.sqrt(np.clip(np.einsum("gi,ij,gj->g", C, cov_b, C), 0, None))
    if np.any(se <= 0) or not np.all(np.isfinite(se)):
        return _fail(G, "nonpositive_se")
    pval = 2 * stats.t.sf(np.abs(est / se), df)
    out = AnalysisResult(est, se, float(df), pval, False)
    if secondary is not None:
        c2 = secondary_basis(secondary[0], secondary[1], T, k)
        C2 = np.zeros((G, p))
        for g in range(G):
            C2[g, p_time + g * k:p_time + (g + 1) * k] = c2
        out.sec_est = C2 @ beta
        out.sec_se = np.sqrt(np.clip(np.einsum("gi,ij,gj->g", C2, cov_b, C2), 0, None))
    return out


def mmrm_visit_contrast(zb: np.ndarray, sex: np.ndarray, group: np.ndarray, Z: np.ndarray,
                        times: np.ndarray, T: float, V0: np.ndarray, G: int,
                        secondary: tuple[np.ndarray, np.ndarray] | None = None) -> AnalysisResult:
    """Saturated treatment-by-visit GLS with explicit endpoint contrast and CR1 SEs.

    Mean at visit j:
        E[Z_ij] = alpha_j + beta_sex,j I(sex_i=1) + beta_zb,j (zb_i - mean zb)
                  + sum_g gamma_{g,j} I(group_i=g)
    Endpoint contrast: c_g' theta = gamma_{g,m} = mu_g(T) - mu_0(T).

    Working covariance is the historical plug-in V0, inverted on each animal's observed
    visit subset. Point estimation is GLS. Standard errors use the animal-level sandwich:

        bread B = (sum_i X_i' V0[o_i,o_i]^{-1} X_i)^{-1}
        score u_i = X_i' V0[o_i,o_i]^{-1} r_i
        meat = sum_i u_i u_i'
        Cov = [N / (N - p_visit)] B meat B
        Var(c_g' theta_hat) = c_g' Cov c_g

    Experimental finite-sample choices (not claimed conservative; not Kenward-Roger):
        p_visit = 3 + G, matching the complete-data per-visit between-animal model
        (the same count as endpoint ANCOVA). Using the full mean-model dimension
        p = (3+G)m as the HC1 divisor overstates complete-data expected variance.
        Reference df = N_T - p_visit, where N_T is the number of animals observed at T.

    Shared study-level random effects that are common across arms at visit j are absorbed
    by alpha_j and cancel in concurrent-control contrasts. Treatment-specific study
    effects would leave within-arm dependence that animal-level clustering does not
    capture; this implementation does not adjust for that.

    Missingness: the observed-data GLS functional is the target. That equals Delta_g(T)
    under complete data, MCAR, or missingness depending only on baseline covariates
    already in the mean. It is not claimed generally consistent under
    prior-outcome-dependent MAR when V0 is misspecified. Sandwich SEs do not restore
    that consistency.
    """
    N, m = Z.shape
    if times[-1] != T:
        raise ValueError("every schedule must include the analysis horizon T as its last visit")
    zb_c = zb - zb.mean()
    sexm = (sex == 1).astype(float)
    p_visit = 3 + G
    p = p_visit * m
    obs = np.isfinite(Z)
    end_ok = obs[:, -1]
    if not end_ok.any():
        return _fail(G, "no_endpoint_data")
    n_end = int(end_ok.sum())
    for g in range(G + 1):
        if int(((group == g) & end_ok).sum()) < 2:
            return _fail(G, f"empty_group_{g}")
    Xf = np.zeros((N, m, p))
    ar = np.arange(m)
    Xf[:, ar, ar] = 1.0
    Xf[:, ar, m + ar] = sexm[:, None]
    Xf[:, ar, 2 * m + ar] = zb_c[:, None]
    for g in range(1, G + 1):
        sel = group == g
        Xf[np.ix_(sel, ar, 3 * m + (g - 1) * m + ar)] = np.eye(m)[None]
    XtVX = np.zeros((p, p))
    XtVy = np.zeros(p)
    groups_by_pattern: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    any_obs = obs.any(1)
    keys = np.packbits(obs, axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    for pid in np.unique(inv):
        rows = np.where((inv == pid) & any_obs)[0]
        if not len(rows):
            continue
        o = obs[rows[0]]
        try:
            Vi = np.linalg.inv(V0[np.ix_(o, o)])
        except np.linalg.LinAlgError:
            return _fail(G, "singular_working_covariance")
        Xp = Xf[rows][:, o, :]
        yp = Z[rows][:, o]
        XtVX += np.einsum("aqi,qr,arj->ij", Xp, Vi, Xp)
        XtVy += np.einsum("aqi,qr,ar->i", Xp, Vi, yp)
        groups_by_pattern.append((rows, Xp, Vi, yp))
    n_animals = int(any_obs.sum())
    df = n_end - p_visit
    if df < 2 or n_animals - p_visit < 2:
        return _fail(G, "insufficient_df")
    try:
        bread = np.linalg.inv(XtVX)
    except np.linalg.LinAlgError:
        return _fail(G, "singular")
    if not np.all(np.isfinite(bread)) or np.linalg.cond(XtVX) > 1e12:
        return _fail(G, "ill_conditioned")
    theta = bread @ XtVy
    C = np.zeros((G, p))
    for g in range(G):
        C[g, 3 * m + (g + 1) * m - 1] = 1.0
    meat = np.zeros((p, p))
    for _, Xp, Vi, yp in groups_by_pattern:
        r = yp - Xp @ theta
        u = np.einsum("aqi,qr,ar->ai", Xp, Vi, r)
        meat += u.T @ u
    cov_b = bread @ meat @ bread * n_animals / max(n_animals - p_visit, 1)
    est = C @ theta
    se = np.sqrt(np.clip(np.einsum("gi,ij,gj->g", C, cov_b, C), 0, None))
    if np.any(se <= 0) or not np.all(np.isfinite(se)):
        return _fail(G, "nonpositive_se")
    pval = 2 * stats.t.sf(np.abs(est / se), df)
    out = AnalysisResult(est, se, float(df), pval, False)
    if secondary is not None:
        w_sched = secondary_visit_weights(secondary[0], secondary[1], times)
        if w_sched is not None:
            C2 = np.zeros((G, p))
            for g in range(G):
                C2[g, 3 * m + g * m:3 * m + (g + 1) * m] = w_sched
            out.sec_est = C2 @ theta
            out.sec_se = np.sqrt(np.clip(np.einsum("gi,ij,gj->g", C2, cov_b, C2), 0, None))
    return out


def holm(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down; returns boolean rejections in the original order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    rej = np.zeros(m, bool)
    for rank, j in enumerate(order):
        if p[j] <= alpha / (m - rank):
            rej[j] = True
        else:
            break
    return rej


def confidence_interval(res: AnalysisResult, level: float) -> tuple[np.ndarray, np.ndarray]:
    if res.failed:
        nan = np.full_like(res.est, np.nan)
        return nan, nan
    q = stats.t.ppf(0.5 + level / 2, res.df)
    return res.est - q * res.se, res.est + q * res.se
