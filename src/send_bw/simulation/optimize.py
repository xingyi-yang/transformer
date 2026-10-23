"""Design evaluation, feasibility, Pareto frontier and finalist re-evaluation (ADEMP).

Aims: choose group sizes / analytical schedules meeting power, FWER, precision and failure criteria.
Data-generating mechanisms: response-model simulators (proposal), fresh batches, independent
generator B, stress scenarios and a synthetic oracle (generator A truth; synthetic only).
Estimand: Delta_g(T) (fixed across candidates). Methods: endpoint ANCOVA, repeated-measures GLS,
and opt-in experimental mmrm_visit_contrast (not a default). Holm. Performance: power
(per comparison / any / all-required), FWER (global and partial nulls),
bias, coverage, CI width, failure rate, with Monte Carlo SEs and Clopper-Pearson intervals.
Failed analyses count as non-rejections and are reported separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from send_bw.models.covariance import HierarchicalCovariance
from send_bw.simulation.analysis import (
    confidence_interval,
    endpoint_ancova,
    holm,
    mmrm_visit_contrast,
    rm_gls,
)
from send_bw.simulation.designs import Candidate, Request
from send_bw.simulation.metrics import prop_summary
from send_bw.simulation.paths import SimBatch, scenarios


def evaluate_candidates(batch: SimBatch, req: Request, cands: list[Candidate], working_cov: HierarchicalCovariance,
                        scen: dict[str, dict] | None = None) -> pd.DataFrame:
    scen = scen or scenarios(req)
    G = len(req.active)
    T = req.horizon_day
    grid = batch.grid
    rows = []
    B = batch.z_ctrl.shape[0]
    sec = req.secondary
    if sec is not None:
        qidx = np.array([int(np.argmin(np.abs(grid - t))) for t in sec[0]])
        if np.max(np.abs(grid[qidx] - sec[0])) > 1e-9:
            raise ValueError("secondary query times must be on the simulation grid")
        sec_shape = (batch.shapes[:, qidx] * sec[1][None]).sum(1)  # (G,) per unit delta
    for cand in cands:
        sel = batch.idx < np.array(cand.n_per_sex)[batch.group]
        tcols = np.array([int(np.argmin(np.abs(grid - t))) for t in cand.schedule])
        times = grid[tcols]
        V0 = working_cov.animal_cov(times)
        for sname, sc in scen.items():
            rec = {"est": np.full((B, G), np.nan), "lo": np.full((B, G), np.nan), "hi": np.full((B, G), np.nan),
                   "rej": np.zeros((B, G), bool), "failed": np.zeros(B, bool), "reasons": [],
                   "sec_est": np.full((B, G), np.nan), "sec_lo": np.full((B, G), np.nan),
                   "sec_hi": np.full((B, G), np.nan)}
            for b in range(B):
                if sc["source"] == "ctrl":
                    eff = np.zeros((len(sel), len(tcols)))
                    act = batch.group > 0
                    eff[act] = (sc["deltas"][batch.group[act] - 1][:, None] * batch.shapes[batch.group[act] - 1][:, tcols])
                    Z = batch.z_ctrl[b][:, tcols] + eff
                else:
                    Z = batch.z_pred[b][:, tcols]
                Z = np.where(batch.obs_mask[b][:, tcols], Z, np.nan)[sel]
                zb, sx, gr = batch.zb[b][sel], batch.sex[sel], batch.group[sel]
                if cand.analysis == "ancova":
                    res = endpoint_ancova(zb, sx, gr, Z[:, -1], G)
                elif cand.analysis == "gls":
                    res = rm_gls(zb, sx, gr, Z, times, T, V0, G, variance=req.gls_variance, secondary=sec)
                elif cand.analysis == "mmrm":
                    res = mmrm_visit_contrast(zb, sx, gr, Z, times, T, V0, G, secondary=sec)
                else:
                    raise ValueError(f"unknown analysis {cand.analysis}")
                if res.failed:
                    rec["failed"][b] = True
                    rec["reasons"].append(res.reason)
                    continue
                lo, hi = confidence_interval(res, req.ci_level)
                rec["est"][b], rec["lo"][b], rec["hi"][b] = res.est, lo, hi
                rec["rej"][b] = holm(res.p, req.alpha)
                if res.sec_est is not None:
                    q = stats.t.ppf(0.5 + req.ci_level / 2, res.df)
                    rec["sec_est"][b] = res.sec_est
                    rec["sec_lo"][b], rec["sec_hi"][b] = res.sec_est - q * res.sec_se, res.sec_est + q * res.sec_se
            truth = sc["deltas"] if sc["deltas"] is not None else None
            row = _summarize(cand, sname, rec, truth, batch.truth_pred if truth is None else None, req)
            if sec is not None and truth is not None and cand.analysis in {"gls", "mmrm"}:
                t2 = truth * sec_shape
                ok = ~rec["failed"]
                if ok.sum() > 1 and np.isfinite(rec["sec_est"][ok]).all():
                    e2 = rec["sec_est"][ok] - t2
                    c2 = (rec["sec_lo"][ok] <= t2) & (t2 <= rec["sec_hi"][ok])
                    row["secondary_bias_max_abs"] = float(np.max(np.abs(e2.mean(0))))
                    row["secondary_coverage_min"] = float(c2.mean(0).min())
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize(cand: Candidate, sname: str, rec: dict, truth: np.ndarray | None, truth_pred: np.ndarray | None,
               req: Request) -> dict[str, Any]:
    G = rec["est"].shape[1]
    nsx = len(req.sexes)
    out: dict[str, Any] = {"candidate": cand.label, "analysis": cand.analysis, "schedule": cand.schedule_name,
                           "n_per_sex": "/".join(map(str, cand.n_per_sex)), "allocation": cand.allocation,
                           "scenario": sname, "n_animals": cand.n_animals(nsx),
                           "n_measurements": cand.n_measurements(nsx), "n_visits": len(cand.schedule),
                           "B": len(rec["failed"])}
    c = req.costs
    out["cost"] = (c.get("per_animal", 1) * out["n_animals"] + c.get("per_measurement", 0) * out["n_measurements"]
                   + c.get("per_day", 0) * req.horizon_day)
    f = prop_summary(rec["failed"])
    out.update({"failure_rate": f["est"], "failure_ci_hi": f["ci_hi"]})
    reasons = pd.Series(rec["reasons"]).value_counts().to_dict() if rec["reasons"] else {}
    out["failure_reasons"] = ";".join(f"{k}:{v}" for k, v in reasons.items())
    rej = rec["rej"]
    names = [g.name for g in req.active]
    if truth is not None:
        nonnull = np.abs(truth) > 0
        for g in range(G):
            ps = prop_summary(rej[:, g])
            key = "power" if nonnull[g] else "type1"
            out[f"{key}_{names[g]}"] = ps["est"]
            out[f"{key}_{names[g]}_ci_lo"] = ps["ci_lo"]
            out[f"{key}_{names[g]}_ci_hi"] = ps["ci_hi"]
        if nonnull.any():
            req_idx = [g for g in req.required if nonnull[g]]
            pa = prop_summary(rej[:, nonnull].any(1))
            pr = prop_summary(rej[:, req_idx].all(1)) if req_idx else None
            out.update({"power_any": pa["est"], "power_any_ci_lo": pa["ci_lo"]})
            if pr:
                out.update({"power_all_required": pr["est"], "power_all_required_ci_lo": pr["ci_lo"],
                            "power_all_required_ci_hi": pr["ci_hi"], "power_all_required_mcse": pr["mcse"]})
        if (~nonnull).any():
            fw = prop_summary(rej[:, ~nonnull].any(1))
            out.update({"fwer": fw["est"], "fwer_ci_lo": fw["ci_lo"], "fwer_ci_hi": fw["ci_hi"],
                        "fwer_mcse": fw["mcse"]})
        tr = np.broadcast_to(truth, rec["est"].shape)
    else:
        tr = truth_pred
        pa = prop_summary(rej.any(1))
        out.update({"assurance_any": pa["est"], "assurance_any_ci_lo": pa["ci_lo"]})
        pr = prop_summary(rej[:, req.required].all(1))
        out.update({"assurance_all_required": pr["est"], "assurance_all_required_ci_lo": pr["ci_lo"]})
    ok = ~rec["failed"]
    if ok.sum() > 1:
        err = rec["est"][ok] - tr[ok]
        cover = (rec["lo"][ok] <= tr[ok]) & (tr[ok] <= rec["hi"][ok])
        out["bias_max_abs"] = float(np.nanmax(np.abs(np.nanmean(err, 0))))
        out["bias_mcse_max"] = float(np.nanmax(np.nanstd(err, 0) / np.sqrt(ok.sum())))
        out["coverage_min"] = float(np.min(cover.mean(0)))
        out["ci_halfwidth_mean"] = float(np.nanmean((rec["hi"][ok] - rec["lo"][ok]) / 2))
        out["emp_se_mean"] = float(np.mean(np.nanstd(rec["est"][ok], 0)))
    return out


def feasibility(results: pd.DataFrame, req: Request) -> pd.DataFrame:
    """One row per candidate with pass/fail per criterion (all must pass)."""
    rows = []
    for _, g in results.groupby("candidate", sort=False):
        alt = g[g["scenario"] == "alternative"].iloc[0]
        nulls = g[g["scenario"].str.contains("null")]
        r = {k: alt[k] for k in ["candidate", "analysis", "schedule", "n_per_sex", "allocation", "n_animals",
                                 "n_measurements", "cost"]}
        p_key = {"all_required": "power_all_required", "any": "power_any"}.get(req.power_requirement,
                                                                               "power_all_required")
        r["power"] = alt.get(p_key)
        r["power_ci_lo"] = alt.get(f"{p_key}_ci_lo")
        r["fwer_max_point"] = float(nulls["fwer"].max())
        r["fwer_max_ci_hi"] = float(nulls["fwer_ci_hi"].max())
        r["coverage_min"] = float(g["coverage_min"].min())
        bias_frac = float(alt.get("bias_max_abs", np.nan) / np.max(np.abs(req.deltas)))
        r["bias_frac"] = bias_frac
        r["failure_rate_max"] = float(g["failure_rate"].max())
        r["ci_halfwidth"] = alt.get("ci_halfwidth_mean")
        if "secondary_coverage_min" in g.columns:
            r["secondary_coverage_min"] = float(g["secondary_coverage_min"].min())  # reported, not a criterion
            r["secondary_bias_max_abs"] = alt.get("secondary_bias_max_abs")
        checks = {
            "pass_power": r["power"] >= req.power_target and r["power_ci_lo"] >= req.power_target - req.power_lower_tolerance,
            "pass_fwer": r["fwer_max_point"] <= req.fwer_max + req.fwer_tolerance
            and r["fwer_max_ci_hi"] <= req.fwer_max + req.fwer_upper_tolerance,
            "pass_coverage": r["coverage_min"] >= req.coverage_min,
            "pass_bias": bias_frac <= req.max_abs_bias_fraction,
            "pass_failures": r["failure_rate_max"] <= req.max_failure_rate,
            "pass_precision": req.precision_max_halfwidth is None or (r["ci_halfwidth"] or np.inf) <= req.precision_max_halfwidth,
        }
        r.update(checks)
        r["feasible"] = all(checks.values())
        if "model_predicted" in set(g["scenario"]):
            mp = g[g["scenario"] == "model_predicted"].iloc[0]
            r["assurance_all_required_model_effect"] = mp.get("assurance_all_required")
        rows.append(r)
    return pd.DataFrame(rows)


def pareto(feas: pd.DataFrame, objectives: tuple[str, ...] = ("n_animals", "n_measurements")) -> pd.DataFrame:
    f = feas[feas["feasible"]].copy()
    if f.empty:
        return f
    keep = []
    vals = f[list(objectives)].to_numpy(float)
    for i in range(len(f)):
        dominated = np.any(np.all(vals <= vals[i], 1) & np.any(vals < vals[i], 1))
        keep.append(not dominated)
    return f[np.array(keep)].sort_values(list(objectives))


def select_finalists(feas: pd.DataFrame, k: int = 3) -> tuple[list[str], dict[str, Any]]:
    """Cheapest feasible designs + comparator (cheapest feasible endpoint-ANCOVA design)."""
    f = feas[feas["feasible"]].sort_values(["cost", "n_animals", "n_measurements"])
    if f.empty:
        return [], {"status": "no_feasible_design"}
    fin = list(f["candidate"].head(k))
    comp = f[f["analysis"] == "ancova"]
    comparator = comp["candidate"].iloc[0] if len(comp) else None
    if comparator and comparator not in fin:
        fin.append(comparator)
    return fin, {"status": "feasible_found", "comparator": comparator}
