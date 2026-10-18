"""End-to-end design workflow shared by `send-bw design` and `send-bw propose-design`.

1. Applicability gates (unsupported contexts are refused, not extrapolated).
2. Proposal: every response model simulates the SAME candidates with the SAME final analysis,
   estimand, multiplicity rule, working covariance and constraints.
3. Finalists (cheapest feasible + endpoint-ANCOVA comparator) are re-evaluated on FRESH seeds under
   the proposing model, the independent generator B, stress scenarios and (synthetic only) the oracle.
4. A design is recommended only if it remains feasible under the fresh proposing-model batch AND
   generator B. 'no_feasible_design' is a valid outcome. Resource savings are claimed only when both
   the recommended design and its comparator meet all requirements in the final evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.simulation.designs import enumerate_candidates, load_request, validate_request
from send_bw.simulation.optimize import evaluate_candidates, feasibility, pareto, select_finalists
from send_bw.simulation.paths import Simulator, scenarios, simulate_batch
from send_bw.utils.artifacts import write_json
from send_bw.utils.seeds import SeedStream


def run_design_workflow(models: dict[str, Any], dcfg: dict[str, Any], out_dir: Path, seed: int,
                        truth_dir: str | None, log) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = SeedStream(seed)
    req = load_request(dcfg["request"])
    if "spline" not in models:
        raise ValueError("the spline model is required: it provides the prespecified working covariance")
    spline = models["spline"]
    cref = req.compound_reference(spline.compound_ref)
    validation = validate_request(req, spline.spec, cref)
    write_json(out_dir / "request_validation.json", validation)
    if validation["status"] != "supported":
        summary = {"status": "not_supported", "validation": validation}
        write_json(out_dir / "design_summary.json", summary)
        log.info(f"[design] request NOT supported: {validation['problems']}")
        return summary
    for m in models.values():
        m.compound_ref = cref
    working_cov = spline.cov
    cands = enumerate_candidates(req)
    n_max = tuple(max(c.n_per_sex[i] for c in cands) for i in range(len(req.groups)))
    grid_t = {t for c in cands for t in c.schedule}
    if req.secondary is not None:
        grid_t |= set(req.secondary[0].tolist())
    grid = np.array(sorted(grid_t))
    B_prop = int(dcfg.get("replicates_proposal", 300))
    B_fin = int(dcfg.get("replicates_final", 600))
    nonconclusive = B_fin < 2000
    proposals: dict[str, Any] = {}
    finalists: set[str] = set()
    for name in [m for m in dcfg.get("proposal_models", list(models)) if m in models]:
        model = models[name]
        unreliable = name == "bayesian" and not model.diagnostics.get("main", {}).get("reliable", False)
        t0 = time.time()
        batch = simulate_batch(Simulator(name, model, "model"), req, n_max, grid, B_prop, ss.seed("proposal", name))
        res = evaluate_candidates(batch, req, cands, working_cov)
        feas = feasibility(res, req)
        par = pareto(feas)
        fin, info = select_finalists(feas, int(dcfg.get("n_finalists", 3)))
        res.to_csv(out_dir / f"proposal_{name}_results.csv", index=False)
        feas.to_csv(out_dir / f"proposal_{name}_feasibility.csv", index=False)
        par.to_csv(out_dir / f"proposal_{name}_pareto.csv", index=False)
        proposals[name] = {"finalists": fin, **info, "n_feasible": int(feas["feasible"].sum()),
                           "n_candidates": len(feas), "unreliable_bayesian": unreliable,
                           "shape_notes": batch.notes, "seconds": time.time() - t0,
                           "effect_shapes": {g.name: dict(zip(grid.tolist(), np.round(batch.shapes[i], 3).tolist(),
                                                              strict=True)) for i, g in enumerate(req.active)}}
        from send_bw.simulation.truth import monte_carlo_true_contrast

        mc = monte_carlo_true_contrast(Simulator(name, model, "model"), req, 300, ss.seed("contrast", name))
        proposals[name]["predicted_contrast_pct_at_T"] = dict(zip([g.name for g in req.active],
                                                                  np.round(100 * np.expm1(mc), 2).tolist(), strict=True))
        finalists.update(fin)
        log.info(f"[design] proposal {name}: {proposals[name]['n_feasible']}/{len(feas)} feasible "
                 f"({time.time() - t0:.0f}s); finalists {fin}")
    oracle_contrast = None
    if truth_dir and (Path(truth_dir) / "params.json").exists() and dcfg.get("oracle", True):
        from send_bw.simulation.truth import monte_carlo_true_contrast, oracle_simulator

        mc = monte_carlo_true_contrast(oracle_simulator(truth_dir, cref), req, 300, ss.seed("contrast", "oracle"))
        oracle_contrast = dict(zip([g.name for g in req.active], np.round(100 * np.expm1(mc), 2).tolist(), strict=True))
    if not finalists:
        summary = {"status": "no_feasible_design", "proposals": proposals, "validation": validation,
                   "replicates_proposal": B_prop, "replicates_final": None, "oracle_contrast_pct_at_T": oracle_contrast,
                   "monte_carlo_nonconclusive": B_prop < 2000,
                   "labels": ["no candidate met all prespecified criteria under any proposing model; "
                              "this is a valid outcome, not a failure"]}
        write_json(out_dir / "design_summary.json", summary)
        return summary

    fin_cands = [c for c in cands if c.label in finalists]
    final_sims: list[Simulator] = [Simulator(f"fresh_{n}", models[n], "model") for n in proposals]
    have_truth = bool(truth_dir) and (Path(truth_dir) / "params.json").exists()
    from send_bw.simulation.truth import stress_simulators

    stress = stress_simulators(truth_dir if have_truth else None, cref)
    final_sims += stress if dcfg.get("stress", True) else stress[:1]  # generator B is always required
    if dcfg.get("oracle", True) and truth_dir:
        from send_bw.simulation.truth import oracle_simulator

        try:
            final_sims.append(oracle_simulator(truth_dir, cref))
        except FileNotFoundError:
            log.info("[design] no truth available; oracle reference skipped")
    scen = {k: v for k, v in scenarios(req).items() if k != "model_predicted"}
    finals = []
    for sim in final_sims:
        t0 = time.time()
        batch = simulate_batch(sim, req, n_max, grid, B_fin, ss.seed("final", sim.name))
        res = evaluate_candidates(batch, req, fin_cands, working_cov, scen)
        feas = feasibility(res, req)
        res.insert(0, "simulator", sim.name)
        feas.insert(0, "simulator", sim.name)
        feas["simulator_label"] = sim.label
        finals.append((res, feas))
        log.info(f"[design] final eval under {sim.name}: {int(feas['feasible'].sum())}/{len(feas)} feasible "
                 f"({time.time() - t0:.0f}s)")
    final_res = pd.concat([f[0] for f in finals], ignore_index=True)
    final_feas = pd.concat([f[1] for f in finals], ignore_index=True)
    final_res.to_csv(out_dir / "final_results.csv", index=False)
    final_feas.to_csv(out_dir / "final_feasibility.csv", index=False)

    ff = final_feas.set_index(["simulator", "candidate"])
    recs = {}
    for name, prop in proposals.items():
        rec = {"proposing_model": name, "unreliable_bayesian": prop["unreliable_bayesian"]}
        ok = []
        for c in prop["finalists"]:
            fresh = bool(ff.loc[(f"fresh_{name}", c), "feasible"])
            indep = bool(ff.loc[("generator_B", c), "feasible"]) if ("generator_B", c) in ff.index else False
            ok.append((c, fresh, indep))
        passing = [c for c, f, i in ok if f and i]
        rec["finalist_checks"] = [{"candidate": c, "fresh_same_model": f, "generator_B": i} for c, f, i in ok]
        if passing:
            best = final_feas[(final_feas["simulator"] == f"fresh_{name}") & final_feas["candidate"].isin(passing)] \
                .sort_values(["cost", "n_animals"]).iloc[0]
            rec["status"] = "recommended" if not prop["unreliable_bayesian"] else "recommended_but_unreliable_model"
            rec["design"] = best["candidate"]
            rec["n_animals"] = int(best["n_animals"])
            rec["n_measurements"] = int(best["n_measurements"])
            rec["cost"] = float(best["cost"])
            rec["under_each_simulator"] = (final_feas[final_feas["candidate"] == best["candidate"]]
                                           [["simulator", "power", "power_ci_lo", "fwer_max_point", "fwer_max_ci_hi",
                                             "coverage_min", "bias_frac", "failure_rate_max", "feasible"]]
                                           .to_dict("records"))
            rec["not_robust_under"] = sorted(final_feas[(final_feas["candidate"] == best["candidate"])
                                                        & ~final_feas["feasible"]]["simulator"].tolist())
            comp = prop.get("comparator")
            if comp and comp != best["candidate"] and ("generator_B", comp) in ff.index:
                comp_ok = bool(ff.loc[(f"fresh_{name}", comp), "feasible"]) and bool(ff.loc[("generator_B", comp), "feasible"])
                rec["comparator"] = {"design": comp, "feasible_final": comp_ok,
                                     "n_animals": int(ff.loc[(f"fresh_{name}", comp), "n_animals"]),
                                     "n_measurements": int(ff.loc[(f"fresh_{name}", comp), "n_measurements"])}
                rec["resource_claim_allowed"] = comp_ok
                if comp_ok:
                    rec["resource_difference_vs_comparator"] = {
                        "animals": rec["n_animals"] - rec["comparator"]["n_animals"],
                        "measurements": rec["n_measurements"] - rec["comparator"]["n_measurements"]}
            else:
                rec["resource_claim_allowed"] = False
                rec["comparator_note"] = "recommended design is itself the endpoint comparator, or no comparator"
        else:
            rec["status"] = "no_feasible_design_after_independent_evaluation"
        recs[name] = rec
    summary = {"status": "evaluated", "request_id": req.request_id, "validation": validation,
               "proposals": proposals, "recommendations": recs, "replicates_proposal": B_prop,
               "replicates_final": B_fin, "oracle_contrast_pct_at_T": oracle_contrast, "simulators": [{"name": s.name, "label": s.label} for s in final_sims],
               "monte_carlo_nonconclusive": nonconclusive,
               "labels": ["SYNTHETIC PROOF OF CONCEPT" if truth_dir else "NOT VALIDATED ON REAL DATA",
                          "conditional power at prespecified effects is reported separately from predictive "
                          "assurance under model-predicted effects",
                          req.welfare_note]}
    write_json(out_dir / "design_summary.json", summary)
    return summary
