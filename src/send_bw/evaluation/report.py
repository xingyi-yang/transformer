"""Reproducible Markdown + HTML report built only from run artifacts."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from send_bw.utils.artifacts import read_json  # noqa: E402


def _r(p: Path) -> Any:
    return read_json(p) if p.exists() else None


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, nd) for x in v) + "]"
    return str(v)


def _table(rows: list[dict], cols: list[str], nd: int = 3) -> list[str]:
    if not rows:
        return ["(none)", ""]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c), nd) for c in cols) + " |")
    return out + [""]


def build_report(run) -> Path:
    d = run.dir
    rep = d / "report"
    plots = rep / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    man = _r(d / "manifest.json") or {}
    synthetic = bool(run.cfg.get("synthetic", {}).get("enabled"))
    L: list[str] = []
    title = "SYNTHETIC PROOF OF CONCEPT - NOT REAL-WORLD VALIDATION" if synthetic else \
        "REAL-DATA RUN - qualification stage must be checked before any use"
    L += [f"# send_bw report: {run.cfg.get('run_name', '')}", "", f"> **{title}**", "",
          "Body weight is the only modelled endpoint. Results cannot establish total animal requirements for a "
          "toxicology program, cannot justify reducing welfare observations or protocol checks, and are not "
          "regulatory-grade.", ""]
    L += ["## Run provenance", "",
          f"- config: `{man.get('config_path')}` (hash `{man.get('config_hash')}`), seed {man.get('seed')}",
          f"- git revision: {man.get('git_revision') or 'not available (git not installed / not a repo)'}",
          "- software: " + ", ".join(f"{k}={v}" for k, v in (man.get("software") or {}).items() if v), ""]
    st = man.get("stages", {})
    L += _table([{"stage": k, **{kk: vv for kk, vv in v.items() if kk in ("seconds", "status", "key")}}
                 for k, v in st.items()], ["stage", "status", "seconds", "key"], 1)

    # ---------------- data
    L += ["## Data ingestion and QC", ""]
    ing = st.get("ingest", {})
    L.append(f"- weights: {ing.get('n_bw')}, animals: {ing.get('n_animals')}, quarantined/flagged records: "
             f"{ing.get('n_quarantine')}, mapping version: {ing.get('mapping_version')}")
    qc = _r(d / "qc" / "qc_report.json") or {}
    L.append(f"- eligible animals: {qc.get('n_eligible_animals')}, modelling observations: {qc.get('n_model_obs')}")
    L.append(f"- QC checks: {qc.get('checks')}")
    L += ["", "Exclusion counts (animals):", ""]
    L += _table([{"reason": k, "n": v} for k, v in (qc.get("exclusion_counts") or {}).items()], ["reason", "n"])
    L += ["Limitations recorded during ingestion:", ""] + [f"- {x}" for x in qc.get("limitations", [])] + [""]
    mr = _r(d / "qc" / "mapping_report.json") or {}
    wtypes = pd.Series([w.get("type") for w in mr.get("warnings", [])]).value_counts().to_dict()
    L += [f"Mapping warnings by type: {wtypes or 'none'}", ""]
    pr = st.get("prepare", {})
    L += [f"Frozen study-level splits: {pr.get('split_counts')} (hash `{pr.get('split_hash')}`); "
          f"compound holdout: {pr.get('compound_holdout')}", ""]
    tm = _r(d / "splits" / "target_manifest.json")
    if tm:
        L += [f"Target-species manifests (frozen before donors): train n={tm.get('train', {}).get('n')} "
              f"hash `{tm.get('train', {}).get('hash')}`; validation n={tm.get('validation', {}).get('n')} "
              f"hash `{tm.get('validation', {}).get('hash')}`; test n={tm.get('test', {}).get('n')} "
              f"hash `{tm.get('test', {}).get('hash')}`. Sharing: {pr.get('sharing', 'context_specific')}; "
              f"donors selected: {pr.get('n_donor_studies', 0)}.", ""]

    # ---------------- models
    L += ["## Models", ""]
    for m in ("spline", "transformer", "bayesian"):
        h = _r(d / "models" / m / "training_history.json")
        s = st.get(f"train_{m}", {})
        if h is None:
            L += [f"### {m}: {s.get('status', 'not_run')}", "", f"{s.get('reason', '')}", ""]
            continue
        L += [f"### {m}", ""]
        if m == "spline":
            L += [f"- ridge alpha {h['alpha']} (validation-selected), train RMSE {h['train_rmse']:.4f}, "
                  f"validation RMSE {h['validation_rmse']:.4f}, OOF RMSE {h['oof_rmse']:.4f}, "
                  f"{h['n_params']} mean parameters, {h['n_members']} study-bootstrap members",
                  "- covariance (OOF residual ML): " + ", ".join(f"{k}={v:.4f}" for k, v in h["covariance"].items()
                                                                  if isinstance(v, float)),
                  f"- covariance fit: {h['covariance'].get('fit_info')}", ""]
        elif m == "transformer":
            L += [f"- architecture {h['arch']} (custom baseline-conditioned transformer; NOT a TFT)",
                  "- members (study-bootstrap except member 0): " + "; ".join(
                      f"m{x['member']}: best val RMSE {x['best_val_rmse']:.4f} after {x['epochs_run']} epochs (seed {x['seed']})"
                      for x in h["members"]),
                  "- covariance layer shared with the spline model (primary mean-function comparison)", ""]
            _plot_curves(h, plots / "transformer_training.png")
            L += ["![transformer training](plots/transformer_training.png)", ""]
        else:
            dg = _r(d / "models" / m / "diagnostics.json") or {}
            mn = dg.get("main", {})
            rel = "RELIABLE (diagnostic targets met)" if mn.get("reliable") else \
                "UNRELIABLE - diagnostics failed; not ranked as a valid benchmark"
            L += [f"- status: **{rel}**",
                  f"- NUTS ({mn.get('chains')} chains, {mn.get('tune')} warmup, {mn.get('draws')} draws, seed {mn.get('seed')}): "
                  f"R-hat max {_fmt(mn.get('rhat_max'))}, bulk ESS min {_fmt(mn.get('ess_bulk_min'), 0)}, "
                  f"tail ESS min {_fmt(mn.get('ess_tail_min'), 0)}, divergences {mn.get('divergences')}, "
                  f"max-treedepth hits {mn.get('maxdepth_hits')}; thresholds {mn.get('thresholds')}",
                  f"- sampler warnings: {mn.get('sampler_warnings') or 'none'}",
                  f"- worst R-hat parameters: {mn.get('worst_rhat_params')}",
                  f"- priors (on training-standardized scale): {mn.get('priors')}",
                  f"- prior predictive (pct change at last observation, 2.5/50/97.5%): "
                  f"{_fmt(mn.get('prior_predictive', {}).get('pct_change_last_obs_q025_q50_q975'), 1)} vs observed "
                  f"{_fmt(mn.get('prior_predictive', {}).get('observed_pct_change_q025_q50_q975'), 1)}; grams "
                  f"{_fmt(mn.get('prior_predictive', {}).get('grams_last_obs_q025_q50_q975'), 0)}; plausible="
                  f"{mn.get('prior_predictive', {}).get('plausible')}",
                  f"- posterior predictive (one-step) p-values by time bin: "
                  f"{mn.get('posterior_predictive', {}).get('by_time_bin')}; extreme bins: "
                  f"{mn.get('posterior_predictive', {}).get('extreme_bins')}",
                  f"- posterior means (log-weight scale): {mn.get('posterior_summary')}", "",
                  "Prior / likelihood / random-effect sensitivity (validation studies):", ""]
            L += _table(dg.get("sensitivity_comparison", []), ["variant", "reliable", "rhat_max",
                                                               "mean_pred_change_day28_validation", "sd_s_logw",
                                                               "sd_a_logw", "sigma_p_logw", "ell"], 4)
            if mn.get("draws", 0) < 1000:
                L += ["> Short Bayesian run: software check, not adequate posterior inference.", ""]

    opt = _r(d / "models" / "optional_models_status.json")
    if opt:
        L += ["### Optional models (not required benchmarks)", ""]
        L += _table([{"model": k, "status": v["status"], "task": v["task"], "reason": v["reason"]}
                     for k, v in opt.items()], ["model", "status", "task", "reason"])

    # ---------------- evaluation
    ev = _r(d / "evaluation" / "metrics.json") or {}
    L += ["## Held-out evaluation", "", "### Primary task: pre-study prediction of new studies", ""]
    rows = []
    for k, v in ev.items():
        if not k.endswith("pre_study"):
            continue
        if v.get("status") == "not_supported" or "observation_weighted" not in v:
            continue
        ow, sb = v["observation_weighted"], v["study_balanced"]
        rows.append({"model": v["model"], "split": v["split"], "studies": v["effective_sample_sizes"]["n_studies"],
                     "obs": v["effective_sample_sizes"]["n_obs"], "RMSE log": ow["rmse_log"], "RMSE g": ow["rmse_g"],
                     "MAE g": ow["mae_g"], "cov80": ow["cov80"], "cov95": ow["cov95"], "CRPS": ow["crps_log"],
                     "SB RMSE": sb["rmse_log"], "SB cov95": sb["cov95"], "energy": v["energy_score_mean"],
                     "grp-mean cov90": v["group_mean_cov90"], "contrast cov90": v["contrast_cov90"],
                     "contrast bias": v["contrast_bias"], "var ratio": v["between_animal_var_ratio_pred_over_obs"],
                     "incr ratio": v["increment_sd_ratio_pred_over_obs"], "max cov90": v["extrema_max_cov90"]})
    L += _table(rows, list(rows[0].keys()) if rows else [], 3)
    L += ["SB = study-balanced. Coverage targets: cov80=0.80, cov95=0.95, group-mean/contrast/max cov90=0.90; "
          "variance and increment ratios target 1. Few test studies -> wide uncertainty.", ""]
    gaps = [{"model": v["model"], "split": v["split"], **g} for k, v in ev.items() if k.endswith("pre_study")
            for g in v["correlation_by_gap"]]
    L += ["Residual correlation by gap (observed vs predicted):", ""]
    L += _table(gaps, ["model", "split", "gap_bin", "corr_obs", "corr_pred", "n_pairs"])
    L += ["### Secondary task: early-history forecasting (kept separate)", ""]
    eh = []
    for k, v in ev.items():
        if k.endswith("early_history"):
            if v.get("status") != "ok":
                eh.append({"model": v["model"], "split": v["split"], "status": v.get("status"), "note": v.get("reason")})
                continue
            cond = v.get("conditioned")
            if isinstance(cond, dict) and "rmse_log" in cond:
                eh.append({"model": v["model"], "split": v["split"], "status": "conditioned", **cond})
            pre = v.get("pre_study")
            if isinstance(pre, dict) and pre.get("status") == "not_supported":
                eh.append({"model": v["model"], "split": v["split"], "status": "not_supported",
                           "note": pre.get("reason")})
            elif isinstance(pre, dict) and "rmse_log" in pre:
                eh.append({"model": v["model"], "split": v["split"], "status": "pre_study", **pre})
    L += _table(eh, ["model", "split", "status", "rmse_log", "cov90", "crps_log", "n_obs", "note"])
    L += ["Success on the secondary task does not establish success on the primary task.", ""]
    tr = _r(d / "evaluation" / "transfer_table.json")
    if tr:
        L += ["### Predose-conditioned paired transfer (sign-defined)", "",
              "Δ_k = S_k − S_0 on the frozen target holdout. On RMSE/CRPS, Δ>0 is harmful. "
              "On contrast bias, Δ = |S_k| − |S_0|. Selective sharing is required later and is not run.", ""]
        rows = []
        for block in tr.values() if isinstance(tr, dict) and "rows" not in tr else [tr]:
            rows.extend(block.get("rows") or [])
        L += _table(rows, ["task", "model", "k", "metric", "S0", "Sk", "delta", "sign"])
    _plot_eval(run, plots)
    if (plots / "trajectories.png").exists():
        L += ["![trajectories](plots/trajectories.png)", ""]

    # ---------------- design
    ds = _r(d / "design" / "design_summary.json")
    L += ["## Experimental design", ""]
    if ds is None:
        L += ["Design stage not run.", ""]
    else:
        L += [f"- request validation: {ds.get('validation')}", f"- overall status: **{ds['status']}**",
              "- power = conditional power at the prespecified meaningful effects (Holm, all required); "
              "'assurance' columns use model-predicted effects and are a separate quantity",
              "- secondary estimand (GLS only): fixed weighted average of contrasts at the request's query "
              "times; reported for bias/coverage, not used for feasibility"]
        L += [f"- replicates: proposal {ds.get('replicates_proposal')}, final {ds.get('replicates_final')}"
              + (" -> **low replication: nonconclusive software check**" if ds.get("monte_carlo_nonconclusive") else ""),
              ""]
        for lab in ds.get("labels", []):
            L.append(f"> {lab}")
        L.append("")
        for name, p in (ds.get("proposals") or {}).items():
            L += [f"### Proposal under {name}", "",
                  f"- feasible candidates {p['n_feasible']}/{p['n_candidates']}; status {p.get('status')}; "
                  f"finalists {p['finalists']}; comparator {p.get('comparator')}"
                  + ("; **Bayesian model unreliable: results not ranked**" if p.get("unreliable_bayesian") else ""),
                  f"- effect-shape notes: {p.get('shape_notes') or 'model-predicted shapes'}",
                  f"- model-predicted contrast at T (%): {p.get('predicted_contrast_pct_at_T')}"
                  + (f"; synthetic oracle: {ds.get('oracle_contrast_pct_at_T')}" if ds.get("oracle_contrast_pct_at_T") else ""),
                  ""]
            fz = d / "design" / f"proposal_{name}_feasibility.csv"
            if fz.exists():
                f = pd.read_csv(fz).sort_values(["cost"])
                cols = ["candidate", "n_animals", "n_measurements", "power", "power_ci_lo", "fwer_max_point",
                        "fwer_max_ci_hi", "coverage_min", "bias_frac", "failure_rate_max", "feasible",
                        "assurance_all_required_model_effect", "secondary_coverage_min", "secondary_bias_max_abs"]
                L += _table(f.to_dict("records"), [c for c in cols if c in f.columns])
        _plot_power(run, plots)
        if (plots / "power_vs_animals.png").exists():
            L += ["![power](plots/power_vs_animals.png)", ""]
        if ds["status"] == "evaluated":
            L += ["### Fresh independent evaluation of finalists", ""]
            for s in ds["simulators"]:
                L.append(f"- `{s['name']}`: {s['label'] or 'fresh batch from the proposing response model'}")
            L.append("")
            ff = pd.read_csv(d / "design" / "final_feasibility.csv")
            L += _table(ff.to_dict("records"), ["simulator", "candidate", "n_animals", "power", "power_ci_lo",
                                                "fwer_max_point", "fwer_max_ci_hi", "coverage_min", "bias_frac",
                                                "failure_rate_max", "feasible"])
            L += ["### Recommendations", ""]
            for name, r in ds["recommendations"].items():
                L.append(f"- **{name}**: {r['status']}" + (f" -> `{r['design']}` ({r['n_animals']} animals, "
                                                          f"{r['n_measurements']} analytical weighings)" if r.get("design") else ""))
                if r.get("design"):
                    L.append(f"  - criteria NOT met under (stress/oracle, reported not vetoing): "
                             f"{r.get('not_robust_under') or 'none'}")
                if r.get("comparator"):
                    L.append(f"  - comparator `{r['comparator']['design']}` feasible in final evaluation: "
                             f"{r['comparator']['feasible_final']}; resource claim allowed: {r['resource_claim_allowed']}"
                             + (f"; difference {r.get('resource_difference_vs_comparator')}" if r.get("resource_claim_allowed") else ""))
            L.append("")
    L += ["## Statistical assumptions not yet validated", "",
          "- Pooled-sex analysis assumes a common treatment effect across sexes.",
          "- The production default final analysis is endpoint ANCOVA. GLS is requestable "
          "and is not a default after locked synthetic diagnostics; it is not design qualification.",
          "- GLS benefit from intermediate visits relies on the fixed quadratic group-difference basis; bias is "
          "only checked under the simulated mechanisms.",
          "- The working covariance is a historical plug-in from training studies; model-based SEs with "
          "df = N_animals - (3 + G k) are an approximation (no Kenward-Roger).",
          "- Missingness is simulated as MAR missed visits and non-informative dropout; informative "
          "termination is not addressed and no power claims are made for it.",
          "- Spline/transformer parameter uncertainty comes from study bootstrap of the mean only; covariance "
          "parameter uncertainty is not propagated for those models.",
          "- Synthetic generators are illustrative and not biologically validated.", ""]
    md = "\n".join(L) + "\n"
    (rep / "report.md").write_text(md, encoding="utf-8")
    (rep / "report.html").write_text(markdown_to_html(md, title), encoding="utf-8")
    return rep / "report.md"


def _plot_curves(h: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    for m in h["members"]:
        c = pd.DataFrame(m["curve"])
        ax.plot(c["epoch"], c["val_rmse"], label=f"member {m['member']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation RMSE (log)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_eval(run, plots: Path) -> None:
    try:
        prep = run.load_prepared()
    except Exception:  # noqa: BLE001
        return
    d = prep["data"].get("test")
    if d is None or not len(d.obs):
        return
    sid = d.studies[0]
    a = d.animals[d.animals["STUDYID"] == sid].reset_index(drop=True)
    o = d.obs[d.obs["STUDYID"] == sid]
    times = np.array(sorted(o["DAY"].unique()), float)
    models = run.available_models()
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], models, strict=True):
        z = run.load_model(name).sample_study(a, None, times, 200, 1)["z"]
        for arm, g in a.groupby("ARM_KEY"):
            idx = g.index.to_numpy()
            obs_m = o[o["USUBJID"].isin(g["USUBJID"])].groupby("DAY")["Y"].mean()
            pm = (z[:, idx] - a.loc[idx, "BASELINE_LOGW"].to_numpy()[None, :, None]).mean(1)
            lo, mid, hi = np.quantile(pm, [0.05, 0.5, 0.95], axis=0)
            line = ax.plot(times, mid, label=arm.split("::")[-1])[0]
            ax.fill_between(times, lo, hi, alpha=0.15, color=line.get_color())
            ax.scatter(obs_m.index, obs_m.values, color=line.get_color(), s=12)
        ax.set_title(f"{name}: test study {sid}", fontsize=9)
        ax.set_xlabel("elapsed day")
    axes[0][0].set_ylabel("mean log change from baseline")
    axes[0][-1].legend(fontsize=6, title="arm", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(plots / "trajectories.png", dpi=110)
    plt.close(fig)


def _plot_power(run, plots: Path) -> None:
    files = sorted((run.dir / "design").glob("proposal_*_feasibility.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for f in files:
        name = f.stem.replace("proposal_", "").replace("_feasibility", "")
        df = pd.read_csv(f)
        for (an, sch), g in df.groupby(["analysis", "schedule"]):
            g = g.sort_values("n_animals")
            ax.plot(g["n_animals"], g["power"], marker="o", ms=3, lw=1, label=f"{name}:{an}:{sch}")
    req_t = run.cfg.get("design", {}).get("power_target_line", 0.8)
    ax.axhline(req_t, ls="--", color="k", lw=0.8)
    ax.set_xlabel("total animals")
    ax.set_ylabel("power (all required, Holm)")
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "power_vs_animals.png", dpi=110)
    plt.close(fig)


def markdown_to_html(md: str, title: str) -> str:
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
           "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;font-size:13px}"
           "table{border-collapse:collapse;font-size:11px}td,th{border:1px solid #ccc;padding:2px 5px}"
           "blockquote{background:#fff3cd;padding:6px;border-left:4px solid #e0a800}</style></head><body>"]
    lines = md.split("\n")
    i = 0
    in_list = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            hdr = [c.strip() for c in ln.strip("|").split("|")]
            out.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in hdr) + "</tr>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</table>")
            continue
        m = re.match(r"^(#+) (.*)", ln)
        if ln.startswith("- ") or ln.startswith("  - "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(ln.strip()[2:])}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if m:
                n = len(m.group(1))
                out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
            elif ln.startswith("> "):
                out.append(f"<blockquote>{_inline(ln[2:])}</blockquote>")
            elif ln.startswith("!["):
                mm = re.match(r"!\[(.*)\]\((.*)\)", ln)
                out.append(f"<img src='{html.escape(mm.group(2))}' alt='{html.escape(mm.group(1))}' style='max-width:100%'>")
            elif ln.strip():
                out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    out.append("</body></html>")
    return "\n".join(out)


def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


__all__ = ["build_report", "markdown_to_html", "json"]
