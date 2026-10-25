"""Controlled synthetic TFT early-history benchmark.

Windows come from observed DAY values. Locked runs/v3_* directories are not written.
Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from send_bw.evaluation.quantile_forecast import QUANTILES, last_value_predictions
from send_bw.sharing import LABELS
from send_bw.tft_examples import (
    assert_windows_match_obs,
    construct_forecast_examples,
    examples_to_tables,
    inventory_model_data,
    quantile_table_from_model,
    score_aligned_quantiles,
    summarize_observation_schedules,
)
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import collect_versions

PRESERVED = (
    "runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research",
    "runs/v3_closeout", "runs/v3_audit", "runs/v3_increment", "runs/v3_phaseb",
    "runs/v3_limitations", "runs/v3_objective", "runs/v3_official_seq",
    "runs/v3_seq_diag", "runs/v3_eval_correct", "runs/v3_cov_chronos",
    "runs/v3_share_repl", "runs/v3_share_abstain", "runs/v3_share_abstain_run",
    "runs/v3_donor_rank",
)
OUTPUT_DEFAULT = "runs/tft_early_hist_bench"


class BenchError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _refuse_preserved(parent: Path) -> None:
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise BenchError("refusing to write into a preserved runs/v3_* directory")
    if parent.resolve().name.startswith("v3_") and "runs" in parent.resolve().parts:
        raise BenchError("refusing to write a new locked-style runs/v3_* directory")


def _compact_score(block: dict[str, Any]) -> dict[str, Any]:
    out = {"status": block.get("status"), "representation": block.get("representation"),
           "n_scored": block.get("n_scored"), "n_failures": block.get("n_failures")}
    for key in ("mae_log", "rmse_log", "mean_quantile_loss", "coverage80", "interval_width80",
                "relative_abs_log", "relative_abs_gram"):
        v = block.get(key)
        if isinstance(v, dict):
            out[key] = {k: v.get(k) for k in ("study_balanced", "pooled", "n_obs", "n_studies") if k in v}
        else:
            out[key] = v
    cr = block.get("crossing_rate")
    out["crossing_rate"] = (cr or {}).get("pooled") if isinstance(cr, dict) else cr
    out["seconds"] = block.get("seconds")
    out["label"] = block.get("label")
    out["uses_early_history"] = block.get("uses_early_history")
    return out


def _select_representatives(pack: dict[str, Any], n: int = 6) -> list[str]:
    ex = pack["examples"]
    if ex.empty:
        return []
    picks: list[str] = []
    for n_fut in sorted(ex["n_future_used"].unique()):
        g = ex.loc[ex["n_future_used"] == n_fut]
        for _sid, sg in g.groupby("STUDYID"):
            picks.append(str(sg.iloc[0]["USUBJID"]))
            if len(picks) >= n:
                return picks
    for u in ex["USUBJID"].astype(str):
        if u not in picks:
            picks.append(u)
        if len(picks) >= n:
            break
    return picks


def _verify_adapter(model, animals: pd.DataFrame, obs: pd.DataFrame, pack: dict[str, Any],
                    hist: pd.DataFrame, fut: pd.DataFrame) -> dict[str, Any]:
    """Independent DAY derivation plus adapter output checks. Does not score skill."""
    checks: list[dict[str, Any]] = []
    reps = _select_representatives(pack, n=min(4, int(pack["n_eligible"])))
    ctx = animals.drop_duplicates("USUBJID")
    ctx = ctx.set_index(ctx["USUBJID"].astype(str))
    for u in reps:
        derived = assert_windows_match_obs(obs, pack, u)
        rec = pack["examples"].loc[pack["examples"]["USUBJID"].astype(str) == u].iloc[0]
        a = ctx.loc[[u]].reset_index(drop=True)
        hh = hist.loc[hist["USUBJID"].astype(str) == u]
        times = np.asarray(rec["target_days"], float)
        pred = model.predict_quantiles(a, times, history=hh)
        got_days = [float(x) for x in pred.sort_values("DAY")["DAY"].to_numpy(float)]
        day_ok = got_days == [float(x) for x in rec["target_days"]]
        uid_ok = set(pred["USUBJID"].astype(str)) == {u}
        qcols = [c for c in pred.columns if str(c).startswith("q")]
        arr = pred.sort_values("DAY")[qcols].to_numpy(float) if qcols else np.zeros((0, 0))
        mono = bool(arr.size == 0 or np.all(np.diff(arr, axis=1) >= -1e-8))
        single = model.predict_quantiles(a, times[:1], history=hh) if len(times) else pred
        multi_same_hist = True
        if len(times) > 1 and len(pred) >= 1 and len(single) >= 1:
            multi_same_hist = abs(float(pred.sort_values("DAY")["q0.5"].iloc[0])
                                  - float(single.sort_values("DAY")["q0.5"].iloc[0])) < 1e-5
        fill_ok = None
        if len(times):
            alt = model._predict_chunk(a, times, hh, as_marginals=False, decoder_logw=0.0)
            ref = model._predict_chunk(a, times, hh, as_marginals=False, decoder_logw=None)
            if len(alt) and len(ref) and "q0.5" in alt.columns:
                fill_ok = bool(np.allclose(
                    alt.sort_values("DAY")["q0.5"].to_numpy(float),
                    ref.sort_values("DAY")["q0.5"].to_numpy(float), atol=1e-5, rtol=1e-5))
        gram_ok = True
        g = obs.loc[obs["USUBJID"].astype(str) == u]
        if "WEIGHT_G" in g.columns:
            for d, z in zip(rec["target_days"], rec["target_logw"], strict=True):
                w = g.loc[np.isclose(g["DAY"].to_numpy(float), float(d)), "WEIGHT_G"]
                if len(w) and abs(float(np.log(w.iloc[0])) - float(z)) > 1e-5:
                    gram_ok = False
        checks.append({
            "USUBJID": u, "derived_encoder_days": derived["encoder_days"],
            "derived_target_days": derived["target_days"],
            "pred_days": got_days, "days_match": day_ok, "usubjid_match": uid_ok,
            "quantiles_ordered": mono, "single_horizon_matches_first_multi": multi_same_hist,
            "dummy_decoder_fill_invariant": fill_ok, "logw_is_ln_weight_g": gram_ok,
            "n_targets": int(len(times)),
        })
    ok = all(c["days_match"] and c["usubjid_match"] and c["quantiles_ordered"]
             and c["single_horizon_matches_first_multi"] and c["logw_is_ln_weight_g"]
             and c["dummy_decoder_fill_invariant"] is not False for c in checks)
    return {
        "ok": ok, "n_checked": len(checks), "rows": checks,
        "scored_pairs_are_example_targets": True,
        "history_not_updated_with_predictions": True,
        "units": "predictions are LOGW = ln(WEIGHT_G in grams); grams = exp(q0.5)",
    }


def _window_contributions(model, data) -> dict[str, Any]:
    """Count PF training windows per study after sliding-window construction."""
    from send_bw.models.tft import _enough_visits, _pf, build_observed_panel

    TimeSeriesDataSet, _, _ = _pf()
    panel = build_observed_panel(data)
    panel = _enough_visits(panel, model.min_encoder_length + 1)
    if panel.empty:
        return {"n_windows": 0, "note": "empty panel"}
    ds = TimeSeriesDataSet.from_parameters(model.dataset_parameters, panel)
    decoded = ds.decoded_index
    decoded["USUBJID"] = decoded["USUBJID"].astype(str)
    key = panel[["USUBJID", "STUDYID"]].drop_duplicates()
    key["USUBJID"] = key["USUBJID"].astype(str)
    merged = decoded.merge(key, on="USUBJID", how="left")
    counts = merged.groupby(merged["STUDYID"].astype(str)).size()
    n = panel.groupby(panel["STUDYID"].astype(str)).size()
    win_share = counts / counts.sum() if counts.sum() else counts
    row_share = n / n.sum() if n.sum() else n
    return {
        "n_windows": int(len(decoded)),
        "n_studies": int(counts.shape[0]),
        "windows_per_study": {str(k): int(v) for k, v in counts.items()},
        "obs_rows_per_study": {str(k): int(v) for k, v in n.items()},
        "intended_row_weights": study_balanced_from_counts(n.to_dict()),
        "window_share": {str(k): float(v) for k, v in win_share.items()},
        "row_share": {str(k): float(v) for k, v in row_share.items()},
        "max_minus_min_window_share": float(win_share.max() - win_share.min()) if len(win_share) else None,
        "note": ("Row-level study-balanced WEIGHT does not equalize window counts. "
                 "Studies with more visits contribute more PF sliding windows."),
    }


def study_balanced_from_counts(counts: dict[str, int]) -> dict[str, float]:
    n = pd.Series(counts, dtype=float)
    w = 1.0 / n
    w = w * len(w) / w.sum()
    return {str(k): float(v) for k, v in w.items()}


def _plot_trajectories(parent: Path, pack: dict[str, Any], hist: pd.DataFrame, fut: pd.DataFrame,
                       preds: dict[str, pd.DataFrame], animals: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    dest = parent / "plots"
    dest.mkdir(parents=True, exist_ok=True)
    reps = _select_representatives(pack, n=6)
    ctx = animals.drop_duplicates("USUBJID").set_index(
        animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    paths = []
    for u in reps:
        rec = pack["examples"].loc[pack["examples"]["USUBJID"].astype(str) == u].iloc[0]
        hh = hist.loc[hist["USUBJID"].astype(str) == u].sort_values("DAY")
        ff = fut.loc[fut["USUBJID"].astype(str) == u].sort_values("DAY")
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.axvline(float(pack["cutoff"]), color="0.6", ls="--", lw=1, label=f"cutoff DAY={pack['cutoff']:g}")
        ax.plot(hh["DAY"], hh["LOGW"], "o-", color="0.15", label="permitted history")
        ax.plot(ff["DAY"], ff["LOGW"], "s", color="0.15", label="observed future")
        colors = {"tft_target_only": "#1f77b4", "tft_pooled": "#d62728",
                  "last_value": "#7f7f7f", "transformer_target_only": "#2ca02c"}
        for name, pred in preds.items():
            pg = pred.loc[pred["USUBJID"].astype(str) == u].sort_values("DAY")
            if pg.empty or "q0.5" not in pg.columns:
                continue
            ax.plot(pg["DAY"], pg["q0.5"], "^-", color=colors.get(name, "C0"), label=f"{name} q0.5", ms=5)
            if "q0.1" in pg.columns and "q0.9" in pg.columns:
                ax.fill_between(pg["DAY"], pg["q0.1"], pg["q0.9"], color=colors.get(name, "C0"),
                                alpha=0.12, linewidth=0)
        sex = str(ctx.loc[u, "SEX"]) if u in ctx.index and "SEX" in ctx.columns else ""
        sid = rec["STUDYID"]
        ax.set_xlabel("DAY (elapsed from study reference)")
        ax.set_ylabel("LOGW = ln(weight grams)")
        ax.set_title(f"{u}  {sid}  {sex}")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = dest / f"{u.replace('/', '_')}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _fit_tft(prep: dict[str, Any], cfg: dict[str, Any], dest: Path, scope: str) -> tuple[Any, dict[str, Any]]:
    from send_bw.models.tft import TFTEarlyHistoryModel

    dest.mkdir(parents=True, exist_ok=True)
    model = TFTEarlyHistoryModel()
    hist = model.fit(prep["data"]["train"], prep["data"]["validation"], {**cfg, "training_scope": scope})
    model.save(dest)
    write_json(dest / "training_history.json", hist)
    return model, hist


def execute_tft_early_hist_bench(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    _refuse_preserved(parent)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise BenchError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise BenchError("design optimization stays off")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise BenchError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    parent.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("benchmark") or {}
    cutoff = float(ev.get("encoder_window_day") or cfg.get("evaluation", {}).get("early_history_window_day") or 14)
    max_enc = int((cfg.get("models") or {}).get("optional", {}).get("tft", {}).get("max_encoder_length", 8))
    max_pred = int((cfg.get("models") or {}).get("optional", {}).get("tft", {}).get("max_prediction_length", 6))
    train_seed = int(cfg.get("train_seed") or ev.get("train_seed") or 82001)
    cells = ev.get("source_cells") or {}
    target_dir = Path(cells.get("target_only") or "")
    pooled_dir = Path(cells.get("pooled") or "")
    if not target_dir.is_dir() or not pooled_dir.is_dir():
        raise BenchError("source_cells.target_only and source_cells.pooled must exist (read-only official_seq cells)")
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": cfg.get("generate_seed"),
        "train_seed": train_seed, "data_source": "existing_official_seq_cells_read_only",
        "target_only_cell": str(target_dir), "pooled_cell": str(pooled_dir),
        "encoder_window_day": cutoff, "max_encoder_length": max_enc,
        "max_prediction_length": max_pred, "chunk_long_horizons": False,
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(PRESERVED), "writes_v3_locks": False,
    }
    write_json(parent / "freeze_lock.json", lock)

    target = load_cell_prepared(target_dir)
    pooled = load_cell_prepared(pooled_dir)
    test = target["data"]["test"]
    tgt_test_ids = set(target["splits"].loc[target["splits"]["SPLIT"] == "test", "STUDYID"].astype(str))
    poo_test_ids = set(pooled["splits"].loc[pooled["splits"]["SPLIT"] == "test", "STUDYID"].astype(str))
    if tgt_test_ids != poo_test_ids:
        raise BenchError(f"test STUDYIDs differ: {sorted(tgt_test_ids)} vs {sorted(poo_test_ids)}")

    inventory = {
        "target_only": {k: inventory_model_data(v) for k, v in target["data"].items()},
        "pooled": {k: inventory_model_data(v) for k, v in pooled["data"].items()},
        "schedules_test": summarize_observation_schedules(test.obs, test.animals),
        "schedules_target_train": summarize_observation_schedules(
            target["data"]["train"].obs, target["data"]["train"].animals),
        "schedules_pooled_train": summarize_observation_schedules(
            pooled["data"]["train"].obs, pooled["data"]["train"].animals),
    }
    write_json(parent / "data_inventory.json", inventory)
    target["splits"].to_csv(parent / "splits_target.csv", index=False)
    pooled["splits"].to_csv(parent / "splits_pooled.csv", index=False)

    pack = construct_forecast_examples(
        test.obs, test.animals, cutoff=cutoff,
        max_encoder_length=max_enc, max_prediction_length=max_pred)
    examples = pack["examples"]
    hist, fut = examples_to_tables(pack)
    examples.to_csv(parent / "examples.csv", index=False)
    pack["exclusions"].to_csv(parent / "exclusions.csv", index=False)
    hist.to_csv(parent / "encoder_obs.csv", index=False)
    fut.to_csv(parent / "target_obs.csv", index=False)
    write_json(parent / "example_summary.json", {
        k: pack[k] for k in pack if k not in {"examples", "exclusions"}
    })

    tft_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("tft") or {})
    tft_cfg.setdefault("seed", train_seed)
    tft_cfg.setdefault("encoder_window_day", cutoff)
    tft_cfg.setdefault("max_encoder_length", max_enc)
    tft_cfg.setdefault("max_prediction_length", max_pred)

    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    train_info: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    t0 = time.time()

    # last-value baseline (same encoder/targets)
    t1 = time.time()
    lv = last_value_predictions(hist, fut, QUANTILES)
    lv.to_csv(parent / "predictions" / "last_value.csv", index=False)
    results["last_value"] = score_aligned_quantiles(fut, lv, representation="last_value")
    results["last_value"].update({
        "seconds": time.time() - t1, "uses_early_history": True,
        "label": "last observed encoder LOGW; degenerate quantiles; no covariates",
        "n_failures": 0,
    })
    preds["last_value"] = lv

    # nonlinear early-history baseline: custom transformer (Gaussian residual conditioning)
    from send_bw.pipeline import model_class

    t1 = time.time()
    try:
        tr = model_class("transformer").load(target_dir / "models" / "transformer")
        pred_tr, fail_tr = quantile_table_from_model(tr, test.animals, pack, hist)
        pred_tr.to_csv(parent / "predictions" / "transformer_target_only.csv", index=False)
        results["transformer_target_only"] = score_aligned_quantiles(
            fut, pred_tr, representation="gaussian")
        results["transformer_target_only"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": ("custom transformer + Gaussian residual conditioning on the same "
                      "observed encoder; not a quantile head; not TFT"),
            "n_failures": len(fail_tr), "failures": fail_tr,
            "checkpoint": str(target_dir / "models" / "transformer"),
        })
        preds["transformer_target_only"] = pred_tr
        failures.extend({"model": "transformer_target_only", **f} for f in fail_tr)
    except Exception as e:  # noqa: BLE001
        results["transformer_target_only"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "seconds": time.time() - t1,
            "label": "custom transformer early-history (load/score failed)",
        }
        failures.append({"model": "transformer_target_only", "reason": str(e)})

    # species-specific TFT
    t1 = time.time()
    try:
        m_to, h_to = _fit_tft(target, tft_cfg, parent / "models" / "tft_target_only", "target_only")
        train_info["tft_target_only"] = h_to
        train_info["tft_target_only_windows"] = _window_contributions(m_to, target["data"]["train"])
        pred_to, fail_to = quantile_table_from_model(m_to, test.animals, pack, hist)
        pred_to.to_csv(parent / "predictions" / "tft_target_only.csv", index=False)
        results["tft_target_only"] = score_aligned_quantiles(fut, pred_to, representation="tft_quantiles")
        results["tft_target_only"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": "TFT fit on target-species train studies only; SPECIES constant",
            "n_failures": len(fail_to), "failures": fail_to,
            "checkpoint": str(parent / "models" / "tft_target_only" / "tft.ckpt"),
            "training_history": {k: h_to.get(k) for k in (
                "n_train_series", "n_val_series", "n_train_obs", "max_epochs",
                "encoder_window_day", "validation_windows", "training_windows")},
        })
        preds["tft_target_only"] = pred_to
        failures.extend({"model": "tft_target_only", **f} for f in fail_to)
        verify = _verify_adapter(m_to, test.animals, test.obs, pack, hist, fut)
        write_json(parent / "correctness.json", verify)
    except Exception as e:  # noqa: BLE001
        verify = {"ok": False, "reason": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2000:]}
        write_json(parent / "correctness.json", verify)
        results["tft_target_only"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_target_only", "reason": str(e)})
        m_to = None

    # pooled TFT with species
    t1 = time.time()
    try:
        m_po, h_po = _fit_tft(pooled, tft_cfg, parent / "models" / "tft_pooled", "pooled")
        train_info["tft_pooled"] = h_po
        train_info["tft_pooled_windows"] = _window_contributions(m_po, pooled["data"]["train"])
        pred_po, fail_po = quantile_table_from_model(m_po, test.animals, pack, hist)
        pred_po.to_csv(parent / "predictions" / "tft_pooled.csv", index=False)
        results["tft_pooled"] = score_aligned_quantiles(fut, pred_po, representation="tft_quantiles")
        results["tft_pooled"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": "TFT fit on target + donor train studies; SPECIES is a static categorical",
            "n_failures": len(fail_po), "failures": fail_po,
            "checkpoint": str(parent / "models" / "tft_pooled" / "tft.ckpt"),
            "training_history": {k: h_po.get(k) for k in (
                "n_train_series", "n_val_series", "n_train_obs", "max_epochs",
                "encoder_window_day", "validation_windows", "training_windows")},
        })
        preds["tft_pooled"] = pred_po
        failures.extend({"model": "tft_pooled", **f} for f in fail_po)
    except Exception as e:  # noqa: BLE001
        results["tft_pooled"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_pooled", "reason": str(e)})

    plots = []
    if preds:
        try:
            plots = _plot_trajectories(parent, pack, hist, fut, preds, test.animals)
        except Exception as e:  # noqa: BLE001
            plots = []
            failures.append({"model": "plots", "reason": str(e)})

    comparison = []
    for name, block in results.items():
        row = {"model": name, **_compact_score(block)}
        comparison.append(row)
    cmp_df = pd.DataFrame(comparison)
    cmp_df.to_csv(parent / "model_comparison.csv", index=False)
    per_study_rows = []
    for name, block in results.items():
        for rec in block.get("per_study") or []:
            per_study_rows.append({"model": name, **rec})
    if per_study_rows:
        pd.DataFrame(per_study_rows).to_csv(parent / "per_study_metrics.csv", index=False)

    out = {
        "labels": list(LABELS) + ["Optional TFT early-history benchmark.", "Synthetic generator only."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "example_summary": {k: pack[k] for k in pack if k not in {"examples", "exclusions"}},
        "inventory": {
            "test_studies": inventory["target_only"]["test"]["studyids"],
            "target_train_studies": inventory["target_only"]["train"]["studyids"],
            "pooled_train_studies": inventory["pooled"]["train"]["studyids"],
            "test_unique_days": inventory["schedules_test"]["unique_days_all"],
        },
        "correctness": read_json(parent / "correctness.json") if (parent / "correctness.json").exists() else {},
        "results": {k: _compact_score(v) if isinstance(v, dict) else v for k, v in results.items()},
        "training": train_info,
        "failures": failures,
        "plots": plots,
        "real_data_accessed": False,
        "design_run": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out, results)
    return out


def _fmt(block: dict[str, Any] | None, key: str) -> str:
    if not block or block.get("status") not in {"ok", None}:
        if block:
            return str(block.get("status") or "n/a")
        return "n/a"
    v = block.get(key)
    if isinstance(v, dict):
        x = v.get("study_balanced")
        return "n/a" if x is None else f"{float(x):.4f}"
    return "n/a" if v is None else f"{float(v):.4f}"


def _write_report(parent: Path, out: dict[str, Any], results: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    lines = [
        "# TFT early-history synthetic benchmark",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "Forecast windows were taken from observed `DAY` values. Unmeasured days were "
        "not created. Long-horizon chunking was not used.",
        "",
        "## What ran",
        "",
        f"- output: `{out.get('output_dir')}`",
        f"- data: read-only official_seq cells (generate_seed={lock.get('generate_seed')})",
        f"- TFT train_seed: {lock.get('train_seed')}",
        f"- cutoff: DAY ≤ {lock.get('encoder_window_day')}",
        f"- encoder visit cap: {lock.get('max_encoder_length')}; "
        f"prediction-length cap: {lock.get('max_prediction_length')} (no chunking)",
        f"- eligible test animals: {(out.get('example_summary') or {}).get('n_eligible')}",
        f"- excluded: {(out.get('example_summary') or {}).get('n_excluded')} "
        f"{(out.get('example_summary') or {}).get('exclude_counts')}",
        f"- correctness: {(out.get('correctness') or {}).get('ok')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Comparison (study-balanced, identical animal/day pairs)",
        "",
        "| model | MAE log | RMSE log | mean pinball | cov80 | width80 | rel gram | crossing | history |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, block in results.items():
        hist = "yes" if block.get("uses_early_history") else "no"
        lines.append(
            f"| {name} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mean_quantile_loss')} | {_fmt(block, 'coverage80')} | "
            f"{_fmt(block, 'interval_width80')} | {_fmt(block, 'relative_abs_gram')} | "
            f"{(block.get('crossing_rate') or {}).get('pooled') if isinstance(block.get('crossing_rate'), dict) else block.get('crossing_rate')} | {hist} |"
        )
    lines += [
        "",
        "## Implementation vs generator",
        "",
        "Successful training and unit tests do not establish forecasting effectiveness. "
        "The table above is a synthetic-generator diagnostic on one holdout. It does not "
        "validate historical experimental studies.",
        "",
        "## Untested on real SEND studies",
        "",
        "No historical experimental body-weight studies were opened. Species transfer, "
        "assay error, and protocol heterogeneity remain untested.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
