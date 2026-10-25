"""Extension of the synthetic TFT early-history pilot.

Adds spline/GAMM and quantile CatBoost, audits study weights, inspects TFT
crossings, and runs a predeclared stability set. Does not overwrite the pilot
or any locked runs/v3_* directory.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec
from send_bw.data.splits import compose_train_with_selected_donors, make_study_splits
from send_bw.evaluation.quantile_forecast import QUANTILES, last_value_predictions
from send_bw.models.base import ModelData
from send_bw.sharing import LABELS
from send_bw.tft_baselines import QuantileCatBoostModel, catboost_status, spline_information_set
from send_bw.tft_crossings import crossing_audit, curve_still_improving, rearrange_quantiles
from send_bw.tft_early_hist_bench import (
    PRESERVED,
    _compact_score,
    _fmt,
    _plot_trajectories,
    _refuse_preserved,
    _verify_adapter,
    load_yaml,
)
from send_bw.tft_examples import (
    construct_forecast_examples,
    examples_to_tables,
    history_lead_report,
    mark_persistence_intervals_unavailable,
    paired_study_differences,
    quantile_table_from_model,
    score_aligned_quantiles,
    summarize_observation_schedules,
)
from send_bw.tft_weighting import audit_training_weights
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/tft_early_hist_extend"
PILOT_DEFAULT = "runs/tft_early_hist_bench"
COMMON_Q = [0.1, 0.5, 0.9]


class ExtendError(ValueError):
    pass


def _refuse_pilot(parent: Path, pilot: Path) -> None:
    _refuse_preserved(parent)
    if parent.resolve() == pilot.resolve():
        raise ExtendError("refusing to overwrite the completed pilot directory")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise ExtendError("refusing to write into a preserved runs/v3_* directory")


def _score(fut: pd.DataFrame, pred: pd.DataFrame, *, representation: str,
           quantiles: list[float] | None = None) -> dict[str, Any]:
    return score_aligned_quantiles(
        fut, pred, representation=representation, quantiles=quantiles or QUANTILES)


def _save_pred(parent: Path, name: str, pred: pd.DataFrame) -> Path:
    dest = parent / "predictions" / f"{name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(dest, index=False)
    return dest


def transformer_provenance(path: Path, splits: pd.DataFrame) -> dict[str, Any]:
    """Document the loaded custom-transformer checkpoint. Not TFT."""
    hist = read_json(path / "training_history.json") if (path / "training_history.json").exists() else {}
    meta = read_json(path / "model.json") if (path / "model.json").exists() else {}
    spec = meta.get("spec") or {}
    members = hist.get("members") or []
    m0 = members[0] if members else {}
    curve = m0.get("curve") or []
    best_epoch = None
    if curve:
        best = min(curve, key=lambda r: r.get("val_rmse", float("inf")))
        best_epoch = best.get("epoch")
    train_ids = sorted(splits.loc[splits["SPLIT"] == "train", "STUDYID"].astype(str).tolist())
    return {
        "checkpoint": str(path),
        "name": meta.get("name"),
        "not_tft": True,
        "label": "pretrained official_seq reference; not refit on TFT train seeds",
        "sharing": spec.get("sharing"),
        "species": spec.get("species"),
        "feature_spec_strains": spec.get("strains"),
        "feature_spec_routes": spec.get("routes"),
        "n_members": len(members) or hist.get("n_members"),
        "member0_seed": m0.get("seed"),
        "epochs_run": m0.get("epochs_run"),
        "best_val_rmse": hist.get("best_val_rmse_member0") or m0.get("best_val_rmse"),
        "best_epoch": best_epoch,
        "checkpoint_selection": "lowest validation RMSE among logged epochs, member 0",
        "training_studies": train_ids,
        "n_training_studies": len(train_ids),
        "preprocessing": "FeatureSpec fit on official_seq training studies only (cell feature_spec.json)",
        "split": "official_seq frozen whole-study splits (target_only cell)",
    }


def load_full_cell(cell_dir: Path) -> dict[str, Any]:
    from send_bw.data.splits import load_frozen_splits

    splits = load_frozen_splits(cell_dir / "splits" / "splits")
    spec = FeatureSpec.from_dict(read_json(cell_dir / "features" / "feature_spec.json"))
    cref = pd.read_parquet(cell_dir / "canonical" / "compound_reference.parquet")
    full = ModelData(
        pd.read_parquet(cell_dir / "cohort" / "animals.parquet"),
        pd.read_parquet(cell_dir / "cohort" / "obs.parquet"),
        cref, spec,
    )
    return {"splits": splits, "spec": spec, "full": full, "compound_ref": cref}


def extra_study_splits(target_full: ModelData, pooled_full: ModelData, *,
                       seed: int, split_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-split the same generated RAT bank. Donors stay train-only for pooled."""
    animals = target_full.animals
    base = animals.drop_duplicates("STUDYID")
    ctx = pd.DataFrame({
        "STUDYID": base["STUDYID"].to_numpy(),
        "STUDY_START": base["REF_DATE"].to_numpy() if "REF_DATE" in base.columns else "2000-01-01",
    })
    cfg = {"method": "random", "validation": 0.2, "test": 0.2, **(split_cfg or {})}
    target_splits = make_study_splits(ctx, cfg, seed)
    donor_ids = []
    if "SPECIES" in pooled_full.animals.columns:
        donor = pooled_full.animals.loc[
            pooled_full.animals["SPECIES"].astype(str).str.upper() != str(target_full.spec.species).upper(),
            "STUDYID"]
        donor_ids = sorted({str(s) for s in donor})
    pooled_splits = compose_train_with_selected_donors(target_splits, donor_ids)
    return {
        "seed": int(seed),
        "target_splits": target_splits,
        "pooled_splits": pooled_splits,
        "donor_ids": donor_ids,
        "split_cfg": cfg,
        "note": ("Whole-study reassignment of the same Generator A bank. "
                 "Official SYNA-010/011 holdout is not reused. Donors are train-only."),
    }


def prepare_from_splits(full: ModelData, splits: pd.DataFrame, *,
                        species: str, sharing: str) -> dict[str, Any]:
    train_ids = splits.loc[splits["SPLIT"] == "train", "STUDYID"]
    train = full.subset(train_ids)
    spec = FeatureSpec.fit(train.animals, train.obs, full.compound_ref, species, sharing=sharing)
    data = {}
    for name in ("train", "validation", "test"):
        ids = splits.loc[splits["SPLIT"] == name, "STUDYID"]
        if not len(ids):
            continue
        sub = full.subset(ids)
        data[name] = ModelData(sub.animals, sub.obs, full.compound_ref, spec)
    return {"spec": spec, "data": data, "splits": splits, "compound_ref": full.compound_ref}


def _fit_tft(prep: dict[str, Any], cfg: dict[str, Any], dest: Path, scope: str) -> tuple[Any, dict[str, Any]]:
    from send_bw.models.tft import TFTEarlyHistoryModel

    dest.mkdir(parents=True, exist_ok=True)
    model = TFTEarlyHistoryModel()
    hist = model.fit(prep["data"]["train"], prep["data"]["validation"], {**cfg, "training_scope": scope})
    model.save(dest)
    write_json(dest / "training_history.json", hist)
    return model, hist


def _information_sets() -> list[dict[str, Any]]:
    return [
        {"model": "last_value",
         "train_targets": "none",
         "test_history": "last encoder LOGW",
         "covariates": "none",
         "intervals": "unavailable (degenerate quantiles)"},
        {"model": "spline_gamm",
         "train_targets": "all observed train days",
         "test_history": "encoder LOGW via Gaussian residual conditioning",
         "covariates": "biology, planned regimen, B-spline time/exposure",
         "intervals": "Gaussian hierarchical residual layer"},
        {"model": "transformer_target_only",
         "train_targets": "all observed train days",
         "test_history": "encoder LOGW via the same Gaussian residual layer",
         "covariates": "baseline-conditioned tokens; not TFT",
         "intervals": "Gaussian hierarchical residual layer"},
        {"model": "catboost_quantile",
         "train_targets": "one row per train animal × cutoff origin × observed future DAY",
         "test_history": "encoder summaries only (no future weights)",
         "covariates": "biology, planned regimen, history summaries, target DAY, elapsed times",
         "intervals": "quantile heads at q0.1/0.5/0.9"},
        {"model": "tft_*",
         "train_targets": "PF sliding observed windows (not only the cutoff origin)",
         "test_history": "observed encoder visits with DAY<=cutoff, last <=max_encoder_length",
         "covariates": "static biology/planned + known-future planned-dose features",
         "intervals": "QuantileLoss; not a Gaussian covariance"},
    ]


def execute_tft_early_hist_extend(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    pilot = Path(cfg.get("pilot_dir") or PILOT_DEFAULT)
    _refuse_pilot(parent, pilot)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ExtendError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise ExtendError("design optimization stays off")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise ExtendError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    parent.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("benchmark") or {}
    cutoff = float(ev.get("encoder_window_day") or 14)
    tft_src = (cfg.get("models") or {}).get("optional", {}).get("tft") or {}
    max_enc = int(tft_src.get("max_encoder_length", 8))
    max_pred = int(tft_src.get("max_prediction_length", 6))
    train_seed = int(cfg.get("train_seed") or 82001)
    extra_train_seeds = [int(s) for s in (ev.get("extra_train_seeds") or [82011])]
    extra_split_seed = int(ev.get("extra_split_seed") or 33001)
    extra_split_tft_seed = int(ev.get("extra_split_tft_train_seed") or train_seed)
    extra_split_mean_seed = int(ev.get("extra_split_mean_seed") or 81001)
    cells = ev.get("source_cells") or {}
    target_dir = Path(cells.get("target_only") or "")
    pooled_dir = Path(cells.get("pooled") or "")
    if not target_dir.is_dir() or not pooled_dir.is_dir():
        raise ExtendError("source_cells.target_only and source_cells.pooled must exist")
    if not pilot.is_dir():
        raise ExtendError(f"completed pilot is required at {pilot}")

    versions = collect_versions()
    versions["packages"]["catboost"] = catboost_status()
    write_json(parent / "versions.json", versions)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)
    (parent / "banks").mkdir(parents=True, exist_ok=True)

    predeclared = {
        "official_holdout": {
            "generate_seed": cfg.get("generate_seed"),
            "test_studies_reference": ["SYNA-010", "SYNA-011"],
            "tft_train_seed": train_seed,
            "role": "reference; reported errors are not used to tune later models",
        },
        "extra_train_seeds": extra_train_seeds,
        "extra_split_seed": extra_split_seed,
        "extra_split_tft_train_seed": extra_split_tft_seed,
        "extra_split_mean_seed": extra_split_mean_seed,
        "declared_before_examining_new_scores": True,
        "tuning_on_official_test_errors": False,
    }
    write_json(parent / "predeclared.json", predeclared)

    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": cfg.get("generate_seed"),
        "train_seed": train_seed, "pilot_dir": str(pilot),
        "target_only_cell": str(target_dir), "pooled_cell": str(pooled_dir),
        "encoder_window_day": cutoff, "max_encoder_length": max_enc,
        "max_prediction_length": max_pred, "chunk_long_horizons": False,
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(PRESERVED) + [str(pilot)],
        "writes_v3_locks": False, "overwrites_pilot": False,
        "predeclared": predeclared,
    }
    write_json(parent / "freeze_lock.json", lock)

    target = load_cell_prepared(target_dir)
    pooled = load_cell_prepared(pooled_dir)
    test = target["data"]["test"]
    pack = construct_forecast_examples(
        test.obs, test.animals, cutoff=cutoff,
        max_encoder_length=max_enc, max_prediction_length=max_pred)
    hist, fut = examples_to_tables(pack)
    pack["examples"].to_csv(parent / "examples.csv", index=False)
    pack["exclusions"].to_csv(parent / "exclusions.csv", index=False)
    hist.to_csv(parent / "encoder_obs.csv", index=False)
    fut.to_csv(parent / "target_obs.csv", index=False)
    lead_rep = history_lead_report(pack)
    write_json(parent / "history_lead_report.json", lead_rep)
    write_json(parent / "example_summary.json", {k: pack[k] for k in pack if k not in {"examples", "exclusions"}})
    write_json(parent / "information_sets.json", {"rows": _information_sets()})
    write_json(parent / "schedules_test.json", summarize_observation_schedules(test.obs, test.animals))

    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    train_info: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    t0 = time.time()

    # --- reference predictions from the completed pilot (do not refit 82001) ---
    for name, representation in (
        ("last_value", "last_value"),
        ("transformer_target_only", "gaussian"),
        ("tft_target_only", "tft_quantiles"),
        ("tft_pooled", "tft_quantiles"),
    ):
        src = pilot / "predictions" / f"{name}.csv"
        if not src.exists():
            failures.append({"model": f"reference_{name}", "reason": f"missing {src}"})
            continue
        pred = pd.read_csv(src)
        _save_pred(parent, f"reference_{name}", pred)
        block = _score(fut, pred, representation=representation)
        block.update({
            "uses_early_history": True,
            "label": f"pilot reference copied from {src}",
            "source": str(src),
            "refit": False,
            "n_failures": 0,
        })
        if name == "last_value":
            mark_persistence_intervals_unavailable(block)
        if name == "transformer_target_only":
            block["provenance"] = transformer_provenance(
                target_dir / "models" / "transformer", target["splits"])
            block["label"] = ("custom transformer Gaussian EH; official_seq checkpoint "
                              "train_seed=81001; pretrained reference, not refit")
        results[f"reference_{name}"] = block
        preds[f"reference_{name}"] = pred

    # --- spline / GAMM on the official holdout ---
    t1 = time.time()
    try:
        from send_bw.pipeline import model_class

        spline = model_class("spline").load(target_dir / "models" / "spline")
        pred_sp, fail_sp = quantile_table_from_model(spline, test.animals, pack, hist)
        _save_pred(parent, "spline_gamm", pred_sp)
        results["spline_gamm"] = _score(fut, pred_sp, representation="gaussian")
        results["spline_gamm"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": ("ridge B-spline mixed-effects / GAMM-equivalent; official_seq "
                      "checkpoint; held-out effects from encoder history only"),
            "n_failures": len(fail_sp), "failures": fail_sp,
            "checkpoint": str(target_dir / "models" / "spline"),
            "information": spline_information_set(),
            "refit": False,
        })
        preds["spline_gamm"] = pred_sp
        failures.extend({"model": "spline_gamm", **f} for f in fail_sp)
    except Exception as e:  # noqa: BLE001
        results["spline_gamm"] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                  "seconds": time.time() - t1}
        failures.append({"model": "spline_gamm", "reason": str(e)})

    # --- quantile CatBoost ---
    t1 = time.time()
    cb_st = catboost_status()
    if not cb_st["ok"]:
        results["catboost_quantile"] = {
            "status": "not_run", "reason": cb_st["reason"], "blocker": cb_st.get("blocker"),
            "label": "quantile CatBoost omitted because the library is unavailable",
        }
    else:
        try:
            cb_cfg = dict((cfg.get("models") or {}).get("catboost") or {})
            cb_cfg.setdefault("seed", extra_split_mean_seed)
            cb_cfg.setdefault("encoder_window_day", cutoff)
            cb_cfg.setdefault("max_encoder_length", max_enc)
            cb_cfg.setdefault("max_prediction_length", max_pred)
            cb = QuantileCatBoostModel()
            h_cb = cb.fit(target["data"]["train"], target["data"]["validation"], cb_cfg)
            cb.save(parent / "models" / "catboost_quantile")
            train_info["catboost_quantile"] = h_cb
            pred_cb, fail_cb = quantile_table_from_model(cb, test.animals, pack, hist)
            _save_pred(parent, "catboost_quantile", pred_cb)
            results["catboost_quantile"] = _score(
                fut, pred_cb, representation="catboost_quantiles", quantiles=COMMON_Q)
            results["catboost_quantile"].update({
                "seconds": time.time() - t1, "uses_early_history": True,
                "label": "quantile CatBoost; cutoff-origin tabular rows; q0.1/0.5/0.9",
                "n_failures": len(fail_cb), "failures": fail_cb,
                "checkpoint": str(parent / "models" / "catboost_quantile"),
                "refit": True,
            })
            preds["catboost_quantile"] = pred_cb
            failures.extend({"model": "catboost_quantile", **f} for f in fail_cb)
        except Exception as e:  # noqa: BLE001
            results["catboost_quantile"] = {
                "status": "failed", "reason": f"{type(e).__name__}: {e}",
                "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:],
            }
            failures.append({"model": "catboost_quantile", "reason": str(e)})

    # --- weight audit (official target-only and pooled train) ---
    try:
        train_info["weight_audit_target_only"] = audit_training_weights(
            target["data"]["train"], max_encoder_length=max_enc,
            max_prediction_length=max_pred, min_encoder_length=1)
        train_info["weight_audit_pooled"] = audit_training_weights(
            pooled["data"]["train"], max_encoder_length=max_enc,
            max_prediction_length=max_pred, min_encoder_length=1)
        write_json(parent / "weight_audit.json", {
            "target_only": train_info["weight_audit_target_only"],
            "pooled": train_info["weight_audit_pooled"],
        })
    except Exception as e:  # noqa: BLE001
        failures.append({"model": "weight_audit", "reason": str(e),
                         "traceback": traceback.format_exc()[-2000:]})

    # --- crossing audit on the official pooled reference ---
    if "reference_tft_pooled" in preds:
        raw_cross = crossing_audit(preds["reference_tft_pooled"], fut, animals=test.animals)
        rearranged = rearrange_quantiles(preds["reference_tft_pooled"])
        _save_pred(parent, "reference_tft_pooled_rearranged", rearranged)
        corr = _score(fut, rearranged, representation="tft_quantiles_rearranged")
        corr.update({
            "label": "pilot pooled TFT after per-row quantile rearrangement (postprocessing only)",
            "training_changed": False,
            "rearrangement": "np.sort_per_row_increasing",
        })
        corr_cross = crossing_audit(rearranged, fut, animals=test.animals)
        results["reference_tft_pooled_rearranged"] = corr
        write_json(parent / "crossing_audit.json", {
            "raw": raw_cross, "rearranged": corr_cross,
            "note": "Rearrangement is postprocessing. It is not improved training.",
        })

    tft_cfg = dict(tft_src)
    tft_cfg.setdefault("seed", train_seed)
    tft_cfg.setdefault("encoder_window_day", cutoff)
    tft_cfg.setdefault("max_encoder_length", max_enc)
    tft_cfg.setdefault("max_prediction_length", max_pred)

    # --- diagnostic replay of pooled 82001 for curves (not a champion) ---
    replay_cfg = dict(tft_cfg)
    replay_cfg["window_balance"] = ""  # original row WEIGHT, same as the pilot
    replay_cfg["seed"] = train_seed
    t1 = time.time()
    try:
        m_rep, h_rep = _fit_tft(pooled, replay_cfg, parent / "models" / "tft_pooled_replay_82001", "pooled")
        train_info["tft_pooled_replay_82001"] = h_rep
        pred_rep, fail_rep = quantile_table_from_model(m_rep, test.animals, pack, hist)
        _save_pred(parent, "tft_pooled_replay_82001", pred_rep)
        results["tft_pooled_replay_82001"] = _score(fut, pred_rep, representation="tft_quantiles")
        results["tft_pooled_replay_82001"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": "diagnostic replay of pooled TFT seed 82001 with logged curves; not a champion",
            "n_failures": len(fail_rep), "champion": False,
        })
        write_json(parent / "replay_curves.json", {
            "curves": h_rep.get("curves"),
            "decision": curve_still_improving(h_rep.get("curves") or []),
        })
        duration = curve_still_improving(h_rep.get("curves") or [])
    except Exception as e:  # noqa: BLE001
        results["tft_pooled_replay_82001"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_pooled_replay_82001", "reason": str(e)})
        duration = {"available": False, "still_improving": None}

    new_tft_cfg = dict(tft_cfg)
    new_tft_cfg["window_balance"] = "study"
    if duration.get("still_improving"):
        new_tft_cfg["max_epochs"] = int(tft_src.get("extended_max_epochs", 16))
        new_tft_cfg["patience"] = int(tft_src.get("extended_patience", 6))
        duration_note = ("validation loss was still best at the last replay epoch; "
                         "new TFT fits use the predeclared extended duration")
    else:
        duration_note = "replay validation did not justify a longer budget; new fits keep 8 epochs"
    write_json(parent / "duration_decision.json", {**duration, "applied": duration_note,
                                                   "new_max_epochs": new_tft_cfg.get("max_epochs")})

    # --- balanced official-holdout TFT (study sampler) ---
    for scope, prep, key in (
        ("target_only", target, "tft_target_only_balanced"),
        ("pooled", pooled, "tft_pooled_balanced"),
    ):
        t1 = time.time()
        try:
            cfg_i = dict(new_tft_cfg)
            cfg_i["seed"] = train_seed
            m, h = _fit_tft(prep, cfg_i, parent / "models" / key, scope)
            train_info[key] = h
            pred, fail = quantile_table_from_model(m, test.animals, pack, hist)
            _save_pred(parent, key, pred)
            results[key] = _score(fut, pred, representation="tft_quantiles")
            results[key].update({
                "seconds": time.time() - t1, "uses_early_history": True,
                "label": f"TFT {scope}; study-balanced window sampler; seed {train_seed}",
                "n_failures": len(fail), "window_balance": "study",
            })
            preds[key] = pred
            if key == "tft_pooled_balanced":
                write_json(parent / "crossing_balanced_pooled.json",
                           crossing_audit(pred, fut, animals=test.animals))
                rearr = rearrange_quantiles(pred)
                _save_pred(parent, f"{key}_rearranged", rearr)
                results[f"{key}_rearranged"] = _score(fut, rearr, representation="tft_quantiles_rearranged")
                results[f"{key}_rearranged"].update({
                    "label": "balanced pooled TFT after rearrangement (postprocessing)",
                    "training_changed": False,
                })
            if key == "tft_target_only_balanced":
                write_json(parent / "correctness.json",
                           _verify_adapter(m, test.animals, test.obs, pack, hist, fut))
        except Exception as e:  # noqa: BLE001
            results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:]}
            failures.append({"model": key, "reason": str(e)})

    # --- extra TFT train seed on the official holdout ---
    for seed in extra_train_seeds:
        for scope, prep, stem in (
            ("target_only", target, "tft_target_only"),
            ("pooled", pooled, "tft_pooled"),
        ):
            key = f"{stem}_seed{seed}"
            t1 = time.time()
            try:
                cfg_i = dict(new_tft_cfg)
                cfg_i["seed"] = seed
                m, h = _fit_tft(prep, cfg_i, parent / "models" / key, scope)
                train_info[key] = h
                pred, fail = quantile_table_from_model(m, test.animals, pack, hist)
                _save_pred(parent, key, pred)
                results[key] = _score(fut, pred, representation="tft_quantiles")
                results[key].update({
                    "seconds": time.time() - t1, "uses_early_history": True,
                    "label": f"TFT {scope}; study sampler; extra train_seed {seed}",
                    "n_failures": len(fail), "train_seed": seed,
                })
                preds[key] = pred
            except Exception as e:  # noqa: BLE001
                results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:]}
                failures.append({"model": key, "reason": str(e)})

    # --- extra whole-study split of the same generated bank ---
    extra_out = _run_extra_split(
        parent, cfg, target_dir, pooled_dir, extra_split_seed, extra_split_tft_seed,
        extra_split_mean_seed, cutoff, max_enc, max_pred, new_tft_cfg)
    write_json(parent / "extra_split.json", extra_out.get("summary") or extra_out)
    if extra_out.get("results"):
        for k, v in extra_out["results"].items():
            results[k] = v
    if extra_out.get("failures"):
        failures.extend(extra_out["failures"])
    if extra_out.get("train_info"):
        train_info.update(extra_out["train_info"])
    if extra_out.get("preds"):
        preds.update(extra_out["preds"])

    plots = []
    plot_preds = {k: v for k, v in preds.items() if k in {
        "reference_last_value", "spline_gamm", "catboost_quantile",
        "reference_transformer_target_only", "reference_tft_target_only",
        "reference_tft_pooled", "tft_target_only_balanced", "tft_pooled_balanced"}}
    if plot_preds:
        try:
            plots = _plot_trajectories(parent, pack, hist, fut, plot_preds, test.animals)
        except Exception as e:  # noqa: BLE001
            failures.append({"model": "plots", "reason": str(e)})

    comparison = pd.DataFrame([{"model": n, **_compact_score(b)} for n, b in results.items()])
    comparison.to_csv(parent / "model_comparison.csv", index=False)
    per_study_rows = []
    for name, block in results.items():
        for rec in block.get("per_study") or []:
            per_study_rows.append({"model": name, **rec})
    if per_study_rows:
        pd.DataFrame(per_study_rows).to_csv(parent / "per_study_metrics.csv", index=False)

    paired = _paired_block(results)
    write_json(parent / "paired_study_diffs.json", paired)
    stability = _stability_block(results, extra_train_seeds, extra_split_seed)
    write_json(parent / "stability.json", stability)

    out = {
        "labels": list(LABELS) + ["Optional TFT early-history extension.", "Synthetic generator only."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "predeclared": predeclared,
        "versions": versions,
        "example_summary": {k: pack[k] for k in pack if k not in {"examples", "exclusions"}},
        "history_lead": lead_rep,
        "information_sets": _information_sets(),
        "correctness": read_json(parent / "correctness.json") if (parent / "correctness.json").exists() else {},
        "results": {k: _compact_score(v) if isinstance(v, dict) else v for k, v in results.items()},
        "training": {k: _train_compact(v) for k, v in train_info.items()},
        "paired": paired,
        "stability": stability,
        "duration_decision": duration_note,
        "failures": failures,
        "plots": plots,
        "real_data_accessed": False,
        "design_run": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out, results)
    return out


def _train_compact(hist: Any) -> Any:
    if not isinstance(hist, dict):
        return hist
    keep = ("name", "training_scope", "n_train_series", "n_val_series", "n_train_obs",
            "n_train_windows", "n_val_windows", "n_optimizer_updates", "epochs_run",
            "max_epochs", "window_balance", "row_weight", "encoder_window_day",
            "train_rows", "val_rows", "curves")
    return {k: hist[k] for k in keep if k in hist}


def _paired_block(results: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        ("reference_tft_target_only", "spline_gamm"),
        ("reference_tft_target_only", "catboost_quantile"),
        ("reference_tft_target_only", "reference_transformer_target_only"),
        ("reference_tft_target_only", "reference_last_value"),
        ("reference_tft_pooled", "reference_tft_target_only"),
        ("tft_pooled_balanced", "tft_target_only_balanced"),
    ]
    out = {}
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        pa, pb = results[a].get("per_study") or [], results[b].get("per_study") or []
        if not pa or not pb:
            continue
        out[f"{b}_minus_{a}"] = paired_study_differences(
            pa, pb, metric="mae_log", name_a=a, name_b=b)
    return out


def _stability_block(results: dict[str, Any], extra_seeds: list[int], split_seed: int) -> dict[str, Any]:
    def _mae(name: str) -> float | None:
        block = results.get(name) or {}
        v = block.get("mae_log")
        if isinstance(v, dict):
            v = v.get("study_balanced")
        return None if v is None else float(v)

    rows = []
    for scope in ("target_only", "pooled"):
        names = [f"reference_tft_{scope}", f"tft_{scope}_balanced"]
        names += [f"tft_{scope}_seed{s}" for s in extra_seeds]
        names += [f"extra_split_tft_{scope}"]
        vals = [{"name": n, "mae_log": _mae(n)} for n in names if _mae(n) is not None]
        maes = [r["mae_log"] for r in vals]
        rows.append({
            "scope": scope, "fits": vals,
            "mean_mae": float(np.mean(maes)) if maes else None,
            "sd_mae": float(np.std(maes, ddof=1)) if len(maes) > 1 else None,
            "n_fits": len(maes),
        })
    return {
        "extra_train_seeds": extra_seeds,
        "extra_split_seed": split_seed,
        "official_holdout_untuned": True,
        "by_scope": rows,
        "note": ("Variability is across predeclared seeds/splits, not across animals "
                 "within a study. Matching train/eval banks only."),
    }


def _run_extra_split(parent: Path, cfg: dict[str, Any], target_dir: Path, pooled_dir: Path,
                     split_seed: int, tft_seed: int, mean_seed: int, cutoff: float,
                     max_enc: int, max_pred: int, tft_cfg: dict[str, Any]) -> dict[str, Any]:
    dest = parent / "banks" / f"split_{split_seed}"
    dest.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    train_info: dict[str, Any] = {}
    try:
        tgt_full = load_full_cell(target_dir)
        poo_full = load_full_cell(pooled_dir)
        split_pack = extra_study_splits(tgt_full["full"], poo_full["full"], seed=split_seed)
        split_pack["target_splits"].to_csv(dest / "splits_target.csv", index=False)
        split_pack["pooled_splits"].to_csv(dest / "splits_pooled.csv", index=False)
        target = prepare_from_splits(
            tgt_full["full"], split_pack["target_splits"], species="RAT", sharing="context_specific")
        pooled = prepare_from_splits(
            poo_full["full"], split_pack["pooled_splits"], species="RAT", sharing="pooled")
        write_json(dest / "feature_spec_target.json", target["spec"].to_dict())
        write_json(dest / "feature_spec_pooled.json", pooled["spec"].to_dict())
        test = target["data"]["test"]
        pack = construct_forecast_examples(
            test.obs, test.animals, cutoff=cutoff,
            max_encoder_length=max_enc, max_prediction_length=max_pred)
        hist, fut = examples_to_tables(pack)
        pack["examples"].to_csv(dest / "examples.csv", index=False)
        fut.to_csv(dest / "target_obs.csv", index=False)
        hist.to_csv(dest / "encoder_obs.csv", index=False)
        write_json(dest / "history_lead_report.json", history_lead_report(pack))
        write_json(dest / "split_summary.json", {
            "seed": split_seed,
            "target_counts": split_pack["target_splits"]["SPLIT"].value_counts().to_dict(),
            "pooled_counts": split_pack["pooled_splits"]["SPLIT"].value_counts().to_dict(),
            "test_studies": sorted(test.animals["STUDYID"].astype(str).unique().tolist()),
            "train_studies_target": target["data"]["train"].studies,
            "train_studies_pooled": pooled["data"]["train"].studies,
            "donor_ids": split_pack["donor_ids"],
            "official_holdout_reused": False,
        })
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "reason": str(e), "traceback": traceback.format_exc()[-2000:],
                "failures": [{"model": "extra_split", "reason": str(e)}]}

    lv = last_value_predictions(hist, fut, QUANTILES)
    lv.to_csv(dest / "last_value.csv", index=False)
    results["extra_split_last_value"] = _score(fut, lv, representation="last_value")
    mark_persistence_intervals_unavailable(results["extra_split_last_value"])
    results["extra_split_last_value"].update({"uses_early_history": True, "label": "extra-split persistence"})
    preds["extra_split_last_value"] = lv

    spline_cfg = dict((cfg.get("models") or {}).get("spline") or {})
    spline_cfg.setdefault("seed", mean_seed)
    tr_cfg = dict((cfg.get("models") or {}).get("transformer") or {})
    tr_cfg.setdefault("seed", mean_seed)
    try:
        from send_bw.pipeline import model_class

        t1 = time.time()
        spline = model_class("spline")()
        train_info["extra_split_spline"] = spline.fit(
            target["data"]["train"], target["data"]["validation"], spline_cfg)
        spline.save(dest / "models" / "spline")
        pred_sp, fail_sp = quantile_table_from_model(spline, test.animals, pack, hist)
        pred_sp.to_csv(dest / "spline_gamm.csv", index=False)
        results["extra_split_spline_gamm"] = _score(fut, pred_sp, representation="gaussian")
        results["extra_split_spline_gamm"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": "extra-split spline/GAMM refit; FeatureSpec refit on new train studies",
            "n_failures": len(fail_sp), "refit": True,
        })
        preds["extra_split_spline_gamm"] = pred_sp
        t1 = time.time()
        tr = model_class("transformer")()
        train_info["extra_split_transformer"] = tr.fit(
            target["data"]["train"], target["data"]["validation"],
            {**tr_cfg, "shared_covariance": spline.cov})
        tr.save(dest / "models" / "transformer")
        pred_tr, fail_tr = quantile_table_from_model(tr, test.animals, pack, hist)
        pred_tr.to_csv(dest / "transformer.csv", index=False)
        results["extra_split_transformer"] = _score(fut, pred_tr, representation="gaussian")
        results["extra_split_transformer"].update({
            "seconds": time.time() - t1, "uses_early_history": True,
            "label": "extra-split custom transformer refit (not the official_seq pretrained checkpoint)",
            "n_failures": len(fail_tr), "refit": True,
        })
        preds["extra_split_transformer"] = pred_tr
    except Exception as e:  # noqa: BLE001
        failures.append({"model": "extra_split_mean", "reason": str(e),
                         "traceback": traceback.format_exc()[-2000:]})

    if catboost_status()["ok"]:
        try:
            t1 = time.time()
            cb_cfg = dict((cfg.get("models") or {}).get("catboost") or {})
            cb_cfg.setdefault("seed", mean_seed)
            cb_cfg.setdefault("encoder_window_day", cutoff)
            cb_cfg.setdefault("max_encoder_length", max_enc)
            cb_cfg.setdefault("max_prediction_length", max_pred)
            cb = QuantileCatBoostModel()
            train_info["extra_split_catboost"] = cb.fit(
                target["data"]["train"], target["data"]["validation"], cb_cfg)
            cb.save(dest / "models" / "catboost")
            pred_cb, fail_cb = quantile_table_from_model(cb, test.animals, pack, hist)
            pred_cb.to_csv(dest / "catboost.csv", index=False)
            results["extra_split_catboost"] = _score(
                fut, pred_cb, representation="catboost_quantiles", quantiles=COMMON_Q)
            results["extra_split_catboost"].update({
                "seconds": time.time() - t1, "uses_early_history": True,
                "label": "extra-split quantile CatBoost", "n_failures": len(fail_cb),
            })
            preds["extra_split_catboost"] = pred_cb
        except Exception as e:  # noqa: BLE001
            failures.append({"model": "extra_split_catboost", "reason": str(e),
                             "traceback": traceback.format_exc()[-2000:]})

    for scope, prep, key in (
        ("target_only", target, "extra_split_tft_target_only"),
        ("pooled", pooled, "extra_split_tft_pooled"),
    ):
        t1 = time.time()
        try:
            cfg_i = dict(tft_cfg)
            cfg_i["seed"] = tft_seed
            m, h = _fit_tft(prep, cfg_i, dest / "models" / key, scope)
            train_info[key] = h
            pred, fail = quantile_table_from_model(m, test.animals, pack, hist)
            pred.to_csv(dest / f"{key}.csv", index=False)
            results[key] = _score(fut, pred, representation="tft_quantiles")
            results[key].update({
                "seconds": time.time() - t1, "uses_early_history": True,
                "label": f"extra-split TFT {scope}; seed {tft_seed}; study sampler",
                "n_failures": len(fail),
            })
            preds[key] = pred
        except Exception as e:  # noqa: BLE001
            results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:]}
            failures.append({"model": key, "reason": str(e)})

    return {
        "status": "ok",
        "results": results,
        "preds": preds,
        "train_info": train_info,
        "failures": failures,
        "summary": {
            "split_seed": split_seed,
            "test_studies": sorted(test.animals["STUDYID"].astype(str).unique().tolist()),
            "n_eligible": pack["n_eligible"],
            "official_holdout_reused": False,
        },
    }


def _write_report(parent: Path, out: dict[str, Any], results: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    lines = [
        "# TFT early-history extension",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "Forecast windows were taken from observed `DAY` values. Unmeasured days were "
        "not created. Long-horizon chunking was not used. The completed pilot and all "
        "`runs/v3_*` directories were left untouched.",
        "",
        "## Predeclared stability plan",
        "",
        "Locked before new scores were examined:",
        "",
        f"- official holdout generate_seed={lock.get('generate_seed')}, TFT train_seed={lock.get('train_seed')}",
        f"- extra TFT train seeds: {(out.get('predeclared') or {}).get('extra_train_seeds')}",
        f"- extra whole-study split seed: {(out.get('predeclared') or {}).get('extra_split_seed')}",
        "- official holdout errors were not used to tune later models",
        "",
        "## What ran",
        "",
        f"- output: `{out.get('output_dir')}`",
        f"- cutoff: DAY ≤ {lock.get('encoder_window_day')}",
        f"- encoder visit cap: {lock.get('max_encoder_length')}; "
        f"prediction-length cap: {lock.get('max_prediction_length')} (no chunking)",
        f"- eligible official-holdout animals: {(out.get('example_summary') or {}).get('n_eligible')}",
        f"- correctness: {(out.get('correctness') or {}).get('ok')}",
        f"- duration decision: {out.get('duration_decision')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Metrics",
        "",
        "- `mae_log` / `rmse_log`: study-balanced |error| and true RMSE on LOGW = ln(WEIGHT_G grams)",
        "- `mae_gram`: study-balanced |exp(point) − WEIGHT_G| in grams",
        "- pinball: quantile loss on LOGW and, separately, on grams at common levels 0.1/0.5/0.9",
        "- `coverage80` and `interval_width80` are reported together on LOGW",
        "- persistence interval metrics are unavailable (degenerate quantiles, not a calibrated interval)",
        "- studies are averaged after within-study aggregation; animals are not independent replications",
        "",
        "## Comparison (study-balanced, official holdout unless prefixed `extra_split_`)",
        "",
        "| model | MAE log | RMSE log | MAE g | pinball | cov80 | width80 | crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in results.items():
        cr = block.get("crossing_rate")
        if isinstance(cr, dict):
            cr = cr.get("pooled")
        lines.append(
            f"| {name} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mae_gram')} | {_fmt(block, 'mean_quantile_loss')} | "
            f"{_fmt(block, 'coverage80')} | {_fmt(block, 'interval_width80')} | {cr} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "Conclusions apply only to Generator A and the evaluated synthetic settings. "
        "Successful training and unit tests are not forecasting effectiveness. "
        "No historical experimental body-weight studies were opened.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
