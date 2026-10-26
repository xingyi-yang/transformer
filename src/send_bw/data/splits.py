"""Frozen study-level splits, created before any transform, tuning or window generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.utils.artifacts import cache_key, read_json, write_json

SPLITS = ("train", "validation", "test", "test_compound_holdout")
CALIBRATED_SPLITS = ("train", "validation", "calibration", "test", "test_compound_holdout")
TARGET_SPLIT_NAMES = ("train", "validation", "test")
CALIBRATED_TARGET_SPLIT_NAMES = ("train", "validation", "calibration", "test")


def make_study_splits(study_context: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Assign whole studies to splits.

    Options: compound holdout (all studies of listed compounds -> test_compound_holdout),
    chronological (latest studies -> test) or random. Fractions apply to the remaining studies.
    """
    studies = study_context[["STUDYID", "STUDY_START"]].copy()
    studies["COMPOUND_ID"] = studies["STUDYID"].map(cfg.get("_study_compound", {}))
    holdout = set(cfg.get("compound_holdout", []) or [])
    rng = np.random.default_rng(seed)
    is_hold = studies["COMPOUND_ID"].isin(holdout)
    rest = studies[~is_hold].copy()
    if cfg.get("method", "random") == "chronological":
        rest = rest.sort_values(["STUDY_START", "STUDYID"])
        ids = rest["STUDYID"].tolist()
    else:
        ids = sorted(rest["STUDYID"].tolist())
        rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(cfg.get("test", 0.2) * n)))
    n_val = max(1, int(round(cfg.get("validation", 0.2) * n)))
    if cfg.get("method", "random") == "chronological":
        test, val, train = ids[n - n_test:], ids[n - n_test - n_val:n - n_test], ids[:n - n_test - n_val]
    else:
        test, val, train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    assign = {**{s: "train" for s in train}, **{s: "validation" for s in val}, **{s: "test" for s in test},
              **{s: "test_compound_holdout" for s in studies.loc[is_hold, "STUDYID"]}}
    out = pd.DataFrame({"STUDYID": list(assign), "SPLIT": list(assign.values())}).sort_values("STUDYID")
    if (out["SPLIT"] == "train").sum() < 2:
        raise ValueError("fewer than 2 training studies; increase n_studies")
    return out.reset_index(drop=True)


def make_study_splits_calibrated(study_context: pd.DataFrame, cfg: dict[str, Any], seed: int,
                                 groups: dict[str, str] | None = None) -> pd.DataFrame:
    """Study-level train / validation / calibration / test assignment.

    Does not replace ``make_study_splits``. Official hashes stay on the
    three-way helper. ``groups`` maps STUDYID -> pair/group key so paired
    dense/sparse siblings stay in one partition. Calibration is separate
    from validation (checkpoint selection) and from test.
    """
    studies = study_context[["STUDYID", "STUDY_START"]].copy()
    studies["COMPOUND_ID"] = studies["STUDYID"].map(cfg.get("_study_compound", {}))
    holdout = set(cfg.get("compound_holdout", []) or [])
    rng = np.random.default_rng(seed)
    is_hold = studies["COMPOUND_ID"].isin(holdout)
    rest = studies[~is_hold].copy()
    group_of = {str(s): str((groups or {}).get(str(s), s)) for s in rest["STUDYID"]}
    members: dict[str, list[str]] = {}
    for sid in rest["STUDYID"].astype(str):
        members.setdefault(group_of[sid], []).append(sid)
    if cfg.get("method", "random") == "chronological":
        start = rest.set_index(rest["STUDYID"].astype(str))["STUDY_START"]
        keys = sorted(members, key=lambda k: (str(start.loc[members[k][0]]), k))
    else:
        keys = sorted(members)
        rng.shuffle(keys)
    n = len(keys)
    n_test = max(1, int(round(cfg.get("test", 0.20) * n)))
    n_val = max(1, int(round(cfg.get("validation", 0.15) * n)))
    n_cal = max(1, int(round(cfg.get("calibration", 0.15) * n)))
    if n_test + n_val + n_cal >= n:
        raise ValueError("calibrated split fractions leave no training groups")
    if cfg.get("method", "random") == "chronological":
        test_k = keys[n - n_test:]
        val_k = keys[n - n_test - n_val:n - n_test]
        cal_k = keys[n - n_test - n_val - n_cal:n - n_test - n_val]
        train_k = keys[: n - n_test - n_val - n_cal]
    else:
        test_k, val_k = keys[:n_test], keys[n_test:n_test + n_val]
        cal_k = keys[n_test + n_val:n_test + n_val + n_cal]
        train_k = keys[n_test + n_val + n_cal:]
    assign: dict[str, str] = {}
    for label, ks in (("test", test_k), ("validation", val_k), ("calibration", cal_k), ("train", train_k)):
        for k in ks:
            for sid in members[k]:
                assign[sid] = label
    for sid in studies.loc[is_hold, "STUDYID"].astype(str):
        assign[sid] = "test_compound_holdout"
    out = pd.DataFrame({"STUDYID": list(assign), "SPLIT": list(assign.values()),
                        "GROUP": [group_of.get(s, s) for s in assign]}).sort_values("STUDYID")
    if (out["SPLIT"] == "train").sum() < 2:
        raise ValueError("fewer than 2 training studies; increase n_studies")
    return out.reset_index(drop=True)


def freeze_splits(splits: pd.DataFrame, path: Path) -> str:
    h = cache_key(splits.sort_values("STUDYID").to_dict("records"))
    splits.to_csv(path.with_suffix(".csv"), index=False)
    write_json(path.with_suffix(".json"), {"hash": h, "n": len(splits),
                                           "counts": splits["SPLIT"].value_counts().to_dict()})
    return h


def load_frozen_splits(path: Path) -> pd.DataFrame:
    splits = pd.read_csv(path.with_suffix(".csv"), dtype=str)
    meta = read_json(path.with_suffix(".json"))
    if cache_key(splits.sort_values("STUDYID").to_dict("records")) != meta["hash"]:
        raise ValueError("frozen split file was modified after freezing")
    return splits


def assert_disjoint(splits: pd.DataFrame) -> None:
    if splits["STUDYID"].duplicated().any():
        raise ValueError("a study appears in more than one split")


def target_manifest(splits: pd.DataFrame) -> dict[str, Any]:
    """Per-split hashes of target-species STUDYIDs. Donors must not appear here."""
    return _manifest(splits, TARGET_SPLIT_NAMES, "target-manifest-v1")


def calibrated_target_manifest(splits: pd.DataFrame) -> dict[str, Any]:
    """Four-way target manifest including calibration. Official three-way helper is unchanged."""
    return _manifest(splits, CALIBRATED_TARGET_SPLIT_NAMES, "target-manifest-calibrated-v1")


def _manifest(splits: pd.DataFrame, names: tuple[str, ...], version: str) -> dict[str, Any]:
    out: dict[str, Any] = {"version": version}
    all_ids: list[str] = []
    for name in names:
        ids = sorted(splits.loc[splits["SPLIT"] == name, "STUDYID"].astype(str).tolist())
        out[name] = {"studyids": ids, "n": len(ids), "hash": cache_key(ids)}
        all_ids.extend(ids)
    out["all_target"] = {"studyids": sorted(all_ids), "n": len(all_ids), "hash": cache_key(sorted(all_ids))}
    return out


def freeze_target_manifests(splits: pd.DataFrame, path: Path) -> dict[str, Any]:
    man = target_manifest(splits)
    write_json(path, man)
    return man


def donor_study_order(study_ids) -> list[str]:
    """Declared nested order: STUDYID sort. First k donors are a prefix of this list."""
    return sorted({str(s) for s in study_ids})


def compose_train_with_donors(target_splits: pd.DataFrame, donor_ids: list[str], k: int) -> pd.DataFrame:
    """Add the first k donors to train only. Validation and test stay target-only."""
    ids = donor_study_order(donor_ids)[: max(0, int(k))]
    return compose_train_with_selected_donors(target_splits, ids)


def compose_train_with_selected_donors(target_splits: pd.DataFrame,
                                       selected_ids: list[str]) -> pd.DataFrame:
    """Add the listed donors to train only. Validation and test stay target-only."""
    ids = [str(s) for s in selected_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected donor STUDYIDs")
    overlap = set(target_splits["STUDYID"].astype(str)) & set(ids)
    if overlap:
        raise ValueError(f"donor studies overlap target splits: {sorted(overlap)}")
    extra = pd.DataFrame({"STUDYID": ids, "SPLIT": "train"})
    out = pd.concat([target_splits[["STUDYID", "SPLIT"]].astype(str), extra], ignore_index=True)
    out = out.sort_values("STUDYID").reset_index(drop=True)
    assert_disjoint(out)
    return out


SPARSE_TARGET_TRAIN_KIND = "sparse_target_study_training"
SPARSE_HISTORY_KIND = "sparse_history_encoder"


def compose_sparse_target_train(target_splits: pd.DataFrame, n_keep: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep the first n_keep target TRAIN studies by STUDYID sort. Val/test stay fixed.

    This is not a sparse observation/history encoder window (W0/W1/Wlast14/Wall14).
    """
    splits = target_splits[["STUDYID", "SPLIT"]].astype(str).copy()
    train_ids = sorted(splits.loc[splits["SPLIT"] == "train", "STUDYID"].tolist())
    n_keep = int(n_keep)
    if n_keep < 2:
        raise ValueError("sparse target-study training requires n_keep >= 2")
    kept = train_ids[:n_keep]
    dropped = train_ids[n_keep:]
    if len(kept) < 2:
        raise ValueError("sparse target-study training would leave fewer than 2 training studies")
    out = splits[~splits["STUDYID"].isin(dropped)].reset_index(drop=True)
    meta: dict[str, Any] = {
        "kind": SPARSE_TARGET_TRAIN_KIND,
        "not": SPARSE_HISTORY_KIND,
        "n_keep": n_keep,
        "n_train_before": len(train_ids),
        "n_train_after": len(kept),
        "kept_train_studyids": kept,
        "dropped_train_studyids": dropped,
        "validation_unchanged": True,
        "test_unchanged": True,
        "validation_studyids": sorted(splits.loc[splits["SPLIT"] == "validation", "STUDYID"].tolist()),
        "test_studyids": sorted(splits.loc[splits["SPLIT"] == "test", "STUDYID"].tolist()),
        "note": (
            "Drops target TRAIN studies only. Validation and test study sets stay frozen. "
            "Distinct from sparse observation/history encoder windows."
        ),
    }
    return out, meta
