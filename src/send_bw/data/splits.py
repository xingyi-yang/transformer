"""Frozen study-level splits, created before any transform, tuning or window generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.utils.artifacts import cache_key, read_json, write_json

SPLITS = ("train", "validation", "test", "test_compound_holdout")
TARGET_SPLIT_NAMES = ("train", "validation", "test")


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
    out: dict[str, Any] = {"version": "target-manifest-v1"}
    all_ids: list[str] = []
    for name in TARGET_SPLIT_NAMES:
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
    overlap = set(target_splits["STUDYID"].astype(str)) & set(ids)
    if overlap:
        raise ValueError(f"donor studies overlap target splits: {sorted(overlap)}")
    extra = pd.DataFrame({"STUDYID": ids, "SPLIT": "train"})
    out = pd.concat([target_splits[["STUDYID", "SPLIT"]].astype(str), extra], ignore_index=True)
    out = out.sort_values("STUDYID").reset_index(drop=True)
    assert_disjoint(out)
    return out
