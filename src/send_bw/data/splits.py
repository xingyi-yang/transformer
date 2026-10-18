"""Frozen study-level splits, created before any transform, tuning or window generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.utils.artifacts import cache_key, read_json, write_json

SPLITS = ("train", "validation", "test", "test_compound_holdout")


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
