from __future__ import annotations

import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from send_bw.data.features import FeatureSpec, availability, check_no_leakage
from send_bw.data.generate import GeneratorA, GeneratorB, exposure_kernel
from send_bw.data.harmonize import harmonize
from send_bw.data.qc import CohortConfig, select_cohort
from send_bw.data.sources import LocalSendSource
from send_bw.data.splits import assert_disjoint, freeze_splits, load_frozen_splits, make_study_splits

SRC = Path(__file__).resolve().parents[2] / "src" / "send_bw"


def test_generator_varies_arms_and_writes_truth_separately(synth):
    t = synth["tables"]
    n_arms = t["arm_regimen"][t["arm_regimen"]["COHORT"] == "MAIN"].groupby("STUDYID").size()
    assert n_arms.nunique() > 1
    root = synth["root"]
    assert (root / "truth" / "params.json").exists()
    assert not any(p.name == "truth" for p in (root / "send").rglob("*"))
    assert (t["bw_obs"]["WEIGHT_G"] > 0).all()


def test_shared_study_effect_drawn_once_per_study(synth):
    st = pd.read_parquet(synth["root"] / "truth" / "studies.parquet")
    assert st["STUDYID"].is_unique


def test_kg_study_converted(synth):
    obs = synth["tables"]["bw_obs"]
    kg = obs[obs["ORIG_UNIT"] == "kg"]
    assert len(kg) > 0 and kg["WEIGHT_G"].between(100, 1000).all()


def test_exposure_kernel_onset_and_recovery():
    t = np.arange(-5, 60, dtype=float)
    h = exposure_kernel(t, 0.0, 27.0, 5.0, 10.0)
    assert (h[t < 0] == 0).all()
    assert h[t == 28][0] > h[t == 7][0]
    assert h[t == 50][0] < h[t == 28][0]


def test_generator_b_differs_from_a():
    a, b = GeneratorA(), GeneratorB()
    t = np.linspace(0, 28, 8)
    ga = a.growth(t, "RAT", "M", "SPRAGUE-DAWLEY", np.log(260))
    gb = b.growth(t, "RAT", "M", "SPRAGUE-DAWLEY", np.log(260))
    assert not np.allclose(ga, gb, atol=0.01)


def test_design_sample_shapes_and_conditioning():
    ctx = pd.DataFrame({"SEX": ["M", "F"] * 3, "ARM_KEY": ["c"] * 6, "BASELINE_LOGW": np.log(250.0),
                        "SPECIES": "RAT", "STRAIN": "SPRAGUE-DAWLEY"})
    reg = pd.DataFrame([{"ARM_KEY": "c", "IS_CONTROL": True, "COMPOUND_ID": None, "PLANNED_DOSE": 0,
                         "INTERVAL_DAYS": 1, "START_DAY": 0, "END_DAY": 27, "ROUTE": "ORAL GAVAGE"}])
    out = GeneratorA().sample_study(ctx, reg, np.array([7.0, 14.0, 28.0]), 5, seed=1)
    assert out["z"].shape == (5, 6, 3) and out["mean"].shape == (5, 6, 3)


def test_splits_are_study_level_and_frozen(synth, tmp_path):
    sc = synth["tables"]["study_context"]
    sp = make_study_splits(sc, {"train": 0.6, "validation": 0.2, "test": 0.2}, seed=3)
    assert_disjoint(sp)
    assert set(sp["SPLIT"]) >= {"train", "validation", "test"}
    p = tmp_path / "splits"
    freeze_splits(sp, p)
    pd.testing.assert_frame_equal(load_frozen_splits(p).reset_index(drop=True), sp.astype(str).reset_index(drop=True))
    tampered = sp.copy()
    tampered.loc[0, "SPLIT"] = "test" if tampered.loc[0, "SPLIT"] != "test" else "train"
    tampered.to_csv(p.with_suffix(".csv"), index=False)
    with pytest.raises(ValueError, match="modified"):
        load_frozen_splits(p)


def test_compound_holdout_split(synth):
    sc = synth["tables"]["study_context"]
    reg = synth["tables"]["arm_regimen"]
    comp = reg.dropna(subset=["COMPOUND_ID"]).groupby("STUDYID")["COMPOUND_ID"].first().to_dict()
    hold = sorted(set(comp.values()))[0]
    sp = make_study_splits(sc, {"compound_holdout": [hold], "_study_compound": comp}, seed=3)
    held = sp[sp["SPLIT"] == "test_compound_holdout"]["STUDYID"]
    assert set(held) == {s for s, c in comp.items() if c == hold}


def test_feature_registry_blocks_truth_and_actual_dosing():
    for c in ["Z0_TRUE", "U_STUDY_SLOPE", "DOSE_DEVIATION"]:
        assert availability(c) == "truth_only"
    assert availability("EXDOSE_ACTUAL") == "optional"
    with pytest.raises(ValueError):
        check_no_leakage(["SEX_M", "Z0_TRUE"])
    with pytest.raises(ValueError):
        check_no_leakage(["EXDOSE_ACTUAL"])
    with pytest.raises(KeyError):
        availability("UNCLASSIFIED_FEATURE")


def test_feature_spec_uses_only_production_features(synth):
    c = select_cohort(synth["tables"], CohortConfig())
    spec = FeatureSpec.fit(c.animals, c.obs, synth["tables"]["compound_reference"], "RAT")
    check_no_leakage(spec.animal_cols + spec.time_cols)
    f = spec.observation_features(c.obs, c.animals, synth["tables"]["compound_reference"])
    assert not set(f.columns) & {"Y", "LOGW", "WEIGHT_G", "Z_LATENT"}


def test_harmonized_output_independent_of_truth_dir(synth, mapping, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(synth["root"], root)
    shutil.rmtree(root / "truth")
    src = LocalSendSource(root / "send", metadata_dir=root / "metadata", domain_aliases=mapping.domain_aliases)
    t2, _ = harmonize(src, mapping)
    a = synth["tables"]["bw_obs"].drop(columns=["SRC_FILE"])
    b = t2["bw_obs"].drop(columns=["SRC_FILE"])
    pd.testing.assert_frame_equal(a, b)


def _imports_and_strings(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods, strs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strs.add(node.value)
    return mods, strs


@pytest.mark.parametrize("sub", ["models", "data/features.py", "data/harmonize.py", "data/qc.py", "data/splits.py"])
def test_fitting_code_never_reads_truth(sub):
    paths = [SRC / sub] if sub.endswith(".py") else list((SRC / sub).glob("*.py"))
    for p in paths:
        mods, strs = _imports_and_strings(p)
        assert "send_bw.simulation.truth" not in mods, p
        assert not any(s in {"truth", "latent_paths.parquet", "params.json"} for s in strs), p
