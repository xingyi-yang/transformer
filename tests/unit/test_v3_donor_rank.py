"""Inner-fold ranking contracts and prefix-abstention closeout. No benchmark train."""

from __future__ import annotations

from pathlib import Path

import pytest

from send_bw.donor_rank import (
    RANK_LABEL,
    RANK_RULE,
    DonorRankError,
    inner_target_train_folds,
    rank_one_donor_candidates,
    recording_score_fn,
    share_or_abstain_after_rank,
)
from send_bw.protected_hashes import compare_snapshots, snapshot_protected
from send_bw.share_policy import RANKED_RULE, decide_abstention, resolve_donors
from send_bw.v3_donor_rank import (
    LOCK_DIR,
    DonorRankLockError,
    execute_v3_donor_rank,
    locked_budget,
    prefix_abstention_closeout,
)
from send_bw.v3_research import load_yaml

FROZEN = "configs/v3_donor_rank.frozen.yaml"


def test_closeout_magnitudes_from_saved_run():
    close = prefix_abstention_closeout()
    assert close["status"] == "ok"
    assert close["retuned"] is False
    assert close["confirmatory"] is False
    assert close["prefix_versus_random_all_equal"] is True
    groups = {(g["dataset_index"], g["model"], g["arm"]): g for g in close["groups"]}
    d0s = groups[(0, "spline", "selected_one")]
    assert d0s["n"] == 4
    assert d0s["mae_signs"]["helpful"] == 4
    assert d0s["rmse_signs"]["helpful"] == 4
    assert d0s["mae_delta_mean"] == pytest.approx(-0.00153378, abs=1e-5)
    assert d0s["rmse_abs_mean"] == pytest.approx(0.00189954, abs=1e-5)
    d1t = groups[(1, "transformer", "selected_one")]
    assert d1t["mae_signs"]["harmful"] == 4
    assert d1t["mae_delta_mean"] == pytest.approx(0.00911473, abs=1e-5)
    d1ta = groups[(1, "transformer", "selected_abstain")]
    assert d1ta["mae_signs"]["harmful"] == 2
    assert d1ta["mae_signs"]["null"] == 2
    abst = close["abstention_counts"]["overall"]
    assert abst["target_only_over_n"] == "9/16"
    assert abst["share_over_n"] == "7/16"


def test_inner_folds_are_target_train_only():
    folds = inner_target_train_folds(["T3", "T1", "T4", "T2"], n_folds=2)
    assert len(folds) == 2
    all_held = [s for f in folds for s in f["heldout_target_ids"]]
    all_fit = [s for f in folds for s in f["fit_target_ids"]]
    assert sorted(all_held) == ["T1", "T2", "T3", "T4"]
    assert "TEST" not in all_held + all_fit
    assert inner_target_train_folds(["T1"], n_folds=2) == []


def test_rank_rejects_test_outcomes_and_scenario_labels():
    score, _ = recording_score_fn({"D-a": 0.2, "D-b": 0.1})
    kwargs = dict(eligible_donors=["D-b", "D-a"], target_train_ids=["T1", "T2", "T3", "T4"],
                  score_fn=score, test_study_ids=["TEST"], validation_study_ids=["VAL"])
    with pytest.raises(DonorRankError, match="generator scenario"):
        rank_one_donor_candidates(**kwargs, generator_scenario="matched")
    with pytest.raises(DonorRankError, match="generator scenario"):
        rank_one_donor_candidates(**kwargs, bank="both")
    with pytest.raises(DonorRankError, match="generator scenario"):
        rank_one_donor_candidates(**kwargs, scenario_label="both")
    with pytest.raises(DonorRankError, match="test outcomes"):
        rank_one_donor_candidates(**kwargs, test_outcomes={"mae": 0.01})
    with pytest.raises(DonorRankError, match="overlap forbidden"):
        rank_one_donor_candidates(
            ["D-a"], ["T1", "TEST"], score, test_study_ids=["TEST"], validation_study_ids=["VAL"])


def test_rank_uses_inner_scores_not_test_or_scenario():
    score, calls = recording_score_fn({"D-a": 0.30, "D-b": 0.10, "D-c": 0.20})
    out = rank_one_donor_candidates(
        ["D-c", "D-a", "D-b"], ["T1", "T2", "T3", "T4"], score,
        test_study_ids=["TEST-1"], validation_study_ids=["VAL-1"],
    )
    assert out["selected"] == ["D-b"]
    assert out["rule"] == RANK_RULE
    assert "not test-informed" in RANK_LABEL
    assert out["test_outcomes_used"] is False
    assert out["generator_scenario_used"] is False
    assert out["outer_validation_used"] is False
    assert out["inner_train_fold_ranked"] is True
    assert out["validation_ranked"] is False
    seen = {s for c in calls for s in c["fit_target_ids"] + c["heldout_target_ids"]}
    assert seen == {"T1", "T2", "T3", "T4"}
    assert "TEST-1" not in seen
    assert "VAL-1" not in seen
    assert {c["donor_id"] for c in calls} == {"D-a", "D-b", "D-c"}
    assert all(c["donor_id"] not in {"TEST-1", "VAL-1"} for c in calls)
    # 3 donors × 2 folds
    assert len(calls) == 6
    # tie-break STUDYID when scores match
    tie, _ = recording_score_fn({"D-a": 0.2, "D-b": 0.2})
    tied = rank_one_donor_candidates(["D-b", "D-a"], ["T1", "T2"], tie)
    assert tied["selected"] == ["D-a"]


def test_outer_abstention_rejects_leaks_and_defaults_target_only():
    with pytest.raises(DonorRankError, match="test outcomes"):
        share_or_abstain_after_rank(0.10, 0.08, test_outcomes={"rmse": 0.01})
    with pytest.raises(DonorRankError, match="generator scenario"):
        share_or_abstain_after_rank(0.10, 0.08, generator_scenario="matched")
    dec = share_or_abstain_after_rank(0.10, 0.12, n_val_studies=2)
    assert dec["decision"] == "target_only"
    assert dec["test_outcomes_used"] is False
    assert dec["generator_scenario_used"] is False
    assert share_or_abstain_after_rank(0.10, 0.08)["decision"] == "share"
    assert decide_abstention(0.10, 0.10)["decision"] == "target_only"


def test_resolve_donors_does_not_rank():
    with pytest.raises(ValueError, match="does not rank"):
        resolve_donors(["D-b", "D-a"], k=1, rule="ranked")
    rec = resolve_donors(["D-b", "D-a"], k=1, rule="ranked", explicit=["D-b"])
    assert rec["rule"] == RANKED_RULE
    assert rec["selected"] == ["D-b"]
    assert rec["learned"] is True
    assert rec["inner_train_fold_ranked"] is True
    assert rec["test_outcomes_used"] is False
    pref = resolve_donors(["D-b", "D-a"], k=1, rule="prefix")
    assert pref["selected"] == ["D-a"]
    assert pref["learned"] is False
    assert pref["inner_train_fold_ranked"] is False


def test_locked_budget_fresh_seeds_and_exact_candidate_fits():
    cfg = load_yaml(FROZEN)
    budget = locked_budget(cfg)
    assert budget["execute"] is False
    assert budget["generate_seeds"] == [78001, 79010]
    assert budget["train_seeds"] == [86001]
    assert budget["n_eligible_donors"] == 3
    assert budget["n_inner_folds"] == 2
    assert budget["banks"] == ["matched", "both"]
    assert budget["densities"] == ["full"]
    assert budget["n_slices"] == 4
    assert budget["n_candidate_fits"] == 48
    assert budget["n_final_model_fits"] == 24
    assert budget["n_model_fits_if_executed"] == 72
    assert budget["additional_fits_for_abstention"] == 0
    assert budget["estimated_runtime_seconds"]["point"] == 240
    assert budget["compute_cap_seconds"] == 600
    assert budget["confirmatory"] is False
    assert budget["training_seed_variability_estimated"] is False
    assert 78001 not in {76001, 77010, 74001, 73001}
    assert 86001 not in {85001, 84001, 83001}


def test_execute_true_is_refused(tmp_path):
    cfg = tmp_path / "exec.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\ngenerate_seed: 78001\ntrain_seed: 86001\n"
        "donor_rank: {execute: true}\n",
        encoding="utf-8")
    with pytest.raises(DonorRankLockError, match="before benchmark execution"):
        execute_v3_donor_rank(cfg)


def test_lock_writes_closeout_and_preserves_artifacts():
    out = execute_v3_donor_rank(FROZEN)
    assert out["status"] == "locked_not_executed"
    assert out["n_candidate_fits"] == 48
    assert out["n_final_model_fits"] == 24
    assert out["n_model_fits_if_executed"] == 72
    assert out["real_data_accessed"] is False
    assert out["design_run"] is False
    assert (out.get("protected") or {}).get("unchanged") is True
    assert Path("docs/v3/share_abstain_run_CLOSEOUT.md").is_file()
    assert Path("docs/v3/donor_rank_protocol.md").is_file()
    assert Path("docs/v3/donor_rank_RESULTS.md").is_file()
    assert Path(LOCK_DIR, "freeze_lock.json").is_file()
    lock = out["freeze_lock"]
    assert lock["execute"] is False
    assert lock["selector_sees_generator_labels"] is False
    assert lock["selector_sees_test_outcomes"] is False
    assert lock["prefix_diagnostic_retuned"] is False
    text = Path("docs/v3/share_abstain_run_CLOSEOUT.md").read_text(encoding="utf-8")
    assert "Closed without retuning" in text
    assert "not confirmatory" in text.lower()
    assert "selected_abstain" in text
    before = snapshot_protected()
    after = snapshot_protected()
    assert compare_snapshots(before, after)["unchanged"] is True
    assert before["files"]["runs/v3_share_abstain_run/SUMMARY.json"] is not None
    assert before["directories"]["runs/v3_share_abstain_run"] is not None
