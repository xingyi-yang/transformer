"""Share-or-abstain contracts: audit, policy, holdouts, hashes. No benchmark train."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from send_bw.data.splits import (
    compose_train_with_donors,
    compose_train_with_selected_donors,
    donor_study_order,
)
from send_bw.evaluation.quantile_forecast import score_quantile_table
from send_bw.protected_hashes import compare_snapshots, snapshot_protected
from send_bw.share_policy import decide_abstention, resolve_donors, val_score
from send_bw.v3_research import load_yaml
from send_bw.v3_share_abstain import (
    ShareAbstainError,
    audit_existing_selection,
    erratum_from_share_repl,
    execute_v3_share_abstain,
    locked_budget,
)

FROZEN = "configs/v3_share_abstain.frozen.yaml"
Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def test_erratum_from_saved_pr14_numbers():
    err = erratum_from_share_repl()
    assert err["status"] == "ok"
    assert err["all_supported"] is True
    ids = {c["id"]: c for c in err["claims"]}
    d1 = ids["pr14_d1_chronos_vs_last_value"]["numbers"]
    assert d1["chronos2_target_only_history_mae"] < d1["last_value_mae"]
    assert d1["chronos2_pooled_history_mae"] < d1["last_value_mae"]
    assert d1["chronos2_zero_shot_history_mae"] > d1["last_value_mae"]
    d0 = ids["pr14_d0_spline_selective"]["numbers"]
    assert d0["selective"]["mae"] < d0["target_only"]["mae"]
    assert d0["selective"]["pinball"] < d0["target_only"]["pinball"]
    assert d0["selective_regularized"]["mae"] < d0["target_only"]["mae"]
    assert d0["selective_regularized"]["pinball"] > d0["target_only"]["pinball"]


def test_audit_records_prefix_and_missing_target_only_bank():
    audit = audit_existing_selection()
    assert audit["no_training"] is True
    assert audit["mechanism"]["test_outcomes_used_for_selection"] is False
    assert audit["mechanism"]["zero_donors_available"] is True
    cells = {c["cell"]: c["donor_order"] for c in audit["pr14_cells"]}
    sel = cells["runs/v3_share_repl/datasets/d0/trains/t0/cells/d0_t0_matched_selective"]
    assert sel["selected"] == ["SYNA-D001"]
    assert sel["studyids"] == ["SYNA-D001", "SYNA-D002", "SYNA-D003"]
    assert sel["selected"] == donor_study_order(sel["studyids"])[:1]
    to = cells["runs/v3_share_repl/datasets/d0/trains/t0/cells/d0_t0_matched_target_only"]
    assert to["k"] == 0
    assert to["selected"] == []
    assert to["studyids"] == []
    man = next(c["target_manifest"] for c in audit["pr14_cells"]
               if c["cell"].endswith("d0_t0_matched_target_only"))
    assert man["train"]["n"] == 7
    assert "single-study" in audit["mechanism"]["target_only_is_not_single_study"]
    hist = next(c["spline_history"] for c in audit["pr14_cells"]
                if c["cell"].endswith("d0_t0_matched_selective_regularized"))
    assert hist["lambda_growth"] == 8.0
    assert hist["covariance_fit_on"] == "target_species_oof"


def test_random_and_selected_have_equal_k():
    eligible = ["D-b", "D-a", "D-c"]
    pref = resolve_donors(eligible, k=1, rule="prefix")
    rnd = resolve_donors(eligible, k=1, rule="random", rng_seed=91001)
    assert pref["k"] == rnd["k"] == 1
    assert pref["selected"] == ["D-a"]
    assert rnd["selected"][0] in {"D-a", "D-b", "D-c"}
    assert pref["test_outcomes_used"] is False
    assert rnd["test_outcomes_used"] is False
    zero = resolve_donors(eligible, k=0, rule="zero", allow_donors=False)
    assert zero["selected"] == []
    assert zero["eligible"] == ["D-a", "D-b", "D-c"]


def test_abstention_defaults_to_target_only():
    assert decide_abstention(0.10, 0.10)["decision"] == "target_only"
    assert decide_abstention(0.10, 0.10)["reason"] == "tie_or_insufficient_improvement"
    assert decide_abstention(0.10, 0.12)["decision"] == "target_only"
    assert decide_abstention(0.10, 0.09)["decision"] == "share"
    assert decide_abstention(None, 0.09)["decision"] == "target_only"
    assert decide_abstention(0.10, 0.09, n_val_studies=0)["decision"] == "target_only"
    assert decide_abstention(0.10, 0.09)["test_outcomes_used"] is False
    assert decide_abstention(0.10, 0.09)["threshold_from_test"] is False


def test_holdouts_unchanged_when_donors_change():
    target = pd.DataFrame({
        "STUDYID": ["T1", "T2", "T3", "T4"],
        "SPLIT": ["train", "train", "validation", "test"],
    })
    a = compose_train_with_donors(target, ["D2", "D1"], 1)
    b = compose_train_with_selected_donors(target, ["D2"])
    assert set(a.loc[a["SPLIT"] == "validation", "STUDYID"]) == {"T3"}
    assert set(b.loc[b["SPLIT"] == "validation", "STUDYID"]) == {"T3"}
    assert set(a.loc[a["SPLIT"] == "test", "STUDYID"]) == {"T4"}
    assert set(b.loc[b["SPLIT"] == "test", "STUDYID"]) == {"T4"}
    assert "D1" in set(a.loc[a["SPLIT"] == "train", "STUDYID"])
    assert "D2" in set(b.loc[b["SPLIT"] == "train", "STUDYID"])
    assert "D1" not in set(b["STUDYID"])


def test_saved_decision_reproduces_chosen_predictions(tmp_path):
    obs = pd.DataFrame({"USUBJID": ["a"], "STUDYID": ["S"], "DAY": [21.0], "LOGW": [1.0]})
    pred = pd.DataFrame({
        "USUBJID": ["a"], "STUDYID": ["S"], "DAY": [21.0],
        "q0.1": [1.0], "q0.25": [1.0], "q0.5": [1.0], "q0.75": [1.0], "q0.9": [1.0],
    })
    src = tmp_path / "spline_target_only.csv"
    dest = tmp_path / "spline_selected_abstain.csv"
    pred.to_csv(src, index=False)
    dest.write_bytes(src.read_bytes())
    copied = pd.read_csv(dest)
    out = score_quantile_table(obs, copied, quantiles=Q, representation="gaussian")
    assert out["status"] == "ok"
    assert out["merge"]["one_to_one"] is True
    assert out["grouping"] == "STUDYID"
    assert out["mae_log"]["study_balanced"] == pytest.approx(0.0)
    assert out["rmse_log"]["study_balanced"] == pytest.approx(0.0)
    dec = decide_abstention(0.05, 0.08)
    assert dec["decision"] == "target_only"
    assert dest.read_bytes() == src.read_bytes()


def test_val_score_ignores_failed_blocks():
    assert val_score({"status": "failed"}) is None
    assert val_score({"status": "ok", "rmse_log": {"study_balanced": 0.2}}) == 0.2


def test_locked_budget_before_execution():
    cfg = load_yaml(FROZEN)
    budget = locked_budget(cfg)
    assert budget["execute"] is False
    assert budget["n_dataset_seeds"] == 2
    assert budget["n_train_seeds"] == 1
    assert budget["generate_seeds"] == [76001, 77010]
    assert budget["train_seeds"] == [85001]
    assert budget["donor_rng_seed"] == 91001
    assert budget["banks"] == ["matched", "both"]
    assert budget["densities"] == ["full", "sparse"]
    assert budget["sparse_target_train_n"] == 4
    assert budget["not_sparse_encoder"] is True
    assert budget["n_trained_cells"] == 34
    assert budget["n_model_fits"] == 68
    assert budget["additional_fits_for_abstention"] == 0
    assert budget["estimated_runtime_seconds"]["point"] == 180
    assert budget["compute_cap_seconds"] == 600
    assert budget["training_seed_variability_estimated"] is False
    assert budget["dataset_variability_estimated"] is True
    assert budget["abstention"]["threshold_from_test"] is False


def test_executor_refuses_preserved_real_design(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_share_repl\ngenerate_seed: 1\ntrain_seed: 2\n",
                   encoding="utf-8")
    with pytest.raises(ShareAbstainError, match="preserved"):
        execute_v3_share_abstain(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\nsource: {type: snowflake}\n",
        encoding="utf-8")
    with pytest.raises(ShareAbstainError, match="real-data"):
        execute_v3_share_abstain(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "design: {request: x.yaml}\n", encoding="utf-8")
    with pytest.raises(ShareAbstainError, match="design"):
        execute_v3_share_abstain(cfg)


def test_protected_snapshot_roundtrip():
    before = snapshot_protected()
    after = snapshot_protected()
    cmp = compare_snapshots(before, after)
    assert cmp["unchanged"] is True
    assert Path("runs/v3_share_repl/SUMMARY.json").is_file()
    assert before["files"]["runs/v3_share_repl/SUMMARY.json"] is not None
