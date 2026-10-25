"""Extension helpers: observed DAY features, weights, crossings, no invented calendars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from send_bw.tft_baselines import build_catboost_table, catboost_quantile_loss, history_summaries
from send_bw.tft_crossings import crossing_audit, curve_still_improving, rearrange_quantiles
from send_bw.tft_early_hist_extend import OUTPUT_DEFAULT, _refuse_pilot, extra_study_splits, load_yaml
from send_bw.tft_examples import (
    history_lead_report,
    mark_persistence_intervals_unavailable,
    paired_study_differences,
    score_aligned_quantiles,
)
from send_bw.tft_weighting import summarize_contributions


def test_config_predeclares_seeds_before_results():
    cfg = load_yaml("configs/tft_early_hist_extend.yaml")
    ev = cfg["benchmark"]
    assert ev["extra_train_seeds"] == [82011]
    assert ev["extra_split_seed"] == 33001
    assert cfg["generate_seed"] == 71001
    assert cfg["train_seed"] == 82001
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["pilot_dir"] == "runs/tft_early_hist_bench"
    assert cfg["output_dir"] != cfg["pilot_dir"]


def test_catboost_quantile_loss_uses_alpha_in_string():
    assert catboost_quantile_loss(0.1) == "Quantile:alpha=0.1"
    assert catboost_quantile_loss(0.9) == "Quantile:alpha=0.9"


def test_trend_uses_actual_day_diff_and_does_not_invent_slope():
    ok = history_summaries([7.0, 14.0], [5.0, 5.14], cutoff=14.0)
    assert ok["trend_missing"] == 0.0
    assert ok["recent_trend"] == pytest.approx((5.14 - 5.0) / (14.0 - 7.0))
    assert ok["history_span"] == pytest.approx(7.0)
    one = history_summaries([14.0], [5.0], cutoff=14.0)
    assert one["trend_missing"] == 1.0
    assert one["recent_trend"] == 0.0
    assert one["var_missing"] == 1.0
    same_day = history_summaries([10.0, 10.0], [5.0, 5.2], cutoff=14.0)
    assert same_day["trend_missing"] == 1.0
    assert same_day["recent_trend"] == 0.0


def test_catboost_rows_use_observed_days_only():
    from send_bw.data.features import FeatureSpec

    animals = pd.DataFrame({
        "STUDYID": ["S1", "S1"], "USUBJID": ["A1", "A2"], "SEX": ["M", "F"],
        "SPECIES": ["RAT", "RAT"], "STRAIN": ["SD", "SD"], "ROUTE": ["ORAL", "ORAL"],
        "COMPOUND_ID": ["C1", "C1"], "IS_CONTROL": [True, True],
        "PLANNED_DOSE": [0.0, 0.0], "INTERVAL_DAYS": [1.0, 1.0],
        "START_DAY": [1.0, 1.0], "END_DAY": [28.0, 28.0], "BASELINE_LOGW": [5.0, 4.8],
    })
    obs = pd.DataFrame({
        "STUDYID": ["S1"] * 5, "USUBJID": ["A1", "A1", "A1", "A2", "A2"],
        "DAY": [7.0, 14.0, 21.0, 10.0, 24.0],
        "LOGW": [5.1, 5.2, 5.3, 4.9, 5.1], "Y": [0.1, 0.2, 0.3, 0.1, 0.3],
    })
    cref = pd.DataFrame({"COMPOUND_ID": ["C1"], "D1": [0.0], "D2": [0.0], "D3": [0.0]})
    spec = FeatureSpec.fit(animals, obs, cref, "RAT")
    table, meta = build_catboost_table(
        obs, animals, spec, cref, cutoff=14.0, max_encoder_length=8, max_prediction_length=6)
    assert meta["fabricated_days"] is False
    assert meta["fabricated_slope"] is False
    days = set(zip(table["USUBJID"], table["target_day"], strict=True))
    assert ("A1", 21.0) in days
    assert ("A2", 24.0) in days
    assert all(d not in {17.0, 18.0} for _, d in days)
    a2 = table.loc[table["USUBJID"] == "A2"].iloc[0]
    assert a2["elapsed_from_last_obs"] == pytest.approx(14.0)
    assert a2["elapsed_from_origin"] == pytest.approx(10.0)


def test_rearrangement_is_sort_and_raw_crossings_are_classified():
    pred = pd.DataFrame({
        "USUBJID": ["A", "A", "B"],
        "STUDYID": ["S", "S", "S"],
        "DAY": [21.0, 28.0, 21.0],
        "q0.1": [1.0, 2.0, 3.0],
        "q0.5": [0.5, 2.1, 3.1],  # first row inner crossing
        "q0.9": [0.2, 2.2, 3.2],  # first row also endpoint reversed
    })
    fut = pd.DataFrame({
        "USUBJID": ["A", "A", "B"], "STUDYID": ["S", "S", "S"],
        "DAY": [21.0, 28.0, 21.0], "LOGW": [1.0, 2.0, 3.0],
        "LEAD_DAYS": [7.0, 14.0, 7.0],
    })
    raw = crossing_audit(pred, fut, levels=[0.1, 0.5, 0.9])
    assert raw["n_obs"] == 3
    assert raw["any_crossing"] == pytest.approx(1.0 / 3.0)
    assert raw["endpoint_reversed"] == pytest.approx(1.0 / 3.0)
    assert raw["inner_adjacent_crossing"] == pytest.approx(1.0 / 3.0)
    assert raw["pair_rates"]["q0.1>q0.9"] == pytest.approx(1.0 / 3.0)
    corr = rearrange_quantiles(pred, [0.1, 0.5, 0.9])
    assert corr["rearranged"].all()
    assert list(corr.loc[0, ["q0.1", "q0.5", "q0.9"]]) == [0.2, 0.5, 1.0]
    after = crossing_audit(corr, fut, levels=[0.1, 0.5, 0.9])
    assert after["any_crossing"] == 0.0


def test_curve_decision_uses_validation_only():
    still = curve_still_improving([
        {"epoch": 0, "val_loss": 1.0}, {"epoch": 1, "val_loss": 0.8},
        {"epoch": 2, "val_loss": 0.7},
    ])
    assert still["still_improving"] is True
    done = curve_still_improving([
        {"epoch": 0, "val_loss": 1.0}, {"epoch": 1, "val_loss": 0.5},
        {"epoch": 2, "val_loss": 0.6},
    ])
    assert done["still_improving"] is False
    assert done["best_epoch"] == 1


def test_effective_weight_uses_decoder_sums_not_window_counts_alone():
    windows = pd.DataFrame({
        "window_i": [0, 1, 2],
        "USUBJID": ["A", "B", "C"],
        "STUDYID": ["S1", "S1", "S2"],
        "n_decoder": [6, 6, 1],
        "decoder_weight_sum": [6.0, 6.0, 1.0],
        "encoder_length": [2, 2, 2],
    })
    rec = summarize_contributions(windows, sampler="uniform")
    assert rec["windows_per_study"]["S1"] == 2
    assert rec["window_count_share"]["S1"] == pytest.approx(2.0 / 3.0)
    assert rec["effective_loss_share_uniform_windows"]["S1"] == pytest.approx(12.0 / 13.0)
    assert rec["effective_loss_share_uniform_windows"]["S1"] != rec["window_count_share"]["S1"]
    # study sampler equalizes study probability; remaining imbalance is decoder length
    assert rec["expected_loss_share_if_study_sampler"]["S1"] == pytest.approx(6.0 / 7.0)


def test_persistence_intervals_marked_unavailable():
    block = {
        "coverage80": {"study_balanced": 0.0},
        "interval_width80": {"study_balanced": 0.0},
    }
    out = mark_persistence_intervals_unavailable(block)
    assert out["coverage80"]["unavailable"] is True
    assert out["coverage80"]["study_balanced"] is None


def test_score_adds_gram_mae_on_actual_pairs():
    fut = pd.DataFrame({
        "USUBJID": ["A", "B"], "STUDYID": ["S1", "S2"], "DAY": [21.0, 21.0],
        "LOGW": [np.log(100.0), np.log(200.0)],
    })
    pred = pd.DataFrame({
        "USUBJID": ["A", "B"], "DAY": [21.0, 21.0],
        "q0.1": [np.log(90.0), np.log(180.0)],
        "q0.5": [np.log(110.0), np.log(200.0)],
        "q0.9": [np.log(120.0), np.log(220.0)],
    })
    sc = score_aligned_quantiles(fut, pred, representation="test", quantiles=[0.1, 0.5, 0.9])
    assert sc["mae_gram"]["study_balanced"] == pytest.approx(5.0)
    assert sc["units"]["mae_gram"] == "grams"


def test_history_lead_report_distinguishes_caps_from_exercised():
    pack = {
        "cutoff": 14.0, "max_encoder_length": 8, "max_prediction_length": 6,
        "examples": pd.DataFrame({
            "n_encoder_used": [2, 2], "n_future_used": [1, 2],
            "n_encoder_available": [2, 3], "n_future_available": [1, 4],
            "truncated_encoder": [False, False], "truncated_future": [False, True],
            "lead_times": [[7.0], [7.0, 14.0]],
        }),
    }
    rep = history_lead_report(pack)
    assert rep["configured"]["max_encoder_length"] == 8
    assert rep["exercised"]["encoder_used_max"] == 2
    assert rep["exercised"]["lead_days_max"] == 14.0
    assert rep["exercised"]["n_animals_truncated_future"] == 1


def test_paired_diffs_are_study_level():
    a = [{"STUDYID": "S1", "mae_log": 0.10, "n_obs": 10},
         {"STUDYID": "S2", "mae_log": 0.20, "n_obs": 5}]
    b = [{"STUDYID": "S1", "mae_log": 0.12, "n_obs": 10},
         {"STUDYID": "S2", "mae_log": 0.16, "n_obs": 5}]
    out = paired_study_differences(a, b, metric="mae_log", name_a="to", name_b="po")
    assert out["n_studies"] == 2
    assert out["mean_delta"] == pytest.approx(-0.01)


def test_extra_split_keeps_donors_in_train_only():
    from send_bw.data.features import FeatureSpec
    from send_bw.models.base import ModelData

    rows = []
    for i, sid in enumerate(["SYNA-001", "SYNA-002", "SYNA-003", "SYNA-004", "SYNA-005"]):
        rows.append({
            "STUDYID": sid, "USUBJID": f"{sid}-1", "SEX": "M", "SPECIES": "RAT",
            "STRAIN": "SD", "ROUTE": "ORAL", "COMPOUND_ID": "C1", "IS_CONTROL": True,
            "PLANNED_DOSE": 0.0, "INTERVAL_DAYS": 1.0, "START_DAY": 1.0, "END_DAY": 28.0,
            "BASELINE_LOGW": 5.0, "REF_DATE": f"2000-01-{i + 1:02d}",
        })
    rows.append({
        "STUDYID": "SYNA-D001", "USUBJID": "D-1", "SEX": "M", "SPECIES": "DOG",
        "STRAIN": "BEAGLE", "ROUTE": "ORAL", "COMPOUND_ID": "C1", "IS_CONTROL": True,
        "PLANNED_DOSE": 0.0, "INTERVAL_DAYS": 1.0, "START_DAY": 1.0, "END_DAY": 28.0,
        "BASELINE_LOGW": 8.0, "REF_DATE": "2000-02-01",
    })
    animals = pd.DataFrame(rows)
    obs = pd.DataFrame({
        "STUDYID": animals["STUDYID"], "USUBJID": animals["USUBJID"],
        "DAY": [7.0] * len(animals), "LOGW": [5.0] * len(animals), "Y": [0.0] * len(animals),
    })
    cref = pd.DataFrame({"COMPOUND_ID": ["C1"], "D1": [0.0], "D2": [0.0], "D3": [0.0]})
    spec = FeatureSpec.fit(animals.loc[animals.SPECIES == "RAT"], obs, cref, "RAT")
    target = ModelData(animals.loc[animals.SPECIES == "RAT"].reset_index(drop=True),
                       obs.loc[obs.STUDYID.isin(animals.loc[animals.SPECIES == "RAT", "STUDYID"])].reset_index(drop=True),
                       cref, spec)
    pooled = ModelData(animals, obs, cref, spec)
    pack = extra_study_splits(target, pooled, seed=33001)
    po = pack["pooled_splits"]
    donors = po.loc[po["STUDYID"] == "SYNA-D001", "SPLIT"].tolist()
    assert donors == ["train"]
    test_ids = set(pack["target_splits"].loc[pack["target_splits"]["SPLIT"] == "test", "STUDYID"])
    assert "SYNA-D001" not in test_ids
    assert pack["seed"] == 33001


def test_refuse_pilot_and_v3_dirs(tmp_path):
    with pytest.raises(Exception, match="pilot|preserved|v3"):
        _refuse_pilot(Path("runs/tft_early_hist_bench"), Path("runs/tft_early_hist_bench"))
    with pytest.raises(Exception, match="preserved|v3"):
        _refuse_pilot(Path("runs/v3_official_seq"), Path("runs/tft_early_hist_bench"))
    _refuse_pilot(tmp_path / "fresh", Path("runs/tft_early_hist_bench"))
