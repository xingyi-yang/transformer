"""Forecast examples must come from observed DAY values, not invented calendars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from send_bw.tft_examples import (
    EXCLUDE_NO_ENCODER,
    EXCLUDE_NO_FUTURE,
    animal_windows,
    assert_windows_match_obs,
    construct_forecast_examples,
    examples_to_tables,
)


def _obs() -> pd.DataFrame:
    # Irregular, animal-specific missing visits. No shared grid assumed.
    rows = [
        ("S1", "A1", 3.0, 5.0),
        ("S1", "A1", 7.0, 5.1),
        ("S1", "A1", 14.0, 5.2),
        ("S1", "A1", 21.0, 5.3),
        ("S1", "A1", 28.0, 5.4),
        ("S1", "A2", 7.0, 4.9),
        ("S1", "A2", 10.0, 5.0),
        ("S1", "A2", 24.0, 5.2),
        ("S2", "B1", 7.0, 6.0),  # no post-cutoff target
        ("S2", "C1", 21.0, 6.1),  # no encoder
    ]
    return pd.DataFrame(rows, columns=["STUDYID", "USUBJID", "DAY", "LOGW"])


def _animals() -> pd.DataFrame:
    return pd.DataFrame({
        "USUBJID": ["A1", "A2", "B1", "C1"],
        "STUDYID": ["S1", "S1", "S2", "S2"],
        "SPECIES": ["RAT", "RAT", "RAT", "RAT"],
        "SEX": ["M", "F", "M", "F"],
    })


def test_windows_use_actual_days_and_drop_unmeasured():
    w = animal_windows(_obs(), "A1", cutoff=14.0, max_encoder_length=8, max_prediction_length=6)
    assert w["eligible"] is True
    assert w["encoder_days"] == [3.0, 7.0, 14.0]
    assert w["target_days"] == [21.0, 28.0]
    assert w["lead_times"] == [7.0, 14.0]
    assert w["last_encoder_day"] == 14.0
    assert 17.0 not in w["target_days"]  # never observed


def test_encoder_and_horizon_caps_do_not_invent_days():
    w = animal_windows(_obs(), "A1", cutoff=14.0, max_encoder_length=2, max_prediction_length=1)
    assert w["encoder_days"] == [7.0, 14.0]
    assert w["target_days"] == [21.0]
    assert w["truncated_encoder"] is True
    assert w["truncated_future"] is True
    assert w["unused_future_days"] == [28.0]


def test_animal_specific_missing_visits_are_preserved():
    w = animal_windows(_obs(), "A2", cutoff=14.0, max_encoder_length=8, max_prediction_length=6)
    assert w["encoder_days"] == [7.0, 10.0]
    assert w["target_days"] == [24.0]
    assert w["lead_times"] == [14.0]
    assert 14.0 not in w["encoder_days"]
    assert 21.0 not in w["target_days"]


def test_exclusions_have_reasons():
    pack = construct_forecast_examples(
        _obs(), _animals(), cutoff=14.0, max_encoder_length=8, max_prediction_length=6)
    assert pack["n_eligible"] == 2
    reasons = set(pack["exclusions"]["exclude_reason"])
    assert EXCLUDE_NO_FUTURE in reasons
    assert EXCLUDE_NO_ENCODER in reasons


def test_tables_only_contain_observed_pairs():
    pack = construct_forecast_examples(
        _obs(), _animals(), cutoff=14.0, max_encoder_length=8, max_prediction_length=6)
    hist, fut = examples_to_tables(pack)
    obs = _obs()
    for _, r in pd.concat([hist, fut], ignore_index=True).iterrows():
        hit = obs[(obs.USUBJID == r.USUBJID) & np.isclose(obs.DAY, r.DAY)]
        assert len(hit) == 1
        assert hit.iloc[0]["LOGW"] == pytest.approx(r.LOGW)
    assert_windows_match_obs(obs, pack, "A2")


def test_another_animals_days_are_not_targets():
    pack = construct_forecast_examples(
        _obs(), _animals(), cutoff=14.0, max_encoder_length=8, max_prediction_length=6)
    a2 = pack["examples"].loc[pack["examples"]["USUBJID"] == "A2"].iloc[0]
    assert 21.0 not in a2["target_days"]
    assert 28.0 not in a2["target_days"]
