"""Errata and counting rules for TFT confirmation and training logs.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from send_bw.models.tft import count_completed_epochs

# Previously reported roles. Confirmation 34011 test studies were new as test,
# but confirmation *training* included official and extra-split test studies.
REPORTED_ROLES: dict[str, list[str]] = {
    "official_train": ["SYNA-002", "SYNA-003", "SYNA-004", "SYNA-006", "SYNA-007", "SYNA-009"],
    "official_validation": ["SYNA-005", "SYNA-012"],
    "official_test": ["SYNA-010", "SYNA-011"],
    "official_compound_holdout": ["SYNA-001", "SYNA-008"],
    "extra_33001_test": ["SYNA-007", "SYNA-010"],
    "extra_33001_validation": ["SYNA-004", "SYNA-012"],
    "confirm_34011_test": ["SYNA-003", "SYNA-004"],
    "confirm_34011_validation": ["SYNA-001", "SYNA-005"],
    "confirm_34011_train": [
        "SYNA-002", "SYNA-006", "SYNA-007", "SYNA-008",
        "SYNA-009", "SYNA-010", "SYNA-011", "SYNA-012",
    ],
}

CONFIRM_FORBIDDEN_PLACED_IN_TRAIN = ["SYNA-007", "SYNA-010", "SYNA-011"]


def roles_for_study(study_id: str) -> list[str]:
    sid = str(study_id)
    return [name for name, ids in REPORTED_ROLES.items() if sid in ids]


def confirm_34011_erratum(splits: pd.DataFrame | None = None) -> dict[str, Any]:
    """The 34011 bank is not an untouched confirmation experiment.

    Test studies SYNA-003/004 had not been reported as test before. Training
    and calibration still used previously reported holdouts.
    """
    if splits is None:
        by = {
            "train": list(REPORTED_ROLES["confirm_34011_train"]),
            "validation": list(REPORTED_ROLES["confirm_34011_validation"]),
            "test": list(REPORTED_ROLES["confirm_34011_test"]),
        }
    else:
        by = {
            name: sorted(splits.loc[splits["SPLIT"] == name, "STUDYID"].astype(str).tolist())
            for name in ("train", "validation", "test")
        }
    leaked_train = sorted(set(by.get("train") or []) & set(CONFIRM_FORBIDDEN_PLACED_IN_TRAIN))
    leaked_val = sorted(set(by.get("validation") or []) & set(
        REPORTED_ROLES["official_test"] + REPORTED_ROLES["official_validation"]
        + REPORTED_ROLES["extra_33001_test"]))
    return {
        "status": "erratum",
        "unsupported_claim": (
            "Confirmation 34011 was an untouched study holdout. Previously "
            "reported test studies were excluded from the confirmation experiment."
        ),
        "supported_claim": (
            "SYNA-003 and SYNA-004 had not been reported as test studies before. "
            "They were official training studies. Confirmation *test* therefore "
            "used new test labels, but confirmation training and validation were "
            "not holdout-free."
        ),
        "forbidden_holdouts_in_train": leaked_train,
        "previously_reported_in_validation": leaked_val,
        "splits": by,
        "roles": {sid: roles_for_study(sid) for sid in
                  sorted({s for ids in by.values() for s in ids})},
        "official_checkpoints": (
            "Official TFT checkpoints were trained on official train "
            "(including SYNA-003/004). Confirm 34011 models were fit fresh "
            "and did not load those weights. Confirm pooled/fine-tune train "
            "sets still contained SYNA-007/010/011."
        ),
        "previous_finetune_default": (
            "PR #22 selected tft_finetune on official validation interval score. "
            "That default failed confirmation: target-only was better and more "
            "stable on 34011 test studies. Fine-tune and CQR stay experimental."
        ),
        "do_not_call_untouched": True,
    }


def series_versus_animals(animals: pd.DataFrame, obs: pd.DataFrame, *,
                          min_encoder_length: int = 1) -> dict[str, Any]:
    """214 animals vs 211 TFT series: short histories fail the encoder filter."""
    min_len = int(min_encoder_length) + 1
    n_animals = int(animals["USUBJID"].nunique())
    counts = obs.groupby(obs["USUBJID"].astype(str)).size()
    keep = counts[counts >= min_len]
    short = counts[counts < min_len]
    dropped = []
    meta = animals.drop_duplicates("USUBJID")
    for uid, n in short.items():
        row = meta.loc[meta["USUBJID"].astype(str) == str(uid)]
        study = str(row["STUDYID"].iloc[0]) if len(row) and "STUDYID" in row.columns else None
        dropped.append({"USUBJID": str(uid), "STUDYID": study, "n_obs": int(n)})
    return {
        "n_animals_in_split": n_animals,
        "n_tft_series": int(len(keep)),
        "n_dropped_short_series": int(len(short)),
        "min_encoder_length": int(min_encoder_length),
        "min_visits_required": min_len,
        "dropped": dropped,
        "note": (
            "n_animals_in_split counts unique USUBJID on the split. n_tft_series "
            "counts animals with at least min_encoder_length+1 observed visits, "
            "the TimeSeriesDataSet filter. Official train is 214 animals / 211 "
            "series because three animals have a single DAY=7 visit."
        ),
    }


def epoch_budget_versus_logged(curve_rows: list[dict[str, Any]], *,
                               max_epochs: int,
                               lightning_current_epoch: int | None = None,
                               finetune_max_epochs: int | None = None) -> dict[str, Any]:
    """16/8 budgets were stored as 17/9 via current_epoch + 1."""
    pack = count_completed_epochs(curve_rows, lightning_current_epoch)
    pack.update({
        "max_epochs_budget": int(max_epochs),
        "finetune_max_epochs_budget": None if finetune_max_epochs is None else int(finetune_max_epochs),
        "official_example": {
            "budget": 16,
            "logged_train_epochs": "0..15 (16 steps)",
            "lightning_current_epoch_after_fit": 16,
            "stored_epochs_run": 17,
            "finetune_budget": 8,
            "stored_finetune_epochs_run": 9,
        },
    })
    return pack


def cqr_metric_policy() -> dict[str, Any]:
    """Pinball on CQR tables uses shifted endpoints, not the TFT QuantileLoss."""
    return {
        "q0_5": "unchanged; MAE and median pinball match the raw table",
        "q0_1_q0_9": "shifted by ±margin; interval score, coverage, and width use these",
        "mean_quantile_loss_on_calibrated": (
            "Average pinball over 0.1/0.25/0.5/0.75/0.9 after the shift. "
            "That is not the training QuantileLoss and is not comparable to raw pinball "
            "as a proper score for the original quantile head."
        ),
        "report": "raw_* from original quantiles; cqr_* from calibrated endpoints",
        "calibration_data": "separate calibration split; not validation, not test",
        "experimental": True,
    }
