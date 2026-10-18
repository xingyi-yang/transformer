"""Shared response-model interface.

All models predict Y(t) = log W(t) - log W_baseline(observed) for t > baseline, conditional on
the observed baseline (also a covariate), sex, biology and the PLANNED regimen. Outputs are
returned on the log-weight scale Z = baseline + Y.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec

REGIMEN_COLS = ["COMPOUND_ID", "IS_CONTROL", "PLANNED_DOSE", "ROUTE", "INTERVAL_DAYS", "START_DAY", "END_DAY"]


@dataclass
class ModelData:
    animals: pd.DataFrame  # eligible animals incl. planned regimen columns
    obs: pd.DataFrame  # post-baseline observations with Y
    compound_ref: pd.DataFrame
    spec: FeatureSpec

    def subset(self, study_ids) -> ModelData:
        ids = set(study_ids)
        return ModelData(self.animals[self.animals["STUDYID"].isin(ids)].reset_index(drop=True),
                         self.obs[self.obs["STUDYID"].isin(ids)].reset_index(drop=True), self.compound_ref, self.spec)

    @property
    def studies(self) -> list[str]:
        return sorted(self.animals["STUDYID"].unique())


def merge_context(context: pd.DataFrame, regimens: pd.DataFrame) -> pd.DataFrame:
    """Attach planned regimen columns to animals via ARM_KEY (many-to-one, no inflation)."""
    ctx = context.drop(columns=[c for c in REGIMEN_COLS if c in context.columns])
    reg = regimens[["ARM_KEY"] + REGIMEN_COLS].drop_duplicates("ARM_KEY")
    out = ctx.merge(reg, on="ARM_KEY", how="left", validate="many_to_one")
    if len(out) != len(ctx):
        raise ValueError("row inflation while merging regimens")
    if out["IS_CONTROL"].isna().any():
        raise ValueError("animals reference arms without planned regimen")
    out["IS_CONTROL"] = out["IS_CONTROL"].astype(bool)
    return out


def study_balanced_weights(study_ids: np.ndarray) -> np.ndarray:
    """Each study contributes equal total weight; weights average to 1."""
    s = pd.Series(study_ids)
    n = s.map(s.value_counts())
    w = 1.0 / n.to_numpy(float)
    return w * len(w) / w.sum()


class ResponseModel(abc.ABC):
    name: str = "abstract"
    task: str = "pre_study"

    @abc.abstractmethod
    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]: ...

    @abc.abstractmethod
    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        """Conditional mean of Z at times for each animal: (n_animals, n_times)."""

    @abc.abstractmethod
    def predict_marginals(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame: ...

    @abc.abstractmethod
    def sample_study(self, context: pd.DataFrame, regimens: pd.DataFrame, times: np.ndarray, n_draws: int,
                     seed: int) -> dict[str, np.ndarray]:
        """Joint trajectories for complete hypothetical studies.

        One parameter realization per simulated study (draw), a new study effect per draw and
        new animal effects per animal. Returns {'z','mean'} of shape (n_draws, n_animals, n_times).
        """

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> ResponseModel: ...
