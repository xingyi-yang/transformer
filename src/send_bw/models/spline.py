"""Conventional hierarchical spline longitudinal model (benchmark A).

Mean: ridge-penalized B-spline time basis interacted with sex, centered baseline and strain, plus
planned-exposure spline terms interacted with log dose, compound descriptors, dosing interval,
route and sex, and post-dosing recovery terms. Study-balanced weights.
Covariance: HierarchicalCovariance fitted on study-level out-of-fold residuals.
Parameter uncertainty: study-bootstrap refits of the mean (covariance parameters are not
bootstrapped; this is a documented limitation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData, ResponseModel, study_balanced_weights
from send_bw.models.covariance import HierarchicalCovariance
from send_bw.models.design import MeanDesign
from send_bw.models.gaussian_layer import GaussianLayerMixin


def ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    Xw = X * w[:, None]
    A = X.T @ Xw + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, Xw.T @ y)


def wrmse(y, yhat, w) -> float:
    return float(np.sqrt(np.sum(w * (y - yhat) ** 2) / np.sum(w)))


def context_features(spec: FeatureSpec, ctx: pd.DataFrame, times: np.ndarray, cref: pd.DataFrame) -> pd.DataFrame:
    ctx = ctx.copy()
    if "USUBJID" not in ctx:
        ctx["USUBJID"] = [f"A{i}" for i in range(len(ctx))]
    if "STUDYID" not in ctx:
        ctx["STUDYID"] = "NEW"
    n, T = len(ctx), len(times)
    obs = pd.DataFrame({"STUDYID": np.repeat(ctx["STUDYID"].to_numpy(), T),
                        "USUBJID": np.repeat(ctx["USUBJID"].to_numpy(), T), "DAY": np.tile(times, n)})
    return spec.observation_features(obs, ctx.reset_index(drop=True), cref)


class SplineModel(GaussianLayerMixin, ResponseModel):
    name = "spline"

    def __init__(self):
        self.spec: FeatureSpec | None = None
        self.design: MeanDesign | None = None
        self.members: list[np.ndarray] = []
        self.cov = HierarchicalCovariance()
        self.compound_ref = pd.DataFrame()
        self.history: dict[str, Any] = {}

    @property
    def n_members(self) -> int:
        return len(self.members)

    def _X(self, data: ModelData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = data.spec.observation_features(data.obs, data.animals, data.compound_ref)
        return self.design.matrix(F), data.obs["Y"].to_numpy(float), study_balanced_weights(data.obs["STUDYID"].to_numpy())

    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        self.spec, self.compound_ref = train.spec, train.compound_ref
        F = train.spec.observation_features(train.obs, train.animals, train.compound_ref)
        self.design = MeanDesign.from_features(F, config.get("n_time_knots", 3), config.get("n_expo_knots", 2))
        X = self.design.matrix(F, fit_scale=True)
        y = train.obs["Y"].to_numpy(float)
        w = study_balanced_weights(train.obs["STUDYID"].to_numpy())
        Xv, yv, wv = self._X(validation)
        grid = config.get("ridge_alphas", [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
        scores = {}
        for a in grid:
            b = ridge(X, y, w, a)
            scores[a] = wrmse(yv, Xv @ b, wv)
        alpha = min(scores, key=scores.get)
        beta0 = ridge(X, y, w, alpha)
        # study-level out-of-fold residuals for the covariance layer
        studies = np.array(train.studies)
        rng = np.random.default_rng(config.get("seed", 0))
        perm = rng.permutation(studies)
        k = min(int(config.get("oof_folds", 5)), len(studies))
        folds = np.array_split(perm, k)
        R = np.zeros_like(y)
        sid = train.obs["STUDYID"].to_numpy()
        for f in folds:
            te = np.isin(sid, f)
            b = ridge(X[~te], y[~te], study_balanced_weights(sid[~te]), alpha)
            R[te] = y[te] - X[te] @ b
        resid = pd.DataFrame({"STUDYID": sid, "USUBJID": train.obs["USUBJID"].to_numpy(),
                              "DAY": train.obs["DAY"].to_numpy(), "R": R})
        self.cov = HierarchicalCovariance.fit(resid)
        # study bootstrap members (one member is applied to a whole simulated study)
        self.members = [beta0]
        for _ in range(int(config.get("n_bootstrap", 10))):
            pick = rng.choice(studies, size=len(studies), replace=True)
            counts = pd.Series(pick).value_counts()
            mult = pd.Series(sid).map(counts).fillna(0).to_numpy()
            keep = mult > 0
            wb = study_balanced_weights(sid[keep]) * mult[keep]
            self.members.append(ridge(X[keep], y[keep], wb, alpha))
        self.history = {"alpha": alpha, "validation_rmse_by_alpha": {str(k): v for k, v in scores.items()},
                        "train_rmse": wrmse(y, X @ beta0, w), "validation_rmse": scores[alpha],
                        "oof_rmse": wrmse(y, y - R, w), "n_params": int(X.shape[1]),
                        "n_members": len(self.members), "covariance": self.cov.to_dict()}
        return self.history

    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        times = np.asarray(times, float)
        F = context_features(self.spec, context, times, self.compound_ref)
        X = self.design.matrix(F)
        beta = self.members[0 if member is None else member]
        yhat = (X @ beta).reshape(len(context), len(times))
        return context["BASELINE_LOGW"].to_numpy(float)[:, None] + yhat

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "members.npy", np.stack(self.members))
        self.compound_ref.to_csv(path / "compound_reference.csv", index=False)
        (path / "model.json").write_text(json.dumps({
            "name": self.name, "spec": self.spec.to_dict(), "design": self.design.to_dict(),
            "cov": self.cov.to_dict(), "history": self.history}, indent=2, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SplineModel:
        path = Path(path)
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        m = cls()
        m.spec = FeatureSpec.from_dict(d["spec"])
        m.design = MeanDesign.from_dict(d["design"])
        m.cov = HierarchicalCovariance.from_dict(d["cov"])
        m.history = d["history"]
        m.members = list(np.load(path / "members.npy"))
        m.compound_ref = pd.read_csv(path / "compound_reference.csv")
        return m
