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


def ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
          penalty: np.ndarray | None = None) -> np.ndarray:
    Xw = X * w[:, None]
    if penalty is None:
        pen = alpha * np.eye(X.shape[1])
    else:
        pen = np.diag(np.asarray(penalty, float))
    A = X.T @ Xw + pen
    return np.linalg.solve(A, Xw.T @ y)


def species_penalty(col_names: list[str], alpha: float, lambda_growth: float,
                    lambda_treatment: float) -> np.ndarray:
    out = []
    for n in col_names:
        if n.startswith("tb_SPECIES_"):
            out.append(float(alpha) * float(lambda_growth))
        elif n.startswith("eb_") and "SPECIES_" in n:
            out.append(float(alpha) * float(lambda_treatment))
        else:
            out.append(float(alpha))
    return np.asarray(out, float)


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
        sharing = getattr(train.spec, "sharing", "context_specific")
        regularized = sharing == "selective_regularized"
        self.design = MeanDesign.from_features(
            F, config.get("n_time_knots", 3), config.get("n_expo_knots", 2),
            species_treatment=regularized)
        X = self.design.matrix(F, fit_scale=True)
        y = train.obs["Y"].to_numpy(float)
        w = study_balanced_weights(train.obs["STUDYID"].to_numpy())
        Xv, yv, wv = self._X(validation)
        grid = config.get("ridge_alphas", [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
        lam_g = float(config.get("lambda_growth", 8.0 if regularized else 1.0))
        lam_t = float(config.get("lambda_treatment", 8.0 if regularized else 1.0))
        scores = {}
        for a in grid:
            pen = species_penalty(self.design.col_names, a, lam_g, lam_t) if regularized else None
            b = ridge(X, y, w, a, pen)
            scores[a] = wrmse(yv, Xv @ b, wv)
        alpha = min(scores, key=scores.get)
        pen0 = species_penalty(self.design.col_names, alpha, lam_g, lam_t) if regularized else None
        beta0 = ridge(X, y, w, alpha, pen0)
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
            b = ridge(X[~te], y[~te], study_balanced_weights(sid[~te]), alpha, pen0)
            R[te] = y[te] - X[te] @ b
        resid = pd.DataFrame({"STUDYID": sid, "USUBJID": train.obs["USUBJID"].to_numpy(),
                              "DAY": train.obs["DAY"].to_numpy(), "R": R})
        cov_resid = resid
        from send_bw.sharing import is_selective

        selective = is_selective(getattr(train.spec, "sharing", "context_specific"))
        if selective and "SPECIES" in train.animals.columns:
            tgt_ids = set(train.animals.loc[
                train.animals["SPECIES"].astype(str).str.upper() == str(train.spec.species).upper(),
                "USUBJID"].astype(str))
            cov_resid = resid[resid["USUBJID"].astype(str).isin(tgt_ids)]
            if cov_resid.empty:
                raise ValueError("selective spline covariance: no target-species residuals")
        self.cov = HierarchicalCovariance.fit(cov_resid)
        # study bootstrap members (one member is applied to a whole simulated study)
        self.members = [beta0]
        for _ in range(int(config.get("n_bootstrap", 10))):
            pick = rng.choice(studies, size=len(studies), replace=True)
            counts = pd.Series(pick).value_counts()
            mult = pd.Series(sid).map(counts).fillna(0).to_numpy()
            keep = mult > 0
            wb = study_balanced_weights(sid[keep]) * mult[keep]
            self.members.append(ridge(X[keep], y[keep], wb, alpha, pen0))
        moment_split = str(config.get("increment_moment_split") or "").lower()
        increment_moment = None
        if moment_split:
            from send_bw.evaluation.increment_moment import match_increment_moment

            if moment_split == "validation":
                moment_data = validation
            elif moment_split == "train":
                moment_data = train
            else:
                raise ValueError("increment_moment_split must be 'validation' or 'train', not a test split")
            increment_moment = match_increment_moment(
                self, moment_data, split_name=moment_split,
                n_draws=int(config.get("increment_moment_draws", 40)),
                seed=int(config.get("seed", 0)) + 31,
            )
        self.history = {"alpha": alpha, "validation_rmse_by_alpha": {str(k): v for k, v in scores.items()},
                        "train_rmse": wrmse(y, X @ beta0, w), "validation_rmse": scores[alpha],
                        "oof_rmse": wrmse(y, y - R, w), "n_params": int(X.shape[1]),
                        "n_members": len(self.members), "covariance": self.cov.to_dict(),
                        "covariance_fit_on": (
                            "target_species_oof" if selective else "train_oof_all_species"
                        ),
                        "lambda_growth": lam_g if regularized else None,
                        "lambda_treatment": lam_t if regularized else None,
                        "increment_moment": increment_moment}
        if selective:
            from send_bw.sharing import selective_record

            self.history["selective_sharing"] = selective_record(
                "spline", "regularized" if regularized else "v1")
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
