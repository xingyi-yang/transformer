"""Learned early-history baselines used by the TFT extension.

Spline/GAMM: the existing ridge B-spline mixed-effects model (SplineModel).
It uses flexible time curves, biology, planned regimen, and hierarchical
study/animal residuals. Held-out personalization uses only the supplied
encoder history (GaussianLayerMixin._condition). Future target weights are
never passed in.

Quantile CatBoost: one row per animal × forecast origin × observed future
target day. History summaries use actual DAY differences. A slope is not
invented when fewer than two encoder visits have a positive span.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData
from send_bw.tft_examples import construct_forecast_examples

CATBOOST_QUANTILES = [0.1, 0.5, 0.9]
CAT_COLS = ["SEX", "STRAIN", "SPECIES", "ROUTE", "COMPOUND_ID", "CTRL"]
NUM_COLS = [
    "ZB_C", "LD", "INV_INTERVAL", "START", "END", "D1", "D2", "D3",
    "latest_logw", "latest_weight_g", "logw_std", "n_encoder", "history_span",
    "recent_trend", "trend_missing", "var_missing",
    "origin_day", "last_obs_day", "target_day",
    "elapsed_from_origin", "elapsed_from_last_obs",
]


def catboost_quantile_loss(q: float) -> str:
    """CatBoost 1.2 takes the quantile level in the loss string, not quantile_alpha=."""
    return f"Quantile:alpha={float(q)}"


def catboost_status() -> dict[str, Any]:
    if importlib.util.find_spec("catboost") is None:
        return {"ok": False, "reason": "optional dependency 'catboost' is not installed",
                "blocker": "pip extra catboost"}
    import catboost

    return {"ok": True, "version": getattr(catboost, "__version__", "present")}


def history_summaries(encoder_days: list[float], encoder_logw: list[float],
                      *, cutoff: float) -> dict[str, float]:
    """Summaries from permitted encoder rows only. Uses actual DAY values."""
    days = np.asarray(encoder_days, float)
    logw = np.asarray(encoder_logw, float)
    n = int(len(days))
    if n == 0:
        raise ValueError("history summaries require at least one encoder visit")
    last_day = float(days[-1])
    latest = float(logw[-1])
    span = float(days[-1] - days[0]) if n >= 2 else 0.0
    if n >= 2:
        std = float(np.std(logw, ddof=1)) if n >= 3 else float(np.std(logw, ddof=0))
        var_missing = 0.0
    else:
        std = 0.0
        var_missing = 1.0
    if n >= 2 and span > 0:
        # most recent adjacent pair; denominator is the actual DAY difference
        recent_trend = float((logw[-1] - logw[-2]) / (days[-1] - days[-2]))
        trend_missing = 0.0
    else:
        recent_trend = 0.0
        trend_missing = 1.0
    return {
        "latest_logw": latest,
        "latest_weight_g": float(np.exp(latest)),
        "logw_std": std,
        "n_encoder": float(n),
        "history_span": span,
        "recent_trend": recent_trend,
        "trend_missing": trend_missing,
        "var_missing": var_missing,
        "origin_day": float(cutoff),
        "last_obs_day": last_day,
    }


def _animal_static(animals: pd.DataFrame, spec: FeatureSpec, cref: pd.DataFrame,
                   usubjid: str) -> dict[str, Any]:
    row = animals.loc[animals["USUBJID"].astype(str) == str(usubjid)]
    if row.empty:
        raise KeyError(usubjid)
    a = row.iloc[[0]].reset_index(drop=True)
    af = spec.animal_features(a, cref).iloc[0]
    ctrl = "C" if bool(a["IS_CONTROL"].iloc[0]) else "A"
    out = {
        "SEX": str(a["SEX"].iloc[0]),
        "STRAIN": str(a["STRAIN"].iloc[0]) if "STRAIN" in a.columns else "UNK",
        "SPECIES": str(a["SPECIES"].iloc[0]) if "SPECIES" in a.columns else str(spec.species),
        "ROUTE": str(a["ROUTE"].iloc[0]) if "ROUTE" in a.columns else "UNK",
        "COMPOUND_ID": str(a["COMPOUND_ID"].iloc[0]) if "COMPOUND_ID" in a.columns else "UNK",
        "CTRL": ctrl,
    }
    for c in ("ZB_C", "LD", "INV_INTERVAL", "START", "END", "D1", "D2", "D3"):
        out[c] = float(af[c]) if c in af.index else 0.0
    return out


def example_feature_rows(rec: dict[str, Any], animals: pd.DataFrame, spec: FeatureSpec,
                         cref: pd.DataFrame, *, cutoff: float) -> list[dict[str, Any]]:
    """One feature row per observed future target day of an example."""
    hist = history_summaries(rec["encoder_days"], rec["encoder_logw"], cutoff=cutoff)
    static = _animal_static(animals, spec, cref, rec["USUBJID"])
    rows = []
    for d, z, lead in zip(rec["target_days"], rec["target_logw"], rec["lead_times"], strict=True):
        item = {
            "USUBJID": rec["USUBJID"],
            "STUDYID": rec["STUDYID"],
            **static,
            **hist,
            "target_day": float(d),
            "elapsed_from_origin": float(d) - float(cutoff),
            "elapsed_from_last_obs": float(lead),
            "y_logw": float(z),
        }
        rows.append(item)
    return rows


def build_catboost_table(obs: pd.DataFrame, animals: pd.DataFrame, spec: FeatureSpec,
                         cref: pd.DataFrame, *, cutoff: float, max_encoder_length: int,
                         max_prediction_length: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rows from observed DAY windows only. One official cutoff origin per animal."""
    pack = construct_forecast_examples(
        obs, animals, cutoff=cutoff, max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length)
    recs = []
    for rec in pack["examples"].to_dict("records"):
        recs.extend(example_feature_rows(rec, animals, spec, cref, cutoff=cutoff))
    table = pd.DataFrame(recs)
    meta = {
        "origins": "cutoff",
        "cutoff": float(cutoff),
        "n_rows": int(len(table)),
        "n_animals": int(table["USUBJID"].nunique()) if len(table) else 0,
        "n_studies": int(table["STUDYID"].nunique()) if len(table) else 0,
        "fabricated_days": False,
        "fabricated_slope": False,
        "n_trend_missing": int(table["trend_missing"].sum()) if len(table) else 0,
        "note": ("Forecast origin is the configured cutoff. Encoder and targets are "
                 "that animal's observed DAY values. Trend is missing unless two "
                 "encoder visits have a positive DAY span."),
    }
    return table, meta


class QuantileCatBoostModel:
    """CatBoost MultiQuantile on early-history tabular rows. Not a TFT."""

    name = "catboost_quantile"
    quantiles = list(CATBOOST_QUANTILES)

    def __init__(self):
        self.models: dict[float, Any] = {}
        self.spec: FeatureSpec | None = None
        self.compound_ref = pd.DataFrame()
        self.cutoff: float = 14.0
        self.max_encoder_length: int = 8
        self.max_prediction_length: int = 6
        self.history: dict[str, Any] = {}
        self.feature_names: list[str] = list(CAT_COLS + NUM_COLS)
        self.cat_cols: list[str] = list(CAT_COLS)

    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        status = catboost_status()
        if not status["ok"]:
            raise RuntimeError(status["reason"])
        from catboost import CatBoostRegressor, Pool

        self.spec, self.compound_ref = train.spec, train.compound_ref
        self.cutoff = float(config.get("encoder_window_day", config.get("cutoff", 14)))
        self.max_encoder_length = int(config.get("max_encoder_length", 8))
        self.max_prediction_length = int(config.get("max_prediction_length", 6))
        self.quantiles = [float(q) for q in config.get("quantiles", CATBOOST_QUANTILES)]
        kw = dict(
            cutoff=self.cutoff, max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length)
        tr, tr_meta = build_catboost_table(train.obs, train.animals, train.spec, train.compound_ref, **kw)
        va, va_meta = build_catboost_table(
            validation.obs, validation.animals, train.spec, train.compound_ref, **kw)
        if tr.empty:
            raise RuntimeError("CatBoost training table is empty")
        x_cols = [c for c in self.feature_names if c in tr.columns]
        self.feature_names = x_cols
        cat_idx = [x_cols.index(c) for c in self.cat_cols if c in x_cols]
        seed = int(config.get("seed", 0))
        common = dict(
            iterations=int(config.get("iterations", 200)),
            depth=int(config.get("depth", 4)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            l2_leaf_reg=float(config.get("l2_leaf_reg", 3.0)),
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        early = int(config.get("early_stopping_rounds", 20))
        self.models = {}
        curves = {}
        for q in self.quantiles:
            m = CatBoostRegressor(**common, loss_function=catboost_quantile_loss(q))
            train_pool = Pool(tr[x_cols], tr["y_logw"].to_numpy(float), cat_features=cat_idx)
            if len(va):
                val_pool = Pool(va[x_cols], va["y_logw"].to_numpy(float), cat_features=cat_idx)
                m.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early, use_best_model=True)
            else:
                m.fit(train_pool)
            self.models[float(q)] = m
            curves[str(q)] = {
                "best_iteration": int(m.get_best_iteration() or m.tree_count_),
                "tree_count": int(m.tree_count_),
            }
        self.history = {
            "name": self.name,
            "library": "catboost.CatBoostRegressor",
            "quantiles": self.quantiles,
            "loss": "separate Quantile heads (not MultiQuantile; one model per level)",
            "train_rows": tr_meta,
            "val_rows": va_meta,
            "curves": curves,
            "features": x_cols,
            "cat_features": [c for c in self.cat_cols if c in x_cols],
            "origins": "cutoff",
            "not_tft": True,
            "information": (
                "Uses permitted biology, planned regimen, and encoder summaries. "
                "Does not see future target weights as features."
            ),
        }
        return self.history

    def predict_quantiles(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.models or self.spec is None:
            raise RuntimeError("CatBoost baseline is not fitted")
        if history is None or not len(history):
            raise ValueError("CatBoost early-history requires observed encoder rows")
        times = np.asarray(times, float).reshape(-1)
        recs = []
        ctx = context.drop_duplicates("USUBJID")
        for _, animal in ctx.iterrows():
            u = str(animal["USUBJID"])
            hh = history.loc[history["USUBJID"].astype(str) == u].sort_values("DAY")
            if hh.empty:
                continue
            enc_days = [float(x) for x in hh["DAY"].to_numpy(float)]
            enc_logw = [float(x) for x in hh["LOGW"].to_numpy(float)]
            hist = history_summaries(enc_days, enc_logw, cutoff=self.cutoff)
            static = _animal_static(ctx, self.spec, self.compound_ref, u)
            last = hist["last_obs_day"]
            for d in times:
                recs.append({
                    "USUBJID": u,
                    "STUDYID": animal["STUDYID"] if "STUDYID" in animal.index else None,
                    **static,
                    **hist,
                    "target_day": float(d),
                    "elapsed_from_origin": float(d) - float(self.cutoff),
                    "elapsed_from_last_obs": float(d) - float(last),
                })
        if not recs:
            return pd.DataFrame(columns=["USUBJID", "DAY", *[f"q{q}" for q in self.quantiles]])
        feats = pd.DataFrame(recs)
        out = feats[["USUBJID"]].copy()
        out["DAY"] = feats["target_day"].to_numpy(float)
        if "STUDYID" in feats.columns:
            out["STUDYID"] = feats["STUDYID"]
        x = feats[self.feature_names]
        for q, m in self.models.items():
            out[f"q{q}"] = np.asarray(m.predict(x), float)
        out["point_definition"] = "catboost_quantile_q0.5"
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        import json

        for q, m in self.models.items():
            m.save_model(str(path / f"q{q}.cbm"))
        self.compound_ref.to_csv(path / "compound_reference.csv", index=False)
        (path / "model.json").write_text(json.dumps({
            "name": self.name, "spec": self.spec.to_dict() if self.spec else {},
            "quantiles": self.quantiles, "cutoff": self.cutoff,
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "feature_names": self.feature_names, "cat_cols": self.cat_cols,
            "history": self.history,
        }, indent=1, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QuantileCatBoostModel:
        path = Path(path)
        import json

        from catboost import CatBoostRegressor

        from send_bw.data.features import FeatureSpec

        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        m = cls()
        m.spec = FeatureSpec.from_dict(d["spec"]) if d.get("spec") else None
        m.quantiles = [float(x) for x in d["quantiles"]]
        m.cutoff = float(d["cutoff"])
        m.max_encoder_length = int(d["max_encoder_length"])
        m.max_prediction_length = int(d["max_prediction_length"])
        m.feature_names = list(d["feature_names"])
        m.cat_cols = list(d["cat_cols"])
        m.history = d.get("history") or {}
        m.compound_ref = pd.read_csv(path / "compound_reference.csv")
        m.models = {}
        for q in m.quantiles:
            cb = CatBoostRegressor()
            cb.load_model(str(path / f"q{q}.cbm"))
            m.models[float(q)] = cb
        return m


def spline_information_set() -> dict[str, Any]:
    return {
        "model": "spline",
        "role": "nonlinear_spline_mixed_effects_gamm",
        "mean": "ridge-penalized B-spline time and exposure design (MeanDesign)",
        "structure": "HierarchicalCovariance study/animal residuals",
        "held_out_effects": (
            "GaussianLayerMixin._condition uses only the supplied encoder residuals. "
            "Future target weights are not in the history table."
        ),
        "integrates_out_when": "no encoder history is supplied (not used on this task)",
        "libraries": "existing send_bw.models.spline; statsmodels/pygam were not added",
        "new_family": False,
    }
