"""Official Chronos-2 adapter (amazon-science/chronos-forecasting Chronos2Pipeline).

Zero-shot inference is a separate artifact from target-only and pooled fine-tuning.
This is not the custom baseline-conditioned transformer and is not a design simulator.

Encoder history is observed visits only (DAY-sorted LOGW). Unobserved calendar days
are not inserted and LOGW is never filled with 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec
from send_bw.evaluation.quantile_forecast import QUANTILES
from send_bw.models.base import ModelData, ResponseModel
from send_bw.models.optional_adapters import BASELINE_ONLY_REASON, DESIGN_REASON

NAME = "chronos2"
OFFICIAL_ID = "amazon/chronos-2"
OFFICIAL_CLASS = "chronos.Chronos2Pipeline"


def verify_chronos2_api() -> dict[str, Any]:
    import importlib.util

    if importlib.util.find_spec("chronos") is None:
        return {"ok": False, "reason": "optional dependency 'chronos-forecasting' not installed",
                "missing": ["chronos"]}
    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        return {"ok": False, "reason": f"chronos import failed: {e}", "missing": ["chronos"]}
    missing = [a for a in ("from_pretrained", "predict_quantiles", "fit", "save_pretrained")
               if not hasattr(Chronos2Pipeline, a)]
    if missing:
        return {"ok": False, "reason": "Chronos2Pipeline API mismatch: " + ", ".join(missing),
                "missing": missing}
    return {
        "ok": True,
        "reason": "verified chronos.Chronos2Pipeline from_pretrained / predict_quantiles / fit / save_pretrained",
        "missing": [],
        "class": f"{Chronos2Pipeline.__module__}.{Chronos2Pipeline.__name__}",
        "forecast_type": str(getattr(Chronos2Pipeline, "forecast_type", "")),
        "official_id": OFFICIAL_ID,
    }


def _require_api() -> dict[str, Any]:
    v = verify_chronos2_api()
    if not v["ok"]:
        raise RuntimeError(v["reason"])
    return v


def observed_series(data: ModelData) -> list[dict[str, Any]]:
    """One observed LOGW path per animal. No interpolated calendar days."""
    out = []
    animals = data.animals.set_index("USUBJID")
    for u, g in data.obs.groupby("USUBJID", sort=False):
        g = g.sort_values("DAY")
        if g.empty:
            continue
        row = animals.loc[u] if u in animals.index else None
        out.append({
            "USUBJID": str(u),
            "STUDYID": str(g["STUDYID"].iloc[0]),
            "SPECIES": None if row is None else str(row.get("SPECIES", "")),
            "DAY": g["DAY"].to_numpy(float),
            "LOGW": g["LOGW"].to_numpy(float),
        })
    return out


class Chronos2EarlyHistoryModel(ResponseModel):
    name = NAME
    task = "early_history"
    supports_early_history = True
    supports_pre_study = False
    supports_design = False

    def __init__(self):
        self.spec: FeatureSpec | None = None
        self.compound_ref = pd.DataFrame()
        self.pipeline = None
        self.quantiles: list[float] = list(QUANTILES)
        self.history: dict[str, Any] = {}
        self.mode: str = "zero_shot"
        self.pretrained_id: str = OFFICIAL_ID
        self.encoder_window_day: float = 14.0

    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        """Fine-tune the official pipeline. Zero-shot uses ``load_pretrained`` instead."""
        _require_api()
        from chronos import Chronos2Pipeline

        self.spec, self.compound_ref = train.spec, train.compound_ref
        self.encoder_window_day = float(config.get("encoder_window_day", 14))
        self.quantiles = [float(q) for q in config.get("quantiles", QUANTILES)]
        self.mode = str(config.get("mode", "finetune"))
        self.pretrained_id = str(config.get("pretrained_id", OFFICIAL_ID))
        device = str(config.get("device_map", "cpu"))
        t0 = time.time()
        pipe = Chronos2Pipeline.from_pretrained(self.pretrained_id, device_map=device)
        train_s = [s["LOGW"] for s in observed_series(train) if len(s["LOGW"]) >= 3]
        val_s = [s["LOGW"] for s in observed_series(validation) if len(s["LOGW"]) >= 3]
        if len(train_s) < 2:
            raise RuntimeError("Chronos-2 fine-tune: fewer than 2 observed training series")
        pred_len = int(config.get("prediction_length", 6))
        out_dir = config.get("finetune_dir")
        steps = int(config.get("num_steps", 80))
        ft_mode = str(config.get("finetune_mode", "lora"))
        self.pipeline = pipe.fit(
            inputs=train_s,
            prediction_length=pred_len,
            validation_inputs=val_s or None,
            finetune_mode=ft_mode,
            learning_rate=float(config.get("lr", 1e-5)),
            num_steps=steps,
            batch_size=int(config.get("batch_size", 8)),
            output_dir=out_dir,
            min_past=int(config.get("min_past", 2)),
        )
        ckpt = None
        if out_dir:
            p = Path(out_dir) / str(config.get("finetuned_ckpt_name", "finetuned-ckpt"))
            ckpt = str(p) if p.exists() or Path(out_dir).exists() else str(out_dir)
        self.history = {
            "name": self.name,
            "task": self.task,
            "library": OFFICIAL_CLASS,
            "pretrained_id": self.pretrained_id,
            "mode": self.mode,
            "finetune_mode": ft_mode,
            "num_steps": steps,
            "n_train_series": len(train_s),
            "n_val_series": len(val_s),
            "prediction_length": pred_len,
            "quantiles": self.quantiles,
            "device_map": device,
            "seconds": time.time() - t0,
            "checkpoint": ckpt,
            "observed_rows_only": True,
            "filled_target_with_zero": False,
            "not_the_required_transformer": True,
            "forecast_type": "quantiles",
            "note": ("official Chronos2Pipeline fine-tune; quantile forecasts only; "
                     "not a trajectory simulator; encoder is observed visits"),
        }
        return self.history

    def load_pretrained(self, config: dict[str, Any]) -> dict[str, Any]:
        _require_api()
        from chronos import Chronos2Pipeline

        self.mode = "zero_shot"
        self.pretrained_id = str(config.get("pretrained_id", OFFICIAL_ID))
        self.quantiles = [float(q) for q in config.get("quantiles", QUANTILES)]
        self.encoder_window_day = float(config.get("encoder_window_day", 14))
        t0 = time.time()
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.pretrained_id, device_map=str(config.get("device_map", "cpu")))
        rev = None
        try:
            rev = getattr(self.pipeline, "name_or_path", None) or self.pretrained_id
        except Exception:  # noqa: BLE001
            rev = self.pretrained_id
        self.history = {
            "name": self.name, "task": self.task, "library": OFFICIAL_CLASS,
            "pretrained_id": self.pretrained_id, "mode": "zero_shot",
            "checkpoint_revision": str(rev),
            "quantiles": self.quantiles,
            "device_map": str(config.get("device_map", "cpu")),
            "seconds": time.time() - t0,
            "observed_rows_only": True, "filled_target_with_zero": False,
            "not_the_required_transformer": True, "forecast_type": "quantiles",
            "note": "official Chronos2Pipeline zero-shot; no fine-tune",
        }
        return self.history

    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        raise NotImplementedError(BASELINE_ONLY_REASON.replace("TFT", "Chronos-2"))

    def sample_study(self, context: pd.DataFrame, regimens: pd.DataFrame, times: np.ndarray, n_draws: int,
                     seed: int) -> dict[str, np.ndarray]:
        raise NotImplementedError(DESIGN_REASON.replace("TFT", "Chronos-2"))

    def predict_marginals(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        q = self.predict_quantiles(context, times, history=history)
        if q.empty:
            return pd.DataFrame(columns=["USUBJID", "DAY", "MEAN", "VAR"])
        mean = q["q0.5"].to_numpy(float)
        if "q0.1" in q.columns and "q0.9" in q.columns:
            sd = (q["q0.9"].to_numpy(float) - q["q0.1"].to_numpy(float)) / (2.0 * 1.2815515655446004)
        else:
            sd = np.full(len(q), 0.05)
        return pd.DataFrame({"USUBJID": q["USUBJID"], "DAY": q["DAY"], "MEAN": mean,
                             "VAR": np.maximum(sd, 1e-6) ** 2})

    def predict_quantiles(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        if history is None or not len(history):
            raise ValueError(BASELINE_ONLY_REASON.replace("TFT", "Chronos-2"))
        if self.pipeline is None:
            raise RuntimeError("Chronos-2 pipeline is not loaded")
        times = np.asarray(times, float)
        hist = history.copy()
        recs = []
        for _, animal in context.reset_index(drop=True).iterrows():
            u = animal["USUBJID"]
            hh = hist.loc[hist["USUBJID"] == u].sort_values("DAY")
            if hh.empty:
                continue
            enc = hh["LOGW"].to_numpy(float)
            fut = times
            if "DAY" in hist.columns:
                # score only requested future times; still no fabricated encoder
                pass
            try:
                qs, _mean = self.pipeline.predict_quantiles(
                    [enc], prediction_length=int(len(fut)), quantile_levels=self.quantiles)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Chronos2Pipeline.predict_quantiles failed: {type(e).__name__}: {e}") from e
            arr = np.asarray(qs[0].detach().cpu() if hasattr(qs[0], "detach") else qs[0], float)
            # (n_variates, horizon, n_q) or (horizon, n_q)
            if arr.ndim == 3:
                arr = arr[0]
            n = min(len(fut), arr.shape[0])
            for j in range(n):
                row = {"USUBJID": u, "DAY": float(fut[j])}
                for k, q in enumerate(self.quantiles):
                    row[f"q{q}"] = float(arr[j, k])
                recs.append(row)
        return pd.DataFrame(recs)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.pipeline is None:
            raise RuntimeError("Chronos-2 pipeline is not loaded")
        self.pipeline.save_pretrained(str(path / "pipeline"))
        if self.compound_ref is not None and len(getattr(self.compound_ref, "columns", [])):
            self.compound_ref.to_csv(path / "compound_reference.csv", index=False)
        (path / "model.json").write_text(json.dumps({
            "name": self.name, "task": self.task, "official_class": OFFICIAL_CLASS,
            "pretrained_id": self.pretrained_id, "mode": self.mode,
            "quantiles": self.quantiles, "encoder_window_day": self.encoder_window_day,
            "history": self.history, "not_the_required_transformer": True,
            "spec": None if self.spec is None else self.spec.to_dict(),
        }, indent=1, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Chronos2EarlyHistoryModel:
        _require_api()
        from chronos import Chronos2Pipeline

        path = Path(path)
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if d.get("name") != NAME:
            raise ValueError(f"refusing to load name={d.get('name')!r} as Chronos-2")
        m = cls()
        m.mode = d.get("mode", "zero_shot")
        m.pretrained_id = d.get("pretrained_id", OFFICIAL_ID)
        m.quantiles = [float(x) for x in d.get("quantiles", QUANTILES)]
        m.encoder_window_day = float(d.get("encoder_window_day", 14))
        m.history = d.get("history") or {}
        if d.get("spec"):
            m.spec = FeatureSpec.from_dict(d["spec"])
        if (path / "compound_reference.csv").exists():
            m.compound_ref = pd.read_csv(path / "compound_reference.csv")
        pipe_dir = path / "pipeline"
        if pipe_dir.exists() and any(pipe_dir.iterdir()):
            m.pipeline = Chronos2Pipeline.from_pretrained(str(pipe_dir), device_map="cpu")
        else:
            # Zero-shot full weights are not committed; reload the official id.
            m.pipeline = Chronos2Pipeline.from_pretrained(
                m.pretrained_id or OFFICIAL_ID, device_map="cpu")
        return m
