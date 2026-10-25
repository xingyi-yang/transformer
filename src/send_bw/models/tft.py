"""Optional Temporal Fusion Transformer adapter (early-history only).

This is a verified PyTorch Forecasting wrapper. It is **not** the required custom
baseline-conditioned transformer in ``models/transformer.py`` and must never be
registered under that name.

Capabilities (honest):
- Condition on **observed** encoder visits (DAY <= window) plus static biology /
  planned-regimen covariates and known-future planned-dose features.
- Score later observations on the secondary early-history task.

Refused (not fabricated):
- Baseline-only / pre-study mean prediction (no observed encoder window).
- ``sample_study`` / design simulation of complete hypothetical studies.

Panel construction passes **observed rows only**. ``allow_missing_timesteps`` is
enabled so an irregular calendar ``DAY`` is acceptable, but this adapter never
inserts unobserved target rows and never fills ``LOGW`` with 0. The PF
``time_idx`` is a dense per-animal visit rank so missing calendar days do not
become fabricated encoder history. Decoder target placeholders are not
observations. Lightning validation keeps only windows whose last encoder
``DAY`` is at the configured cutoff and whose decoder starts after it.
Training still uses sliding observed windows (DAY/TAU are known) so the
model can learn early-to-late dynamics; those windows are not the scored
task. Predict re-clips history to the cutoff, refuses unseen species, and
chunks horizons to the trained ``max_prediction_length``.

Variance is a quantile-spread approximation from QuantileLoss, not a learned
Gaussian covariance. Installing the extra is required; the default environment
must stay ``not_run``.
"""

from __future__ import annotations

import inspect
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData, ResponseModel, study_balanced_weights
from send_bw.models.optional_adapters import BASELINE_ONLY_REASON, DESIGN_REASON, verify_tft_api

NAME = "tft"
TARGET = "LOGW"
GROUP_IDS = ["USUBJID"]
STATIC_CATS = ["SEX", "STRAIN", "SPECIES", "ROUTE", "COMPOUND_ID", "CTRL"]
STATIC_REALS = ["ZB_C", "LD", "INV_INTERVAL", "START", "END", "D1", "D2", "D3"]
KNOWN_REALS = ["DAY", "TAU", "EXPO", "OFF", "LCUM"]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
_Z_10_90 = 1.2815515655446004


def _require_api() -> dict[str, Any]:
    verified = verify_tft_api()
    if not verified["ok"]:
        raise RuntimeError(verified["reason"])
    return verified


def _pf():
    """Import verified PyTorch Forecasting symbols (after ``verify_tft_api``)."""
    _require_api()
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    return TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss


def _trainer_cls():
    _require_api()
    try:
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:  # pragma: no cover
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    return Trainer, EarlyStopping, ModelCheckpoint


def _call_dataset_factory(fn, *args, predict: bool = False, **overrides):
    """Use from_dataset / from_parameters with whichever predict kwarg the install has."""
    params = inspect.signature(fn).parameters
    kw = dict(overrides)
    if "stop_randomization" in params:
        kw.setdefault("stop_randomization", True)
    if "predict" in params:
        kw["predict"] = predict
    elif "predict_mode" in params:
        kw["predict_mode"] = predict
    return fn(*args, **kw)


def count_completed_epochs(curve_rows: list[dict[str, Any]] | None,
                           lightning_current_epoch: int | None = None) -> dict[str, Any]:
    """Count logged training epochs. Do not add 1 to Lightning ``current_epoch``.

    Lightning's ``current_epoch`` is 0-based and is incremented after an epoch
    ends, so a finished ``max_epochs=16`` fit often has ``current_epoch==16``.
    Prior histories stored ``epochs_run = current_epoch + 1`` (16-budget → 17).
    """
    trained = sorted({
        int(r["epoch"]) for r in (curve_rows or [])
        if r.get("train_loss") is not None
    })
    cur = None if lightning_current_epoch is None else int(lightning_current_epoch)
    return {
        "epochs_completed": len(trained),
        "trained_epoch_ids": trained,
        "lightning_current_epoch": cur,
        "epochs_run_legacy_plus_one": None if cur is None else cur + 1,
        "note": ("epochs_completed counts callback rows that logged train_loss. "
                 "Legacy epochs_run was lightning current_epoch + 1."),
    }


def _make_trainer(tmp: str, max_epochs: int, callbacks: list):
    Trainer, _, _ = _trainer_cls()
    params = inspect.signature(Trainer.__init__).parameters
    kw: dict[str, Any] = {"max_epochs": max_epochs, "logger": False, "callbacks": callbacks,
                          "default_root_dir": tmp}
    for key, val in (("accelerator", "cpu"), ("devices", 1), ("enable_checkpointing", True),
                     ("enable_progress_bar", False), ("enable_model_summary", False)):
        if key in params:
            kw[key] = val
    return Trainer(**kw)


def quantiles_to_mean_var(q: np.ndarray, levels: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Map QuantileLoss output to a mean and an approximate variance.

    MEAN is the 0.5 quantile. VAR uses the 0.10–0.90 spread as a Gaussian-scale
    approximation; this is not a fitted residual covariance.
    """
    q = np.asarray(q, float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    levels = [float(x) for x in levels]
    i50 = levels.index(0.5)
    mean = q[..., i50]
    if 0.1 in levels and 0.9 in levels:
        sd = (q[..., levels.index(0.9)] - q[..., levels.index(0.1)]) / (2.0 * _Z_10_90)
    else:
        i_hi = int(np.argmax(levels))
        i_lo = int(np.argmin(levels))
        sd = (q[..., i_hi] - q[..., i_lo]) / 3.289707  # ~0.02–0.98 if those are the ends
    var = np.maximum(sd, 1e-6) ** 2
    return mean, var


def _ctrl_label(is_control: pd.Series) -> pd.Series:
    return pd.Series(np.where(is_control.astype(bool), "C", "A"), index=is_control.index)


def cat_label(value: Any) -> str:
    """Stable categorical token. NaN must not become the string 'nan' at predict time."""
    if value is None:
        return "UNK"
    try:
        if pd.isna(value):
            return "UNK"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat", "<na>", "null"}:
        return "UNK"
    return text


def _normalize_static_cats(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in STATIC_CATS:
        if c not in out.columns:
            out[c] = "UNK"
        out[c] = [cat_label(v) for v in out[c].to_numpy()]
    return out


def build_target_normalizer(panel: pd.DataFrame, config: dict[str, Any] | None = None):
    """Fit-time target scale. LOGW is already ln(grams); do not apply relu.

    Multi-species panels use a SPECIES-grouped normalizer so RAT (~5.6) and DOG (~9.1)
    are not forced through one global center/scale. Single-species panels use a
    global standardizer. Encoder-window scaling is opt-in (short encoders are noisy).
    """
    from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer

    cfg = config or {}
    kind = str(cfg.get("target_normalizer") or "species_or_global").strip().lower()
    n_sp = int(panel["SPECIES"].nunique()) if "SPECIES" in panel.columns else 1
    if kind in {"encoder", "encoder_normalizer"}:
        return EncoderNormalizer(method="standard", center=True, transformation=None)
    if kind in {"species", "species_or_global"} and n_sp > 1:
        return GroupNormalizer(
            groups=["SPECIES"], method="standard", center=True,
            scale_by_group=True, transformation=None)
    return GroupNormalizer(method="standard", center=True, transformation=None)


def dataset_group_ids(panel: pd.DataFrame, config: dict[str, Any] | None = None) -> list[str]:
    """Series identity for TimeSeriesDataSet.

    PF ``GroupNormalizer.get_parameters`` only sees ``group_ids``. A SPECIES-grouped
    scale therefore requires SPECIES as the first group id. USUBJID remains in the
    key so each animal is still its own series.
    """
    if "SPECIES" in panel.columns and int(panel["SPECIES"].nunique()) > 1:
        kind = str((config or {}).get("target_normalizer") or "species_or_global").strip().lower()
        if kind in {"species", "species_or_global"}:
            return ["SPECIES", "USUBJID"]
    return list(GROUP_IDS)


def uses_target_scales(panel: pd.DataFrame, config: dict[str, Any] | None = None) -> bool:
    """Group/encoder scales are network inputs; a single global standardizer is not."""
    cfg = config or {}
    if cfg.get("add_target_scales") is not None:
        return bool(cfg.get("add_target_scales"))
    kind = str(cfg.get("target_normalizer") or "species_or_global").strip().lower()
    if kind in {"encoder", "encoder_normalizer"}:
        return True
    n_sp = int(panel["SPECIES"].nunique()) if "SPECIES" in panel.columns else 1
    return n_sp > 1


def normalizer_record(norm: Any) -> dict[str, Any]:
    """JSON-safe snapshot of a fitted PF target normalizer."""
    rec: dict[str, Any] = {
        "class": type(norm).__name__ if norm is not None else None,
        "transformation": getattr(norm, "transformation", None),
        "groups": list(getattr(norm, "groups", None) or []) or None,
        "scale_by_group": getattr(norm, "scale_by_group", None),
        "method": getattr(norm, "method", None),
        "center": getattr(norm, "center", None),
    }
    for attr in ("center_", "scale_", "norm_"):
        if hasattr(norm, attr):
            rec[attr] = _jsonable(getattr(norm, attr))
    return rec


def categorical_encoder_classes(parameters: dict[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, enc in (parameters or {}).get("categorical_encoders", {}).items():
        classes = getattr(enc, "classes_", None)
        if classes is None:
            classes = getattr(enc, "classes", None)
        if classes is None:
            continue
        if isinstance(classes, dict):
            out[str(name)] = [str(k) for k in classes]
        else:
            out[str(name)] = [str(x) for x in classes]
    return out


def _jsonable(val: Any) -> Any:
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        if val.size <= 32:
            return val.tolist()
        return {"shape": list(val.shape), "mean": float(np.mean(val))}
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonable(x) for x in val]
    return str(val)


def _set_learning_rate(net: Any, lr: float) -> None:
    if hasattr(net, "hparams") and hasattr(net.hparams, "learning_rate"):
        net.hparams.learning_rate = float(lr)
    if hasattr(net, "learning_rate"):
        try:
            net.learning_rate = float(lr)
        except (TypeError, AttributeError):
            pass


def validate_forecast_times(times: np.ndarray, history: pd.DataFrame) -> np.ndarray:
    """Require finite, unique, strictly increasing times later than supplied history."""
    times = np.asarray(times, float).reshape(-1)
    if times.size == 0:
        raise ValueError("prediction times are empty")
    if not np.isfinite(times).all():
        raise ValueError("prediction times must be finite")
    if times.size != len(np.unique(times)):
        raise ValueError("prediction times must be unique")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("prediction times must be strictly increasing")
    if len(history) and float(times.min()) <= float(np.asarray(history["DAY"], float).max()):
        raise ValueError("prediction times must be later than the supplied history")
    return times


def clip_encoder_history(history: pd.DataFrame, window_day: float) -> pd.DataFrame:
    """Keep observed encoder rows with DAY <= window. Does not invent visits."""
    if "DAY" not in history.columns:
        raise ValueError("early-history encoder requires DAY")
    return history.loc[history["DAY"].to_numpy(float) <= float(window_day)].copy()


def chunk_forecast_times(times: np.ndarray, max_len: int) -> list[np.ndarray]:
    """Split a horizon into trained-length decoder chunks. Encoder is not rolled."""
    times = np.asarray(times, float).reshape(-1)
    max_len = max(1, int(max_len))
    return [times[i:i + max_len] for i in range(0, len(times), max_len)]


def early_history_window_mask(decoded: pd.DataFrame, panel: pd.DataFrame,
                              window_day: float) -> np.ndarray:
    """Keep PF windows whose last encoder DAY is at the cutoff and decoder starts after it."""
    key = panel[["USUBJID", "time_idx", "DAY"]].drop_duplicates()
    key = key.assign(USUBJID=key["USUBJID"].astype(str), time_idx=key["time_idx"].astype(int))
    d = decoded[["USUBJID", "time_idx_first_prediction"]].copy()
    d["USUBJID"] = d["USUBJID"].astype(str)
    d["enc_idx"] = d["time_idx_first_prediction"].astype(int) - 1
    d["dec_idx"] = d["time_idx_first_prediction"].astype(int)
    enc = d.merge(key, left_on=["USUBJID", "enc_idx"], right_on=["USUBJID", "time_idx"],
                  how="left", sort=False)
    dec = d.merge(key, left_on=["USUBJID", "dec_idx"], right_on=["USUBJID", "time_idx"],
                  how="left", sort=False)
    return ((enc["DAY"].to_numpy(float) <= float(window_day))
            & (dec["DAY"].to_numpy(float) > float(window_day)))


def build_observed_panel(data: ModelData) -> pd.DataFrame:
    """One row per **observed** post-baseline weight; no filled calendar days."""
    if data.obs.empty:
        return pd.DataFrame()
    animals = data.animals.reset_index(drop=True)
    af = data.spec.animal_features(animals, data.compound_ref)
    af = af.reset_index(drop=True)
    af["USUBJID"] = animals["USUBJID"].to_numpy()
    static_cols = ["USUBJID", "STUDYID", "SEX", "STRAIN", "ROUTE", "COMPOUND_ID", "IS_CONTROL"]
    if "SPECIES" in animals.columns:
        static_cols.append("SPECIES")
    static = animals[static_cols].copy()
    static["CTRL"] = _ctrl_label(static["IS_CONTROL"])
    static = static.merge(af[["USUBJID", *STATIC_REALS]], on="USUBJID", how="left", validate="one_to_one")
    obs = data.obs[["USUBJID", "STUDYID", "DAY", "LOGW"]].copy()
    rows = animals.set_index("USUBJID").loc[obs["USUBJID"]].reset_index()
    tf = data.spec.time_features(obs["DAY"].to_numpy(), rows)
    panel = pd.concat([obs.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)
    panel = panel.merge(static.drop(columns=["STUDYID"]), on="USUBJID", how="left", validate="many_to_one")
    w = pd.Series(study_balanced_weights(panel["STUDYID"].to_numpy()), index=panel.index)
    panel["WEIGHT"] = w.to_numpy(float)
    panel = panel.sort_values(["USUBJID", "DAY"], kind="stable").reset_index(drop=True)
    panel["time_idx"] = panel.groupby("USUBJID", sort=False).cumcount().astype(int)
    panel = _normalize_static_cats(panel)
    for c in STATIC_REALS + KNOWN_REALS + ["WEIGHT"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
    return panel


class TFTEarlyHistoryModel(ResponseModel):
    name = NAME
    task = "early_history"
    supports_early_history = True
    supports_pre_study = False
    supports_design = False

    def __init__(self):
        self.spec: FeatureSpec | None = None
        self.compound_ref = pd.DataFrame()
        self.dataset_parameters: dict[str, Any] | None = None
        self.net = None
        self.quantiles: list[float] = list(QUANTILES)
        self.history: dict[str, Any] = {}
        self.encoder_window_day: float = 14.0
        self.max_encoder_length: int = 8
        self.max_prediction_length: int = 6
        self.min_encoder_length: int = 1

    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss = _pf()
        _, EarlyStopping, ModelCheckpoint = _trainer_cls()
        self.spec, self.compound_ref = train.spec, train.compound_ref
        self.encoder_window_day = float(config.get("encoder_window_day", config.get("early_history_window_day", 14)))
        self.max_encoder_length = int(config.get("max_encoder_length", 8))
        self.max_prediction_length = int(config.get("max_prediction_length", 6))
        self.min_encoder_length = int(config.get("min_encoder_length", 1))
        self.quantiles = [float(x) for x in config.get("quantiles", QUANTILES)]
        if 0.5 not in self.quantiles:
            raise ValueError("TFT QuantileLoss quantiles must include 0.5 (used as MEAN)")

        train_panel = build_observed_panel(train)
        val_panel = build_observed_panel(validation)
        min_len = self.min_encoder_length + 1
        train_panel = _enough_visits(train_panel, min_len)
        val_panel = _enough_visits(val_panel, min_len)
        if train_panel.empty or train_panel["USUBJID"].nunique() < 2:
            raise RuntimeError("TFT training panel has too few observed series after the encoder-length filter")
        if val_panel.empty:
            raise RuntimeError("TFT validation panel is empty after the encoder-length filter")

        window_balance = str(config.get("window_balance") or "").strip().lower()
        if window_balance == "study":
            train_panel = _unit_row_weights(train_panel)
            val_panel = _unit_row_weights(val_panel)

        seed = int(config.get("seed", 0))
        import torch

        torch.set_num_threads(int(config.get("torch_threads", 4)))
        torch.manual_seed(seed)
        np.random.seed(seed)

        from pytorch_forecasting.data import NaNLabelEncoder

        group_ids = dataset_group_ids(train_panel, config)
        cat_enc = {c: NaNLabelEncoder(add_nan=True) for c in list(group_ids) + list(STATIC_CATS)}
        target_normalizer = build_target_normalizer(train_panel, config)
        add_scales = uses_target_scales(train_panel, config)
        ds_kwargs = dict(
            time_idx="time_idx",
            target=TARGET,
            group_ids=list(group_ids),
            weight="WEIGHT",
            max_encoder_length=self.max_encoder_length,
            min_encoder_length=self.min_encoder_length,
            max_prediction_length=self.max_prediction_length,
            min_prediction_length=1,
            static_categoricals=list(STATIC_CATS),
            static_reals=list(STATIC_REALS),
            time_varying_known_reals=list(KNOWN_REALS),
            time_varying_unknown_reals=[TARGET],
            allow_missing_timesteps=True,  # irregular DAY is allowed; we do not insert unobserved Y
            add_relative_time_idx=True,
            add_target_scales=add_scales,
            add_encoder_length=True,
            categorical_encoders=cat_enc,
            target_normalizer=target_normalizer,
        )
        training = TimeSeriesDataSet(train_panel, **ds_kwargs)
        validation_ds = _call_dataset_factory(TimeSeriesDataSet.from_dataset, training, val_panel, predict=False)
        try:
            validation_ds = validation_ds.filter(
                lambda decoded, panel=val_panel, window=self.encoder_window_day:
                early_history_window_mask(decoded, panel, window))
        except ValueError as e:
            raise RuntimeError(
                "TFT validation has no windows whose encoder ends at DAY <= "
                f"{self.encoder_window_day} with later observed targets") from e
        self.dataset_parameters = training.get_parameters()
        fitted_norm = self.dataset_parameters.get("target_normalizer", target_normalizer)
        batch_size = int(config.get("batch_size", 64))
        if window_balance == "study":
            sampler = _study_window_sampler(training, train_panel)
            train_dl = training.to_dataloader(
                train=True, batch_size=batch_size, num_workers=0, sampler=sampler, shuffle=False)
        else:
            train_dl = training.to_dataloader(
                train=True, batch_size=batch_size, num_workers=0)
        val_dl = validation_ds.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0)

        loss = QuantileLoss(quantiles=self.quantiles)
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=int(config.get("hidden_size", 16)),
            lstm_layers=int(config.get("lstm_layers", 1)),
            dropout=float(config.get("dropout", 0.1)),
            attention_head_size=int(config.get("attention_head_size", 2)),
            hidden_continuous_size=int(config.get("hidden_continuous_size", 8)),
            learning_rate=float(config.get("lr", 3e-3)),
            loss=loss,
            output_size=len(self.quantiles),
            optimizer=str(config.get("optimizer", "adamw")),
            reduce_on_plateau_patience=int(config.get("reduce_on_plateau_patience", 4)),
            log_interval=-1,
            log_val_interval=-1,
        )
        curve = _loss_callback_cls()()
        with tempfile.TemporaryDirectory(prefix="send_bw_tft_") as tmp:
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=int(config.get("patience", 6)), mode="min"),
                ModelCheckpoint(dirpath=tmp, filename="best", monitor="val_loss", mode="min", save_top_k=1),
                curve,
            ]
            trainer = _make_trainer(tmp, int(config.get("max_epochs", 20)), callbacks)
            trainer.fit(tft, train_dl, val_dl)
            best = callbacks[1].best_model_path
            if best:
                self.net = TemporalFusionTransformer.load_from_checkpoint(best)
            else:
                self.net = tft
            self.net.eval()
            self._best_checkpoint_bytes = Path(best).read_bytes() if best else None
            n_updates = int(getattr(trainer, "global_step", 0) or 0)
            epoch_count = count_completed_epochs(
                curve.rows, int(getattr(trainer, "current_epoch", 0) or 0))
            epochs_run = int(epoch_count["epochs_completed"])

        n_train_windows = int(len(training))
        n_animals_split = int(train.animals["USUBJID"].nunique()) if "USUBJID" in train.animals.columns else 0
        n_train_series = int(train_panel["USUBJID"].nunique())
        self.history = {
            "name": self.name,
            "task": self.task,
            "library": "pytorch_forecasting.TemporalFusionTransformer",
            "official_class": "pytorch_forecasting.TemporalFusionTransformer",
            "training_scope": str(config.get("training_scope", "unspecified")),
            "not_the_required_transformer": True,
            "n_animals_in_split": n_animals_split,
            "n_train_series": n_train_series,
            "n_dropped_short_series": max(0, n_animals_split - n_train_series),
            "n_val_series": int(val_panel["USUBJID"].nunique()),
            "n_train_obs": int(len(train_panel)),
            "n_train_windows": n_train_windows,
            "n_val_windows": int(len(validation_ds)),
            "n_optimizer_updates": n_updates,
            "epochs_run": epochs_run,
            "epoch_count": epoch_count,
            "batch_size": batch_size,
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "min_encoder_length": self.min_encoder_length,
            "encoder_window_day": self.encoder_window_day,
            "quantiles": self.quantiles,
            "allow_missing_timesteps": True,
            "observed_rows_only": True,
            "filled_target_with_zero": False,
            "validation_windows": "early_history_encoder_day",
            "training_windows": "sliding_observed_visits",
            "window_balance": window_balance or "row_study_WEIGHT_uniform_windows",
            "row_weight": 1.0 if window_balance == "study" else "study_balanced_per_obs_row",
            "double_balancing": False,
            "max_epochs": int(config.get("max_epochs", 20)),
            "target_normalizer": normalizer_record(fitted_norm),
            "add_target_scales": add_scales,
            "group_ids": list(group_ids),
            "cat_label": "UNK_not_nan",
            "categorical_classes": categorical_encoder_classes(self.dataset_parameters),
            "curves": curve.rows,
            "note": ("early-history only; baseline-only pre-study and design simulation are not supported; "
                     "VAR is a quantile-spread approximation"),
        }
        return self.history

    def finetune(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        """Continue a fitted pooled net on target-species data.

        Uses ``TimeSeriesDataSet.from_parameters`` so SPECIES / COMPOUND_ID classes
        from pretraining (including donor species) are not rebuilt. Feature scaling
        stays on the pretrained ``FeatureSpec``. Learning rate is lowered.
        """
        if self.net is None or self.dataset_parameters is None or self.spec is None:
            raise RuntimeError("finetune requires a fitted pooled TFT")
        TimeSeriesDataSet, TemporalFusionTransformer, _QuantileLoss = _pf()
        _, EarlyStopping, ModelCheckpoint = _trainer_cls()
        kept_classes = categorical_encoder_classes(self.dataset_parameters)
        train_md = ModelData(train.animals, train.obs, self.compound_ref, self.spec)
        val_md = ModelData(validation.animals, validation.obs, self.compound_ref, self.spec)
        train_panel = _enough_visits(build_observed_panel(train_md), self.min_encoder_length + 1)
        val_panel = _enough_visits(build_observed_panel(val_md), self.min_encoder_length + 1)
        if train_panel.empty or val_panel.empty:
            raise RuntimeError("TFT fine-tune panel is empty after the encoder-length filter")
        window_balance = str(config.get("window_balance") or "").strip().lower()
        if window_balance == "study":
            train_panel = _unit_row_weights(train_panel)
            val_panel = _unit_row_weights(val_panel)
        seed = int(config.get("seed", 0))
        import torch

        torch.set_num_threads(int(config.get("torch_threads", 4)))
        torch.manual_seed(seed)
        np.random.seed(seed)
        training = _call_dataset_factory(
            TimeSeriesDataSet.from_parameters, self.dataset_parameters, train_panel, predict=False)
        validation_ds = _call_dataset_factory(
            TimeSeriesDataSet.from_parameters, self.dataset_parameters, val_panel, predict=False)
        try:
            validation_ds = validation_ds.filter(
                lambda decoded, panel=val_panel, window=self.encoder_window_day:
                early_history_window_mask(decoded, panel, window))
        except ValueError as e:
            raise RuntimeError(
                "TFT fine-tune validation has no windows whose encoder ends at DAY <= "
                f"{self.encoder_window_day} with later observed targets") from e
        after_classes = categorical_encoder_classes(self.dataset_parameters)
        for name, before in kept_classes.items():
            after = after_classes.get(name) or []
            missing = sorted(set(before) - set(after))
            if missing:
                raise RuntimeError(f"finetune dropped {name} classes {missing}; from_parameters must keep them")
        batch_size = int(config.get("batch_size", 64))
        if window_balance == "study":
            sampler = _study_window_sampler(training, train_panel)
            train_dl = training.to_dataloader(
                train=True, batch_size=batch_size, num_workers=0, sampler=sampler, shuffle=False)
        else:
            train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dl = validation_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        lr = float(config.get("finetune_lr", config.get("lr", 1e-3)))
        _set_learning_rate(self.net, lr)
        self.net.train()
        curve = _loss_callback_cls()()
        max_epochs = int(config.get("finetune_max_epochs", config.get("max_epochs", 8)))
        with tempfile.TemporaryDirectory(prefix="send_bw_tft_ft_") as tmp:
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=int(config.get("finetune_patience",
                              config.get("patience", 4))), mode="min"),
                ModelCheckpoint(dirpath=tmp, filename="best", monitor="val_loss", mode="min", save_top_k=1),
                curve,
            ]
            trainer = _make_trainer(tmp, max_epochs, callbacks)
            trainer.fit(self.net, train_dl, val_dl)
            best = callbacks[1].best_model_path
            if best:
                self.net = TemporalFusionTransformer.load_from_checkpoint(best)
            self.net.eval()
            self._best_checkpoint_bytes = Path(best).read_bytes() if best else None
            n_updates = int(getattr(trainer, "global_step", 0) or 0)
            epoch_count = count_completed_epochs(
                curve.rows, int(getattr(trainer, "current_epoch", 0) or 0))
            epochs_run = int(epoch_count["epochs_completed"])
        pre = dict(self.history)
        self.history = {
            **pre,
            "finetuned": True,
            "finetune_lr": lr,
            "finetune_max_epochs": max_epochs,
            "finetune_epochs_run": epochs_run,
            "finetune_epoch_count": epoch_count,
            "finetune_n_optimizer_updates": n_updates,
            "finetune_n_train_series": int(train_panel["USUBJID"].nunique()),
            "finetune_n_val_series": int(val_panel["USUBJID"].nunique()),
            "finetune_n_train_obs": int(len(train_panel)),
            "finetune_n_train_windows": int(len(training)),
            "finetune_n_val_windows": int(len(validation_ds)),
            "finetune_from_parameters": True,
            "finetune_species_classes_kept": after_classes.get("SPECIES"),
            "finetune_curves": curve.rows,
            "training_scope": str(config.get("training_scope", "pooled_then_target_finetune")),
        }
        return self.history

    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        raise NotImplementedError(BASELINE_ONLY_REASON)

    def sample_study(self, context: pd.DataFrame, regimens: pd.DataFrame, times: np.ndarray, n_draws: int,
                     seed: int) -> dict[str, np.ndarray]:
        raise NotImplementedError(DESIGN_REASON)

    def predict_marginals(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        return self._run_predict(context, times, history, as_marginals=True)

    def predict_quantiles(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Official TFT quantile forecasts. Not a trajectory simulator."""
        return self._run_predict(context, times, history, as_marginals=False)

    def _run_predict(self, context: pd.DataFrame, times: np.ndarray,
                     history: pd.DataFrame | None, *, as_marginals: bool) -> pd.DataFrame:
        if history is None or not len(history):
            raise ValueError(BASELINE_ONLY_REASON)
        if self.net is None or self.dataset_parameters is None or self.spec is None:
            raise RuntimeError("TFT adapter is not fitted")
        if "LOGW" not in history.columns:
            raise ValueError("early-history encoder requires LOGW")
        hist = clip_encoder_history(history, self.encoder_window_day)
        if hist.empty:
            raise ValueError(BASELINE_ONLY_REASON)
        times = validate_forecast_times(times, hist)
        self._refuse_unseen_species(context)
        parts = [self._predict_chunk(context, chunk, hist, as_marginals=as_marginals)
                 for chunk in chunk_forecast_times(times, self.max_prediction_length)]
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            cols = ["USUBJID", "DAY"] + [f"q{q}" for q in self.quantiles]
            if as_marginals:
                cols += ["MEAN", "VAR"]
            return pd.DataFrame(columns=cols)
        return pd.concat(parts, ignore_index=True)

    def _refuse_unseen_species(self, context: pd.DataFrame) -> None:
        if self.spec is None or "SPECIES" not in context.columns:
            return
        allowed = {str(s).upper() for s in self.spec.allowed_species()}
        seen = {str(s).upper() for s in context["SPECIES"].dropna()}
        bad = seen - allowed
        if bad:
            raise ValueError(f"species {sorted(bad)} not supported (trained on {sorted(allowed)})")

    def _predict_chunk(self, context: pd.DataFrame, times: np.ndarray, hist: pd.DataFrame,
                       *, as_marginals: bool, decoder_logw: float | None = None) -> pd.DataFrame:
        TimeSeriesDataSet, _, _ = _pf()
        ctx = context.reset_index(drop=True)
        rows = []
        for _, animal in ctx.iterrows():
            u = animal["USUBJID"]
            hh = hist.loc[hist["USUBJID"] == u].sort_values("DAY")
            if hh.empty:
                continue  # no observed encoder window: do not fabricate
            last = float(hh["LOGW"].to_numpy(float)[-1])
            fill = last if decoder_logw is None else float(decoder_logw)
            enc = self._animal_rows(animal, hh["DAY"].to_numpy(float), hh["LOGW"].to_numpy(float))
            dec = self._animal_rows(animal, times, np.repeat(fill, len(times)))
            both = pd.concat([enc, dec], ignore_index=True)
            both["time_idx"] = np.arange(len(both), dtype=int)
            rows.append(both)
        if not rows:
            return pd.DataFrame()
        panel = pd.concat(rows, ignore_index=True)
        n_dec = int(len(times))
        # Keep the trained encoder cap. PF predict_mode uses the last max_encoder_length
        # observed visits; expanding the cap would be an untrained sequence length.
        pred_ds = _call_dataset_factory(
            TimeSeriesDataSet.from_parameters, self.dataset_parameters, panel, predict=True,
            min_prediction_length=n_dec, max_prediction_length=n_dec,
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length)
        raw = self.net.predict(pred_ds, mode="quantiles", return_index=True)
        q, index = _unpack_predict(raw, pred_ds)
        return self._align_quantiles(q, index, panel, times, as_marginals=as_marginals)

    def _align_quantiles(self, q: np.ndarray, index: pd.DataFrame | None, panel: pd.DataFrame,
                         times: np.ndarray, *, as_marginals: bool) -> pd.DataFrame:
        if index is None:
            raise RuntimeError("TFT predict() did not return an index; cannot align USUBJID/DAY")
        out = index.copy()
        if "USUBJID" not in out.columns:
            raise RuntimeError("TFT predict index is missing USUBJID")
        key = panel[["USUBJID", "time_idx", "DAY"]].drop_duplicates()
        tcol = "time_idx" if "time_idx" in out.columns else None
        recs = []
        for i, row in out.reset_index(drop=True).iterrows():
            u = row["USUBJID"]
            n_h = q.shape[1] if q.ndim > 1 else 1
            for j in range(n_h):
                if tcol is None:
                    day = float(times[min(j, len(times) - 1)])
                else:
                    ti = int(row[tcol]) + j if q.ndim > 1 else int(row[tcol])
                    hit = key[(key["USUBJID"] == u) & (key["time_idx"] == ti)]
                    day = float(hit["DAY"].iloc[0]) if len(hit) else float(times[min(j, len(times) - 1)])
                if q.ndim >= 3:
                    vec = q[i, j]
                elif q.ndim == 2:
                    vec = q[i] if q.shape[1] == len(self.quantiles) and n_h == 1 else q[i, j] if q.shape[0] == len(out) else q[j]
                else:
                    vec = q
                vec = np.atleast_1d(np.asarray(vec, float))
                item = {"USUBJID": u, "DAY": day}
                for k, lev in enumerate(self.quantiles):
                    if k < len(vec):
                        item[f"q{lev}"] = float(vec[k])
                if as_marginals:
                    mean, var = quantiles_to_mean_var(vec.reshape(1, -1), self.quantiles)
                    item["MEAN"] = float(mean[0])
                    item["VAR"] = float(var[0])
                recs.append(item)
        cols = ["USUBJID", "DAY"] + [f"q{q}" for q in self.quantiles]
        if as_marginals:
            cols += ["MEAN", "VAR"]
        return pd.DataFrame(recs)[ [c for c in cols if c in pd.DataFrame(recs).columns] ]

    def _animal_rows(self, animal: pd.Series, days: np.ndarray, logw: np.ndarray) -> pd.DataFrame:
        n = len(days)
        one = pd.DataFrame([animal.to_dict()] * n).reset_index(drop=True)
        one["CTRL"] = _ctrl_label(one["IS_CONTROL"])
        af = self.spec.animal_features(one, self.compound_ref).reset_index(drop=True)
        tf = self.spec.time_features(np.asarray(days, float), one)
        out = pd.DataFrame({
            "USUBJID": one["USUBJID"].to_numpy(), "STUDYID": one["STUDYID"].to_numpy(),
            "DAY": np.asarray(days, float), "LOGW": np.asarray(logw, float),
            "WEIGHT": np.ones(n, float),
        })
        for c in STATIC_CATS:
            if c in one.columns:
                out[c] = [cat_label(v) for v in one[c].to_numpy()]
            elif c == "SPECIES" and self.spec is not None:
                out[c] = np.full(n, cat_label(self.spec.species), dtype=object)
            else:
                out[c] = np.full(n, "UNK", dtype=object)
        for c in STATIC_REALS:
            out[c] = af[c].to_numpy(float)
        for c in KNOWN_REALS:
            if c != "DAY":
                out[c] = tf[c].to_numpy(float)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.net is None:
            raise RuntimeError("TFT adapter is not fitted")
        self.compound_ref.to_csv(path / "compound_reference.csv", index=False)
        (path / "dataset_parameters.pkl").write_bytes(pickle.dumps(self.dataset_parameters, protocol=4))
        ckpt = path / "tft.ckpt"
        if getattr(self, "_best_checkpoint_bytes", None):
            ckpt.write_bytes(self._best_checkpoint_bytes)
        else:
            trainer = _make_trainer(str(path), 1, [])
            trainer.strategy.connect(self.net)
            trainer.save_checkpoint(str(ckpt))
        (path / "model.json").write_text(json.dumps({
            "name": self.name, "task": self.task, "spec": self.spec.to_dict(),
            "quantiles": self.quantiles, "encoder_window_day": self.encoder_window_day,
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "min_encoder_length": self.min_encoder_length,
            "history": self.history,
            "not_the_required_transformer": True,
        }, indent=1, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TFTEarlyHistoryModel:
        path = Path(path)
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if d.get("name") != NAME:
            raise ValueError(f"refusing to load name={d.get('name')!r} as the optional TFT adapter")
        m = cls()
        m.spec = FeatureSpec.from_dict(d["spec"])
        m.quantiles = [float(x) for x in d["quantiles"]]
        m.encoder_window_day = float(d["encoder_window_day"])
        m.max_encoder_length = int(d["max_encoder_length"])
        m.max_prediction_length = int(d["max_prediction_length"])
        m.min_encoder_length = int(d["min_encoder_length"])
        m.history = d.get("history") or {}
        m.compound_ref = pd.read_csv(path / "compound_reference.csv")
        m.dataset_parameters = pickle.loads((path / "dataset_parameters.pkl").read_bytes())
        _, TemporalFusionTransformer, _ = _pf()
        m.net = TemporalFusionTransformer.load_from_checkpoint(path / "tft.ckpt")
        m.net.eval()
        return m


def _unit_row_weights(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["WEIGHT"] = 1.0
    return out


def _study_window_sampler(dataset, panel: pd.DataFrame):
    """Equal study probability over PF windows. Requires WEIGHT=1 (no double balance)."""
    from torch.utils.data import WeightedRandomSampler

    decoded = dataset.decoded_index.copy()
    decoded["USUBJID"] = decoded["USUBJID"].astype(str)
    key = panel[["USUBJID", "STUDYID"]].drop_duplicates()
    key["USUBJID"] = key["USUBJID"].astype(str)
    merged = decoded.merge(key, on="USUBJID", how="left")
    n = merged.groupby(merged["STUDYID"].astype(str)).size()
    w = merged["STUDYID"].astype(str).map(lambda s: 1.0 / float(n[s])).to_numpy(float)
    return WeightedRandomSampler(weights=w, num_samples=int(len(merged)), replacement=True)


def _loss_callback_cls():
    try:
        from lightning.pytorch.callbacks import Callback
    except ImportError:  # pragma: no cover
        from pytorch_lightning.callbacks import Callback

    class _LossCurveCallback(Callback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict[str, Any]] = []

        def on_validation_epoch_end(self, trainer, pl_module):
            row: dict[str, Any] = {"epoch": int(trainer.current_epoch)}
            for key, val in (trainer.callback_metrics or {}).items():
                if "loss" not in str(key):
                    continue
                try:
                    row[str(key)] = float(val.detach().cpu()) if hasattr(val, "detach") else float(val)
                except (TypeError, ValueError):
                    continue
            self.rows.append(row)

    return _LossCurveCallback


def _enough_visits(panel: pd.DataFrame, min_len: int) -> pd.DataFrame:
    if panel.empty:
        return panel
    n = panel.groupby("USUBJID")["time_idx"].transform("count")
    return panel.loc[n >= min_len].reset_index(drop=True)


def _unpack_predict(raw: Any, dataset) -> tuple[np.ndarray, pd.DataFrame | None]:
    """Accept the several return shapes PF ``predict(..., return_index=True)`` has used."""
    index = None
    pred = raw
    if hasattr(raw, "output") and hasattr(raw, "index"):
        pred, index = raw.output, raw.index
    elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
        pred, index = raw[0], raw[1]
    elif hasattr(raw, "prediction"):
        pred = raw.prediction
        index = getattr(raw, "index", None)
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred, float)
    if index is None:
        try:
            index = dataset.decoded_index
        except Exception:  # noqa: BLE001
            index = None
    if index is not None and not isinstance(index, pd.DataFrame):
        try:
            index = pd.DataFrame(index)
        except Exception:  # noqa: BLE001
            index = None
    return pred, index
