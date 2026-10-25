"""Baseline-conditioned transformer (benchmark C). Custom architecture (NOT a TFT).

Context tokens (one per animal, pre-study information only):
  biology [sex, strain one-hot], baseline [centered baseline log weight],
  compound [curated descriptors, control flag], regimen [log dose, 1/interval, start, end, route].
Query tokens: actual query time and planned-exposure features at that time.
Attention: Attn(Q,K,V) = softmax(QK'/sqrt(d_k) + M)V with mask M that lets every query attend
only to the context tokens and itself, so a prediction at time t never depends on which other
times are queried (schedule invariance) and no outcome ever enters the inputs.
Loss: study-balanced weighted MSE on observed targets; padding masked. AdamW, early stopping
on validation loss. Parameter uncertainty: members trained on study-bootstrap resamples.
The joint trajectory layer (covariance) is SHARED with the spline model for the primary
comparison, so differences isolate the mean function.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData, ResponseModel, study_balanced_weights
from send_bw.models.covariance import HierarchicalCovariance
from send_bw.models.gaussian_layer import GaussianLayerMixin


def token_groups(spec: FeatureSpec) -> dict[str, list[str]]:
    groups = {
        "bio": ["SEX_M"] + [c for c in spec.animal_cols if c.startswith("STRAIN_")],
        "baseline": ["ZB_C"],
        "compound": ["D1", "D2", "D3", "IS_CONTROL"],
        "regimen": ["LD", "INV_INTERVAL", "START", "END"] + [c for c in spec.animal_cols if c.startswith("ROUTE_")],
    }
    species_cols = [c for c in spec.animal_cols if c.startswith("SPECIES_")]
    if species_cols:
        groups["species"] = species_cols
    return groups


QUERY_COLS = ["TAU", "EXPO", "OFF", "LCUM", "SQRT_TAU"]


class BaselineTransformer(nn.Module):
    def __init__(self, group_dims: dict[str, int], q_dim: int, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1, ff_mult: int = 2, treat_dev_dim: int = 0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.groups = list(group_dims)
        self.ctx_proj = nn.ModuleDict({g: nn.Linear(max(d, 1), d_model) for g, d in group_dims.items()})
        self.q_proj = nn.Sequential(nn.Linear(q_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.type_emb = nn.Parameter(torch.zeros(len(group_dims) + 1, d_model))
        nn.init.normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=ff_mult * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.n_heads = n_heads
        self.treat_dev = nn.Linear(treat_dev_dim, 1) if treat_dev_dim else None

    def forward(self, ctx: dict[str, torch.Tensor], q: torch.Tensor, q_pad: torch.Tensor) -> torch.Tensor:
        B, L, _ = q.shape
        toks = [self.ctx_proj[g](ctx[g]).unsqueeze(1) + self.type_emb[i] for i, g in enumerate(self.groups)]
        C = len(toks)
        x = torch.cat(toks + [self.q_proj(q) + self.type_emb[-1]], dim=1)
        S = C + L
        allow = torch.zeros(S, S, dtype=torch.bool)
        allow[:, :C] = True
        allow[C:, C:] = torch.eye(L, dtype=torch.bool)
        attn_mask = ~allow  # True = attention not allowed
        pad = torch.cat([torch.zeros(B, C, dtype=torch.bool), q_pad], dim=1)
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad)
        out = self.head(h[:, C:, :]).squeeze(-1)
        if self.treat_dev is not None and "species" in ctx and "regimen" in ctx:
            td = self.treat_dev(torch.cat([ctx["species"], ctx["regimen"]], dim=-1))
            out = out + td
        return out


class TransformerModel(GaussianLayerMixin, ResponseModel):
    name = "transformer"

    def __init__(self):
        self.spec: FeatureSpec | None = None
        self.cov = HierarchicalCovariance()
        self.compound_ref = pd.DataFrame()
        self.nets: list[BaselineTransformer] = []
        self.norm: dict[str, Any] = {}
        self.arch: dict[str, Any] = {}
        self.history: dict[str, Any] = {}

    @property
    def n_members(self) -> int:
        return len(self.nets)

    # ---------------------------------------------------------------- tensors
    def _features(self, animals: pd.DataFrame, days: list[np.ndarray]) -> tuple[dict, np.ndarray, np.ndarray]:
        af = self.spec.animal_features(animals.reset_index(drop=True), self.compound_ref)
        L = max(len(d) for d in days)
        n = len(animals)
        Q = np.zeros((n, L, len(QUERY_COLS)))
        pad = np.ones((n, L), bool)
        rows = animals.reset_index(drop=True)
        for i, d in enumerate(days):
            if not len(d):
                continue
            rep = rows.iloc[[i] * len(d)].reset_index(drop=True)
            tf = self.spec.time_features(d, rep)
            tf["SQRT_TAU"] = np.sqrt(np.clip(tf["TAU"], 0, None))
            Q[i, :len(d)] = tf[QUERY_COLS].to_numpy()
            pad[i, :len(d)] = False
        ctx = {g: af[cols].to_numpy(float) if cols else np.zeros((n, 1)) for g, cols in token_groups(self.spec).items()}
        return ctx, Q, pad

    def _normalize(self, ctx, Q, fit: bool = False):
        if fit:
            self.norm = {"ctx": {g: (v.mean(0).tolist(), np.where(v.std(0) < 1e-8, 1.0, v.std(0)).tolist())
                                 for g, v in ctx.items()},
                         "q": (Q.reshape(-1, Q.shape[-1]).mean(0).tolist(),
                               np.where(Q.reshape(-1, Q.shape[-1]).std(0) < 1e-8, 1.0,
                                        Q.reshape(-1, Q.shape[-1]).std(0)).tolist())}
        c = {g: torch.tensor((v - np.array(self.norm["ctx"][g][0])) / np.array(self.norm["ctx"][g][1]),
                             dtype=torch.float32) for g, v in ctx.items()}
        q = torch.tensor((Q - np.array(self.norm["q"][0])) / np.array(self.norm["q"][1]), dtype=torch.float32)
        return c, q

    def _data(self, d: ModelData, fit_norm: bool = False):
        animals = d.animals.reset_index(drop=True)
        g = d.obs.groupby("USUBJID")
        days, ys, ws = [], [], []
        w_all = pd.Series(study_balanced_weights(d.obs["STUDYID"].to_numpy()), index=d.obs.index)
        for u in animals["USUBJID"]:
            if u in g.groups:
                o = g.get_group(u).sort_values("DAY")
                days.append(o["DAY"].to_numpy(float))
                ys.append(o["Y"].to_numpy(float))
                ws.append(w_all.loc[o.index].to_numpy(float))
            else:
                days.append(np.zeros(0))
                ys.append(np.zeros(0))
                ws.append(np.zeros(0))
        ctx, Q, pad = self._features(animals, days)
        c, q = self._normalize(ctx, Q, fit=fit_norm)
        L = Q.shape[1]
        Y = np.zeros((len(animals), L))
        W = np.zeros((len(animals), L))
        for i, (y, w) in enumerate(zip(ys, ws, strict=True)):
            Y[i, :len(y)] = y
            W[i, :len(w)] = w
        return c, q, torch.tensor(pad), torch.tensor(Y, dtype=torch.float32), torch.tensor(W, dtype=torch.float32), \
            animals["STUDYID"].to_numpy()

    # ---------------------------------------------------------------- training
    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        self.spec, self.compound_ref = train.spec, train.compound_ref
        shared = config.get("shared_covariance")
        if shared is None:
            raise ValueError("transformer requires the shared covariance layer (fit the spline model first)")
        self.cov = copy.deepcopy(shared)
        torch.set_num_threads(int(config.get("torch_threads", 4)))
        c, q, pad, Y, W, sids = self._data(train, fit_norm=True)
        self.y_scale = float(Y[W > 0].std())
        cv, qv, padv, Yv, Wv, _ = self._data(validation)
        sharing = getattr(self.spec, "sharing", "context_specific")
        regularized = sharing == "selective_regularized"
        treat_dim = 0
        if regularized and "species" in c and "regimen" in c:
            treat_dim = int(c["species"].shape[1] + c["regimen"].shape[1])
        self.arch = {"group_dims": {g: int(v.shape[1]) for g, v in c.items()}, "q_dim": q.shape[-1],
                     "d_model": int(config.get("d_model", 32)), "n_heads": int(config.get("n_heads", 2)),
                     "n_layers": int(config.get("n_layers", 1)), "dropout": float(config.get("dropout", 0.1)),
                     "treat_dev_dim": treat_dim}
        seed = int(config.get("seed", 0))
        studies = np.unique(sids)
        rng = np.random.default_rng(seed)
        from send_bw.sharing import is_selective

        selective = is_selective(sharing)
        tgt_mask = None
        if selective and "SPECIES" in train.animals.columns:
            tgt_mask = (train.animals.reset_index(drop=True)["SPECIES"].astype(str).str.upper()
                        == str(self.spec.species).upper()).to_numpy()
        self.nets, hist = [], []
        for mbr in range(int(config.get("n_members", 3))):
            if mbr == 0:
                mult = np.ones(len(sids))
            else:
                pick = pd.Series(rng.choice(studies, size=len(studies), replace=True)).value_counts()
                mult = pd.Series(sids).map(pick).fillna(0).to_numpy()
            Wm = W * torch.tensor(mult, dtype=torch.float32)[:, None]
            if selective and tgt_mask is not None and tgt_mask.any():
                cfg1 = dict(config)
                cfg1["freeze_groups"] = []
                c1 = {g: v[tgt_mask] for g, v in c.items()}
                net, h1 = self._train_one(c1, q[tgt_mask], pad[tgt_mask], Y[tgt_mask], Wm[tgt_mask],
                                          cv, qv, padv, Yv, Wv, cfg1, seed + 1000 * mbr)
                cfg2 = dict(config)
                cfg2["init_state"] = net.state_dict()
                if regularized:
                    cfg2["freeze_groups"] = []
                    cfg2["pull_state"] = copy.deepcopy(net.state_dict())
                    cfg2["pull_groups"] = ["bio", "baseline", "species"]
                    cfg2["lambda_growth"] = float(config.get("lambda_growth", 8.0))
                    cfg2["lambda_treatment"] = float(config.get("lambda_treatment", 8.0))
                else:
                    cfg2["freeze_groups"] = ["bio", "baseline", "species"]
                net, h2 = self._train_one(c, q, pad, Y, Wm, cv, qv, padv, Yv, Wv, cfg2,
                                          seed + 1000 * mbr + 17)
                h = {"phase1": h1, "phase2": h2, "best_val_rmse": h2["best_val_rmse"],
                     "epochs_run": int(h1["epochs_run"]) + int(h2["epochs_run"]),
                     "selective_sharing": True, "variant": "regularized" if regularized else "v1"}
            else:
                net, h = self._train_one(c, q, pad, Y, Wm, cv, qv, padv, Yv, Wv, config,
                                         seed + 1000 * mbr)
            self.nets.append(net)
            h["member"] = mbr
            hist.append(h)
        self.history = {"members": hist, "y_scale": self.y_scale, "arch": self.arch,
                        "best_val_rmse_member0": hist[0]["best_val_rmse"]}
        if regularized:
            self.history["lambda_growth"] = float(config.get("lambda_growth", 8.0))
            self.history["lambda_treatment"] = float(config.get("lambda_treatment", 8.0))
        if selective:
            from send_bw.sharing import selective_record

            self.history["selective_sharing"] = selective_record(
                "transformer", "regularized" if regularized else "v1")
        return self.history

    def _train_one(self, c, q, pad, Y, W, cv, qv, padv, Yv, Wv, cfg, seed):
        torch.manual_seed(seed)
        net = BaselineTransformer(**self.arch)
        if cfg.get("init_state") is not None:
            net.load_state_dict(cfg["init_state"])
        freeze = set(cfg.get("freeze_groups") or [])
        if freeze:
            for g, mod in net.ctx_proj.items():
                if g in freeze:
                    for p in mod.parameters():
                        p.requires_grad = False
        params = [p for p in net.parameters() if p.requires_grad]
        if not params:
            raise ValueError("transformer selective training: no unfrozen parameters")
        opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 3e-3)),
                                weight_decay=float(cfg.get("weight_decay", 1e-3)))
        n = Y.shape[0]
        bs = int(cfg.get("batch_size", 64))
        best, best_state, bad = np.inf, None, 0
        curve = []
        gen = torch.Generator().manual_seed(seed)
        keep = torch.where(W.sum(1) > 0)[0]
        for epoch in range(int(cfg.get("max_epochs", 80))):
            net.train()
            perm = keep[torch.randperm(len(keep), generator=gen)]
            tot = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                pred = net({g: v[idx] for g, v in c.items()}, q[idx], pad[idx])
                w = W[idx]
                loss = (w * (pred - Y[idx] / self.y_scale) ** 2).sum() / w.sum().clamp_min(1e-8)
                pull = cfg.get("pull_state")
                if pull:
                    extra = torch.zeros((), dtype=loss.dtype)
                    n_pen = 0
                    for g in cfg.get("pull_groups") or []:
                        if g not in net.ctx_proj:
                            continue
                        for name, p in net.ctx_proj[g].named_parameters():
                            key = f"ctx_proj.{g}.{name}"
                            if key in pull:
                                extra = extra + ((p - pull[key].to(p.device)) ** 2).sum()
                                n_pen += int(p.numel())
                    if n_pen:
                        loss = loss + float(cfg.get("lambda_growth", 8.0)) * extra / n_pen
                if net.treat_dev is not None:
                    extra_t = torch.zeros((), dtype=loss.dtype)
                    n_t = 0
                    for p in net.treat_dev.parameters():
                        extra_t = extra_t + (p ** 2).sum()
                        n_t += int(p.numel())
                    if n_t:
                        loss = loss + float(cfg.get("lambda_treatment", 8.0)) * extra_t / n_t
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            net.eval()
            with torch.no_grad():
                pv = net(cv, qv, padv) * self.y_scale
                vl = float(((Wv * (pv - Yv) ** 2).sum() / Wv.sum()).sqrt())
            curve.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_rmse": vl})
            if vl < best - 1e-6:
                best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= int(cfg.get("patience", 10)):
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net, {"best_val_rmse": best, "epochs_run": len(curve), "curve": curve, "seed": seed,
                     "freeze_groups": list(cfg.get("freeze_groups") or [])}

    # ---------------------------------------------------------------- prediction
    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        times = np.asarray(times, float)
        ctx, Q, pad = self._features(context, [times] * len(context))
        c, q = self._normalize(ctx, Q)
        net = self.nets[0 if member is None else member]
        with torch.no_grad():
            out = []
            for i in range(0, len(context), 512):
                sl = slice(i, i + 512)
                out.append(net({g: v[sl] for g, v in c.items()}, q[sl], torch.tensor(pad[sl])).numpy())
        y = np.concatenate(out) * self.y_scale
        return context["BASELINE_LOGW"].to_numpy(float)[:, None] + y

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save([n.state_dict() for n in self.nets], path / "members.pt")
        self.compound_ref.to_csv(path / "compound_reference.csv", index=False)
        hist = {k: v for k, v in self.history.items()}
        (path / "model.json").write_text(json.dumps({
            "name": self.name, "spec": self.spec.to_dict(), "cov": self.cov.to_dict(), "norm": self.norm,
            "arch": self.arch, "y_scale": self.y_scale, "history": hist}, indent=1, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TransformerModel:
        path = Path(path)
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        m = cls()
        m.spec = FeatureSpec.from_dict(d["spec"])
        m.cov = HierarchicalCovariance.from_dict(d["cov"])
        m.norm = d["norm"]
        m.arch = d["arch"]
        m.y_scale = d["y_scale"]
        m.history = d["history"]
        m.compound_ref = pd.read_csv(path / "compound_reference.csv")
        for sd in torch.load(path / "members.pt", weights_only=True):
            arch = dict(m.arch)
            arch.setdefault("treat_dev_dim", 0)
            net = BaselineTransformer(**arch)
            net.load_state_dict(sd)
            net.eval()
            m.nets.append(net)
        return m
