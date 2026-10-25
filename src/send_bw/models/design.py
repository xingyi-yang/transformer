"""Fixed design-matrix construction for the spline and Bayesian mean functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


def bspline_basis(x: np.ndarray, knots_interior: list[float], lo: float, hi: float, degree: int = 3) -> np.ndarray:
    t = np.concatenate([[lo] * (degree + 1), knots_interior, [hi] * (degree + 1)])
    xx = np.clip(np.asarray(x, float), lo, hi - 1e-9)
    return BSpline.design_matrix(xx, t, degree).toarray()


@dataclass
class MeanDesign:
    """Spline mean design. Effect terms are zero for controls and before dosing starts."""

    tau_hi: float
    expo_hi: float
    off_hi: float
    time_knots: list[float]
    expo_knots: list[float]
    off_knots: list[float]
    strain_cols: list[str]
    route_cols: list[str]
    species_cols: list[str] = field(default_factory=list)
    rich: bool = True
    species_treatment: bool = False
    col_names: list[str] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)

    @staticmethod
    def from_features(F: pd.DataFrame, n_time_knots: int = 3, n_expo_knots: int = 2, rich: bool = True,
                      species_treatment: bool = False) -> MeanDesign:
        tau_hi = float(F["TAU"].max()) + 1e-6
        expo_hi = max(float(F["EXPO"].max()), 1e-3) + 1e-6
        off_hi = max(float(F["OFF"].max()), 1e-3) + 1e-6
        tk = list(np.quantile(F["TAU"], np.linspace(0, 1, n_time_knots + 2)[1:-1]))
        ek_src = F.loc[F["EXPO"] > 0, "EXPO"]
        ek = list(np.quantile(ek_src, np.linspace(0, 1, n_expo_knots + 2)[1:-1])) if len(ek_src) else []
        d = MeanDesign(tau_hi, expo_hi, off_hi, sorted(set(np.round(tk, 4))), sorted(set(np.round(ek, 4))), [],
                       [c for c in F.columns if c.startswith("STRAIN_")],
                       [c for c in F.columns if c.startswith("ROUTE_")],
                       [c for c in F.columns if c.startswith("SPECIES_")], rich,
                       species_treatment)
        return d

    def _blocks(self, F: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        TB = bspline_basis(F["TAU"], self.time_knots, 0.0, self.tau_hi, 3 if self.rich else 2)
        EB = bspline_basis(F["EXPO"], self.expo_knots, 0.0, self.expo_hi, 2)[:, 1:]
        OB = bspline_basis(F["OFF"], self.off_knots, 0.0, self.off_hi, 1)[:, 1:]
        sex = F["SEX_M"].to_numpy()[:, None]
        zb = F["ZB_C"].to_numpy()[:, None]
        ld = F["LD"].to_numpy()[:, None]
        blocks = [("tb", TB), ("tb_sex", TB * sex), ("tb_zb", TB * zb)]
        for c in self.strain_cols:
            blocks.append((f"tb_{c}", TB * F[c].to_numpy()[:, None]))
        for c in self.species_cols:
            blocks.append((f"tb_{c}", TB * F[c].to_numpy()[:, None]))
        blocks += [("eb_ld", EB * ld), ("eb_ld2", EB * ld**2)]
        for dname in ["D1", "D2", "D3"]:
            blocks.append((f"eb_ld_{dname}", EB * ld * F[dname].to_numpy()[:, None]))
        if self.rich:
            blocks.append(("eb_ld_inv", EB * ld * F["INV_INTERVAL"].to_numpy()[:, None]))
            for c in self.route_cols:
                blocks.append((f"eb_ld_{c}", EB * ld * F[c].to_numpy()[:, None]))
            blocks.append(("eb_ld_sex", EB * ld * sex))
            blocks.append(("ob_ld", OB * ld))
            blocks.append(("ob_ld_D1", OB * ld * F["D1"].to_numpy()[:, None]))
        else:
            blocks.append(("ob_ld", OB * ld))
        if self.species_treatment:
            for c in self.species_cols:
                sp = F[c].to_numpy()[:, None]
                blocks.append((f"eb_ld_{c}", EB * ld * sp))
                for dname in ["D1", "D2", "D3"]:
                    blocks.append((f"eb_ld_{dname}_{c}", EB * ld * F[dname].to_numpy()[:, None] * sp))
        return blocks

    def matrix(self, F: pd.DataFrame, fit_scale: bool = False) -> np.ndarray:
        blocks = self._blocks(F)
        X = np.concatenate([b for _, b in blocks], axis=1)
        names = [f"{n}[{i}]" for n, b in blocks for i in range(b.shape[1])]
        if fit_scale:
            s = X.std(0)
            s[s < 1e-8] = 1.0
            self.scale = s.tolist()
            self.col_names = names
        return X / np.asarray(self.scale)[None, :]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> MeanDesign:
        raw = dict(d)
        raw.setdefault("species_cols", [])
        raw.setdefault("species_treatment", False)
        return MeanDesign(**raw)
