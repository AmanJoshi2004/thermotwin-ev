"""Physics-motivated features and causal moving averages.

The same recursion runs offline (vectorised with scipy.signal.lfilter) and online
(`StreamingFeatures`, one sample at a time), so training and serving see identical
features. `tests/test_features.py` checks that they match.

EWMA:      mu_t  = (1 - a) mu_{t-1} + a x_t,              a = 2 / (span + 1), mu_0 = x_0
EW var:    v_t   = (1 - a) (v_{t-1} + a (x_t - mu_{t-1})^2),   v_0 = 0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import EWMA_SPANS, EWSTD_COLS, GROUP, INPUTS

DERIVED: list[str] = ["i_s", "u_s", "s_app", "p_el", "i_s_sq", "i_s_speed", "s_speed"]
BASE: list[str] = INPUTS + DERIVED


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add current/voltage magnitudes, power and loss proxies. Returns a copy."""
    out = df.copy()
    i_d, i_q = out["i_d"].to_numpy(np.float64), out["i_q"].to_numpy(np.float64)
    u_d, u_q = out["u_d"].to_numpy(np.float64), out["u_q"].to_numpy(np.float64)
    n = out["motor_speed"].to_numpy(np.float64)
    i_s = np.sqrt(i_d**2 + i_q**2)
    u_s = np.sqrt(u_d**2 + u_q**2)
    out["i_s"] = i_s
    out["u_s"] = u_s
    out["s_app"] = 1.5 * u_s * i_s / 1000.0  # kVA
    out["p_el"] = 1.5 * (u_d * i_d + u_q * i_q) / 1000.0  # kW
    out["i_s_sq"] = i_s**2 / 1000.0  # copper-loss proxy
    out["i_s_speed"] = i_s * n / 1e5
    out["s_speed"] = out["s_app"].to_numpy() * n / 1e4  # iron-loss proxy
    return out


def derived_row(sample: dict[str, float]) -> dict[str, float]:
    """Same as add_derived for one sample (used online)."""
    i_s = float(np.hypot(sample["i_d"], sample["i_q"]))
    u_s = float(np.hypot(sample["u_d"], sample["u_q"]))
    s_app = 1.5 * u_s * i_s / 1000.0
    n = float(sample["motor_speed"])
    row = {c: float(sample[c]) for c in INPUTS}
    row.update(
        i_s=i_s,
        u_s=u_s,
        s_app=s_app,
        p_el=1.5 * (sample["u_d"] * sample["i_d"] + sample["u_q"] * sample["i_q"]) / 1000.0,
        i_s_sq=i_s**2 / 1000.0,
        i_s_speed=i_s * n / 1e5,
        s_speed=s_app * n / 1e4,
    )
    return row


def _alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def _ewma_1d(x: np.ndarray, span: int) -> np.ndarray:
    a = _alpha(span)
    zi = np.array([(1.0 - a) * x[0]])
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=zi)
    return y


def _ewstd_1d(x: np.ndarray, mu: np.ndarray, span: int) -> np.ndarray:
    a = _alpha(span)
    prev_mu = np.concatenate([[x[0]], mu[:-1]])
    d2 = (x - prev_mu) ** 2
    v = lfilter([(1.0 - a) * a], [1.0, -(1.0 - a)], d2)
    return np.sqrt(np.maximum(v, 0.0))


@dataclass
class FeatureSpec:
    spans: list[int] = field(default_factory=lambda: list(EWMA_SPANS))
    ewstd_cols: list[str] = field(default_factory=lambda: list(EWSTD_COLS))
    base: list[str] = field(default_factory=lambda: list(BASE))

    def names(self) -> list[str]:
        cols = list(self.base)
        for s in self.spans:
            cols += [f"{c}_ewma_{s}" for c in self.base]
        for s in self.spans:
            cols += [f"{c}_ewstd_{s}" for c in self.ewstd_cols]
        return cols


def make_features(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Raw + derived + EWMA/EW-std features, computed causally and per session.

    Returns a float32 frame aligned with df's index (targets are not included).
    """
    spec = spec or FeatureSpec()
    d = add_derived(df) if "i_s" not in df else df
    blocks = []
    for _, g in d.groupby(GROUP, sort=False):
        cols: dict[str, np.ndarray] = {c: g[c].to_numpy(np.float64) for c in spec.base}
        out = dict(cols)
        mus: dict[tuple[str, int], np.ndarray] = {}
        for s in spec.spans:
            for c in spec.base:
                mu = _ewma_1d(cols[c], s)
                mus[(c, s)] = mu
                out[f"{c}_ewma_{s}"] = mu
        for s in spec.spans:
            for c in spec.ewstd_cols:
                out[f"{c}_ewstd_{s}"] = _ewstd_1d(cols[c], mus[(c, s)], s)
        blocks.append(pd.DataFrame(out, index=g.index)[spec.names()])
    feats = pd.concat(blocks).loc[df.index]
    return feats.astype(np.float32)


class StreamingFeatures:
    """Online version of make_features for one session, one sample at a time."""

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()
        self.alphas = {s: _alpha(s) for s in self.spec.spans}
        self.mu: dict[tuple[str, int], float] = {}
        self.var: dict[tuple[str, int], float] = {}
        self.started = False

    def reset(self) -> None:
        self.mu.clear()
        self.var.clear()
        self.started = False

    def update(self, sample: dict[str, float]) -> np.ndarray:
        row = derived_row(sample)
        out: dict[str, float] = {c: row[c] for c in self.spec.base}
        for s in self.spec.spans:
            a = self.alphas[s]
            for c in self.spec.base:
                x = row[c]
                if not self.started:
                    prev = x
                    self.var[(c, s)] = 0.0
                else:
                    prev = self.mu[(c, s)]
                    self.var[(c, s)] = (1.0 - a) * (self.var[(c, s)] + a * (x - prev) ** 2)
                self.mu[(c, s)] = (1.0 - a) * prev + a * x
                out[f"{c}_ewma_{s}"] = self.mu[(c, s)]
        for s in self.spec.spans:
            for c in self.spec.ewstd_cols:
                out[f"{c}_ewstd_{s}"] = float(np.sqrt(max(self.var[(c, s)], 0.0)))
        self.started = True
        return np.array([out[n] for n in self.spec.names()], dtype=np.float32)
