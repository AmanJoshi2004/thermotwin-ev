"""Thermal headroom: how long the motor can hold its current load, and how much current
it can sustain, before the conformal upper bound of any temperature reaches its limit.

Works with the stateful sequence models (TNN, LPTN, GRU): we copy the live state and roll
the model forward on hypothetical future inputs.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_LIMITS, DT_S, GROUP, TARGETS
from .models.sequence import SeqModel


def _future(sample: dict, seconds: float, current_a: float | None = None) -> pd.DataFrame:
    n = max(1, int(seconds / DT_S))
    s = dict(sample)
    if current_a is not None:
        i_s = math.hypot(s["i_d"], s["i_q"])
        if i_s < 1e-6:
            s["i_d"], s["i_q"] = 0.0, current_a
        else:
            k = current_a / i_s
            s["i_d"], s["i_q"] = s["i_d"] * k, s["i_q"] * k
    df = pd.DataFrame([s] * n)
    df[GROUP] = 0
    return df


@dataclass
class HeadroomEngine:
    model: SeqModel
    q_upper: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in TARGETS})
    limits: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    horizon_s: float = 1800.0
    hold_s: float = 60.0
    i_max: float = 500.0

    def __post_init__(self):
        if not isinstance(self.model, SeqModel):
            raise TypeError("headroom needs a sequence model (tnn, lptn or gru)")
        self.streamer = self.model.streamer()
        self.q = np.array([self.q_upper.get(t, 0.0) for t in TARGETS])
        self.lim = np.array([self.limits[t] for t in TARGETS])
        self.last_sample: dict | None = None

    def reset(self, init: np.ndarray | None = None):
        self.streamer.reset(init)

    def sync(self, history: pd.DataFrame, init: np.ndarray | None = None) -> np.ndarray:
        """Jump to the end of a recorded history (one session) in one vectorised pass."""
        hist = history.copy()
        hist[GROUP] = 0
        p, states = self.model.predict(hist, init={0: init} if init is not None else None, return_state=True)
        self.streamer.reset()
        self.streamer.state = states[0]
        self.streamer.last = p.iloc[-1].to_numpy()
        self.last_sample = {c: float(hist[c].iloc[-1]) for c in ["ambient", "coolant", "u_d", "u_q", "i_d", "i_q",
                                                                  "motor_speed"]}
        return self.streamer.last

    def step(self, sample: dict) -> dict:
        """Advance with one 2 Hz sample. Returns estimates and upper bounds in C."""
        est = self.streamer.step(sample)
        self.last_sample = sample
        upper = est + self.q
        return {"estimate": dict(zip(TARGETS, est.round(2).tolist())),
                "upper": dict(zip(TARGETS, upper.round(2).tolist())),
                "alarm": {t: bool(u >= l) for t, u, l in zip(TARGETS, upper, self.lim)}}

    def _roll(self, sample: dict, seconds: float, current_a: float | None = None) -> np.ndarray:
        state = copy.deepcopy(self.streamer.state)
        return self.model.rollout(state, _future(sample, seconds, current_a)) + self.q

    def time_to_limit(self) -> dict[str, float | None]:
        """Seconds until each upper bound reaches its limit if the present load is held.
        None = not within the horizon."""
        if self.last_sample is None or self.streamer.state is None:
            return {t: None for t in TARGETS}
        up = self._roll(self.last_sample, self.horizon_s)
        step_s = self.model.ds * DT_S
        out = {}
        for j, t in enumerate(TARGETS):
            hit = np.flatnonzero(up[:, j] >= self.lim[j])
            out[t] = float((hit[0] + 1) * step_s) if len(hit) else None
        return out

    def max_current(self, iters: int = 10) -> float | None:
        """Largest stator current (A) that keeps every upper bound under its limit for hold_s."""
        if self.last_sample is None or self.streamer.state is None:
            return None
        lo, hi = 0.0, self.i_max
        if np.any(self._roll(self.last_sample, self.hold_s, 0.0).max(axis=0) >= self.lim):
            return 0.0
        if np.all(self._roll(self.last_sample, self.hold_s, hi).max(axis=0) < self.lim):
            return hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ok = np.all(self._roll(self.last_sample, self.hold_s, mid).max(axis=0) < self.lim)
            lo, hi = (mid, hi) if ok else (lo, mid)
        return round(lo, 1)
