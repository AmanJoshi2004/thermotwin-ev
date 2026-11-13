"""ThermoBench physics property tests: pass/fail rules that need no thermometer.

Each test builds inputs, runs the model starting from the coolant temperature, and checks
a physical expectation. Thresholds are design choices; state them in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TARGETS
from .scenarios import Scenario, build, constant_point


def _final(model, df: pd.DataFrame, minutes: float = 10.0) -> np.ndarray:
    p = model.predict(df)
    k = int(minutes * 60 * 2)
    return p[TARGETS].iloc[-k:].mean().to_numpy()


def coolant_shift(model, delta: float = 10.0, tol: float = 3.0) -> dict:
    """Raise coolant by `delta`: steady-state temperatures rise by delta +- tol."""
    a = constant_point(3000, 100, 180, 25, 40)
    b = constant_point(3000, 100, 180, 25, 40 + delta)
    diff = _final(model, b) - _final(model, a)
    ok = bool(np.all(np.abs(diff - delta) <= tol))
    return {"test": "coolant_shift", "passed": ok,
            "detail": "rise per target: " + ", ".join(f"{t}={d:.1f}" for t, d in zip(TARGETS, diff))}


def current_monotone(model, torques=(40, 100, 160, 220), tol: float = 0.5) -> dict:
    """More torque (current) at the same speed never predicts lower steady temperatures."""
    finals = np.array([_final(model, constant_point(2500, tq, 120, 25, 45)) for tq in torques])
    steps = np.diff(finals, axis=0)
    ok = bool(np.all(steps >= -tol))
    worst = steps.min(axis=0)
    return {"test": "current_monotone", "passed": ok,
            "detail": "smallest step per target: " + ", ".join(f"{t}={w:.1f}" for t, w in zip(TARGETS, worst))}


def cooldown(model, tol: float = 10.0) -> dict:
    """After hard driving and 60 min at standstill, temperatures fall to within tol of coolant."""
    s = Scenario(name="cooldown", cycle="us06", repeats=3, idle_after_min=60, ambient_c=25, coolant_c=45)
    df, _ = build(s, pid=1)
    p = model.predict(df)
    end = p[TARGETS].iloc[-120:].mean().to_numpy()
    peak = p[TARGETS].max().to_numpy()
    ok = bool(np.all(end - 45.0 <= tol) and np.all(end <= peak))
    return {"test": "cooldown", "passed": ok,
            "detail": "end minus coolant: " + ", ".join(f"{t}={e - 45:.1f}" for t, e in zip(TARGETS, end))}


def long_run_bounded(model, hours: float = 6.0, drift_tol: float = 2.0) -> dict:
    """Six hours of a repeated cycle: predictions stay in a sane range and settle (no drift)."""
    reps = int(hours * 3600 / 1370)
    s = Scenario(name="long", cycle="udds", repeats=reps, ambient_c=25, coolant_c=45)
    df, _ = build(s, pid=1)
    p = model.predict(df)[TARGETS].to_numpy()
    sane = bool(np.all(p > 45 - 10) and np.all(p < 250))
    n = len(p)
    one = 1370 * 2
    drift = np.abs(p[n - one:].mean(axis=0) - p[n - 2 * one: n - one].mean(axis=0))
    ok = sane and bool(np.all(drift <= drift_tol))
    return {"test": "long_run_bounded", "passed": ok,
            "detail": f"range ok={sane}; last-cycle drift: " + ", ".join(f"{t}={d:.2f}" for t, d in zip(TARGETS, drift))}


def deterministic(model) -> dict:
    df = constant_point(3000, 120, 20, 25, 45)
    a, b = model.predict(df).to_numpy(), model.predict(df).to_numpy()
    ok = bool(np.allclose(a, b))
    return {"test": "deterministic", "passed": ok, "detail": f"max diff {np.abs(a - b).max():.2e}"}


ALL = [coolant_shift, current_monotone, cooldown, long_run_bounded, deterministic]


def run_all(model) -> pd.DataFrame:
    return pd.DataFrame([t(model) for t in ALL])
