"""Accuracy, safety and alarm metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DT_S, GROUP, TARGETS


def regression_table(y_true: pd.DataFrame, y_pred: pd.DataFrame, under_k: float = 5.0) -> pd.DataFrame:
    """Per-target MSE, MAE, max |error| and share of time under-estimating by more than under_k."""
    rows = {}
    for t in [c for c in TARGETS if c in y_pred]:
        e = y_pred[t].to_numpy(np.float64) - y_true[t].to_numpy(np.float64)
        rows[t] = {
            "mse_K2": float(np.mean(e**2)),
            "mae_K": float(np.mean(np.abs(e))),
            "max_abs_K": float(np.max(np.abs(e))),
            f"under_{under_k:g}K_pct": float(100.0 * np.mean(e < -under_k)),
        }
    tab = pd.DataFrame(rows).T
    tab.loc["mean"] = tab.mean()
    return tab


def coverage(y_true: pd.DataFrame, upper: pd.DataFrame) -> pd.Series:
    return pd.Series({t: float(np.mean(y_true[t].to_numpy() <= upper[t].to_numpy())) for t in upper})


def _onsets(flag: np.ndarray) -> np.ndarray:
    f = flag.astype(np.int8)
    return np.flatnonzero(np.diff(np.concatenate([[0], f])) == 1)


def alarm_metrics(
    y_true: pd.DataFrame,
    signal: pd.DataFrame,
    groups: pd.Series,
    thresholds: dict[str, float],
    lookback_s: float = 600.0,
    dt_s: float = DT_S,
) -> pd.DataFrame:
    """Event-level alarm scoring.

    An *event* starts when the true temperature crosses its threshold upward. It is
    caught if the alarm signal (prediction or upper bound) crossed the threshold at most
    lookback_s before, and at the latest at the event itself. Lead time = how early.
    A *false alarm* is an alarm onset with no true crossing in the following lookback_s.
    """
    out = {}
    look = int(lookback_s / dt_s)
    for t, thr in thresholds.items():
        caught, leads, events, false_alarms, hours = 0, [], 0, 0, 0.0
        for pid in pd.unique(groups):
            m = (groups == pid).to_numpy()
            yt = y_true[t].to_numpy()[m] >= thr
            ys = signal[t].to_numpy()[m] >= thr
            hours += m.sum() * dt_s / 3600.0
            ev = _onsets(yt)
            al = _onsets(ys)
            events += len(ev)
            for e in ev:
                before = al[al <= e]
                last = before.max() if len(before) else None
                # caught if the alarm is on at the event, or went on within the lookback window
                if ys[e] or (last is not None and e - last <= look):
                    caught += 1
                    leads.append(min(e - last, look) * dt_s if last is not None else 0.0)
            for a in al:
                if not yt[a : a + look + 1].any():
                    false_alarms += 1
        out[t] = {
            "threshold_C": thr,
            "events": events,
            "recall": caught / events if events else np.nan,
            "median_lead_s": float(np.median(leads)) if leads else np.nan,
            "false_alarms_per_h": false_alarms / hours if hours else np.nan,
        }
    return pd.DataFrame(out).T


def per_profile_error(y_true: pd.DataFrame, y_pred: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    e = (y_pred[TARGETS] - y_true[TARGETS]).abs()
    e[GROUP] = groups.to_numpy()
    return e.groupby(GROUP).agg(["mean", "max"])
