"""ThermoBench tier B2: sensor faults you inject on purpose."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GROUP

ELECTRICAL = ["u_d", "u_q", "i_d", "i_q", "motor_speed"]
FAULTS = ["clean", "noise", "stuck_coolant", "dropout_10pct", "ambient_bias_2C"]


def apply(df: pd.DataFrame, kind: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    if kind == "clean":
        return out
    if kind == "noise":
        for c in ELECTRICAL:
            out[c] = out[c] + rng.normal(0, 0.02 * float(df[c].std()), len(df))
        return out
    if kind == "stuck_coolant":
        for _, idx in out.groupby(GROUP).groups.items():
            idx = list(idx)
            k = int(0.2 * len(idx))
            out.loc[idx[k:], "coolant"] = out.loc[idx[k], "coolant"]
        return out
    if kind == "dropout_10pct":
        # dropped samples keep the last received value (sample and hold)
        drop = rng.random(len(df)) < 0.10
        cols = ["ambient", "coolant"] + ELECTRICAL
        for _, idx in out.groupby(GROUP).groups.items():
            idx = np.asarray(list(idx))
            d = drop[out.index.get_indexer(idx)]
            d[0] = False
            block = out.loc[idx, cols].to_numpy(copy=True)
            for k in np.flatnonzero(d):
                block[k] = block[k - 1]
            out.loc[idx, cols] = block
        return out
    if kind == "ambient_bias_2C":
        out["ambient"] = out["ambient"] + 2.0
        return out
    raise ValueError(f"unknown fault {kind!r}; choose from {FAULTS}")
