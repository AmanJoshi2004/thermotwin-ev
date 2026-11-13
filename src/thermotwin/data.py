"""Loading and validating the Kaggle electric motor temperature data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUP, INPUTS, TARGETS

REQUIRED = INPUTS + TARGETS + [GROUP]


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, require_targets: bool = True) -> None:
    """Raise DataValidationError if the frame is not usable."""
    needed = INPUTS + [GROUP] + (TARGETS if require_targets else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    cols = [c for c in needed if c != GROUP]
    if df[cols].isna().any().any():
        bad = df[cols].columns[df[cols].isna().any()].tolist()
        raise DataValidationError(f"NaN values in: {bad}")
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise DataValidationError("non-finite values found")
    temps = [c for c in ["ambient", "coolant"] + (TARGETS if require_targets else []) if c in df]
    lo, hi = df[temps].min().min(), df[temps].max().max()
    if lo < -40 or hi > 250:
        raise DataValidationError(f"temperatures outside [-40, 250] C: min={lo:.1f}, max={hi:.1f}")


def load(path: str | Path, require_targets: bool = True) -> pd.DataFrame:
    """Load measures_v2.csv (or a Parquet copy), validate it and sort it by session.

    Row order inside each session is time order, so it is preserved.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    validate(df, require_targets=require_targets)
    df[GROUP] = df[GROUP].astype(int)
    # Stable sort keeps time order inside a session.
    df = df.sort_values(GROUP, kind="stable").reset_index(drop=True)
    float_cols = [c for c in df.columns if c != GROUP]
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def to_parquet(csv_path: str | Path, out_path: str | Path) -> Path:
    df = load(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: duration and temperature ranges."""
    g = df.groupby(GROUP)
    out = pd.DataFrame(
        {
            "hours": g.size() / 2.0 / 3600.0,
            "pm_max": g["pm"].max() if "pm" in df else np.nan,
            "winding_max": g["stator_winding"].max() if "stator_winding" in df else np.nan,
            "coolant_mean": g["coolant"].mean(),
            "speed_max": g["motor_speed"].max(),
        }
    )
    return out.round(2)
