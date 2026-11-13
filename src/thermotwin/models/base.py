"""Common model interface.

Every model:
    fit(train_df, val_df=None)            -> self
    predict(df, init=None)                -> DataFrame[TARGETS] aligned with df.index
    n_params()                            -> int

`init` (sequence models only) maps profile_id -> 4 starting temperatures. If omitted,
sequence models start from the measured targets when they are in df (as on the test
bench) and otherwise from the coolant temperature (as in a car after a long stop).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import TARGETS


class BaseModel:
    name: str = "base"
    kind: str = "row"  # "row" (tabular, per sample) or "seq" (recurrent with state)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, **kw) -> "BaseModel":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, init: dict[int, np.ndarray] | None = None, **kw) -> pd.DataFrame:
        raise NotImplementedError

    def n_params(self) -> int:
        return 0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseModel":
        return joblib.load(path)

    @staticmethod
    def _frame(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(values, dtype=np.float64), index=index, columns=TARGETS)
