"""Row-wise models (rungs 0-3). Memory comes only from the EWMA features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import INPUTS, TARGETS
from ..features import FeatureSpec, StreamingFeatures, make_features
from .base import BaseModel


class CoolantFloor(BaseModel):
    """Rung 0: target = coolant + a constant offset per target."""

    name = "coolant_floor"

    def fit(self, train, val=None, **kw):
        self.offset_ = (train[TARGETS].sub(train["coolant"], axis=0)).mean().to_numpy()
        return self

    def predict(self, df, init=None, **kw):
        v = df["coolant"].to_numpy()[:, None] + self.offset_[None, :]
        return self._frame(v, df.index)

    def predict_row(self, sample: dict) -> np.ndarray:
        return sample["coolant"] + self.offset_

    def n_params(self):
        return len(TARGETS)


class LinearModel(BaseModel):
    """Rung 1 (features='raw') and rung 2 (features='ewma'): ridge regression."""

    def __init__(self, features: str = "ewma", alpha: float = 1.0, spec: FeatureSpec | None = None):
        assert features in {"raw", "ewma"}
        self.features = features
        self.alpha = alpha
        self.spec = spec or FeatureSpec()
        self.name = f"linear_{features}"

    def _X(self, df: pd.DataFrame, feats: pd.DataFrame | None = None) -> np.ndarray:
        if self.features == "raw":
            return df[INPUTS].to_numpy(np.float64)
        f = feats if feats is not None else make_features(df, self.spec)
        return f.to_numpy(np.float64)

    def fit(self, train, val=None, feats=None, **kw):
        self.pipe_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.pipe_.fit(self._X(train, feats), train[TARGETS].to_numpy(np.float64))
        return self

    def predict(self, df, init=None, feats=None, **kw):
        return self._frame(self.pipe_.predict(self._X(df, feats)), df.index)

    def streamer(self):
        return _RowStreamer(self)

    def n_params(self):
        r = self.pipe_[-1]
        return int(r.coef_.size + r.intercept_.size)


class LGBMModel(BaseModel):
    """Rung 3: one LightGBM regressor per target on the EWMA feature set."""

    name = "lgbm"

    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.05, num_leaves: int = 63,
                 row_stride: int = 4, spec: FeatureSpec | None = None, seed: int = 0):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                           verbose=-1)
        self.row_stride = row_stride  # train on every k-th row: neighbours are near duplicates
        self.spec = spec or FeatureSpec()
        self.features = "ewma"

    def fit(self, train, val=None, feats=None, **kw):
        import lightgbm as lgb

        f = feats if feats is not None else make_features(train, self.spec)
        sl = slice(None, None, self.row_stride)
        X = f.iloc[sl].copy()
        self.feature_names_ = list(f.columns)
        self.models_ = {}
        fv = make_features(val, self.spec) if val is not None else None
        for t in TARGETS:
            m = lgb.LGBMRegressor(**self.params)
            fit_kw = {}
            if fv is not None:
                fit_kw = dict(eval_set=[(fv.iloc[sl].copy(), val[t].iloc[sl].copy())],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            m.fit(X, train[t].iloc[sl].copy(), **fit_kw)
            self.models_[t] = m
        return self

    def predict(self, df, init=None, feats=None, **kw):
        f = feats if feats is not None else make_features(df, self.spec)
        v = np.column_stack([self.models_[t].predict(f) for t in TARGETS])
        return self._frame(v, df.index)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({t: m.booster_.feature_importance("gain") for t, m in self.models_.items()},
                            index=self.feature_names_)

    def streamer(self):
        return _RowStreamer(self)

    def n_params(self):
        # number of tree nodes, the usual size measure for tree ensembles
        return int(sum(m.booster_.trees_to_dataframe().shape[0] for m in self.models_.values()))


class _RowStreamer:
    """Online inference for row-wise models: streaming features + per-sample predict."""

    def __init__(self, model):
        self.model = model
        self.feats = StreamingFeatures(getattr(model, "spec", None))

    def reset(self):
        self.feats.reset()

    def step(self, sample: dict) -> np.ndarray:
        x = self.feats.update(sample)
        m = self.model
        if isinstance(m, LinearModel):
            if m.features == "raw":
                xr = np.array([[sample[c] for c in INPUTS]], dtype=np.float64)
                return m.pipe_.predict(xr)[0]
            return m.pipe_.predict(x[None, :].astype(np.float64))[0]
        if isinstance(m, LGBMModel):
            xf = pd.DataFrame(x[None, :], columns=m.feature_names_)
            return np.array([m.models_[t].predict(xf)[0] for t in TARGETS])
        raise TypeError(type(m))


