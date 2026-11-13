"""One-sided conformal upper bounds.

Split conformal: on calibration sessions, take residuals r = y - y_hat and set
q = the ceil((n+1)(1-alpha))/n empirical quantile. Then y <= y_hat + q holds with
probability >= 1 - alpha if calibration and test data are exchangeable.

Time series are not exchangeable, so also report coverage on the test suite, and use
AdaptiveConformal online when the true temperature becomes observable (e.g. on a bench).
Residuals inside a session are strongly correlated; we subsample them with `stride`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGETS


def fit_upper(y_true: pd.DataFrame, y_pred: pd.DataFrame, alpha: float = 0.05, stride: int = 20) -> dict[str, float]:
    q = {}
    for t in TARGETS:
        r = (y_true[t].to_numpy() - y_pred[t].to_numpy())[::stride]
        n = len(r)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q[t] = float(np.quantile(r, level, method="higher"))
    return q


def apply_upper(y_pred: pd.DataFrame, q: dict[str, float]) -> pd.DataFrame:
    return y_pred.assign(**{t: y_pred[t] + q[t] for t in TARGETS})


class AdaptiveConformal:
    """Adaptive conformal inference (Gibbs & Candes, 2021) for one target, one-sided.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t), err_t = 1 if y_t > upper_t.
    """

    def __init__(self, residuals: np.ndarray, alpha: float = 0.05, gamma: float = 0.005, window: int = 5000):
        self.alpha = alpha
        self.alpha_t = alpha
        self.gamma = gamma
        self.window = window
        self.res = list(np.asarray(residuals, dtype=float)[-window:])

    def offset(self) -> float:
        a = float(np.clip(self.alpha_t, 1e-4, 0.999))
        return float(np.quantile(self.res, 1 - a))

    def update(self, y_true: float, y_pred: float) -> float:
        err = float(y_true > y_pred + self.offset())
        self.alpha_t += self.gamma * (self.alpha - err)
        self.res.append(y_true - y_pred)
        if len(self.res) > self.window:
            self.res.pop(0)
        return self.offset()
