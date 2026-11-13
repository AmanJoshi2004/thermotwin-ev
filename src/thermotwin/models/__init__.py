"""Model registry: the six-rung ladder plus the fitted reference network."""

from __future__ import annotations

from .base import BaseModel
from .sequence import GRUModel, LPTNModel, TNNModel
from .tabular import CoolantFloor, LGBMModel, LinearModel

LADDER = ["coolant_floor", "linear_raw", "linear_ewma", "lgbm", "gru", "tnn", "lptn"]


def make(name: str, **kw) -> BaseModel:
    if name == "coolant_floor":
        return CoolantFloor()
    if name == "linear_raw":
        return LinearModel("raw", **kw)
    if name == "linear_ewma":
        return LinearModel("ewma", **kw)
    if name == "lgbm":
        return LGBMModel(**kw)
    if name == "gru":
        return GRUModel(**kw)
    if name == "tnn":
        return TNNModel(**kw)
    if name == "lptn":
        return LPTNModel(**kw)
    raise ValueError(f"unknown model {name!r}; choose from {LADDER}")


__all__ = ["BaseModel", "CoolantFloor", "LinearModel", "LGBMModel", "GRUModel", "TNNModel",
           "LPTNModel", "LADDER", "make"]
