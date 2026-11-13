"""Recurrent models with a state (rungs 4-5 and the fitted reference LPTN).

- GRUModel: black-box recurrent network (rung 4).
- TNNModel: thermal neural network (rung 5, proposed). Its state is the 4 temperatures;
  small networks output thermal conductances and power losses, and learnable heat
  capacities integrate them (after Kirchgaessner, Wallscheid & Boecker, 2023).
- LPTNModel: the same structure with constant conductances and non-negative linear loss
  terms. Few parameters, physically monotone. Used as the ThermoBench reference simulator.

All three are trained on session data downsampled by window averaging (default 4 -> 0.5 Hz)
with truncated backpropagation through time, and run in physical time (dt in seconds),
so the TNN and LPTN can also step at 2 Hz online.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import DT_S, GROUP, TARGETS
from ..features import add_derived
from .base import BaseModel

TEMP_COLS = ["ambient", "coolant"]
OTHER_COLS = ["u_d", "u_q", "i_d", "i_q", "motor_speed", "i_s", "u_s", "p_el", "i_s_sq"]
T_SCALE = 100.0  # degrees C per model unit


# --------------------------------------------------------------------------------------
# data handling
# --------------------------------------------------------------------------------------
def _downsample(a: np.ndarray, k: int) -> np.ndarray:
    n = (len(a) // k) * k
    if n == 0:
        return a[:1]
    head = a[:n].reshape(-1, k, *a.shape[1:]).mean(axis=1)
    if n < len(a):
        head = np.concatenate([head, a[n:].mean(axis=0, keepdims=True)])
    return head


class _Prep:
    """Scales inputs, downsamples per session and builds padded [T, B, F] tensors."""

    def __init__(self, ds: int):
        self.ds = ds

    def fit(self, df: pd.DataFrame) -> "_Prep":
        d = add_derived(df)
        self.scale_ = np.maximum(np.abs(d[OTHER_COLS].to_numpy(np.float64)).max(axis=0), 1e-6)
        return self

    def transform_rows(self, d: pd.DataFrame) -> np.ndarray:
        t = d[TEMP_COLS].to_numpy(np.float64) / T_SCALE
        o = d[OTHER_COLS].to_numpy(np.float64) / self.scale_
        return np.concatenate([t, o], axis=1).astype(np.float32)

    def sessions(self, df: pd.DataFrame, with_targets: bool):
        d = add_derived(df)
        out = []
        for pid, g in d.groupby(GROUP, sort=False):
            x = _downsample(self.transform_rows(g), self.ds)
            y = _downsample(g[TARGETS].to_numpy(np.float32) / T_SCALE, self.ds) if with_targets else None
            out.append((int(pid), g.index, x, y))
        return out

    @staticmethod
    def pad(seqs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        L = max(len(s) for s in seqs)
        F = seqs[0].shape[1]
        X = np.zeros((L, len(seqs), F), dtype=np.float32)
        M = np.zeros((L, len(seqs)), dtype=np.float32)
        for b, s in enumerate(seqs):
            X[: len(s), b] = s
            M[: len(s), b] = 1.0
        return torch.from_numpy(X), torch.from_numpy(M)


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------
N_NODES = len(TARGETS) + len(TEMP_COLS)  # 4 states + ambient + coolant
PAIRS = [(i, j) for i in range(N_NODES) for j in range(i + 1, N_NODES) if i < len(TARGETS)]


class TNNCell(nn.Module):
    """dT_i/dt = C_i^-1 * ( P_i + sum_j G_ij (T_j - T_i) )."""

    def __init__(self, n_in: int, hidden: int = 16):
        super().__init__()
        n_t = len(TARGETS)
        self.cond = nn.Sequential(nn.Linear(n_in + n_t, hidden), nn.Tanh(), nn.Linear(hidden, len(PAIRS)), nn.Sigmoid())
        self.loss = nn.Sequential(nn.Linear(n_in + n_t, hidden), nn.Tanh(), nn.Linear(hidden, n_t))
        self.log_inv_cap = nn.Parameter(torch.full((n_t,), math.log(2e-3)))
        i, j = zip(*PAIRS)
        self.register_buffer("pi", torch.tensor(i))
        self.register_buffer("pj", torch.tensor(j))

    def forward(self, x: torch.Tensor, T: torch.Tensor, dt: float) -> torch.Tensor:
        z = torch.cat([x, T], dim=-1)
        g = self.cond(z)  # [B, pairs]
        p = torch.abs(self.loss(z))  # [B, 4]
        nodes = torch.cat([T, x[:, : len(TEMP_COLS)]], dim=-1)  # [B, 6]
        flow = g * (nodes[:, self.pj] - nodes[:, self.pi])  # heat from j into i
        q = p.clone()
        q = q.index_add(1, self.pi, flow)
        back = -flow[:, self.pj < len(TARGETS)]
        q = q.index_add(1, self.pj[self.pj < len(TARGETS)], back)
        return T + dt * torch.exp(self.log_inv_cap) * q


class LPTNCell(nn.Module):
    """Physics-only network: constant conductances, non-negative linear losses."""

    def __init__(self, i_s_idx: int, speed_idx: int):
        super().__init__()
        n_t = len(TARGETS)
        self.i_s_idx, self.speed_idx = i_s_idx, speed_idx
        self.g_raw = nn.Parameter(torch.full((len(PAIRS),), -2.0))
        self.g_air_speed = nn.Parameter(torch.tensor(-3.0))
        self.w_raw = nn.Parameter(torch.full((n_t, 4), -3.0))
        self.log_inv_cap = nn.Parameter(torch.full((n_t,), math.log(2e-3)))
        i, j = zip(*PAIRS)
        self.register_buffer("pi", torch.tensor(i))
        self.register_buffer("pj", torch.tensor(j))
        # index of the rotor-to-tooth pair (pm=0, stator_tooth=2): airgap convection grows with speed
        self.air = PAIRS.index((0, 2))

    def forward(self, x: torch.Tensor, T: torch.Tensor, dt: float) -> torch.Tensor:
        i_s = x[:, self.i_s_idx].abs()
        n = x[:, self.speed_idx].abs()
        phi = torch.stack([i_s**2, n**1.5, n**2, i_s**2 * n], dim=-1)  # loss proxies, all >= 0
        p = phi @ nn.functional.softplus(self.w_raw).T
        g = nn.functional.softplus(self.g_raw).expand(x.shape[0], -1).clone()
        g[:, self.air] = g[:, self.air] + nn.functional.softplus(self.g_air_speed) * n
        nodes = torch.cat([T, x[:, : len(TEMP_COLS)]], dim=-1)
        flow = g * (nodes[:, self.pj] - nodes[:, self.pi])
        q = p.index_add(1, self.pi, flow)
        back = -flow[:, self.pj < len(TARGETS)]
        q = q.index_add(1, self.pj[self.pj < len(TARGETS)], back)
        return T + dt * torch.exp(self.log_inv_cap) * q


class _ThermalNet(nn.Module):
    """Wraps a cell: state = temperatures (scaled)."""

    def __init__(self, cell: nn.Module):
        super().__init__()
        self.cell = cell

    def init_state(self, y0: torch.Tensor) -> torch.Tensor:
        return y0

    def forward(self, X: torch.Tensor, state: torch.Tensor, dt: float):
        outs = []
        T = state
        for k in range(X.shape[0]):
            T = self.cell(X[k], T, dt)
            T = T.clamp(-0.5, 3.0)
            outs.append(T)
        return torch.stack(outs), T


class _GRUNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.h0 = nn.Linear(len(TARGETS), hidden)
        self.gru = nn.GRU(n_in, hidden)
        self.head = nn.Linear(hidden, len(TARGETS))

    def init_state(self, y0: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.h0(y0)).unsqueeze(0)

    def forward(self, X: torch.Tensor, state: torch.Tensor, dt: float):
        h, hn = self.gru(X, state)
        coolant = X[..., 1:2]
        return coolant + self.head(h), hn


# --------------------------------------------------------------------------------------
# trainer / wrapper
# --------------------------------------------------------------------------------------
class SeqModel(BaseModel):
    kind = "seq"

    def __init__(self, ds: int = 4, chunk: int = 256, epochs: int = 40, lr: float = 5e-3,
                 seed: int = 0, verbose: bool = True, max_minutes: float | None = None):
        self.ds = ds
        self.chunk = chunk
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.max_minutes = max_minutes

    # subclasses build self.net_
    def _build(self, n_in: int) -> nn.Module:
        raise NotImplementedError

    @property
    def dt_train(self) -> float:
        return DT_S * self.ds

    def fit(self, train, val=None, **kw):
        torch.manual_seed(self.seed)
        self.prep_ = _Prep(self.ds).fit(train)
        sess = self.prep_.sessions(train, with_targets=True)
        X, M = _Prep.pad([s[2] for s in sess])
        Y, _ = _Prep.pad([s[3] for s in sess])
        self.net_ = self._build(X.shape[-1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        best, best_state, t0 = np.inf, None, time.time()
        self.history_ = []
        for ep in range(self.epochs):
            self.net_.train()
            state = self.net_.init_state(Y[0])
            tot, cnt = 0.0, 0.0
            for s in range(0, X.shape[0], self.chunk):
                x, y, m = X[s : s + self.chunk], Y[s : s + self.chunk], M[s : s + self.chunk]
                out, state = self.net_(x, state, self.dt_train)
                err = ((out - y) ** 2).mean(-1) * m
                loss = err.sum() / m.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                state = state.detach()
                tot += float(err.detach().sum())
                cnt += float(m.sum())
            sched.step()
            train_mse = tot / cnt * T_SCALE**2
            val_mse = self._val_mse(val) if val is not None else train_mse
            self.history_.append({"epoch": ep, "train_mse_K2": train_mse, "val_mse_K2": val_mse})
            if val_mse < best:
                best, best_state = val_mse, copy.deepcopy(self.net_.state_dict())
            if self.verbose:
                print(f"  [{self.name}] epoch {ep + 1:3d}  train {train_mse:8.2f} K2  val {val_mse:8.2f} K2")
            if self.max_minutes and (time.time() - t0) / 60 > self.max_minutes:
                break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _val_mse(self, val: pd.DataFrame) -> float:
        p = self.predict(val)
        return float(((p[TARGETS] - val[TARGETS]) ** 2).to_numpy().mean())

    def _init_for(self, pid: int, g: pd.DataFrame, init: dict | None) -> np.ndarray:
        if init and pid in init:
            return np.asarray(init[pid], dtype=np.float32)
        if all(t in g for t in TARGETS):
            return g[TARGETS].iloc[: self.ds].to_numpy(np.float32).mean(axis=0)
        return np.full(len(TARGETS), float(g["coolant"].iloc[0]), dtype=np.float32)

    @torch.no_grad()
    def predict(self, df, init=None, return_state=False, **kw):
        self.net_.eval()
        d = add_derived(df)
        out = np.empty((len(df), len(TARGETS)))
        pos = pd.Series(np.arange(len(df)), index=df.index)
        states = {}
        for pid, g in d.groupby(GROUP, sort=False):
            x = _downsample(self.prep_.transform_rows(g), self.ds)
            y0 = torch.from_numpy(self._init_for(int(pid), g, init) / T_SCALE)[None, :]
            st = self.net_.init_state(y0)
            pred, st = self.net_(torch.from_numpy(x)[:, None, :], st, self.dt_train)
            p = pred[:, 0, :].numpy() * T_SCALE
            # window k covers samples [k*ds, (k+1)*ds); place its value at the window end
            centers = np.minimum(np.arange(len(p)) * self.ds + self.ds - 1, len(g) - 1)
            idx = np.arange(len(g))
            vals = np.column_stack([np.interp(idx, centers, p[:, j]) for j in range(p.shape[1])])
            out[pos.loc[g.index].to_numpy()] = vals
            states[int(pid)] = st
        res = self._frame(out, df.index)
        return (res, states) if return_state else res

    @torch.no_grad()
    def rollout(self, state, future: pd.DataFrame) -> np.ndarray:
        """Continue from a state over future rows (one session). Returns [n_windows, 4] in C."""
        self.net_.eval()
        x = _downsample(self.prep_.transform_rows(add_derived(future)), self.ds)
        pred, _ = self.net_(torch.from_numpy(x)[:, None, :], state, self.dt_train)
        return pred[:, 0, :].numpy() * T_SCALE

    def streamer(self):
        return _SeqStreamer(self)

    def n_params(self):
        return int(sum(p.numel() for p in self.net_.parameters()))


class GRUModel(SeqModel):
    name = "gru"

    def __init__(self, hidden: int = 32, **kw):
        super().__init__(**kw)
        self.hidden = hidden

    def _build(self, n_in):
        return _GRUNet(n_in, self.hidden)


class TNNModel(SeqModel):
    name = "tnn"

    def __init__(self, hidden: int = 16, **kw):
        kw.setdefault("lr", 1e-2)
        super().__init__(**kw)
        self.hidden = hidden

    def _build(self, n_in):
        return _ThermalNet(TNNCell(n_in, self.hidden))


class LPTNModel(SeqModel):
    name = "lptn"

    def __init__(self, **kw):
        kw.setdefault("lr", 2e-2)
        super().__init__(**kw)

    def _build(self, n_in):
        i_s = len(TEMP_COLS) + OTHER_COLS.index("i_s")
        n = len(TEMP_COLS) + OTHER_COLS.index("motor_speed")
        return _ThermalNet(LPTNCell(i_s, n))


class _SeqStreamer:
    """Online inference: average `ds` samples, then advance the state one step."""

    def __init__(self, model: SeqModel):
        self.m = model
        self.reset()

    def reset(self, init: np.ndarray | None = None):
        self.buf: list[np.ndarray] = []
        self.state = None
        self.init = init
        self.last = None

    @torch.no_grad()
    def step(self, sample: dict) -> np.ndarray:
        row = pd.DataFrame([sample])
        x = self.m.prep_.transform_rows(add_derived(row))[0]
        if self.state is None:
            y0 = self.init if self.init is not None else np.full(len(TARGETS), sample["coolant"])
            self.state = self.m.net_.init_state(torch.tensor(np.asarray(y0, np.float32) / T_SCALE)[None, :])
            self.last = np.asarray(y0, dtype=np.float64)
        self.buf.append(x)
        if len(self.buf) == self.m.ds:
            xa = torch.from_numpy(np.mean(self.buf, axis=0, dtype=np.float32))[None, None, :]
            pred, self.state = self.m.net_(xa, self.state, self.m.dt_train)
            self.last = pred[0, 0].numpy().astype(np.float64) * T_SCALE
            self.buf = []
        return self.last
