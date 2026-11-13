"""Simple PMSM electrical model and a 4-node lumped-parameter thermal network (LPTN).

Used for two things:
1. `synthetic_dataset()` makes fake data shaped like the Kaggle file, so tests and CI
   run without downloading anything.
2. The ThermoBench scenario generator turns speed/torque demands into d/q currents and
   voltages with `MotorModel`.

All parameters are ILLUSTRATIVE. They give plausible magnitudes for a ~50 kW traction
motor but are not the parameters of the Paderborn test motor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DT_S, GROUP, TARGETS


@dataclass
class MotorModel:
    pole_pairs: int = 4
    psi_pm: float = 0.11  # Vs, permanent magnet flux linkage
    l_d: float = 0.30e-3  # H
    l_q: float = 0.50e-3  # H
    r_s: float = 0.010  # Ohm at 20 C
    u_max: float = 160.0  # V, max phase voltage amplitude (DC link / sqrt(3))
    i_max: float = 450.0  # A
    torque_min: float = -240.0
    torque_max: float = 260.0

    def operate(self, speed_rpm: np.ndarray, torque_nm: np.ndarray) -> pd.DataFrame:
        """Map speed and torque demand to i_d, i_q, u_d, u_q (steady state, per sample).

        Below base speed: i_d = 0 control. Above it: field weakening keeps |u| <= u_max.
        """
        n = np.asarray(speed_rpm, dtype=np.float64)
        t = np.clip(np.asarray(torque_nm, dtype=np.float64), self.torque_min, self.torque_max)
        w = n * 2.0 * np.pi / 60.0 * self.pole_pairs  # electrical rad/s
        aw = np.abs(w) + 1e-9
        i_d_fw = -(aw * self.psi_pm - self.u_max) / (aw * self.l_d)
        i_d = np.clip(np.minimum(0.0, i_d_fw), -0.8 * self.i_max, 0.0)
        flux = self.psi_pm + (self.l_d - self.l_q) * i_d
        i_q = t / (1.5 * self.pole_pairs * flux)
        i_q_lim = np.sqrt(np.maximum(self.i_max**2 - i_d**2, 0.0))
        i_q = np.clip(i_q, -i_q_lim, i_q_lim)
        u_d = self.r_s * i_d - w * self.l_q * i_q
        u_q = self.r_s * i_q + w * (self.l_d * i_d + self.psi_pm)
        return pd.DataFrame({"u_d": u_d, "u_q": u_q, "i_d": i_d, "i_q": i_q, "motor_speed": n})


@dataclass
class ThermalParams:
    # Heat capacities, J/K
    c_pm: float = 6000.0
    c_yoke: float = 20000.0
    c_tooth: float = 6000.0
    c_wind: float = 4000.0
    # Thermal conductances, W/K
    g_yoke_cool: float = 90.0
    g_tooth_yoke: float = 60.0
    g_wind_tooth: float = 30.0
    g_wind_yoke: float = 15.0
    g_pm_tooth0: float = 3.0
    g_pm_tooth_per_krpm: float = 1.0
    g_pm_cool: float = 1.5
    g_amb: float = 0.5
    # Loss coefficients
    r_s20: float = 0.008
    alpha_cu: float = 0.0039
    k_iron: float = 0.12  # W per Hz^1.5 (scaled by flux ratio^2)
    k_pm: float = 1.8e-3  # rotor eddy losses, W per Hz^2 at reference current
    i_ref: float = 300.0


def simulate_lptn(
    inputs: pd.DataFrame,
    init: np.ndarray | None = None,
    params: ThermalParams | None = None,
    motor: MotorModel | None = None,
    dt_s: float = DT_S,
) -> np.ndarray:
    """Integrate the 4-node network (forward Euler). Returns array [n, 4] in TARGETS order."""
    p = params or ThermalParams()
    m = motor or MotorModel()
    amb = inputs["ambient"].to_numpy(np.float64)
    cool = inputs["coolant"].to_numpy(np.float64)
    i_d = inputs["i_d"].to_numpy(np.float64)
    i_q = inputs["i_q"].to_numpy(np.float64)
    n = inputs["motor_speed"].to_numpy(np.float64)
    i_s2 = i_d**2 + i_q**2
    f_el = np.abs(n) / 60.0 * m.pole_pairs
    flux_ratio = np.clip((m.psi_pm + m.l_d * i_d) / m.psi_pm, 0.2, 1.2)
    p_iron = p.k_iron * f_el**1.5 * flux_ratio**2
    p_pm = p.k_pm * f_el**2 * (0.3 + i_s2 / p.i_ref**2)

    T = np.empty((len(inputs), 4))
    x = np.array(init, dtype=np.float64) if init is not None else np.full(4, cool[0])
    for k in range(len(inputs)):
        t_pm, t_y, t_t, t_w = x
        r = p.r_s20 * (1.0 + p.alpha_cu * (t_w - 20.0))
        p_cu = 1.5 * r * i_s2[k]
        g_air = p.g_pm_tooth0 + p.g_pm_tooth_per_krpm * abs(n[k]) / 1000.0
        q_pm = p_pm[k] + g_air * (t_t - t_pm) + p.g_pm_cool * (cool[k] - t_pm) + p.g_amb * (amb[k] - t_pm)
        q_y = (
            0.6 * p_iron[k]
            + p.g_yoke_cool * (cool[k] - t_y)
            + p.g_tooth_yoke * (t_t - t_y)
            + p.g_wind_yoke * (t_w - t_y)
            + p.g_amb * (amb[k] - t_y)
        )
        q_t = (
            0.4 * p_iron[k]
            + p.g_tooth_yoke * (t_y - t_t)
            + p.g_wind_tooth * (t_w - t_t)
            + g_air * (t_pm - t_t)
        )
        q_w = p_cu + p.g_wind_tooth * (t_t - t_w) + p.g_wind_yoke * (t_y - t_w) + p.g_amb * (amb[k] - t_w)
        x = x + dt_s * np.array([q_pm / p.c_pm, q_y / p.c_yoke, q_t / p.c_tooth, q_w / p.c_wind])
        T[k] = x
    return T


def random_walk_profile(
    hours: float, rng: np.random.Generator, dt_s: float = DT_S
) -> tuple[np.ndarray, np.ndarray]:
    """Speed/torque references as piecewise-constant random walks with ramps,
    like the hand-designed excitation of the Kaggle data."""
    n = int(hours * 3600 / dt_s)
    speed = np.empty(n)
    torque = np.empty(n)
    s, t = 0.0, 0.0
    k = 0
    while k < n:
        hold = int(rng.uniform(20, 400) / dt_s)
        s_target = float(rng.choice([0, rng.uniform(500, 6000)], p=[0.1, 0.9]))
        t_target = float(rng.uniform(-80, 240)) if s_target > 0 else 0.0
        ramp = max(1, int(rng.uniform(5, 40) / dt_s))
        for j in range(min(hold, n - k)):
            a = min(1.0, (j + 1) / ramp)
            speed[k] = s + a * (s_target - s)
            torque[k] = t + a * (t_target - t)
            k += 1
        s, t = speed[k - 1], torque[k - 1]
    return speed, torque


def synthetic_dataset(
    n_profiles: int = 8,
    hours: tuple[float, float] = (0.5, 1.0),
    seed: int = 0,
    noise: bool = True,
) -> pd.DataFrame:
    """Fake data with the same columns as measures_v2.csv. For tests and demos only."""
    rng = np.random.default_rng(seed)
    motor = MotorModel()
    frames = []
    for pid in range(1, n_profiles + 1):
        h = rng.uniform(*hours)
        speed, torque = random_walk_profile(h, rng)
        n = len(speed)
        el = motor.operate(speed, torque)
        steps = np.arange(n)
        amb = 22.0 + rng.uniform(-3, 5) + 1.5 * np.sin(steps / (n + 1) * np.pi)
        cool_base = rng.uniform(18, 60)
        cool = cool_base + 3.0 * np.sin(steps / 3000.0 + rng.uniform(0, 6))
        df = el.assign(ambient=amb, coolant=cool)
        temps = simulate_lptn(df, init=np.full(4, cool[0] + rng.uniform(0, 5)), motor=motor)
        df = df.assign(**{t: temps[:, j] for j, t in enumerate(TARGETS)})
        df["torque"] = torque
        df[GROUP] = pid
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if noise:
        for c, s in [("u_d", 0.5), ("u_q", 0.5), ("i_d", 0.5), ("i_q", 0.5), ("motor_speed", 2.0)]:
            out[c] += rng.normal(0, s, len(out))
        for c in TARGETS + ["coolant", "ambient"]:
            out[c] += rng.normal(0, 0.1, len(out))
    cols = ["u_q", "coolant", "stator_winding", "u_d", "stator_tooth", "motor_speed", "i_d", "i_q", "pm",
            "stator_yoke", "ambient", "torque", GROUP]
    out = out[cols]
    float_cols = [c for c in cols if c != GROUP]
    out[float_cols] = out[float_cols].astype(np.float32)
    return out
