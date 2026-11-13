"""ThermoBench tier B1: driving scenarios you generate yourself.

Pipeline: drive cycle (vehicle speed) -> vehicle model (wheel force) -> motor speed and
torque -> MotorModel (d/q currents and voltages) -> the 7 model inputs at 2 Hz.

Vehicle and motor parameters are ILLUSTRATIVE (a compact EV sized so the 260 Nm,
6,000 rpm motor in the Kaggle data could drive it). State them in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DT_S, GROUP
from ..physics import MotorModel
from .cycles import load_cycle


@dataclass
class Vehicle:
    mass_kg: float = 1300.0
    cd_a_m2: float = 0.60
    c_rr: float = 0.010
    wheel_radius_m: float = 0.31
    gear_ratio: float = 5.0
    efficiency: float = 0.95
    air_density: float = 1.2

    def motor_demand(self, speed_mps: np.ndarray, dt_s: float, grade: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(speed_mps, dtype=float)
        a = np.gradient(v, dt_s)
        g = 9.81
        theta = np.arctan(grade)
        force = (self.mass_kg * a
                 + 0.5 * self.air_density * self.cd_a_m2 * v**2
                 + self.c_rr * self.mass_kg * g * np.cos(theta) * (v > 0.1)
                 + self.mass_kg * g * np.sin(theta))
        wheel_torque = force * self.wheel_radius_m
        motor_torque = np.where(wheel_torque >= 0, wheel_torque / (self.gear_ratio * self.efficiency),
                                wheel_torque * self.efficiency / self.gear_ratio)
        rpm = v / self.wheel_radius_m * self.gear_ratio * 60.0 / (2.0 * np.pi)
        return rpm, motor_torque


@dataclass
class Scenario:
    name: str
    cycle: str | None = "udds"  # udds, hwfet, us06, or None for a constant operating point
    repeats: int = 3
    ambient_c: float = 25.0
    coolant_c: float = 45.0
    weak_pump: bool = False  # coolant heats up with load instead of staying regulated
    grade: float = 0.0
    const_speed_kmh: float | None = None  # for cycle=None
    const_minutes: float = 0.0
    idle_after_min: float = 0.0  # standstill at the end (cool-down)
    tags: list[str] = field(default_factory=list)


def _to_2hz(t: np.ndarray, x: np.ndarray, dt: float = DT_S) -> tuple[np.ndarray, np.ndarray]:
    tt = np.arange(0.0, t[-1] + 1e-9, dt)
    return tt, np.interp(tt, t, x)


def build(s: Scenario, pid: int, cycles_dir: str = "data/cycles", vehicle: Vehicle | None = None,
          motor: MotorModel | None = None) -> tuple[pd.DataFrame, bool]:
    """Return (inputs at 2 Hz with a profile_id column, used_real_cycle)."""
    veh = vehicle or Vehicle()
    mot = motor or MotorModel()
    real = True
    if s.cycle is not None:
        cyc, real = load_cycle(s.cycle, cycles_dir)
        one_t, one_v = _to_2hz(cyc["time_s"].to_numpy(), cyc["speed_mps"].to_numpy())
        v = np.tile(one_v, s.repeats)
    else:
        v = np.full(int(s.const_minutes * 60 / DT_S), (s.const_speed_kmh or 0.0) / 3.6)
    if s.idle_after_min:
        v = np.concatenate([v, np.zeros(int(s.idle_after_min * 60 / DT_S))])
    rpm, torque = veh.motor_demand(v, DT_S, s.grade)
    # a vehicle at standstill on a slope is held by the brakes, not the motor
    torque = np.where(v < 0.1, 0.0, torque)
    el = mot.operate(rpm, torque)
    n = len(el)
    amb = np.full(n, s.ambient_c)
    cool = np.full(n, s.coolant_c)
    if s.weak_pump:
        load = np.hypot(el["i_d"], el["i_q"]).to_numpy() / mot.i_max
        a = 2.0 / (3600 / DT_S + 1)  # 30 min time constant
        ew = np.empty(n)
        acc = 0.0
        for k in range(n):
            acc = (1 - a) * acc + a * load[k]
            ew[k] = acc
        cool = cool + 60.0 * ew
    df = el.assign(ambient=amb, coolant=cool, torque=np.clip(torque, mot.torque_min, mot.torque_max))
    df[GROUP] = pid
    df["vehicle_speed_kmh"] = v * 3.6
    return df, real


CLIMATES = {
    "winter": dict(ambient_c=0.0, coolant_c=20.0),
    "mild": dict(ambient_c=25.0, coolant_c=45.0),
    "desert": dict(ambient_c=40.0, coolant_c=65.0),
}


def catalog() -> list[Scenario]:
    """The default ThermoBench B1 scenarios (each 20 min to ~2 h)."""
    out = []
    for cyc, reps in [("udds", 3), ("hwfet", 4), ("us06", 5)]:
        for cname, c in CLIMATES.items():
            out.append(Scenario(name=f"{cyc}_{cname}", cycle=cyc, repeats=reps, tags=[cyc, cname], **c))
        out.append(Scenario(name=f"{cyc}_weak_pump", cycle=cyc, repeats=reps, weak_pump=True,
                            tags=[cyc, "weak_pump"], **CLIMATES["mild"]))
    out.append(Scenario(name="hill_climb_6pct", cycle=None, const_speed_kmh=60, const_minutes=30, grade=0.06,
                        tags=["hill"], **CLIMATES["desert"]))
    out.append(Scenario(name="us06_then_cooldown", cycle="us06", repeats=4, idle_after_min=60,
                        tags=["cooldown"], **CLIMATES["mild"]))
    return out


def build_catalog(cycles_dir: str = "data/cycles", scenarios: list[Scenario] | None = None) -> tuple[pd.DataFrame, dict]:
    scen = scenarios or catalog()
    frames, meta = [], {}
    for i, s in enumerate(scen, start=1001):
        df, real = build(s, i, cycles_dir)
        frames.append(df)
        meta[i] = {"name": s.name, "real_cycle": real, "tags": s.tags, "hours": len(df) * DT_S / 3600}
    return pd.concat(frames, ignore_index=True), meta


def constant_point(speed_rpm: float, torque_nm: float, minutes: float, ambient_c: float, coolant_c: float,
                   pid: int = 1, motor: MotorModel | None = None) -> pd.DataFrame:
    """A constant motor operating point, used by the physics property tests."""
    mot = motor or MotorModel()
    n = int(minutes * 60 / DT_S)
    el = mot.operate(np.full(n, speed_rpm), np.full(n, torque_nm))
    df = el.assign(ambient=ambient_c, coolant=coolant_c)
    df[GROUP] = pid
    return df
