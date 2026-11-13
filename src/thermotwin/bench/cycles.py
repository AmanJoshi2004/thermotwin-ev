"""EPA drive cycles (public speed-versus-time traces, not ML datasets).

Download them once with `python scripts/download_epa_cycles.py`. The files are tab
separated: time in seconds and speed in mph, after two header lines. If a file is
missing, a clearly labelled *approximate* cycle is generated instead so that code and
tests still run; use the real files for the report.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EPA_BASE = "https://www.epa.gov/sites/default/files/2015-10"
EPA_FILES = {
    "udds": "uddscol.txt",  # city
    "hwfet": "hwycol.txt",  # highway
    "us06": "us06col.txt",  # aggressive
}
MPH = 0.44704


def parse_epa(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            t, v = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        rows.append((t, v * MPH))
    if len(rows) < 50:
        raise ValueError(f"could not parse a drive cycle from {path}")
    return pd.DataFrame(rows, columns=["time_s", "speed_mps"])


def approximate_cycle(name: str, seed: int = 0) -> pd.DataFrame:
    """Modal stand-in (accelerate, cruise, brake, idle) with roughly the cycle's character."""
    rng = np.random.default_rng(seed)
    cfg = {
        "udds": dict(dur=1370, vmax=(8, 25), accel=1.2, idle=(5, 40), cruise=(10, 60)),
        "hwfet": dict(dur=765, vmax=(20, 27), accel=0.8, idle=(0, 0), cruise=(60, 200)),
        "us06": dict(dur=600, vmax=(15, 36), accel=3.0, idle=(3, 15), cruise=(5, 40)),
    }[name]
    v, out = 0.0, []
    while len(out) < cfg["dur"]:
        target = rng.uniform(*cfg["vmax"])
        while v < target:
            v = min(target, v + cfg["accel"] * rng.uniform(0.5, 1.0))
            out.append(v)
        out += [target + rng.normal(0, 0.3) for _ in range(int(rng.uniform(*cfg["cruise"])))]
        v = out[-1]
        stop = name != "hwfet" and rng.random() < 0.6
        floor = 0.0 if stop else rng.uniform(0.4, 0.8) * v
        while v > floor:
            v = max(floor, v - cfg["accel"] * rng.uniform(0.6, 1.0))
            out.append(v)
        if stop:
            out += [0.0] * int(rng.uniform(*cfg["idle"]))
    s = np.clip(np.array(out[: cfg["dur"]]), 0, None)
    s[0] = 0.0
    return pd.DataFrame({"time_s": np.arange(len(s), dtype=float), "speed_mps": s})


def load_cycle(name: str, cycles_dir: str | Path = "data/cycles") -> tuple[pd.DataFrame, bool]:
    """Return (cycle, is_real). is_real is False when the approximate stand-in was used."""
    path = Path(cycles_dir) / EPA_FILES[name]
    if path.exists():
        return parse_epa(path), True
    warnings.warn(f"{path} not found: using an APPROXIMATE {name} cycle. Run scripts/download_epa_cycles.py.")
    return approximate_cycle(name), False
