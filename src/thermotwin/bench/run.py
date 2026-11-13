"""Run ThermoBench on a trained run folder.

    python -m thermotwin.bench.run --run runs/main --data data/measures_v2.csv

B1  generated driving scenarios: agreement with the reference network (fitted LPTN) and
    peak temperatures; plus physics property tests for every model.
B2  sensor faults injected into the held-out development sessions (needs --data).
B3  over-temperature alarm events on the development sessions (needs --data).

Writes CSV tables and report.md into <run>/thermobench/.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import conformal, data, metrics
from ..config import GROUP, TARGETS
from ..models import BaseModel
from . import faults, property_tests, scenarios


def _load_models(run: Path) -> dict[str, BaseModel]:
    return {p.stem: BaseModel.load(p) for p in sorted((run / "models").glob("*.joblib"))}


def run_b1(models: dict, reference: str, cycles_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scen, meta = scenarios.build_catalog(cycles_dir)
        notes = sorted({str(x.message) for x in w})
    ref = models[reference].predict(scen)
    agree, peaks = [], []
    for name, m in models.items():
        p = m.predict(scen)
        for pid, info in meta.items():
            mask = (scen[GROUP] == pid).to_numpy()
            peaks.append({"model": name, "scenario": info["name"],
                          **{f"peak_{t}": float(p.loc[mask, t].max()) for t in TARGETS}})
            if name == reference:
                continue
            e = (p.loc[mask, TARGETS] - ref.loc[mask, TARGETS]).to_numpy()
            agree.append({"model": name, "scenario": info["name"], "real_cycle": info["real_cycle"],
                          "mse_vs_ref_K2": float((e**2).mean()), "max_abs_vs_ref_K": float(np.abs(e).max())})
    return pd.DataFrame(agree), pd.DataFrame(peaks), {"scenarios": meta, "warnings": notes}


def run_b2(models: dict, dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in faults.FAULTS:
        d = faults.apply(dev, kind)
        for name, m in models.items():
            tab = metrics.regression_table(dev, m.predict(d))
            rows.append({"fault": kind, "model": name, "mse_K2": tab.loc["mean", "mse_K2"],
                         "max_abs_K": tab.loc[TARGETS, "max_abs_K"].max()})
    out = pd.DataFrame(rows)
    clean = out[out.fault == "clean"].set_index("model")["mse_K2"]
    out["mse_increase_K2"] = out["mse_K2"] - out["model"].map(clean)
    return out


def run_b3(models: dict, dev: pd.DataFrame, conf: dict, quantile: float = 0.95) -> pd.DataFrame:
    thr = {t: float(dev[t].quantile(quantile)) for t in TARGETS}
    rows = []
    for name, m in models.items():
        p = m.predict(dev)
        signals = {"prediction": p}
        if name in conf:
            signals["upper_bound"] = conformal.apply_upper(p, conf[name])
        for sig, s in signals.items():
            tab = metrics.alarm_metrics(dev, s, dev[GROUP], thr)
            tab = tab.assign(model=name, signal=sig).reset_index(names="target")
            rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder written by thermotwin.train")
    ap.add_argument("--reference", default="lptn", help="model used as the reference simulator in B1")
    ap.add_argument("--data", help="measures_v2.csv/parquet, for B2 and B3 on the development sessions")
    ap.add_argument("--cycles-dir", default="data/cycles")
    ap.add_argument("--skip-properties", action="store_true")
    a = ap.parse_args(argv)

    run = Path(a.run)
    out = run / "thermobench"
    out.mkdir(exist_ok=True)
    models = _load_models(run)
    if a.reference not in models:
        raise SystemExit(f"reference model {a.reference!r} not found in {run / 'models'}")
    conf = json.loads((run / "conformal.json").read_text()) if (run / "conformal.json").exists() else {}

    print("B1: driving scenarios")
    agree, peaks, meta = run_b1(models, a.reference, a.cycles_dir)
    agree.to_csv(out / "b1_agreement.csv", index=False)
    peaks.to_csv(out / "b1_peaks.csv", index=False)
    (out / "b1_scenarios.json").write_text(json.dumps(meta, indent=2, default=str))

    props = pd.DataFrame()
    if not a.skip_properties:
        print("B1: physics property tests")
        props = pd.concat([property_tests.run_all(m).assign(model=n) for n, m in models.items()], ignore_index=True)
        props.to_csv(out / "b1_properties.csv", index=False)

    b2 = b3 = None
    if a.data:
        split = json.loads((run / "split.json").read_text())
        df = data.load(a.data)
        dev = df[df[GROUP].isin(split["development"])]
        print("B2: sensor faults")
        b2 = run_b2(models, dev)
        b2.to_csv(out / "b2_faults.csv", index=False)
        print("B3: alarm events")
        b3 = run_b3(models, dev, conf)
        b3.to_csv(out / "b3_alarms.csv", index=False)

    lines = ["# ThermoBench report", ""]
    if meta["warnings"]:
        lines += ["**Warning:** approximate drive cycles were used. Download the EPA files for the final report.", ""]
    lines += ["## B1 agreement with the reference network (mean over scenarios)", "",
              agree.groupby("model")[["mse_vs_ref_K2", "max_abs_vs_ref_K"]].mean().round(2).to_markdown(), ""]
    if len(props):
        lines += ["## B1 physics property tests", "",
                  props.pivot(index="model", columns="test", values="passed").to_markdown(), ""]
    if b2 is not None:
        lines += ["## B2 sensor faults: mean MSE (K²)", "",
                  b2.pivot(index="model", columns="fault", values="mse_K2").round(2).to_markdown(), ""]
    if b3 is not None:
        piv = b3.groupby(["model", "signal"])[["recall", "median_lead_s", "false_alarms_per_h"]].mean().round(2)
        lines += ["## B3 alarm events (mean over targets)", "", piv.to_markdown(), ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
