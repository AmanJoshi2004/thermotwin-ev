"""Experiments behind the eight insights in the project plan.

    python -m thermotwin.experiments leakage        --data data/measures_v2.csv
    python -m thermotwin.experiments learning-curve --data data/measures_v2.csv
    python -m thermotwin.experiments sensor-value   --data data/measures_v2.csv
    python -m thermotwin.experiments importance     --data data/measures_v2.csv
    python -m thermotwin.experiments error-anatomy  --data data/measures_v2.csv --run runs/main
    python -m thermotwin.experiments all            --synthetic

Each writes a CSV and a PNG into --out (default runs/experiments).
They use the fast row-wise models so they finish on a laptop; swap in others if you like.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data, metrics, models, splits
from .config import GROUP, LITERATURE_TEST_PROFILES_TNN, TARGETS
from .features import FeatureSpec, add_derived, make_features
from .physics import synthetic_dataset

GREY, BLUE = "#9a9a9a", "#2a6fdb"


def _mean_mse(y, p) -> float:
    return float(metrics.regression_table(y, p).loc["mean", "mse_K2"])


def leakage(df: pd.DataFrame, dev_ids: list[int], out: Path) -> pd.DataFrame:
    """Insight 5: a random row split looks far better than an honest session split."""
    rows = []
    for name in ["linear_raw", "linear_ewma", "lgbm"]:
        kw = {"n_estimators": 300} if name == "lgbm" else {}
        tr, te = splits.random_row_split(df, test_frac=0.2)
        # features are computed on the full sessions first, then split by row (as a naive notebook would)
        f_all = make_features(df) if name != "linear_raw" else None
        m = models.make(name, **kw)
        if f_all is not None:
            m.fit(tr, feats=f_all.loc[tr.index])
            p = m.predict(te, feats=f_all.loc[te.index])
        else:
            m.fit(tr)
            p = m.predict(te)
        rows.append({"model": name, "split": "random rows", "mse_K2": _mean_mse(te, p)})
        tr, te = splits.by_profiles(df, dev_ids)
        m = models.make(name, **kw).fit(tr)
        rows.append({"model": name, "split": "whole sessions", "mse_K2": _mean_mse(te, m.predict(te))})
    res = pd.DataFrame(rows)
    piv = res.pivot(index="model", columns="split", values="mse_K2")
    ax = piv[["random rows", "whole sessions"]].plot.bar(color=[GREY, BLUE], figsize=(7, 4), rot=0)
    ax.set_ylabel("mean MSE, K²")
    ax.set_title("Random row splits hide most of the error")
    plt.tight_layout()
    plt.savefig(out / "leakage.png", dpi=150)
    plt.close()
    return res


def learning_curve(df: pd.DataFrame, dev_ids: list[int], out: Path, steps: int = 6) -> pd.DataFrame:
    """Insight 6: how many hours of bench data are needed."""
    rest, dev = splits.by_profiles(df, dev_ids)
    ids = rest[GROUP].unique()
    rng = np.random.default_rng(0)
    order = rng.permutation(ids)
    rows = []
    for frac in np.linspace(1.0 / steps, 1.0, steps):
        take = order[: max(1, int(round(frac * len(order))))]
        tr = rest[rest[GROUP].isin(take)]
        hours = len(tr) / 2 / 3600
        for name in ["linear_ewma", "lgbm"]:
            m = models.make(name, **({"n_estimators": 300} if name == "lgbm" else {})).fit(tr)
            rows.append({"model": name, "train_hours": hours, "mse_K2": _mean_mse(dev, m.predict(dev))})
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in [("linear_ewma", BLUE), ("lgbm", GREY)]:
        r = res[res.model == name]
        ax.plot(r.train_hours, r.mse_K2, "o-", color=c, label=name)
    ax.set_xlabel("hours of training data")
    ax.set_ylabel("mean MSE, K²")
    ax.set_title("Error against bench time")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "learning_curve.png", dpi=150)
    plt.close()
    return res


def sensor_value(df: pd.DataFrame, dev_ids: list[int], out: Path) -> pd.DataFrame:
    """Insight 8: does one stator-winding thermistor help the rotor estimate more than a bigger model?"""
    tr, te = splits.by_profiles(df, dev_ids)
    rows = []
    base = FeatureSpec()
    with_sensor = FeatureSpec(base=base.base + ["stator_winding"])
    for label, spec in [("no extra sensor", base), ("+ winding sensor", with_sensor)]:
        for name in ["linear_ewma", "lgbm"]:
            m = models.make(name, spec=spec, **({"n_estimators": 300} if name == "lgbm" else {})).fit(
                tr, feats=make_features(tr, spec))
            p = m.predict(te, feats=make_features(te, spec))
            tab = metrics.regression_table(te, p)
            rows.append({"inputs": label, "model": name, "pm_mse_K2": tab.loc["pm", "mse_K2"],
                         "pm_max_abs_K": tab.loc["pm", "max_abs_K"]})
    res = pd.DataFrame(rows)
    piv = res.pivot(index="model", columns="inputs", values="pm_mse_K2")
    ax = piv.plot.bar(color=[BLUE, GREY], figsize=(7, 4), rot=0)
    ax.set_ylabel("rotor MSE, K²")
    ax.set_title("Value of one extra stator sensor for the rotor estimate")
    plt.tight_layout()
    plt.savefig(out / "sensor_value.png", dpi=150)
    plt.close()
    return res


def importance(df: pd.DataFrame, dev_ids: list[int], out: Path) -> pd.DataFrame:
    """Insights 3-4: which signals and which time scales each target relies on (LightGBM gain)."""
    tr, _ = splits.by_profiles(df, dev_ids)
    m = models.make("lgbm", n_estimators=300).fit(tr)
    imp = m.importance()
    imp = imp / imp.sum()

    def span_of(f: str) -> str:
        for key in ("_ewma_", "_ewstd_"):
            if key in f:
                return f"{int(f.split(key)[1]) / 120:g} min"
        return "instant"

    def signal_of(f: str) -> str:
        return f.split("_ewma_")[0].split("_ewstd_")[0]

    by_span = imp.groupby(imp.index.map(span_of)).sum()
    by_signal = imp.groupby(imp.index.map(signal_of)).sum()
    order = ["instant"] + sorted([s for s in by_span.index if s != "instant"], key=lambda s: float(s.split()[0]))
    ax = by_span.loc[order].T.plot.bar(stacked=True, figsize=(8, 4), rot=0, colormap="Blues")
    ax.set_ylabel("share of LightGBM gain")
    ax.set_title("Time scale each target relies on")
    plt.tight_layout()
    plt.savefig(out / "importance_by_span.png", dpi=150)
    plt.close()
    by_signal.to_csv(out / "importance_by_signal.csv")
    return by_span.loc[order]


def error_anatomy(df: pd.DataFrame, run: Path, out: Path) -> pd.DataFrame:
    """Insight 7: error against current transients and time since session start."""
    pr = pd.read_parquet(run / "predictions.parquet")
    dev = df[df[GROUP].isin(pr[GROUP].unique())].reset_index(drop=True)
    d = add_derived(dev)
    di = d.groupby(GROUP)["i_s"].diff().abs().fillna(0).to_numpy() * 2  # A/s
    t_min = d.groupby(GROUP).cumcount().to_numpy() / 120.0
    rows = []
    model_names = sorted({c.split("__")[0] for c in pr.columns if "__" in c})
    for name in model_names:
        e = np.abs(pr[f"{name}__pm"].to_numpy() - pr["pm"].to_numpy())
        for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 1e9)]:
            m = (t_min >= lo) & (t_min < hi)
            rows.append({"model": name, "bucket": f"minute {lo:g}-{hi if hi < 1e9 else 'end'}",
                         "kind": "time since start", "pm_mae_K": float(e[m].mean()) if m.any() else np.nan})
        q = np.quantile(di, [0, 0.5, 0.9, 0.99, 1.0])
        for lo, hi, lab in zip(q[:-1], q[1:], ["calm (<p50)", "p50-p90", "p90-p99", "fastest 1%"]):
            m = (di >= lo) & (di <= hi)
            rows.append({"model": name, "bucket": lab, "kind": "current change rate",
                         "pm_mae_K": float(e[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["leakage", "learning-curve", "sensor-value", "importance",
                                           "error-anatomy", "all"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--run", help="training run folder (for error-anatomy)")
    ap.add_argument("--dev-profiles", type=int, nargs="*")
    ap.add_argument("--out", default="runs/experiments")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        df = synthetic_dataset(n_profiles=10, hours=(0.4, 0.8), seed=3)
        dev_ids = a.dev_profiles or [9, 10]
    else:
        df = data.load(a.data)
        dev_ids = a.dev_profiles or LITERATURE_TEST_PROFILES_TNN
    todo = ["leakage", "learning-curve", "sensor-value", "importance"] if a.experiment == "all" else [a.experiment]
    if a.experiment == "all" and a.run:
        todo.append("error-anatomy")
    for name in todo:
        print(f"== {name}")
        if name == "error-anatomy":
            if not a.run:
                raise SystemExit("error-anatomy needs --run")
            res = error_anatomy(df, Path(a.run), out)
        else:
            res = {"leakage": leakage, "learning-curve": learning_curve, "sensor-value": sensor_value,
                   "importance": importance}[name](df, dev_ids, out)
        res.to_csv(out / f"{name.replace('-', '_')}.csv")
        print(res.round(3).to_string())
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
