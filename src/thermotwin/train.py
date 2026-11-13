"""Train and compare the model ladder on session-based splits.

Examples
--------
    # real data (download first, see README)
    python -m thermotwin.train --data data/measures_v2.csv --out runs/main

    # quick run on fake data, no download needed
    python -m thermotwin.train --synthetic --quick --out runs/smoke

Outputs in --out:
    models/<name>.joblib    trained models
    summary.csv             one row per model: size, time, mean and worst errors
    metrics.csv             per model and target
    predictions.parquet     development-set truth and every model's prediction
    conformal.json          one-sided 95% upper-bound offsets per model and target
    split.json              which sessions went where
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import conformal, data, metrics, models, splits
from .config import GROUP, LITERATURE_TEST_PROFILES_TNN, TARGETS
from .features import FeatureSpec, make_features
from .physics import synthetic_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="path to measures_v2.csv or a .parquet copy")
    src.add_argument("--synthetic", action="store_true", help="use generated fake data")
    p.add_argument("--models", default=",".join(models.LADDER), help="comma-separated subset of the ladder")
    p.add_argument("--dev-profiles", type=int, nargs="*", default=None,
                   help="held-out development sessions (default: literature 60 62 74, or 2 random for synthetic)")
    p.add_argument("--cal-profiles", type=int, nargs="*", default=None,
                   help="sessions used for early stopping and conformal calibration (default: 3 random from train)")
    p.add_argument("--epochs", type=int, default=40, help="epochs for gru/tnn/lptn")
    p.add_argument("--quick", action="store_true", help="few epochs and trees, for smoke tests")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def model_kwargs(name: str, args) -> dict:
    if name in {"gru", "tnn", "lptn"}:
        return {"epochs": 5 if args.quick else args.epochs, "seed": args.seed}
    if name == "lgbm":
        return {"n_estimators": 60 if args.quick else 600, "seed": args.seed}
    return {}


def main(argv=None) -> Path:
    args = parse_args(argv)
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        df = synthetic_dataset(n_profiles=6 if args.quick else 16, hours=(0.4, 0.8) if args.quick else (0.5, 1.5),
                               seed=args.seed)
        dev_ids = args.dev_profiles or splits.random_profiles(df, 1 if args.quick else 2, seed=args.seed)
    else:
        df = data.load(args.data)
        dev_ids = args.dev_profiles or LITERATURE_TEST_PROFILES_TNN

    rest, dev = splits.by_profiles(df, dev_ids)
    cal_ids = args.cal_profiles or splits.random_profiles(rest, 1 if args.quick else 3, seed=args.seed + 1)
    train, cal = splits.by_profiles(rest, cal_ids)
    print(f"sessions: train {train[GROUP].nunique()}, calibration {cal_ids}, development {dev_ids}")
    (out / "split.json").write_text(json.dumps(
        {"train": sorted(train[GROUP].unique().tolist()), "calibration": cal_ids, "development": dev_ids}, indent=2))

    spec = FeatureSpec()
    feat_cache: dict[str, pd.DataFrame] = {}

    def feats(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        if name not in feat_cache:
            feat_cache[name] = make_features(frame, spec)
        return feat_cache[name]

    preds = {"truth": dev[TARGETS].reset_index(drop=True)}
    rows, per_target, conf = [], [], {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n== {name}")
        m = models.make(name, **model_kwargs(name, args))
        t0 = time.time()
        uses_feats = name in {"linear_ewma", "lgbm"}
        if uses_feats:
            m.fit(train, cal if name == "lgbm" else None, feats=feats("train", train))
            p_dev = m.predict(dev, feats=feats("dev", dev))
            p_cal = m.predict(cal, feats=feats("cal", cal))
        else:
            m.fit(train, cal) if m.kind == "seq" else m.fit(train)
            p_dev, p_cal = m.predict(dev), m.predict(cal)
        secs = time.time() - t0
        m.save(out / "models" / f"{name}.joblib")

        q = conformal.fit_upper(cal, p_cal)
        conf[name] = q
        tab = metrics.regression_table(dev, p_dev)
        cov = metrics.coverage(dev, conformal.apply_upper(p_dev, q))
        rows.append({
            "model": name,
            "params": m.n_params(),
            "train_s": round(secs, 1),
            "mse_K2": tab.loc["mean", "mse_K2"],
            "mae_K": tab.loc["mean", "mae_K"],
            "max_abs_K": tab.loc[TARGETS, "max_abs_K"].max(),
            "pm_mse_K2": tab.loc["pm", "mse_K2"],
            "pm_max_abs_K": tab.loc["pm", "max_abs_K"],
            "upper_coverage": float(cov.mean()),
        })
        per_target.append(tab.assign(model=name).reset_index(names="target"))
        for t in TARGETS:
            preds[f"{name}__{t}"] = p_dev[t].reset_index(drop=True)
        print(tab.round(2).to_string())

    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(per_target).round(4).to_csv(out / "metrics.csv", index=False)
    pr = pd.DataFrame({k: v for k, v in preds.items() if k != "truth"})
    pr = pd.concat([preds["truth"], pr, dev[[GROUP]].reset_index(drop=True)], axis=1)
    pr.to_parquet(out / "predictions.parquet", index=False)
    (out / "conformal.json").write_text(json.dumps(conf, indent=2))
    print("\n" + summary.to_string(index=False))
    print(f"\nsaved to {out.resolve()}")
    return out


if __name__ == "__main__":
    main()
