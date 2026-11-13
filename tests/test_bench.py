import warnings

import numpy as np
import pandas as pd

from thermotwin import conformal, metrics, models, splits
from thermotwin.bench import faults, property_tests, scenarios
from thermotwin.bench.cycles import approximate_cycle
from thermotwin.config import GROUP, INPUTS, TARGETS
from thermotwin.headroom import HeadroomEngine


def test_scenarios_have_model_inputs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, meta = scenarios.build_catalog()
    assert set(INPUTS + [GROUP]).issubset(df.columns)
    assert np.isfinite(df[INPUTS].to_numpy()).all()
    assert len(meta) == len(scenarios.catalog())
    assert df["motor_speed"].max() <= 6500


def test_approximate_cycles_are_plausible():
    for name in ["udds", "hwfet", "us06"]:
        c = approximate_cycle(name)
        assert c.speed_mps.min() >= 0 and c.speed_mps.max() < 45


def test_faults_keep_shape(small_df):
    for k in faults.FAULTS:
        d = faults.apply(small_df, k)
        assert d.shape == small_df.shape
        assert np.isfinite(d[INPUTS].to_numpy()).all()


def test_property_tests_run(small_df):
    m = models.make("linear_ewma").fit(small_df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = property_tests.run_all(m)
    assert set(res.columns) == {"test", "passed", "detail"}
    assert res.loc[res.test == "deterministic", "passed"].item()


def test_conformal_upper_covers(small_df):
    tr, cal = splits.by_profiles(small_df, [4])
    m = models.make("linear_ewma").fit(tr)
    p = m.predict(cal)
    q = conformal.fit_upper(cal, p, alpha=0.1, stride=1)
    cov = metrics.coverage(cal, conformal.apply_upper(p, q))
    assert (cov >= 0.89).all()


def test_alarm_metrics_simple():
    y = pd.DataFrame({t: np.r_[np.zeros(100), np.ones(100) * 10] for t in TARGETS})
    s = pd.DataFrame({t: np.r_[np.zeros(90), np.ones(110) * 10] for t in TARGETS})
    g = pd.Series(np.zeros(200, dtype=int))
    tab = metrics.alarm_metrics(y, s, g, {t: 5.0 for t in TARGETS}, lookback_s=60)
    assert (tab["recall"] == 1.0).all()
    assert np.allclose(tab["median_lead_s"], 5.0)


def test_headroom_engine(small_df):
    tr, te = splits.by_profiles(small_df, [4])
    m = models.make("lptn", epochs=2, verbose=False).fit(tr)
    eng = HeadroomEngine(m, limits={t: 500.0 for t in TARGETS}, horizon_s=120, hold_s=30)
    eng.sync(te.iloc[:200])
    ttl = eng.time_to_limit()
    assert all(v is None for v in ttl.values())  # 500 C is never reached
    assert eng.max_current() == eng.i_max
    eng2 = HeadroomEngine(m, limits={t: -100.0 for t in TARGETS})
    eng2.sync(te.iloc[:200])
    assert eng2.max_current() == 0.0
