import numpy as np
import pytest

from thermotwin import metrics, models, splits
from thermotwin.config import GROUP, INPUTS, TARGETS


@pytest.mark.parametrize("name", models.LADDER)
def test_each_model_fits_and_predicts(small_df, name, tmp_path):
    tr, te = splits.by_profiles(small_df, [4])
    kw = {"epochs": 2, "verbose": False} if name in {"gru", "tnn", "lptn"} else {}
    if name == "lgbm":
        kw = {"n_estimators": 20}
    m = models.make(name, **kw).fit(tr)
    p = m.predict(te)
    assert list(p.columns) == TARGETS
    assert p.index.equals(te.index)
    assert np.isfinite(p.to_numpy()).all()
    assert m.n_params() > 0
    path = m.save(tmp_path / f"{name}.joblib")
    np.testing.assert_allclose(models.BaseModel.load(path).predict(te).to_numpy(), p.to_numpy())


def test_ewma_beats_raw_linear():
    """Insight 1 in miniature: memory features matter (needs a few sessions of data)."""
    from thermotwin.physics import synthetic_dataset

    df = synthetic_dataset(n_profiles=10, hours=(0.4, 0.8), seed=3)
    tr, te = splits.by_profiles(df, [9, 10])
    raw = metrics.regression_table(te, models.make("linear_raw").fit(tr).predict(te)).loc["mean", "mse_K2"]
    ewma = metrics.regression_table(te, models.make("linear_ewma").fit(tr).predict(te)).loc["mean", "mse_K2"]
    assert ewma < raw


def test_seq_streamer_matches_batch(small_df):
    tr, te = splits.by_profiles(small_df, [4])
    m = models.make("tnn", epochs=1, verbose=False).fit(tr)
    one = te.iloc[:400].copy()
    init = one[TARGETS].iloc[: m.ds].mean().to_numpy()
    batch = m.predict(one)
    s = m.streamer()
    s.reset(init)
    online = np.vstack([s.step(r) for r in one[INPUTS].to_dict(orient="records")])
    # compare at window ends, where both have just stepped
    ends = np.arange(m.ds - 1, len(one), m.ds)
    np.testing.assert_allclose(online[ends], batch.to_numpy()[ends], atol=1e-3)


def test_sessions_never_mix(small_df):
    tr, te = splits.by_profiles(small_df, [2, 3])
    assert set(tr[GROUP]).isdisjoint(set(te[GROUP]))
