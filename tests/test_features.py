import numpy as np

from thermotwin.config import GROUP, INPUTS
from thermotwin.features import FeatureSpec, StreamingFeatures, make_features


def test_streaming_matches_batch(small_df):
    """Training/serving parity: online features equal offline features."""
    spec = FeatureSpec(spans=[20, 200])
    one = small_df[small_df[GROUP] == small_df[GROUP].iloc[0]].iloc[:600]
    batch = make_features(one, spec).to_numpy()
    sf = StreamingFeatures(spec)
    online = np.vstack([sf.update(r) for r in one[INPUTS].to_dict(orient="records")])
    np.testing.assert_allclose(online, batch, rtol=1e-4, atol=1e-3)


def test_features_are_causal(small_df):
    """Changing the future must not change past features."""
    one = small_df[small_df[GROUP] == small_df[GROUP].iloc[0]].iloc[:400].copy()
    a = make_features(one).to_numpy()
    one.iloc[300:, one.columns.get_loc("i_q")] += 100.0
    b = make_features(one).to_numpy()
    np.testing.assert_allclose(a[:300], b[:300])
    assert not np.allclose(a[300:], b[300:])


def test_features_reset_per_session(small_df):
    """The first sample of each session starts fresh (no carry-over between sessions)."""
    f = make_features(small_df)
    starts = small_df.groupby(GROUP).head(1).index
    col = "i_s_ewma_7200"
    np.testing.assert_allclose(f.loc[starts, col].to_numpy(), f.loc[starts, "i_s"].to_numpy(), rtol=1e-5)
