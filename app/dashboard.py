"""ThermoTwin dashboard.

    streamlit run app/dashboard.py

Pick a trained run, a model and a session (Kaggle development session or a ThermoBench
scenario). Scrub through time, see the four temperatures with their upper bounds and
limits, the thermal headroom at that moment, and try what-if changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermotwin import conformal, data  # noqa: E402
from thermotwin.bench import scenarios  # noqa: E402
from thermotwin.config import DEFAULT_LIMITS, GROUP, TARGETS  # noqa: E402
from thermotwin.headroom import HeadroomEngine  # noqa: E402
from thermotwin.models import BaseModel  # noqa: E402

LABELS = {"pm": "Rotor magnet", "stator_yoke": "Stator yoke", "stator_tooth": "Stator tooth",
          "stator_winding": "Stator winding"}

st.set_page_config(page_title="ThermoTwin", layout="wide")
st.title("ThermoTwin")
st.caption("Virtual temperature sensor and thermal headroom for an EV traction motor")


@st.cache_resource
def load_model(path: str) -> BaseModel:
    return BaseModel.load(path)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return data.load(path)


@st.cache_data
def scenario_frame(name: str) -> pd.DataFrame:
    sc = {s.name: s for s in scenarios.catalog()}[name]
    df, _ = scenarios.build(sc, pid=1)
    return df


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    run = Path(st.text_input("Run folder", "runs/latest"))
    found = sorted(p.stem for p in (run / "models").glob("*.joblib")) if (run / "models").exists() else []
    if not found:
        st.error(f"No models in {run / 'models'}. Train first: python -m thermotwin.train --synthetic --quick "
                 f"--out {run}")
        st.stop()
    default = found.index("tnn") if "tnn" in found else 0
    model_name = st.selectbox("Model", found, index=default)
    conf = json.loads((run / "conformal.json").read_text()) if (run / "conformal.json").exists() else {}
    q = {t: float(conf.get(model_name, {}).get(t, 0.0)) for t in TARGETS}

    source = st.radio("Session", ["ThermoBench scenario", "Recorded session"])
    if source == "Recorded session":
        path = st.text_input("Data file", "data/measures_v2.csv")
        if not Path(path).exists():
            st.warning("Data file not found")
            st.stop()
        df_all = load_data(path)
        split_file = run / "split.json"
        dev_ids = json.loads(split_file.read_text())["development"] if split_file.exists() else []
        ids = sorted(df_all[GROUP].unique())
        pid = st.selectbox("Session (development sessions first)", dev_ids + [i for i in ids if i not in dev_ids])
        df = df_all[df_all[GROUP] == pid].reset_index(drop=True)
    else:
        names = [s.name for s in scenarios.catalog()]
        df = scenario_frame(st.selectbox("Scenario", names, index=names.index("us06_desert"))).copy()

    st.subheader("What if")
    d_amb = st.slider("Ambient change (°C)", -20, 20, 0)
    d_cool = st.slider("Coolant change (°C)", -20, 20, 0)
    load = st.slider("Current scale", 0.5, 1.5, 1.0, 0.05)

    st.subheader("Limits (°C)")
    limits = {t: st.number_input(LABELS[t], value=float(DEFAULT_LIMITS[t]), step=5.0) for t in TARGETS}

has_truth = all(t in df for t in TARGETS) and d_amb == 0 and d_cool == 0 and load == 1.0
x = df.copy()
x["ambient"] += d_amb
x["coolant"] += d_cool
x["i_d"] *= load
x["i_q"] *= load
x[GROUP] = 0
model = load_model(str(run / "models" / f"{model_name}.joblib"))


@st.cache_data(show_spinner="Running the model")
def run_model(_model, key: tuple, frame: pd.DataFrame) -> pd.DataFrame:
    # Sequence models start from the measured temperatures when the frame has them,
    # otherwise from the coolant temperature.
    return _model.predict(frame)


pred = run_model(model, (str(run), model_name, d_amb, d_cool, load, len(x), float(x["coolant"].iloc[0])), x)
upper = conformal.apply_upper(pred, q)
minutes = np.arange(len(x)) / 120.0

pos_min = st.slider("Time (minutes)", 0.0, float(minutes[-1]), float(minutes[-1]) * 0.5, 0.5)
pos = int(min(len(x) - 1, pos_min * 120))

# ---------------------------------------------------------------- headline numbers
cols = st.columns(4)
for c, t in zip(cols, TARGETS):
    est, up, lim = pred[t].iloc[pos], upper[t].iloc[pos], limits[t]
    c.metric(LABELS[t], f"{est:.1f} °C", f"{lim - up:+.1f} K to limit (upper bound)",
             delta_color="normal" if up < lim else "inverse")

# ---------------------------------------------------------------- headroom
if model.kind == "seq":
    eng = HeadroomEngine(model, q_upper=q, limits=limits)
    eng.sync(x.iloc[: pos + 1])
    ttl = eng.time_to_limit()
    imax = eng.max_current()
    h = st.columns(5)
    for c, t in zip(h, TARGETS):
        v = ttl[t]
        c.metric(f"{LABELS[t]}: time to limit", "> 30 min" if v is None else f"{v / 60:.1f} min")
    i_now = float(np.hypot(x["i_d"].iloc[pos], x["i_q"].iloc[pos]))
    h[4].metric("Max current for the next 60 s", f"{imax:.0f} A" if imax is not None else "n/a", f"now {i_now:.0f} A",
                delta_color="off")
else:
    st.info("Headroom needs a sequence model (tnn, lptn or gru). Pick one in the sidebar.")

# ---------------------------------------------------------------- charts
step = max(1, len(x) // 1500)
charts = []
for t in TARGETS:
    frame = pd.DataFrame({"minute": minutes, "estimate": pred[t], "upper": upper[t]})
    if has_truth:
        frame["measured"] = df[t].to_numpy()
    frame = frame.iloc[::step]
    base = alt.Chart(frame).encode(x=alt.X("minute:Q", title="minutes"))
    band = base.mark_area(opacity=0.15, color="#2a6fdb").encode(y=alt.Y("estimate:Q", title="°C",
                                                                        scale=alt.Scale(zero=False)), y2="upper:Q")
    layers = [band, base.mark_line(color="#2a6fdb").encode(y="estimate:Q")]
    if has_truth:
        layers.append(base.mark_line(color="#8a8a8a", strokeDash=[4, 3]).encode(y="measured:Q"))
    rule = alt.Chart(pd.DataFrame({"lim": [limits[t]]})).mark_rule(color="#c0392b", strokeDash=[6, 4]).encode(y="lim:Q")
    now = alt.Chart(pd.DataFrame({"m": [pos_min]})).mark_rule(color="#444").encode(x="m:Q")
    charts.append(alt.layer(*layers, rule, now).properties(title=LABELS[t], height=220))
st.altair_chart(alt.vconcat(alt.hconcat(charts[0], charts[1]), alt.hconcat(charts[2], charts[3])),
                width="stretch")
st.caption("Blue: estimate with its one-sided 95% upper bound. Grey dashed: measured (recorded sessions). "
           "Red dashed: limit. Vertical line: selected time.")

# ---------------------------------------------------------------- alarms and accuracy
over = pd.DataFrame({t: upper[t] >= limits[t] for t in TARGETS})
c1, c2 = st.columns(2)
with c1:
    st.subheader("Alarm time")
    st.dataframe(pd.DataFrame({"minutes over limit (upper bound)": over.sum() / 120.0}).round(1))
with c2:
    if has_truth:
        st.subheader("Accuracy on this session")
        e = pred[TARGETS].to_numpy() - df[TARGETS].to_numpy()
        st.dataframe(pd.DataFrame({"MAE (K)": np.abs(e).mean(0), "max |error| (K)": np.abs(e).max(0)},
                                  index=[LABELS[t] for t in TARGETS]).round(2))
    else:
        st.caption("No measured temperatures for this session (scenario or what-if).")
