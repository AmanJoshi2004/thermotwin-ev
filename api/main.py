"""ThermoTwin inference API.

Run:
    THERMOTWIN_RUN=runs/main THERMOTWIN_MODEL=tnn uvicorn api.main:app --port 8000

Endpoints
    GET  /health                    model name, size, limits
    POST /predict                   a batch of samples (one session) -> estimates + upper bounds
    WS   /stream                    send one sample per message at 2 Hz; receive estimates,
                                    upper bounds, alarms, and headroom every `headroom_every` samples
A sample is a JSON object with: ambient, coolant, u_d, u_q, i_d, i_q, motor_speed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel as PydModel, Field

from thermotwin import conformal
from thermotwin.config import DEFAULT_LIMITS, GROUP, INPUTS, TARGETS
from thermotwin.headroom import HeadroomEngine
from thermotwin.models import BaseModel

RUN = Path(os.environ.get("THERMOTWIN_RUN", "runs/latest"))
MODEL_NAME = os.environ.get("THERMOTWIN_MODEL", "tnn")
HEADROOM_EVERY = int(os.environ.get("THERMOTWIN_HEADROOM_EVERY", "10"))

app = FastAPI(title="ThermoTwin", version="0.1.0")
_state: dict = {}


def _load():
    if "model" not in _state:
        path = RUN / "models" / f"{MODEL_NAME}.joblib"
        if not path.exists():
            raise HTTPException(503, f"model not found at {path}; train first (see README)")
        _state["model"] = BaseModel.load(path)
        conf = RUN / "conformal.json"
        q = json.loads(conf.read_text()).get(MODEL_NAME, {}) if conf.exists() else {}
        _state["q"] = {t: float(q.get(t, 0.0)) for t in TARGETS}
    return _state["model"], _state["q"]


class Sample(PydModel):
    ambient: float
    coolant: float
    u_d: float
    u_q: float
    i_d: float
    i_q: float
    motor_speed: float


class Batch(PydModel):
    samples: list[Sample] = Field(..., min_length=1)
    init: list[float] | None = Field(None, description="starting temperatures pm, yoke, tooth, winding (C)")


@app.get("/health")
def health():
    m, q = _load()
    return {"model": MODEL_NAME, "params": m.n_params(), "upper_offsets_C": q, "limits_C": DEFAULT_LIMITS}


@app.post("/predict")
def predict(batch: Batch):
    m, q = _load()
    df = pd.DataFrame([s.model_dump() for s in batch.samples])[INPUTS]
    df[GROUP] = 0
    init = {0: batch.init} if batch.init else None
    p = m.predict(df, init=init) if m.kind == "seq" else m.predict(df)
    up = conformal.apply_upper(p, q)
    return {"estimate": p.round(2).to_dict(orient="list"), "upper": up.round(2).to_dict(orient="list")}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    m, q = _load()
    eng = HeadroomEngine(m, q_upper=q) if m.kind == "seq" else None
    row = m.streamer() if eng is None else None
    k = 0
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("reset"):
                (eng.reset(msg.get("init")) if eng else row.reset())
                k = 0
                continue
            sample = Sample(**msg).model_dump()
            if eng:
                out = eng.step(sample)
                if k % HEADROOM_EVERY == 0:
                    out["time_to_limit_s"] = eng.time_to_limit()
                    out["max_current_A"] = eng.max_current()
            else:
                est = row.step(sample)
                out = {"estimate": dict(zip(TARGETS, est.round(2).tolist()))}
            out["k"] = k
            k += 1
            await ws.send_text(json.dumps(out))
    except WebSocketDisconnect:
        return
