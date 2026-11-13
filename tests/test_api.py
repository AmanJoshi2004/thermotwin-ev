import json
import os

import pytest

from thermotwin import models, splits


@pytest.fixture(scope="module")
def client(small_df, tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    tr, _ = splits.by_profiles(small_df, [4])
    models.make("lptn", epochs=1, verbose=False).fit(tr).save(run / "models" / "lptn.joblib")
    (run / "conformal.json").write_text(json.dumps({"lptn": {"pm": 1.0}}))
    os.environ["THERMOTWIN_RUN"] = str(run)
    os.environ["THERMOTWIN_MODEL"] = "lptn"
    import importlib

    import api.main as main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app)


SAMPLE = dict(ambient=25, coolant=45, u_d=-50, u_q=100, i_d=-20, i_q=200, motor_speed=3000)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model"] == "lptn"


def test_predict(client):
    r = client.post("/predict", json={"samples": [SAMPLE] * 10})
    body = r.json()
    assert r.status_code == 200
    assert len(body["estimate"]["pm"]) == 10
    assert body["upper"]["pm"][0] == pytest.approx(body["estimate"]["pm"][0] + 1.0, abs=0.02)


def test_stream_gives_headroom(client):
    with client.websocket_connect("/stream") as ws:
        ws.send_text(json.dumps(SAMPLE))
        out = json.loads(ws.receive_text())
    assert "time_to_limit_s" in out and "max_current_A" in out
