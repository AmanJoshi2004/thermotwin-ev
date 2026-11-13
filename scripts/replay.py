"""Stream one session (or a ThermoBench scenario) to the API like a CAN bus feed.

    python scripts/replay.py --data data/measures_v2.csv --profile 60 --speedup 50
    python scripts/replay.py --scenario us06_desert --speedup 50
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets

from thermotwin import data
from thermotwin.bench import scenarios
from thermotwin.config import GROUP, INPUTS, TARGETS


async def run(rows, url: str, speedup: float):
    async with websockets.connect(url) as ws:
        for k, r in enumerate(rows):
            await ws.send(json.dumps(r))
            out = json.loads(await ws.recv())
            if "time_to_limit_s" in out:
                est = ", ".join(f"{t}={out['estimate'][t]:.1f}" for t in TARGETS)
                print(f"t={k / 2:7.0f}s  {est}  max_current={out.get('max_current_A')}")
            await asyncio.sleep(0.5 / speedup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--profile", type=int)
    ap.add_argument("--scenario", help="name from thermotwin.bench.scenarios.catalog()")
    ap.add_argument("--url", default="ws://localhost:8000/stream")
    ap.add_argument("--speedup", type=float, default=20.0)
    a = ap.parse_args()
    if a.scenario:
        sc = {s.name: s for s in scenarios.catalog()}[a.scenario]
        df, _ = scenarios.build(sc, 1)
    else:
        df = data.load(a.data)
        df = df[df[GROUP] == a.profile]
    asyncio.run(run(df[INPUTS].to_dict(orient="records"), a.url, a.speedup))


if __name__ == "__main__":
    main()
