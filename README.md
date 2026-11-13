# ThermoTwin

**A virtual temperature sensor and thermal headroom forecaster for EV traction motors.**
CS 441 (Applied Machine Learning) final project.

An electric car cannot put a temperature sensor on its spinning rotor, so it guesses the
magnet temperature and cuts power early to be safe. ThermoTwin estimates the four internal
temperatures of a permanent magnet synchronous motor (rotor magnet, stator yoke, stator
tooth, stator winding) every 0.5 s from signals the controller already has. It then answers
the practical question: **how long can the motor hold this load, and how much current can it
take, before something overheats?**

- **Training data:** [Electric Motor Temperature (Kaggle)](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature),
  185 h of 2 Hz test-bench data from a 52 kW motor, 69 sessions.
- **Test data:** *ThermoBench*, a test suite we design ourselves (the course does not allow
  testing on a ready-made ML dataset). See [Testing](#testing-thermobench).
- **Models:** a six-rung ladder from a trivial baseline to a physics-structured thermal neural
  network, plus a fitted reference thermal network.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan, insights and timeline.

---

## Quick start (5 minutes, no download)

```bash
git clone <this repo> && cd thermotwin
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; skip if you have torch
pip install -r requirements.txt && pip install -e .

pytest -q                                                # 23 tests on generated data
python -m thermotwin.train --synthetic --quick --out runs/latest
streamlit run app/dashboard.py                          # opens http://localhost:8501
```

`--synthetic` uses fake data from a built-in physics simulator, so everything runs before
you download the real dataset. **Never report synthetic numbers as results.**

## Real data

1. Get the data (needs a free Kaggle account):
   ```bash
   pip install kaggle   # put your API token in ~/.kaggle/kaggle.json (Kaggle > Settings > API)
   kaggle datasets download -d wkirgsn/electric-motor-temperature -p data --unzip
   ```
   or download it from the Kaggle page and put `measures_v2.csv` in `data/`.
2. Get the EPA drive cycles for the test scenarios:
   ```bash
   python scripts/download_epa_cycles.py
   ```
3. Train and compare all models (CPU; the sequence models take the longest, roughly
   5-15 min each on a laptop, so start with `--models coolant_floor,linear_raw,linear_ewma,lgbm`):
   ```bash
   python -m thermotwin.train --data data/measures_v2.csv --out runs/main
   ```
4. Run the test suite and the insight experiments:
   ```bash
   python -m thermotwin.bench.run --run runs/main --data data/measures_v2.csv
   python -m thermotwin.experiments all --data data/measures_v2.csv --run runs/main --out runs/experiments
   ```
5. Demo:
   ```bash
   streamlit run app/dashboard.py          # set "Run folder" to runs/main in the sidebar
   THERMOTWIN_RUN=runs/main uvicorn api.main:app --port 8000
   python scripts/replay.py --scenario us06_desert --speedup 50
   ```

## What is in the repo

```
src/thermotwin/
  config.py          columns, splits, EWMA spans, demo temperature limits
  data.py            load + validate measures_v2.csv
  features.py        physics features + causal moving averages (batch and streaming, identical)
  splits.py          session-based splits (never random rows)
  models/
    tabular.py       rung 0 coolant floor, rung 1-2 ridge regression, rung 3 LightGBM
    sequence.py      rung 4 GRU, rung 5 thermal neural network (TNN), fitted LPTN reference
  train.py           train + compare the ladder -> runs/<name>/
  conformal.py       one-sided 95% upper bounds (+ adaptive conformal)
  headroom.py        time to limit, max sustainable current
  experiments.py     leakage, learning curve, sensor value, importance, error anatomy
  physics.py         illustrative motor + thermal simulator (synthetic data, scenarios)
  bench/             ThermoBench: drive cycles, scenarios, property tests, faults, runner
api/main.py          FastAPI: /health, /predict, WebSocket /stream
app/dashboard.py     Streamlit dashboard
scripts/             EPA cycle download, live replay client
tests/               unit tests (run in GitHub Actions on every push)
```

## The model ladder

| Rung | Model | Question it answers |
| --- | --- | --- |
| 0 | `coolant_floor`: coolant + offset | How much is there to learn? |
| 1 | `linear_raw`: ridge on 7 raw signals | Can instantaneous signals explain temperature? |
| 2 | `linear_ewma`: ridge + moving averages | Is memory what matters? |
| 3 | `lgbm`: LightGBM on the same features | What does non-linearity add? |
| 4 | `gru`: recurrent network | Can a network learn the memory itself? |
| 5 | `tnn`: thermal neural network (**proposed**) | Does physics structure beat black boxes? |
| ref | `lptn`: fitted 4-node thermal network | Reference simulator for ThermoBench |

Inputs: `ambient, coolant, u_d, u_q, i_d, i_q, motor_speed`. `torque` is excluded (derived
from currents, not measurable in a car). Targets: `pm, stator_yoke, stator_tooth, stator_winding`.

## Testing: ThermoBench

The Kaggle sessions are used for training and development only. The official test suite:

- **B1 driving scenarios:** EPA UDDS, HWFET and US06 cycles, through our own vehicle and
  motor model, in winter, mild, desert and weak-pump conditions, plus a hill climb and a
  cool-down. Scored against the fitted reference network and with physics property tests
  (coolant shift, more current means more heat, cool-down, no drift, determinism).
- **B2 sensor faults:** noise, stuck coolant sensor, 10% dropped samples, 2 °C ambient bias.
- **B3 alarm events:** over-temperature events, scored by recall, warning lead time and
  false alarms per hour.

Confirm with course staff that B2/B3 count as self-designed test data; B1 does on its own.

## Working together

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and a suggested split of work.

## Credits

Dataset: W. Kirchgässner, O. Wallscheid, J. Böcker, Paderborn University (cite as the Kaggle
page asks). TNN idea: Kirchgässner et al., *Thermal neural networks: Lumped-parameter thermal
modeling with state-space machine learning*, 2023. Our implementation is written from the
paper. AI assistance (Claude) was used to draft code and documentation, as the course allows.
