# Project plan (summary)

## Problem

Estimate the rotor magnet and stator temperatures of an EV traction motor from signals the
controller already measures, and turn them into a thermal headroom forecast. The rotor cannot
be measured in a production car; overheating magnets demagnetise permanently.

## Course rule on test data

> Your test data cannot be from an existing benchmark or dataset prepared for machine
> learning. The test data can be derived from online sources, but it must be designed by you
> to reflect an evaluation of a ML application/model that you design.

So: train on Kaggle, test on ThermoBench (self-generated scenarios, property tests, injected
faults, alarm events). Ask the TA to confirm B2/B3.

## Hypotheses to test (insights)

1. Memory beats model size (rung 1 vs 2; benchmark: linear + EWMA 3.10 K² vs k-NN 26.10).
2. The rotor is the hardest target, the stator yoke the easiest.
3. Each part reacts on its own time scale (importance by EWMA span).
4. Coolant sets the baseline, current sets the rise.
5. Random row splits hide most of the error (`experiments leakage`).
6. About 70 h of bench data is enough for simple models (`experiments learning-curve`).
7. Errors cluster in fast transients and cold starts (`experiments error-anatomy`).
8. One stator sensor may help more than a bigger model (`experiments sensor-value`).

## Success criteria (proposed)

| Metric | Rotor | Stator |
| --- | --- | --- |
| MAE | ≤ 2 K | ≤ 1.5 K |
| Max abs error | ≤ 8 K | ≤ 6 K |
| Under-estimating by > 5 K | < 1% of time | < 1% |
| Upper-bound coverage | ≥ 95% | ≥ 95% |
| Alarm recall / false alarms / lead | ≥ 95% / ≤ 1 per h / ≥ 30 s | same |

## Timeline

- Week 1: data, EDA, splits, rungs 0-2, email the TA.
- Week 2: LightGBM, GRU, TNN, learning curves, drive-cycle generator. **Freeze models.**
- Week 3: ThermoBench, conformal bounds, API, dashboard.
- Week 4: final test run, error analysis, report, demo video.

## Key references

- Kirchgässner et al., data-driven PM temperature estimation benchmark (arXiv 2001.06246)
- Kirchgässner et al., thermal neural networks (arXiv 2103.16323)
- EPA dynamometer drive schedules: https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
