# Working together

## Branch workflow

1. Pull the latest `main`: `git checkout main && git pull`
2. Make a branch per task: `git checkout -b feature/learning-curve`
3. Commit small, clear changes: `git commit -m "Add learning curve plot"`
4. Push and open a pull request: `git push -u origin feature/learning-curve`
5. The other person reviews; GitHub Actions must be green; then merge.

Never commit `data/` or `runs/` (they are in `.gitignore`). Share result tables and figures by
copying them into `docs/results/` when they are final.

## Suggested split for two people

| Week | Person A: data and tabular models | Person B: sequence models and app |
| --- | --- | --- |
| 1 | Download data, EDA notebook, rungs 0-2 on real data | Read the TNN paper, run `gru`/`tnn`/`lptn` on real data |
| 2 | LightGBM + SHAP, leakage and learning-curve experiments | Tune TNN, fit the reference LPTN, download EPA cycles |
| 3 | B2/B3 results, sensor-value and error-anatomy experiments | ThermoBench B1, dashboard polish, API demo |
| 4 | Report sections 2, 5, 6 | Report sections 3, 4, 7; demo video |

Both: email course staff in week 1 to confirm the test-data plan.

## Checklist before the final report

- [ ] All numbers come from `--data` runs, not `--synthetic`
- [ ] Real EPA cycles used (ThermoBench report has no "approximate" warning)
- [ ] Development sessions and calibration sessions listed in the report
- [ ] Limits and vehicle/motor parameters stated as assumptions
- [ ] AI use acknowledged as required by the course
