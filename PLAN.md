# ColdSnaps — forecasting campus building energy from weather

Goal: forecast daily building electricity (kW) 1–14 days ahead from weather,
with emphasis on cold snaps, using TimesFM 3.0 (`google/timesfm-3.0-pytorch`)
and showing whether/when it beats simpler models.

## Data notes (known issues)

- `merged_energy_weather.csv`: 184 buildings, daily, 2021-09-21 → 2026-09-22.
- **Multi-meter buildings**: 51 buildings have >1 meter (`target`). Filtering by
  `building` alone interleaves meters on the same date. Default pipeline keeps
  the 133 single-meter buildings (same set as
  `building_daily_normal_weekdays_features.csv`); `meter_policy="sum"` adds the
  rest by summing meters. Open question: some sub-meters (Solar, ATT/T-Mobile
  telecom) may be generation or tenant loads and need per-building rules.
- **Weather** is campus-wide (same for every building on a date) and missing only
  for 2021-09-21 → 2021-12-30 and 2026-08-02 → 2026-09-22; we trim to the
  covered range. Backfill from Open-Meteo archive later if needed.
- **Units**: temperatures are °F, but `hdd_c` / `cdd_c` / `cold_intensity_c` in
  `building_daily_normal_weekdays_features.csv` use an 18 °C base on °F values
  (a 24 °F day shows 6 "cooling degree days"). The new pipeline computes
  `hdd`/`cdd` with a 65 °F base. `hdd_coefficient` / `cdd_coefficient` in
  `building_features_clean.csv` are probably affected too.
- Partial days (`n_points` < 75% of the building's typical count) are set to
  missing; interior gaps ≤2 days are interpolated.

## Phases

| # | Work | Deliverable | Status |
|---|------|-------------|--------|
| 0 | Package code into `src/coldsnaps/`; meter fix; weather trimming; °F degree days | `load_panel()` | done |
| 1 | Rolling-origin backtest (origin every 7 days, 14-day horizon, test = 2025-08-01 → 2026-08-01); MAE, RMSE, WAPE, bias, skill vs. naive; cold-day and building-type slices | `scripts/run_backtest.py`, `results/<run>/` | done |
| 2 | Baselines vs. TimesFM with/without weather | comparison table | run it |
| 3 | Realistic weather: Open-Meteo *Historical Forecast API* as future covariates instead of observed weather | error vs. lead time | todo |
| 4 | Uncertainty & peaks: quantiles (`return_quantiles=True`), CRPS/coverage, `max_kw` target, cold-snap event scoring | peak-risk metrics | todo |
| 5 | Extension (pick one): **fine-tune TimesFM (LoRA)** ← recommended; hierarchical campus forecasting / cold-start buildings; what-if weather scenarios with sanity checks | headline result | todo |
| 6 | Dashboard or write-up | demo | todo |

## Evaluation protocol

- Models only see `kw` strictly before the origin. Global models (LightGBM,
  later fine-tuned TimesFM) are trained only on origins before `test_start`.
- Future weather is currently *observed* → upper bound on achievable accuracy.
  Phase 3 replaces it with archived forecasts.
- Headline metrics: overall WAPE, median per-building WAPE, MAE skill vs.
  seasonal naive, the same on cold target days (mean ≤ 20 °F).

## Models

- `seasonal_naive` — same weekday last week.
- `degree_day` — per-building OLS on trailing 365 days: HDD + CDD + weekday.
- `lgbm` — one global LightGBM on scale-normalised targets with weather,
  calendar, recent-level and building-type features.
- `timesfm` / `timesfm_weather` — zero-shot TimesFM 3.0, weather as
  past+future covariates (from `model /Model_1.ipynb`).

## Ideas backlog

- Academic calendar (term/break/holiday flags exist in the features file only
  for normal weekdays — rebuild for all dates).
- Hourly data, if the raw meter feed is available, for true peak forecasting.
- Check TimesFM's response to synthetic cold snaps against degree-day slopes.
