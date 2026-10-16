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
| 2 | Baselines vs. TimesFM with/without weather | comparison table | baselines done; TimesFM code ready, full run pending (GPU) |
| 3 | Realistic weather: archived Open-Meteo forecasts (*Previous Runs API*) as future covariates instead of observed weather | error vs. lead time | code done (`forecast_weather.py`, `--weather forecast`); run pending |
| 4 | Uncertainty & peaks: quantiles (`return_quantiles=True`), CRPS/coverage, `max_kw` target, cold-snap event scoring | peak-risk metrics | code done (`metrics.py`, `--target max_kw`); run pending |
| 5 | Extension (pick one): **fine-tune TimesFM (LoRA)** ← recommended; hierarchical campus forecasting / cold-start buildings; what-if weather scenarios with sanity checks | headline result | what-if scenarios built (`scenarios.py`, `scripts/run_scenarios.py`) since CPU-only; run pending. LoRA back on the table with the GPU |
| 6 | Dashboard or write-up | demo | todo |

## Results so far

Test 2025-08-01 → 2026-08-01, 5,736 origins × 14 days, 125 buildings
(`results/baseline/report.txt`):

| model | WAPE | median bldg WAPE | skill vs naive | skill on cold days |
|---|---|---|---|---|
| lgbm | 5.2% | 6.8% | +25% | +30% |
| seasonal_naive | 7.0% | 9.5% | 0 | 0 |
| degree_day | 8.2% | 10.7% | −17% | −22% |

The degree-day regression losing to naive suggests electric load is only weakly
weather-driven (heating is likely steam/district energy, not metered here).
Weather probably matters more through occupancy/calendar and in a few building
types — worth checking per building.

TimesFM smoke test (AnabelTaylorHall, Dec–Feb, 11 origins): with weather MAE
1.20 vs 1.84 without vs 2.46 naive. Too small to conclude anything.

## Status (2026-10-16)

All phase 2–5 code is written and tested (`pytest`: 9 pass), but none of the
long runs has finished: on the CPU-only laptop TimesFM takes ~0.26 s/task
without weather and ~0.7 s/task with 3 weather covariates (≈4 h for the queue
below), and the first attempt died with the session. Next: run on the NVIDIA
Spark (needs a `--device cuda` option, a larger batch size, and a
`requirements.txt` without the CPU-only torch index).

Run queue (each writes `results/<out>/predictions.parquet` + `report.txt`):

```
python scripts/run_backtest.py --models seasonal_naive,degree_day,lgbm,timesfm,timesfm_weather --out phase2
python scripts/run_backtest.py --models seasonal_naive,degree_day,lgbm,timesfm_weather --weather forecast --out phase3_forecast_weather
python scripts/run_backtest.py --models seasonal_naive,lgbm,timesfm,timesfm_weather --target max_kw --out phase4_max_kw
python scripts/run_scenarios.py --n-buildings 40
```

Early peak-day result, from the existing `results/baseline` predictions:
during the 18-day Jan 23 – Feb 9 2026 cold snap, seasonal naive under-forecasts
87% of buildings' peak days at 1–7 day lead (mean error −10%); LightGBM's mean
absolute peak-day error is 11% (1–7 d) and 14% (8–14 d).

## What was built (phases 2–5)

**Backtest plumbing** (`backtest.py`)
- Results carry quantile columns `q10 … q90` (NaN for point-only models);
  models opt in with `predict_quantiles(panel, tasks, horizon)`.
- `target="kw" | "max_kw"` threads through `make_tasks`, `run_backtest`
  and every model's `fit`.
- Each `Task` holds its own `future_weather` frame; models read that instead
  of `panel.weather` for horizon dates, so the weather source is pluggable
  (`future_weather=` callable: observed, forecast, or scenario).

**Models** (`models.py`)
- TimesFM always returns its 9 native quantiles (0.1 … 0.9).
- `timesfm_weather` now uses 3 covariates (`temperature_2m_mean`, `hdd`,
  `cdd`): ~3.5× faster on CPU than all 11. The old setup is
  `timesfm_weather_all`.
- LightGBM gets split-conformal quantiles: fit on data before the last 182
  training days, take per-horizon quantiles of scale-normalised residuals on
  those days, then refit on everything.

**Realistic weather** (`forecast_weather.py`, phase 3)
- Source: Open-Meteo Previous Runs API, cached in
  `data/openmeteo_previous_runs.parquet` (2025-06 → 2026-08, leads 0–7).
  Leads beyond 7 days are not archived for any model (checked GFS, ECMWF,
  GEM), so horizon days 8–14 use day-of-year climatology (±7 days, prior
  years only).
- The CSV weather is from a nearby grid point (~1 °F warmer), so we add the
  forecast *error* to the observed value:
  `obs[d] + (forecast_lead_h[d] − forecast_lead_0[d])`.
- Lead mapping: horizon day h uses the run from h days earlier (h=1 is the
  origin day, run from the day before).
- Mean-temperature forecast MAE grows from 1.9 °F (lead 1) to 4.3 °F (lead 7).
  Climatology completely misses the late-Jan 2026 snap — exactly the realistic
  case to measure.
- Only `predict` uses forecasts; LightGBM still trains on observed weather.

**Uncertainty & peaks** (`metrics.py`, phase 4)
- `summarize_quantiles`: CRPS (2 × mean pinball loss over the 9 levels),
  WAPE-style scaled CRPS, 80% interval coverage/width, share below q10 /
  above q90.
- `cold_snap_events`: runs of ≥3 days with mean ≤ 20 °F. Test year has three:
  Jan 1–3, Jan 19–21, Jan 23 – Feb 9 2026 (min 1.2 °F).
- `exceedance_scores`: "high-load day" = above the building's trailing-year
  p90. Point forecasts scored as warnings (precision/recall); quantile models
  also get Brier score and Brier skill for P(exceed).
- `peak_day_scores`: signed % error on each building's peak day within each
  snap, by lead (1–7 d vs 8–14 d).
- Both reports are appended to `report.txt` by `run_backtest.py`.

**What-if cold snaps** (`scenarios.py`, `scripts/run_scenarios.py`, phase 5)
- Lowers all temperature columns by Δ °F (default 10, 20) on horizon days
  4–10 for winter origins, recomputing HDD/rolling/streak features. In
  winter each −1 °F is +1 HDD.
- Compares each model's response (% of mean load per 10 °F) against the
  building's empirical winter HDD slope: sign agreement, Spearman rank
  correlation across buildings, and "leak" onto non-snap days (should be ≈0).
- Chosen over LoRA because the laptop has no GPU; with the Spark, LoRA
  fine-tuning is feasible again.

## Evaluation protocol

- Models only see `kw` strictly before the origin. Global models (LightGBM,
  later fine-tuned TimesFM) are trained only on origins before `test_start`.
- Future weather defaults to *observed* → upper bound on achievable accuracy.
  `--weather forecast` uses archived forecasts instead (see below).
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
- ~~Check TimesFM's response to synthetic cold snaps against degree-day slopes.~~
  → phase 5 scenarios.
- Train LightGBM on forecast (not observed) weather so train/test match under
  `--weather forecast`.
