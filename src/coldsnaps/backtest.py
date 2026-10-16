"""Rolling-origin backtest.

An origin is the first forecast day. A model sees `kw` strictly before the
origin and weather for the whole context + horizon window (observed weather for
now, i.e. a "perfect forecast" upper bound -- see PLAN.md phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import Panel, load_building_meta

# Quantile levels stored in backtest results (TimesFM's native output heads).
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def qcol(q: float) -> str:
    return f"q{round(q * 100):02d}"


@dataclass
class Task:
    building: str
    origin: pd.Timestamp
    context: pd.Series  # kw indexed by date, ends the day before origin
    target: np.ndarray  # actual kw for origin .. origin + horizon - 1
    future_weather: pd.DataFrame  # weather the model may use for the horizon dates

    @property
    def future_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origin, periods=len(self.target), freq="D")


# (origin, horizon dates) -> weather frame for those dates, shaped like panel.weather.
WeatherFn = Callable[[pd.Timestamp, pd.DatetimeIndex], pd.DataFrame]


def make_tasks(
    panel: Panel,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    horizon: int,
    step: int,
    buildings: list[str] | None = None,
    min_history: int = 90,
    target: str = "kw",
    future_weather: WeatherFn | None = None,
) -> list[Task]:
    """Origins every `step` days in [start, end - horizon]. Skips windows with
    missing actuals or too little non-missing history.

    future_weather: defaults to observed weather (a perfect-forecast upper
    bound); pass forecast_weather.ForecastWeather for realistic forecasts."""
    future_weather = future_weather or (lambda o, dates: panel.weather.loc[dates])
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last_origin = min(end, panel.weather.index.max()) - pd.Timedelta(days=horizon - 1)
    origins = pd.date_range(start, last_origin, freq=f"{step}D")
    tasks = []
    for b in buildings or panel.buildings:
        s = panel.series(b, target)
        for o in origins:
            ctx = s.loc[: o - pd.Timedelta(days=1)]
            tgt = s.reindex(pd.date_range(o, periods=horizon, freq="D")).to_numpy()
            if np.isnan(tgt).any() or ctx.notna().sum() < min_history:
                continue
            dates = pd.date_range(o, periods=horizon, freq="D")
            tasks.append(Task(b, o, ctx, tgt, future_weather(o, dates)))
    return tasks


def run_backtest(
    panel: Panel,
    models: list,
    test_start: str,
    test_end: str | None = None,
    horizon: int = 14,
    step: int = 7,
    buildings: list[str] | None = None,
    target: str = "kw",
    future_weather: WeatherFn | None = None,
) -> pd.DataFrame:
    """Fit each model on data before `test_start`, forecast every test task,
    and return long-format results (one row per model/task/horizon day).

    Models with a `predict_quantiles` method also contribute q10..q90 columns
    (NaN for point-only models)."""
    test_end = test_end or panel.weather.index.max()
    tasks = make_tasks(panel, test_start, test_end, horizon, step, buildings, target=target,
                       future_weather=future_weather)
    if not tasks:
        raise ValueError("no valid backtest tasks; check dates/buildings")
    print(f"{len(tasks)} tasks across {len({t.building for t in tasks})} buildings")

    rows = []
    for m in models:
        m.fit(panel, train_end=pd.Timestamp(test_start), buildings=buildings, target=target)
        if hasattr(m, "predict_quantiles"):
            preds, qs = m.predict_quantiles(panel, tasks, horizon)
            qs = np.asarray(qs, dtype=float)
            assert qs.shape == (len(tasks), horizon, len(QUANTILES)), (m.name, qs.shape)
        else:
            preds, qs = m.predict(panel, tasks, horizon), None
        preds = np.asarray(preds, dtype=float)
        assert preds.shape == (len(tasks), horizon), (m.name, preds.shape)
        for i, (t, p) in enumerate(zip(tasks, preds)):
            df = pd.DataFrame({
                "model": m.name,
                "building": t.building,
                "origin": t.origin,
                "date": t.future_dates,
                "h": np.arange(1, horizon + 1),
                "y": t.target,
                "yhat": p,
            })
            for j, q in enumerate(QUANTILES):
                df[qcol(q)] = qs[i, :, j] if qs is not None else np.nan
            rows.append(df)
        print(f"  {m.name}: done")

    res = pd.concat(rows, ignore_index=True)
    res = res.join(panel.weather[["temperature_2m_mean"]], on="date")
    res = res.join(load_building_meta(), on="building")
    res["building_type"] = res["building_type"].fillna("Unknown")
    return res
