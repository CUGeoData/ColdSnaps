"""Rolling-origin backtest.

An origin is the first forecast day. A model sees `kw` strictly before the
origin and weather for the whole context + horizon window (observed weather for
now, i.e. a "perfect forecast" upper bound -- see PLAN.md phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel, load_building_meta


@dataclass
class Task:
    building: str
    origin: pd.Timestamp
    context: pd.Series  # kw indexed by date, ends the day before origin
    target: np.ndarray  # actual kw for origin .. origin + horizon - 1

    @property
    def future_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origin, periods=len(self.target), freq="D")


def make_tasks(
    panel: Panel,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    horizon: int,
    step: int,
    buildings: list[str] | None = None,
    min_history: int = 90,
) -> list[Task]:
    """Origins every `step` days in [start, end - horizon]. Skips windows with
    missing actuals or too little non-missing history."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last_origin = min(end, panel.weather.index.max()) - pd.Timedelta(days=horizon - 1)
    origins = pd.date_range(start, last_origin, freq=f"{step}D")
    tasks = []
    for b in buildings or panel.buildings:
        s = panel.series(b)
        for o in origins:
            ctx = s.loc[: o - pd.Timedelta(days=1)]
            tgt = s.reindex(pd.date_range(o, periods=horizon, freq="D")).to_numpy()
            if np.isnan(tgt).any() or ctx.notna().sum() < min_history:
                continue
            tasks.append(Task(b, o, ctx, tgt))
    return tasks


def run_backtest(
    panel: Panel,
    models: list,
    test_start: str,
    test_end: str | None = None,
    horizon: int = 14,
    step: int = 7,
    buildings: list[str] | None = None,
) -> pd.DataFrame:
    """Fit each model on data before `test_start`, forecast every test task,
    and return long-format results (one row per model/task/horizon day)."""
    test_end = test_end or panel.weather.index.max()
    tasks = make_tasks(panel, test_start, test_end, horizon, step, buildings)
    if not tasks:
        raise ValueError("no valid backtest tasks; check dates/buildings")
    print(f"{len(tasks)} tasks across {len({t.building for t in tasks})} buildings")

    rows = []
    for m in models:
        m.fit(panel, train_end=pd.Timestamp(test_start), buildings=buildings)
        preds = np.asarray(m.predict(panel, tasks, horizon), dtype=float)
        assert preds.shape == (len(tasks), horizon), (m.name, preds.shape)
        for t, p in zip(tasks, preds):
            rows.append(pd.DataFrame({
                "model": m.name,
                "building": t.building,
                "origin": t.origin,
                "date": t.future_dates,
                "h": np.arange(1, horizon + 1),
                "y": t.target,
                "yhat": p,
            }))
        print(f"  {m.name}: done")

    res = pd.concat(rows, ignore_index=True)
    res = res.join(panel.weather[["temperature_2m_mean"]], on="date")
    res = res.join(load_building_meta(), on="building")
    res["building_type"] = res["building_type"].fillna("Unknown")
    return res
