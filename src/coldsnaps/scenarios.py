"""What-if weather: inject a synthetic cold snap into the forecast horizon and
check whether each model's response is physically sensible.

A scenario lowers every temperature column by `delta_f` on a block of horizon
days; derived features (HDD, rolling means, cold streaks) are recomputed. In
winter (mean temp < 65 F) each -1 F adds exactly +1 HDD, so a model's response
can be compared with the building's empirical HDD slope.

Sanity checks per model:
  * sign     -- does colder weather move load the same way as history says?
  * scale    -- does sensitivity across buildings line up with the empirical
                slope (correlation, ratio)?
  * locality -- days outside the snap block should barely move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel, add_weather_features

TEMP_COLS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
             "apparent_temperature_max", "apparent_temperature_min"]


class ColdSnapScenario:
    """WeatherFn: observed weather with temperatures shifted by -delta_f on
    horizon days `snap_days` (1-based, inclusive range)."""

    def __init__(self, observed: pd.DataFrame, delta_f: float, snap_days: tuple[int, int] = (4, 10)):
        self.obs = observed
        self.delta_f = delta_f
        self.snap_days = snap_days

    def __call__(self, origin: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
        base_cols = [c for c in self.obs.columns if c not in _DERIVED]
        past = self.obs.loc[origin - pd.Timedelta(days=30): origin - pd.Timedelta(days=1), base_cols]
        fut = self.obs.loc[dates, base_cols].copy()
        a, b = self.snap_days
        fut.iloc[a - 1:b, [fut.columns.get_loc(c) for c in TEMP_COLS]] -= self.delta_f
        return add_weather_features(pd.concat([past, fut])).loc[dates]


_DERIVED = {"hdd", "cdd", "hdd_x_wind", "temp_mean_3d", "temp_mean_7d", "temp_change_1d",
            "consecutive_cold_days", "weekday", "month", "doy_sin", "doy_cos"}


def empirical_hdd_slope(panel: Panel, building: str, before: pd.Timestamp, days: int = 365) -> tuple[float, float]:
    """(kW per HDD, mean kW) from OLS kw ~ HDD + CDD + weekday on the trailing
    `days` before `before`, using only winter days so HDD is the active term."""
    s = panel.series(building).loc[before - pd.Timedelta(days=days): before - pd.Timedelta(days=1)].dropna()
    w = panel.weather.loc[s.index]
    keep = (w["temperature_2m_mean"] < 55).to_numpy()
    s, w = s[keep], w[keep]
    if len(s) < 60:
        return np.nan, s.mean() if len(s) else np.nan
    wd = np.eye(7)[w["weekday"].to_numpy()][:, 1:]
    X = np.column_stack([np.ones(len(w)), w["hdd"], wd])
    coef, *_ = np.linalg.lstsq(X, s.to_numpy(), rcond=None)
    return coef[1], s.mean()


def scenario_sensitivity(results: dict[float, pd.DataFrame], snap_days: tuple[int, int]) -> pd.DataFrame:
    """Per (model, building, origin): mean change in forecast per +1 HDD on snap
    days, and mean absolute change on non-snap days (both in kW per degree F).

    results: {delta_f: long backtest results}, must include delta_f == 0."""
    base = results[0.0].set_index(["model", "building", "origin", "h"])["yhat"]
    a, b = snap_days
    rows = []
    for d, r in results.items():
        if d == 0:
            continue
        diff = (r.set_index(["model", "building", "origin", "h"])["yhat"] - base).rename("diff").reset_index()
        diff["in_snap"] = diff["h"].between(a, b)
        g = diff.groupby(["model", "building", "origin"])
        rows.append(pd.DataFrame({
            "delta_f": d,
            "kw_per_f": g.apply(lambda x: x.loc[x["in_snap"], "diff"].mean() / d, include_groups=False),
            "leak_kw_per_f": g.apply(lambda x: x.loc[~x["in_snap"], "diff"].abs().mean() / d,
                                     include_groups=False),
        }))
    return pd.concat(rows).reset_index()
