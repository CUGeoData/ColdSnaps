"""Realistic future weather for backtests: what was actually forecast, not what
happened.

Source: Open-Meteo *Previous Runs API*, which archives each hourly value as
forecast 1..7 days earlier (`<var>_previous_day<k>`). Longer leads are not
archived, so horizons 8-14 fall back to day-of-year climatology from the years
before the origin -- roughly what a two-week forecast is worth anyway.

The CSV weather comes from a slightly different grid point than our request
(~1 F warmer), so rather than swapping in the forecast directly we add the
forecast *error* to the observed CSV value:

    future[d, lead k] = observed[d] + (forecast_k[d] - forecast_0[d])

which keeps one consistent source while carrying the real forecast error.

Lead mapping: horizon h (h=1 is the origin day) uses lead k=h, i.e. a model
run from the day before the origin -- the latest run available when the
forecast is made with kw known through origin-1.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, WEATHER_COLS, add_weather_features

LAT, LON = 42.45, -76.48  # Cornell campus, Ithaca NY
MAX_LEAD = 7
CACHE = DATA_DIR / "openmeteo_previous_runs.parquet"

HOURLY = ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
          "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]


def _fetch_lead(lead: int, start: str, end: str) -> pd.DataFrame:
    suffix = "" if lead == 0 else f"_previous_day{lead}"
    url = (
        "https://previous-runs-api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
        f"&hourly={','.join(v + suffix for v in HOURLY)}"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
        "&timezone=America/New_York"
    )
    with urllib.request.urlopen(url, timeout=120) as r:
        h = json.load(r)["hourly"]
    df = pd.DataFrame({v: h[v + suffix] for v in HOURLY}, index=pd.to_datetime(h["time"]), dtype=float)
    return _daily(df)


def _daily(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly -> the daily columns of the CSV (WEATHER_COLS)."""
    g = h.groupby(h.index.normalize())
    rad = np.deg2rad(h["wind_direction_10m"])
    u, v = np.sin(rad).groupby(h.index.normalize()).mean(), np.cos(rad).groupby(h.index.normalize()).mean()
    d = pd.DataFrame({
        "temperature_2m_max": g["temperature_2m"].max(),
        "temperature_2m_min": g["temperature_2m"].min(),
        "temperature_2m_mean": g["temperature_2m"].mean(),
        "apparent_temperature_max": g["apparent_temperature"].max(),
        "apparent_temperature_min": g["apparent_temperature"].min(),
        "precipitation_sum": g["precipitation"].sum(min_count=24),
        "rain_sum": g["rain"].sum(min_count=24),
        "snowfall_sum": g["snowfall"].sum(min_count=24),
        "windspeed_10m_max": g["wind_speed_10m"].max(),
        "windgusts_10m_max": g["wind_gusts_10m"].max(),
        "winddirection_10m_dominant": np.rad2deg(np.arctan2(u, v)) % 360,
    })
    # Days with any missing hour are unreliable.
    d[g["temperature_2m"].count() < 24] = np.nan
    return d.rename_axis("date")


def load_forecasts(start: str = "2025-06-01", end: str = "2026-08-15", refresh: bool = False) -> pd.DataFrame:
    """Daily forecasts by lead, long format: index (lead, date), WEATHER_COLS.
    Cached to data/ after the first download."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    frames = {k: _fetch_lead(k, start, end) for k in range(MAX_LEAD + 1)}
    out = pd.concat(frames, names=["lead"])
    out.to_parquet(CACHE)
    return out


# Variables bounded below by zero after adding forecast error.
_NONNEG = ["precipitation_sum", "rain_sum", "snowfall_sum", "windspeed_10m_max", "windgusts_10m_max"]


class ForecastWeather:
    """Callable future-weather provider for `make_tasks(future_weather=...)`.

    Returns, for an origin and its horizon dates, a frame shaped like
    `panel.weather.loc[dates]` but built only from information available
    before the origin.
    """

    def __init__(self, observed: pd.DataFrame, forecasts: pd.DataFrame | None = None,
                 clim_window: int = 7):
        self.obs = observed[WEATHER_COLS]
        fc = forecasts if forecasts is not None else load_forecasts()
        self.err = {k: (fc.loc[k] - fc.loc[0])[WEATHER_COLS] for k in range(1, MAX_LEAD + 1)}
        self.clim_window = clim_window
        self._cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def climatology(self, origin: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Mean of observed weather within +-clim_window days of each date's
        day-of-year, over years strictly before the origin."""
        hist = self.obs.loc[: origin - pd.Timedelta(days=1)]
        doy = hist.index.dayofyear.to_numpy()
        rows = []
        for d in dates:
            dist = np.abs(doy - d.dayofyear)
            dist = np.minimum(dist, 365 - dist)
            sel = hist[(dist <= self.clim_window) & (hist.index < d - pd.Timedelta(days=300))]
            row = sel.mean()
            # Wind direction is circular: an arithmetic mean is meaningless.
            row["winddirection_10m_dominant"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows, index=dates)

    def raw(self, origin: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
        out = self.climatology(origin, dates)
        for h, d in enumerate(dates, start=1):
            if h > MAX_LEAD or d not in self.err[h].index or d not in self.obs.index:
                continue
            e = self.err[h].loc[d]
            if e.isna().any():
                continue
            row = self.obs.loc[d] + e
            row["winddirection_10m_dominant"] = (self.obs.loc[d, "winddirection_10m_dominant"]
                                                 + e["winddirection_10m_dominant"]) % 360
            out.loc[d] = row
        out[_NONNEG] = out[_NONNEG].clip(lower=0)
        return out

    def __call__(self, origin: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
        key = (origin, len(dates))
        if key not in self._cache:
            past = self.obs.loc[origin - pd.Timedelta(days=30): origin - pd.Timedelta(days=1)]
            full = pd.concat([past, self.raw(origin, dates)])
            # Rolling/streak features must see forecast values for future days.
            self._cache[key] = add_weather_features(full).loc[dates]
        return self._cache[key]
