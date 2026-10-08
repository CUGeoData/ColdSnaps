"""Load the merged energy/weather CSV into a clean, gap-aware daily panel.

Panel layout: one row per (building, date), contiguous daily dates per building,
`kw` NaN where the meter had no (or too little) data. Weather is campus-wide and
lives in its own date-indexed frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

WEATHER_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
]

# Temperatures in the source data are Fahrenheit.
DEGREE_DAY_BASE_F = 65.0


@dataclass
class Panel:
    energy: pd.DataFrame  # columns: building, date, kw, max_kw, n_points
    weather: pd.DataFrame  # index: date, columns: WEATHER_COLS + derived
    buildings: list[str]

    def series(self, building: str) -> pd.Series:
        e = self.energy
        return e.loc[e["building"] == building].set_index("date")["kw"]


def add_weather_features(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    t = w["temperature_2m_mean"]
    w["hdd"] = (DEGREE_DAY_BASE_F - t).clip(lower=0)
    w["cdd"] = (t - DEGREE_DAY_BASE_F).clip(lower=0)
    w["hdd_x_wind"] = w["hdd"] * w["windspeed_10m_max"]
    w["temp_mean_3d"] = t.rolling(3, min_periods=1).mean()
    w["temp_mean_7d"] = t.rolling(7, min_periods=1).mean()
    w["temp_change_1d"] = t.diff()
    # Consecutive days with mean temperature below freezing.
    cold = (t < 32).astype(int)
    w["consecutive_cold_days"] = cold.groupby((cold == 0).cumsum()).cumsum()
    w["weekday"] = w.index.dayofweek
    w["month"] = w.index.month
    w["doy_sin"] = np.sin(2 * np.pi * w.index.dayofyear / 365.25)
    w["doy_cos"] = np.cos(2 * np.pi * w.index.dayofyear / 365.25)
    return w


def _combine_meters(df: pd.DataFrame, meter_policy: str) -> pd.DataFrame:
    n_meters = df.groupby("building")["target"].nunique()
    if meter_policy == "single":
        keep = n_meters.index[n_meters == 1]
        return df[df["building"].isin(keep)].drop(columns="target")
    if meter_policy == "sum":
        # Only sum dates where every meter of the building reported, so a
        # missing sub-meter doesn't show up as a drop in load. Note max_kw of a
        # sum is approximated by the sum of maxes (an upper bound).
        df = df.copy()
        df["n_meters"] = df["building"].map(n_meters)
        g = df.groupby(["building", "date"])
        out = g.agg(
            mean_kw=("mean_kw", "sum"),
            max_kw=("max_kw", "sum"),
            n_points=("n_points", "min"),
            n_reported=("target", "nunique"),
            n_meters=("n_meters", "first"),
        ).reset_index()
        out = out[out["n_reported"] == out["n_meters"]]
        return out.drop(columns=["n_reported", "n_meters"])
    raise ValueError(f"unknown meter_policy {meter_policy!r}")


def load_panel(
    path: Path | str = DATA_DIR / "merged_energy_weather.csv",
    meter_policy: str = "single",
    min_coverage: float = 0.75,
    max_interp_gap: int = 2,
) -> Panel:
    """Build the daily panel.

    meter_policy: "single" keeps buildings with exactly one meter (matches
        building_daily_normal_weekdays_features.csv); "sum" also includes
        multi-meter buildings by summing meters.
    min_coverage: a day's mean is dropped when its n_points is below this
        fraction of the building's typical n_points (partial days bias means).
    max_interp_gap: interior gaps up to this many days are linearly filled.
    """
    raw = pd.read_csv(path, parse_dates=["date"])

    weather = (
        raw.groupby("date")[WEATHER_COLS].first().sort_index().asfreq("D")
    )
    has_weather = weather["temperature_2m_mean"].notna()
    lo, hi = has_weather.idxmax(), has_weather[::-1].idxmax()
    weather = weather.loc[lo:hi].interpolate(limit=max_interp_gap)
    weather = add_weather_features(weather)

    e = _combine_meters(raw[["date", "building", "target", "mean_kw", "max_kw", "n_points"]], meter_policy)
    e = e[(e["date"] >= lo) & (e["date"] <= hi)]

    typical = e.groupby("building")["n_points"].transform("median")
    partial = e["n_points"] < min_coverage * typical
    e.loc[partial, ["mean_kw", "max_kw"]] = np.nan

    frames = []
    for b, g in e.groupby("building"):
        g = g.set_index("date").sort_index()
        g = g[~g.index.duplicated()]
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx)
        for c in ("mean_kw", "max_kw"):
            g[c] = g[c].interpolate(limit=max_interp_gap, limit_area="inside")
        g["building"] = b
        frames.append(g.rename_axis("date").reset_index())
    energy = pd.concat(frames, ignore_index=True).rename(columns={"mean_kw": "kw"})
    energy = energy[["building", "date", "kw", "max_kw", "n_points"]]

    return Panel(energy=energy, weather=weather, buildings=sorted(energy["building"].unique()))


def load_building_meta() -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "building_labels.csv")
    return labels.set_index("building")[["building_type"]]
