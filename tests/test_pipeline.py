import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coldsnaps.backtest import make_tasks  # noqa: E402
from coldsnaps.data import load_panel  # noqa: E402
from coldsnaps.models import DegreeDayRegression, SeasonalNaive  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return load_panel()


def test_one_row_per_building_day(panel):
    assert not panel.energy.duplicated(["building", "date"]).any()
    # 133 single-meter buildings minus KaneSportsComplex & MeinigFieldhouse,
    # whose data starts after the weather record ends.
    assert len(panel.buildings) == 131


def test_weather_complete_and_degree_days_in_fahrenheit(panel):
    w = panel.weather
    assert w["temperature_2m_mean"].notna().all()
    hot = w[w["temperature_2m_mean"] > 75]
    assert (hot["hdd"] == 0).all() and (hot["cdd"] > 0).all()
    freezing = w[w["temperature_2m_mean"] < 20]
    assert (freezing["cdd"] == 0).all() and (freezing["hdd"] > 45).all()


def test_tasks_have_no_leakage(panel):
    tasks = make_tasks(panel, "2025-08-01", "2025-10-01", horizon=14, step=7, buildings=panel.buildings[:5])
    assert tasks
    for t in tasks:
        assert t.context.index.max() < t.origin
        assert not np.isnan(t.target).any()
        actual = panel.series(t.building).reindex(t.future_dates).to_numpy()
        np.testing.assert_allclose(t.target, actual)


@pytest.mark.parametrize("model", [SeasonalNaive(), DegreeDayRegression()])
def test_baselines_shape(panel, model):
    tasks = make_tasks(panel, "2025-08-01", "2025-09-15", 14, 7, panel.buildings[:3])
    p = model.predict(panel, tasks, 14)
    assert p.shape == (len(tasks), 14) and np.isfinite(p).all()


def test_sum_policy_adds_multi_meter_buildings():
    p = load_panel(meter_policy="sum")
    assert len(p.buildings) > 133
    assert "KlarmanHall" in p.buildings


def test_forecast_weather_uses_only_past(panel):
    from coldsnaps.forecast_weather import CACHE, MAX_LEAD, ForecastWeather

    if not CACHE.exists():
        pytest.skip("Open-Meteo forecast cache not downloaded")
    fw = ForecastWeather(panel.weather)
    o = pd.Timestamp("2026-01-20")
    dates = pd.date_range(o, periods=14)
    w = fw(o, dates)
    assert list(w.index) == list(dates) and w["temperature_2m_mean"].notna().all()
    # Climatology days must not change if the actual future weather changes.
    altered = panel.weather.copy()
    altered.loc[o:, "temperature_2m_mean"] += 30
    w2 = ForecastWeather(altered)(o, dates)
    np.testing.assert_allclose(w["temperature_2m_mean"].iloc[MAX_LEAD:], w2["temperature_2m_mean"].iloc[MAX_LEAD:])


def test_lgbm_quantiles_ordered(panel):
    from coldsnaps.models import LightGBMGlobal

    b = panel.buildings[:4]
    m = LightGBMGlobal(n_estimators=20, train_step=14, calib_days=120)
    m.fit(panel, pd.Timestamp("2025-08-01"), buildings=b)
    tasks = make_tasks(panel, "2025-08-01", "2025-09-15", 14, 7, b)
    point, q = m.predict_quantiles(panel, tasks, 14)
    assert q.shape == (len(tasks), 14, 9)
    assert (np.diff(q, axis=-1) >= -1e-9).all()


def test_cold_snap_scenario_only_touches_snap_days(panel):
    from coldsnaps.scenarios import ColdSnapScenario

    o = pd.Timestamp("2026-01-05")
    dates = pd.date_range(o, periods=14)
    base = ColdSnapScenario(panel.weather, 0)(o, dates)
    cold = ColdSnapScenario(panel.weather, 10, snap_days=(4, 10))(o, dates)
    d_temp = (cold["temperature_2m_mean"] - base["temperature_2m_mean"]).to_numpy()
    np.testing.assert_allclose(d_temp, [0] * 3 + [-10] * 7 + [0] * 4)
    # Winter: every degree colder is one more HDD.
    np.testing.assert_allclose((cold["hdd"] - base["hdd"]).iloc[3:10], 10)
    np.testing.assert_allclose(base["temperature_2m_mean"], panel.weather.loc[dates, "temperature_2m_mean"])
