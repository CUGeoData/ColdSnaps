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
    assert len(panel.buildings) == 133


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
