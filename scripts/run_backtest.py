"""Run the rolling-origin backtest and write results + a text report.

Example (fast, no TimesFM):
    python scripts/run_backtest.py --models seasonal_naive,degree_day,lgbm
Example (TimesFM on a few buildings, CPU is slow):
    python scripts/run_backtest.py --models seasonal_naive,timesfm,timesfm_weather \
        --buildings AnabelTaylorHall,120MapleAve
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldsnaps.backtest import run_backtest  # noqa: E402
from coldsnaps.data import load_panel  # noqa: E402
from coldsnaps.forecast_weather import ForecastWeather  # noqa: E402
from coldsnaps.metrics import peak_report, report  # noqa: E402
from coldsnaps.models import REGISTRY  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="seasonal_naive,degree_day,lgbm")
    p.add_argument("--test-start", default="2025-08-01")
    p.add_argument("--test-end", default=None)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--step", type=int, default=7)
    p.add_argument("--buildings", default=None, help="comma-separated; default all")
    p.add_argument("--meter-policy", default="single", choices=["single", "sum"])
    p.add_argument("--target", default="kw", choices=["kw", "max_kw"])
    p.add_argument("--weather", default="observed", choices=["observed", "forecast"],
                   help="future weather: observed (upper bound) or archived Open-Meteo forecasts")
    p.add_argument("--out", default="baseline")
    a = p.parse_args()

    panel = load_panel(meter_policy=a.meter_policy)
    buildings = a.buildings.split(",") if a.buildings else None
    models = [REGISTRY[m]() for m in a.models.split(",")]
    future_weather = ForecastWeather(panel.weather) if a.weather == "forecast" else None
    res = run_backtest(panel, models, a.test_start, a.test_end, a.horizon, a.step, buildings,
                       target=a.target, future_weather=future_weather)

    out = ROOT / "results" / a.out
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / "predictions.parquet", index=False)
    text = report(res) + "\n\n" + peak_report(res, panel, a.test_start, a.target)
    (out / "report.txt").write_text(text)
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
