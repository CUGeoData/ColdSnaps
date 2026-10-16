"""Synthetic cold-snap what-if test (PLAN.md phase 5).

Lowers temperatures by delta F on horizon days 4-10 for winter origins and
compares each model's load response with the building's empirical HDD slope.

Example:
    python scripts/run_scenarios.py --n-buildings 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldsnaps.backtest import make_tasks  # noqa: E402
from coldsnaps.data import load_panel  # noqa: E402
from coldsnaps.models import make_model  # noqa: E402
from coldsnaps.scenarios import ColdSnapScenario, empirical_hdd_slope, scenario_sensitivity  # noqa: E402

SNAP_DAYS = (4, 10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="degree_day,lgbm,timesfm_weather")
    p.add_argument("--deltas", default="0,10,20")
    p.add_argument("--origins", default="2025-12-01,2025-12-22,2026-01-12,2026-02-02,2026-02-23")
    p.add_argument("--n-buildings", type=int, default=40)
    p.add_argument("--train-end", default="2025-08-01")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="TimesFM device: auto, cuda or cpu")
    p.add_argument("--batch-size", type=int, default=None, help="TimesFM batch size (default 32)")
    p.add_argument("--out", default="scenarios")
    a = p.parse_args()

    panel = load_panel()
    rng = np.random.default_rng(a.seed)
    buildings = sorted(rng.choice(panel.buildings, size=min(a.n_buildings, len(panel.buildings)),
                                  replace=False))
    origins = pd.to_datetime(a.origins.split(","))
    deltas = [float(d) for d in a.deltas.split(",")]
    horizon = 14

    results = {d: [] for d in deltas}
    for name in a.models.split(","):
        m = make_model(name, a.device, a.batch_size)
        # Global models train on all buildings (as in the backtest), only before train_end.
        m.fit(panel, pd.Timestamp(a.train_end))
        for d in deltas:
            scen = ColdSnapScenario(panel.weather, d, SNAP_DAYS)
            tasks = [t for o in origins
                     for t in make_tasks(panel, o, o + pd.Timedelta(days=horizon - 1), horizon, 7,
                                         buildings, future_weather=scen)]
            preds = np.asarray(m.predict(panel, tasks, horizon))
            for t, yhat in zip(tasks, preds):
                results[d].append(pd.DataFrame({"model": name, "building": t.building, "origin": t.origin,
                                                "h": np.arange(1, horizon + 1), "yhat": yhat}))
            print(f"{name} delta={d:g}: {len(tasks)} tasks")
    results = {d: pd.concat(r, ignore_index=True) for d, r in results.items()}

    sens = scenario_sensitivity(results, SNAP_DAYS)
    emp = pd.DataFrame([(b, o, *empirical_hdd_slope(panel, b, o)) for b in buildings for o in origins],
                       columns=["building", "origin", "emp_kw_per_hdd", "mean_kw"])
    sens = sens.merge(emp, on=["building", "origin"])
    # Express everything as % of the building's mean load per 10 F colder.
    sens["model_pct_per_10f"] = sens["kw_per_f"] / sens["mean_kw"] * 1000
    sens["emp_pct_per_10f"] = sens["emp_kw_per_hdd"] / sens["mean_kw"] * 1000
    sens["leak_pct_per_10f"] = sens["leak_kw_per_f"] / sens["mean_kw"] * 1000

    out = ROOT / "results" / a.out
    out.mkdir(parents=True, exist_ok=True)
    sens.to_parquet(out / "sensitivity.parquet", index=False)
    text = report(sens)
    (out / "report.txt").write_text(text)
    print(text)


def report(sens: pd.DataFrame) -> str:
    pd.set_option("display.width", 160)
    # Average over origins, so each building counts once per model and delta.
    b = sens.groupby(["model", "delta_f", "building"])[
        ["model_pct_per_10f", "emp_pct_per_10f", "leak_pct_per_10f"]].mean().reset_index()
    # Only judge the sign where history shows a clear heating response.
    clear = b["emp_pct_per_10f"].abs() >= 1.0

    def score(g):
        c = g[clear.loc[g.index]]
        return pd.Series({
            "buildings": len(g),
            "median_model_pct": g["model_pct_per_10f"].median(),
            "median_emp_pct": g["emp_pct_per_10f"].median(),
            "sign_agree": (np.sign(c["model_pct_per_10f"]) == np.sign(c["emp_pct_per_10f"])).mean(),
            "spearman_vs_emp": g["model_pct_per_10f"].corr(g["emp_pct_per_10f"], method="spearman"),
            "median_leak_pct": g["leak_pct_per_10f"].median(),
        })

    s = b.groupby(["model", "delta_f"]).apply(score, include_groups=False)
    return "\n".join([
        f"Synthetic cold snap on horizon days {SNAP_DAYS[0]}-{SNAP_DAYS[1]}; units: % of mean load per 10F colder.",
        "sign_agree: share of buildings (with |empirical slope| >= 1%/10F) where the model moves the same way.",
        "spearman_vs_emp: rank correlation across buildings between model and empirical sensitivity.",
        "median_leak_pct: response on non-snap days (should be ~0).\n",
        s.round(3).to_string(),
    ])


if __name__ == "__main__":
    main()
