"""Accuracy summaries over long-format backtest results."""
from __future__ import annotations

import numpy as np
import pandas as pd

# A target day counts as "cold" at or below this daily mean temperature (F).
COLD_DAY_F = 20.0


def _scores(g: pd.DataFrame) -> pd.Series:
    err = g["yhat"] - g["y"]
    return pd.Series({
        "n": len(g),
        "mae": err.abs().mean(),
        "rmse": np.sqrt((err**2).mean()),
        "wape": err.abs().sum() / g["y"].abs().sum(),
        "bias_pct": err.sum() / g["y"].abs().sum() * 100,
    })


def summarize(res: pd.DataFrame, by: list[str] | None = None, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Scores per model (and optional extra grouping). `skill` is
    1 - MAE / baseline MAE on the same rows, so >0 beats the baseline."""
    res = res.assign(cold=res["temperature_2m_mean"] <= COLD_DAY_F)
    keys = ["model"] + (by or [])
    out = res.groupby(keys).apply(_scores, include_groups=False)

    # Per-building WAPE, then median: keeps big buildings from dominating.
    per_b = res.groupby(keys + ["building"]).apply(
        lambda g: (g["yhat"] - g["y"]).abs().sum() / g["y"].abs().sum(), include_groups=False
    )
    out["median_building_wape"] = per_b.groupby(level=keys).median()

    if baseline in res["model"].unique():
        base = out.xs(baseline, level="model")["mae"]
        if by:
            out["skill"] = 1 - out["mae"] / base.reindex(out.index.droplevel("model")).to_numpy()
        else:
            out["skill"] = 1 - out["mae"] / base.iloc[0]
    return out.sort_values("mae") if not by else out


def report(res: pd.DataFrame) -> str:
    pd.set_option("display.width", 160)
    parts = [
        "== Overall", summarize(res).round(4).to_string(),
        f"\n== Cold target days (mean temp <= {COLD_DAY_F:g}F)",
        summarize(res[res["temperature_2m_mean"] <= COLD_DAY_F]).round(4).to_string(),
        "\n== By building type (median building WAPE)",
        summarize(res, ["building_type"])["median_building_wape"].unstack(0).round(4).to_string(),
        "\n== By horizon (MAE)",
        summarize(res, ["h"])["mae"].unstack(0).round(3).to_string(),
    ]
    return "\n".join(parts)
