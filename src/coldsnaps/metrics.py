"""Accuracy summaries over long-format backtest results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import QUANTILES, qcol

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
    # Buildings with ~zero load have undefined WAPE; leave them out of the median.
    g = res.assign(abs_err=(res["yhat"] - res["y"]).abs(), abs_y=res["y"].abs())
    sums = g.groupby(keys + ["building"])[["abs_err", "abs_y"]].sum()
    per_b = (sums["abs_err"] / sums["abs_y"]).where(sums["abs_y"] > 1e-6)
    out["median_building_wape"] = per_b.groupby(level=keys).median()

    if baseline in res["model"].unique():
        if by:
            base = out.xs(baseline, level="model")["mae"]
            out["skill"] = 1 - out["mae"] / base.reindex(out.index.droplevel("model")).to_numpy()
        else:
            out["skill"] = 1 - out["mae"] / out.loc[baseline, "mae"]
    return out.sort_values("mae") if not by else out


def _prob_scores(g: pd.DataFrame) -> pd.Series:
    y = g["y"].to_numpy()
    q = g[[qcol(a) for a in QUANTILES]].to_numpy()
    levels = np.asarray(QUANTILES)
    # CRPS approximated by twice the mean pinball loss over the quantile levels.
    diff = y[:, None] - q
    pinball = np.maximum(levels * diff, (levels - 1) * diff)
    crps = 2 * pinball.mean(axis=1)
    lo, hi = g[qcol(0.1)].to_numpy(), g[qcol(0.9)].to_numpy()
    return pd.Series({
        "n": len(g),
        "crps": crps.mean(),
        # Scaled like WAPE so buildings of different size are comparable.
        "crps_scaled": crps.sum() / np.abs(y).sum(),
        "cover80": ((y >= lo) & (y <= hi)).mean(),
        "width80_pct": (hi - lo).sum() / np.abs(y).sum() * 100,
        "below_q10": (y < lo).mean(),
        "above_q90": (y > hi).mean(),
    })


def summarize_quantiles(res: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Probabilistic scores for models that produced quantiles. An 80%
    interval (q10..q90) is well calibrated when cover80 ~= 0.80."""
    if qcol(0.5) not in res:
        return pd.DataFrame()
    res = res[res[qcol(0.5)].notna()]
    if res.empty:
        return pd.DataFrame()
    return res.groupby(["model"] + (by or [])).apply(_prob_scores, include_groups=False)


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
    probs = summarize_quantiles(res)
    if not probs.empty:
        cold = res[res["temperature_2m_mean"] <= COLD_DAY_F]
        parts += [
            "\n== Probabilistic (quantile models)", probs.round(4).to_string(),
            f"\n== Probabilistic, cold target days (mean temp <= {COLD_DAY_F:g}F)",
            summarize_quantiles(cold).round(4).to_string(),
        ]
    return "\n".join(parts)


# ---- Cold-snap events and peak risk -------------------------------------------------

def cold_snap_events(weather: pd.DataFrame, threshold_f: float = COLD_DAY_F, min_days: int = 3) -> pd.DataFrame:
    """Runs of >= min_days consecutive days with mean temperature <= threshold_f."""
    t = weather["temperature_2m_mean"]
    cold = t <= threshold_f
    run_id = (~cold).cumsum()[cold]
    ev = t[cold].groupby(run_id).agg(start=lambda s: s.index.min(), end=lambda s: s.index.max(),
                                      days="size", min_temp="min")
    ev = ev[ev["days"] >= min_days].reset_index(drop=True)
    ev.index.name = "event"
    return ev


def tag_events(res: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Event id for each result row whose target date falls in a cold snap, else NaN."""
    ev = pd.Series(np.nan, index=res.index)
    for i, e in events.iterrows():
        ev[(res["date"] >= e["start"]) & (res["date"] <= e["end"])] = i
    return ev


def high_load_thresholds(panel, before: str | pd.Timestamp, q: float = 0.9,
                         target: str = "kw", days: int = 365) -> pd.Series:
    """Per-building 'high load day' threshold: the q-quantile of daily load over
    the year before `before` (so it is known when forecasting)."""
    before = pd.Timestamp(before)
    e = panel.energy
    e = e[(e["date"] < before) & (e["date"] >= before - pd.Timedelta(days=days))]
    return e.groupby("building")[target].quantile(q)


def _prob_exceed(g: pd.DataFrame, thr: np.ndarray) -> np.ndarray:
    """P(y > thr) from the quantile forecast, interpolating the CDF linearly
    between quantile levels and capping the tails at 0.05 / 0.95."""
    q = g[[qcol(a) for a in QUANTILES]].to_numpy()
    levels = np.asarray(QUANTILES)
    cdf = np.array([np.interp(t, qi, levels, left=0.05, right=0.95) for t, qi in zip(thr, q)])
    return 1 - cdf


def exceedance_scores(res: pd.DataFrame, thresholds: pd.Series, by: list[str] | None = None) -> pd.DataFrame:
    """Can the model warn about high-load days? Point forecasts are scored as a
    yes/no warning (yhat > threshold): precision, recall. Quantile models also
    get a Brier score for P(y > threshold) and Brier skill vs. always
    predicting the base rate."""
    res = res[res["building"].isin(thresholds.index)].copy()
    res["thr"] = res["building"].map(thresholds)
    res["event_obs"] = res["y"] > res["thr"]
    res["warn"] = res["yhat"] > res["thr"]
    has_q = qcol(0.5) in res
    if has_q:
        res["p"] = np.nan
        m = res[qcol(0.5)].notna()
        res.loc[m, "p"] = _prob_exceed(res[m], res.loc[m, "thr"].to_numpy())

    def score(g):
        obs, warn = g["event_obs"], g["warn"]
        out = {
            "n": len(g),
            "base_rate": obs.mean(),
            "precision": (obs & warn).sum() / max(warn.sum(), 1),
            "recall": (obs & warn).sum() / max(obs.sum(), 1),
        }
        if has_q and g["p"].notna().all():
            brier = ((g["p"] - obs) ** 2).mean()
            ref = ((obs.mean() - obs) ** 2).mean()
            out["brier"] = brier
            out["brier_skill"] = 1 - brier / ref if ref > 0 else np.nan
        return pd.Series(out)

    return res.groupby(["model"] + (by or [])).apply(score, include_groups=False)


def peak_day_scores(res: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """For each building and cold snap, take the day of highest actual load
    and score every forecast made for that day: signed % error (negative =
    under-forecast the peak), by lead bucket."""
    res = res.assign(event=tag_events(res, events)).dropna(subset=["event"])
    actual = res.groupby(["building", "event", "date"])["y"].first().reset_index()
    peak = actual.loc[actual.groupby(["building", "event"])["y"].idxmax(), ["building", "date"]]
    r = res.merge(peak, on=["building", "date"])
    r = r[r["y"] > 1e-6]
    r["pct_err"] = (r["yhat"] - r["y"]) / r["y"] * 100
    r["lead"] = np.where(r["h"] <= 7, "1-7d", "8-14d")
    return r.groupby(["model", "lead"]).agg(
        n=("pct_err", "size"),
        mean_abs_pct_err=("pct_err", lambda s: s.abs().mean()),
        mean_pct_err=("pct_err", "mean"),
        under_share=("pct_err", lambda s: (s < 0).mean()),
    )


def peak_report(res: pd.DataFrame, panel, test_start: str | pd.Timestamp, target: str = "kw") -> str:
    events = cold_snap_events(panel.weather.loc[res["date"].min():res["date"].max()])
    thr = high_load_thresholds(panel, test_start, target=target)
    cold = res.assign(event=tag_events(res, events)).dropna(subset=["event"])
    parts = [
        f"== Cold snaps in test period (>= 3 days with mean <= {COLD_DAY_F:g}F)",
        events.to_string(),
        "\n== High-load day warnings, all test days (threshold = building's trailing-year p90)",
        exceedance_scores(res, thr).round(4).to_string(),
        "\n== High-load day warnings, cold-snap days only",
        exceedance_scores(cold, thr).round(4).to_string(),
        "\n== Error on each building's peak day within each cold snap (%)",
        peak_day_scores(res, events).round(2).to_string(),
    ]
    return "\n".join(parts)
