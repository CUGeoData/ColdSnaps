"""Forecasters sharing one interface:

    fit(panel, train_end, buildings, target)  -- may use only data before train_end
    predict(panel, tasks, horizon)            -- returns array (len(tasks), horizon)

Optionally predict_quantiles(panel, tasks, horizon) -> (point, (n, horizon, 9))
at backtest.QUANTILES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import QUANTILES, Task, make_tasks
from .data import WEATHER_COLS, Panel, load_building_meta


def _filled(context: pd.Series) -> np.ndarray:
    return context.interpolate(limit_direction="both").to_numpy()


class SeasonalNaive:
    """Same weekday last week."""

    name = "seasonal_naive"

    def fit(self, panel, train_end, buildings=None, target="kw"):
        pass

    def predict(self, panel, tasks, horizon):
        out = []
        for t in tasks:
            last_week = _filled(t.context)[-7:]
            out.append([last_week[i % 7] for i in range(horizon)])
        return np.array(out)


class DegreeDayRegression:
    """Per-building OLS on the trailing year: kw ~ HDD + CDD + weekday."""

    name = "degree_day"

    def __init__(self, window: int = 365):
        self.window = window

    def fit(self, panel, train_end, buildings=None, target="kw"):
        pass

    @staticmethod
    def _design(w: pd.DataFrame) -> np.ndarray:
        wd = np.eye(7)[w["weekday"].to_numpy()][:, 1:]
        return np.column_stack([np.ones(len(w)), w["hdd"], w["cdd"], wd])

    def predict(self, panel, tasks, horizon):
        out = []
        for t in tasks:
            hist = t.context.iloc[-self.window:].dropna()
            X = self._design(panel.weather.loc[hist.index])
            coef, *_ = np.linalg.lstsq(X, hist.to_numpy(), rcond=None)
            out.append(self._design(t.future_weather) @ coef)
        return np.array(out)


LGBM_WEATHER = WEATHER_COLS + ["hdd", "cdd", "hdd_x_wind", "temp_mean_3d", "temp_mean_7d",
                               "temp_change_1d", "consecutive_cold_days"]


class LightGBMGlobal:
    """One model across all buildings. Predicts kw relative to the building's
    trailing 28-day mean, so it can share patterns across building sizes."""

    name = "lgbm"

    def __init__(self, horizon: int = 14, train_step: int = 3, calib_days: int = 182, **params):
        self.horizon = horizon
        self.train_step = train_step
        self.calib_days = calib_days
        self.params = {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 63,
                       "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
                       "colsample_bytree": 0.8, "verbose": -1, **params}
        self.meta = load_building_meta()["building_type"]

    def _features(self, panel: Panel, tasks: list[Task], horizon: int):
        rows, scales = [], []
        for t in tasks:
            ctx = _filled(t.context)
            scale = max(np.nanmean(ctx[-28:]), 1e-3)
            hist_w = panel.weather.loc[t.context.index[-365:]]
            hist_y = ctx[-365:]
            # Building's trailing heating sensitivity, normalised by scale.
            if hist_w["hdd"].std() > 0 and np.std(hist_y) > 0:
                heat_corr = np.corrcoef(hist_w["hdd"], hist_y)[0, 1]
            else:
                heat_corr = 0.0
            fw = t.future_weather[LGBM_WEATHER + ["weekday", "month", "doy_sin", "doy_cos"]]
            f = fw.reset_index(drop=True)
            f["h"] = np.arange(1, horizon + 1)
            f["last_ratio"] = ctx[-1] / scale
            f["mean7_ratio"] = ctx[-7:].mean() / scale
            f["same_wd_ratio"] = ctx[-7:][np.arange(horizon) % 7] / scale
            f["hdd_at_origin"] = panel.weather["hdd"].loc[t.context.index[-7:]].mean()
            f["heat_corr"] = heat_corr
            f["log_scale"] = np.log(scale)
            f["building_type"] = self.meta.get(t.building, "Unknown")
            rows.append(f)
            scales.append(scale)
        X = pd.concat(rows, ignore_index=True)
        X["building_type"] = X["building_type"].astype(
            pd.CategoricalDtype(sorted(set(self.meta.unique()) | {"Unknown"})))
        return X, np.repeat(scales, horizon)

    def _fit_on(self, panel, start, end, buildings, target):
        import lightgbm as lgb

        tasks = make_tasks(panel, start, end, self.horizon, self.train_step, buildings, target=target)
        X, scale = self._features(panel, tasks, self.horizon)
        y = np.concatenate([t.target for t in tasks]) / scale
        return lgb.LGBMRegressor(objective="l1", **self.params).fit(X, y)

    def fit(self, panel, train_end, buildings=None, target="kw"):
        start = panel.weather.index.min() + pd.Timedelta(days=120)
        cal_start = train_end - pd.Timedelta(days=self.calib_days)
        # Split-conformal style quantiles: fit on data before the calibration
        # window, take per-horizon quantiles of scale-normalised residuals on
        # it, then refit the point model on everything.
        if self.calib_days:
            early = self._fit_on(panel, start, cal_start - pd.Timedelta(days=self.horizon), buildings, target)
            cal = make_tasks(panel, cal_start, train_end - pd.Timedelta(days=1), self.horizon,
                             self.train_step, buildings, target=target)
            X, scale = self._features(panel, cal, self.horizon)
            y = np.concatenate([t.target for t in cal]) / scale
            resid = (y - early.predict(X)).reshape(len(cal), self.horizon)
            self.resid_q = np.quantile(resid, QUANTILES, axis=0).T  # (horizon, 9)
        self.model = self._fit_on(panel, start, train_end - pd.Timedelta(days=1), buildings, target)

    def predict(self, panel, tasks, horizon):
        X, scale = self._features(panel, tasks, horizon)
        return (self.model.predict(X) * scale).reshape(len(tasks), horizon)

    def predict_quantiles(self, panel, tasks, horizon):
        X, scale = self._features(panel, tasks, horizon)
        ratio = self.model.predict(X).reshape(len(tasks), horizon)
        scale = scale.reshape(len(tasks), horizon)
        qs = (ratio[..., None] + self.resid_q[None, :horizon]) * scale[..., None]
        return ratio * scale, np.maximum(qs, 0)


TIMESFM_COVARIATES = ["temperature_2m_mean", "hdd", "cdd"]


class TimesFM:
    """google/timesfm-3.0 zero-shot, optionally with weather as past+future
    covariates (the approach from Model_1.ipynb)."""

    def __init__(self, use_weather: bool = True, context_len: int = 512, batch_size: int = 32,
                 covariates: list[str] | None = None, checkpoint: str = "google/timesfm-3.0-pytorch",
                 device: str = "cpu", name: str | None = None):
        self.use_weather = use_weather
        self.context_len = context_len
        self.batch_size = batch_size
        # Compute scales with the number of variates: 11 covariates is ~3.5x
        # slower on CPU than these 3.
        self.covariates = covariates or TIMESFM_COVARIATES
        self.checkpoint = checkpoint
        self.device = device
        self.name = name or ("timesfm_weather" if use_weather else "timesfm")
        self._model = None

    def fit(self, panel, train_end, buildings=None, target="kw"):
        if self._model is None:
            import timesfm3

            cfg = timesfm3.ModelConfig(checkpoint_path=self.checkpoint, device=self.device)
            self._model = timesfm3.TimesFM3Evaluator(cfg)

    def predict(self, panel, tasks, horizon):
        return self.predict_quantiles(panel, tasks, horizon)[0]

    def predict_quantiles(self, panel, tasks, horizon):
        """Median forecast (n, horizon) and quantiles (n, horizon, 9) at
        backtest.QUANTILES, which match TimesFM's output heads."""
        out, qs = [], []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            contexts, covs = [], []
            for t in batch:
                ctx = t.context.iloc[-self.context_len:]
                contexts.append(_filled(ctx).astype(np.float32))
                if self.use_weather:
                    w = pd.concat([panel.weather.loc[ctx.index, self.covariates],
                                   t.future_weather[self.covariates]])
                    covs.append(w.to_numpy(np.float32).T)
            kwargs = {"past_future_covariates": covs} if self.use_weather else {}
            res = self._model.predict_batch(contexts=contexts, horizon=horizon,
                                            return_quantiles=True, **kwargs)
            for r in res:
                out.append(np.asarray(r.forecast).reshape(-1)[:horizon])
                qs.append(np.asarray(r.quantiles).reshape(-1, len(QUANTILES))[:horizon])
            print(f"    timesfm {min(i + self.batch_size, len(tasks))}/{len(tasks)}", end="\r")
        print()
        return np.array(out), np.array(qs)


REGISTRY = {
    "seasonal_naive": SeasonalNaive,
    "degree_day": DegreeDayRegression,
    "lgbm": LightGBMGlobal,
    "timesfm": lambda **kw: TimesFM(use_weather=False, **kw),
    "timesfm_weather": lambda **kw: TimesFM(use_weather=True, **kw),
    "timesfm_weather_all": lambda **kw: TimesFM(use_weather=True, covariates=WEATHER_COLS,
                                                name="timesfm_weather_all", **kw),
}


def make_model(name: str, device: str = "auto", batch_size: int | None = None):
    """Instantiate a REGISTRY model; device/batch_size only apply to TimesFM.
    device="auto" picks CUDA when available."""
    if not name.startswith("timesfm"):
        return REGISTRY[name]()
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"device": device}
    if batch_size:
        kw["batch_size"] = batch_size
    return REGISTRY[name](**kw)
