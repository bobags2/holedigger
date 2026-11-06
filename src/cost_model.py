"""Cost as a learnable target: predict the spread a trade will actually pay.

Why this module
---------------
Every "dead on cost" verdict in this project has been computed with a *fixed* cost: the
median spread of the symbol, or the spread at the candidate's own bar. Neither is a model.
The spread is a state variable — it varies through the session by 2x or more, tracks tick
activity, and blows out when volatility does — and it is the one thing in this project with
real predictive signal. So predict it, and let the sweep's net-of-cost columns use the
prediction instead of a median.

What "cost" means here
----------------------
The target is the **round-trip half-spread** at the bar where a trade enters: crossing the
spread once on entry, once on exit, at whatever the book is at that moment. The target is
``spread_med_bp`` from the tick-derived bars, which is exactly what the sweep already
charges per trade — so a better prediction is a drop-in replacement.

The model
---------
Gradient-boosted regression on ``lightgbm`` when installed, falling back to
``sklearn.ensemble.HistGradientBoostingRegressor`` when it is not; if neither is installed
the module raises, so no silent "no model" path exists. Features are deliberately *state*
features, all available at entry time (no lookahead):

* ``hour`` / ``tod_sin`` / ``tod_cos`` — time of day (the dominant term, and the thing a
  per-symbol median throws away)
* ``tick_n`` — activity, log-scaled
* ``ret_abs``, ``ret_abs_lag`` — current and previous bar absolute return (volatility)
* ``range_bp`` — the bar's high-low range in bp
* ``spread_lag`` — the previous bar's spread (persistence; available at entry)

The split is **contiguous in time**, never random: a random split would let tomorrow's
spread predict today's and the out-of-sample score would be optimistic by construction.
The test asserts the model beats the per-candidate median baseline on the held-out tail,
which is the honest bar.

Usage::

    from src import cost_model
    m = cost_model.fit_cost_model("XPBUSD")
    m.predict_cost(df)          # same frame, predicted half-spread in bp
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import mt5_clock

FEATURES = ["hour", "tod_sin", "tod_cos", "log_ticks", "ret_abs",
            "ret_abs_lag", "range_bp", "spread_lag"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """State features available at entry time, from a bar frame with ts/ohlc/spread."""
    close = df["close"].to_numpy(dtype=float)
    ret = np.zeros(len(df), dtype=float)
    ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    ret_abs = np.abs(ret)
    rng = (df["high"].to_numpy(float) - df["low"].to_numpy(float)) / np.maximum(close, 1e-12) * 1e4
    hour = df["ts"].dt.hour.to_numpy() + df["ts"].dt.minute.to_numpy() / 60.0
    ticks = df["tick_n"].to_numpy(dtype=float) if "tick_n" in df else np.ones(len(df))
    spread = df["spread_med_bp"].to_numpy(dtype=float) if "spread_med_bp" in df else np.full(len(df), np.nan)
    lag = np.full(len(df), np.nan)
    lag[1:] = ret_abs[:-1]
    slag = np.full(len(df), np.nan)
    slag[1:] = spread[:-1]
    out = pd.DataFrame({
        "hour": hour,
        "tod_sin": np.sin(2 * np.pi * hour / 24.0),
        "tod_cos": np.cos(2 * np.pi * hour / 24.0),
        "log_ticks": np.log1p(ticks),
        "ret_abs": ret_abs,
        "ret_abs_lag": lag,
        "range_bp": rng,
        "spread_lag": slag,
    }, index=df.index)
    return out[FEATURES]


@dataclass
class CostModel:
    """A fitted spread predictor plus the baseline it must beat."""

    symbol: str
    model: object
    train_rows: int
    test_rows: int
    baseline_test_mae: float
    test_mae: float
    feature_names: list[str]

    def predict_cost(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted half-spread (bp) per bar of ``df``.

        Clamped at zero: a spread cannot be negative, and a raw regressor does predict small
        negatives on symbols whose true spread sits at 0 most of the time (gold: measured).
        Charging a negative cost would *subsidise* a trade, which is the one direction this
        correction must never fail in.
        """
        X = build_features(df)
        pred = np.asarray(self.model.predict(X), dtype=float)
        return np.maximum(pred, 0.0)

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "test_mae_bp": round(float(self.test_mae), 4),
            "baseline_mae_bp": round(float(self.baseline_test_mae), 4),
            "mae_improvement": round(
                float(1.0 - self.test_mae / max(self.baseline_test_mae, 1e-12)), 4
            ),
            "features": self.feature_names,
        }


def _make_model():
    """GBDT: lightgbm if present, else sklearn's histogram GBM. Never a silent fallback."""
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=0, verbose=-1,
        )
    except ImportError:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0, random_state=0,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cost model needs lightgbm or scikit-learn; install one "
            "(`pip install lightgbm` is the faster of the two on this data)"
        ) from exc


def fit_cost_model(
    symbol: str,
    *,
    test_fraction: float = 0.25,
    parquet=None,
) -> CostModel:
    """Fit on the first ``1 - test_fraction`` of the panel, test on the held-out tail.

    The split is contiguous in time on purpose: a random split would let later bars inform
    earlier ones and the out-of-sample number would be optimistic by construction.
    """
    from src import config

    path = parquet or (config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    df = mt5_clock.load_mt5_utc(path)
    df = df.dropna(subset=["spread_med_bp"]).reset_index(drop=True)
    if len(df) < 500:
        raise ValueError(f"{symbol}: only {len(df)} usable bars")
    n = len(df)
    cut = int(n * (1.0 - test_fraction))

    feat = build_features(df)
    y = df["spread_med_bp"].to_numpy(dtype=float)
    tr, te = np.arange(0, cut), np.arange(cut, n)

    model = _make_model()
    model.fit(feat.iloc[tr], y[tr])

    pred = np.asarray(model.predict(feat.iloc[te]), dtype=float)
    mae = float(np.mean(np.abs(pred - y[te])))
    # baseline: the per-symbol training median, charged at every trade - what the sweep
    # does today (via the adopted spread) and what mt5_audit reports
    baseline = float(np.median(y[tr]))
    baseline_mae = float(np.mean(np.abs(np.full(len(te), baseline) - y[te])))
    return CostModel(
        symbol=symbol, model=model, train_rows=len(tr), test_rows=len(te),
        baseline_test_mae=baseline_mae, test_mae=mae,
        feature_names=list(FEATURES),
    )
