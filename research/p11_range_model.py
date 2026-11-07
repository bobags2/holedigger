"""Next-period RANGE prediction — the second quantity that is genuinely predictable here.

Why this module exists, stated as a hypothesis it can fail:

The repo's search for a directional edge has closed eight families with zero survivors, and the
reason is uniform: the effects it can measure (~0.45 bp per ranking leg, 3-4 bp conditional
drift) are smaller than the toll (~10 bp median equity entry+exit). Every one of those is a
directional claim. RANGE is the other quantity, and it is not a directional claim at all:

  - The published cost probe (``artifacts/cost_probe.json``) already shows the mean absolute
    move varies 3.27 bp -> 8.46 bp across spread-state buckets on GBPUSD, a 2.6x spread in
    *representable* opportunity. That variation is what a range model harvests.
  - Range is bounded below by zero and roughly lognormal, so HAR-style features are the natural
    fit and the loss (QLIKE) has a known good functional form.
  - Range does not require predicting direction, so the alpha is not fighting the direction
    question the rest of the repo has already lost.

WHAT IT DELIVERS: a next-session range forecast per instrument, scored out-of-sample on QLIKE
and out-of-sample R2, plus an **opportunity score** = forecast range divided by the instrument's
own measured toll (``src.venue_cost``). That score is the net-per-trade-relevant quantity this
whole project has been trying to raise, and it is computable without any directional edge.

WHAT IT DOES NOT CLAIM: that the range forecast is an alpha. It is a sizing and filtering
input. A high score means "this instrument, this session, has a range worth paying its toll",
which is a statement about whether to trade, not which way.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, venue_cost as vc


def har_features(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """HAR-style realised-range features, all causal, all from the panel itself.

    Realised range over 1h / 4h / 1d of *bars already elapsed*, plus the intraday position and
    the time-of-day cyclical encoding. No forward information: every column at row ``i`` is
    computed from rows ``< i``.
    """
    close = bars["close"].to_numpy(float)
    hi, lo = bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    rng = (hi - lo) / close * 1e4                                    # bp per bar
    vol = bars["vol"].to_numpy(float) if "vol" in bars else np.ones(len(bars))
    s = lambda a: pd.Series(a).rolling(window, min_periods=window // 2).mean().to_numpy() if False else None
    def roll_mean(a, window):
        return pd.Series(a).rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()
    # HAR: 1h, 4h, 1d of FEATURE-COMPLETE bars (a 5-min bar -> 12 / 48 / 288 lookbacks)
    med = len(bars) // 5 if len(bars) else 12
    h1 = roll_mean(rng, 12)
    h4 = roll_mean(rng, 48)
    d1 = roll_mean(rng, 288)
    hour = bars["ts"].dt.hour.to_numpy() + bars["ts"].dt.minute.to_numpy() / 60.0
    dow = bars["ts"].dt.dayofweek.to_numpy(float)
    pos = (bars["close"].to_numpy(float) - lo) / np.where(hi - lo > 0, hi - lo, np.nan)
    vol_roll = roll_mean(vol, 48)
    vol_ratio = vol / np.where(vol_roll > 0, vol_roll, np.nan)
    return pd.DataFrame({
        "rv_1h": h1, "rv_4h": h4, "rv_1d": d1,
        "tod_sin": np.sin(2 * np.pi * hour / 24.0),
        "tod_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow": dow, "close_pos": pos,
        "vol_ratio": vol_ratio,
    }, index=bars.index)


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss for a volatility forecast: y/p - log(y/p) - 1. Scale-free, and the
    standard loss for realised-variance forecasting (positive, 0 at a perfect forecast)."""
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred > 0) & (y_true > 0)
    if not m.any():
        return float("nan")
    r = y_true[m] / y_pred[m]
    return float((r - np.log(r) - 1.0).mean())


def run(symbol: str, *, horizon_bars: int = 12, train_frac: float = 0.5) -> dict:
    """Predict the next ``horizon_bars`` realised range for one instrument."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as e:
        raise RuntimeError("sklearn required") from e

    path = config.PARQUET_DIR / f"{symbol}_M1.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{symbol}: no long bar panel")
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"])
    if "volume" in df.columns:
        df = df.rename(columns={"volume": "vol"})
    df = df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)
    if len(df) < 8000:
        raise RuntimeError(f"{symbol}: only {len(df)} bars")

    # target: realised high-low range over the NEXT horizon_bars, in bp
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    fwd_hi = pd.Series(hi).rolling(horizon_bars).max().shift(-horizon_bars).to_numpy()
    fwd_lo = pd.Series(lo).rolling(horizon_bars).min().shift(-horizon_bars).to_numpy()
    y = (fwd_hi - fwd_lo) / df["close"].to_numpy(float) * 1e4

    X = har_features(df, symbol)
    cut = int(len(df) * train_frac)
    Xtr, ytr = X.iloc[:cut].to_numpy(), y[:cut]
    Xte, yte = X.iloc[cut:].to_numpy(), y[cut:]
    mtr = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr) & (ytr > 0)
    mte = np.isfinite(Xte).all(axis=1) & np.isfinite(yte) & (yte > 0)
    Xtr, ytr, Xte, yte = Xtr[mtr], ytr[mtr], Xte[mte], yte[mte]
    if len(ytr) < 2000 or len(yte) < 500:
        raise RuntimeError(f"{symbol}: too few usable rows")

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=50, random_state=0)
    model.fit(Xtr, ytr)
    pred = np.maximum(model.predict(Xte), 1e-6)
    # baseline 1: unconditional mean of the training target
    base = np.full(len(yte), float(np.mean(ytr)))
    # baseline 2: HAR's own d1 (yesterday's daily range), the textbook benchmark
    r2 = 1.0 - float(((yte - pred) ** 2).sum() / ((yte - base) ** 2).sum())
    try:
        toll = vc.load_cost(symbol).roundtrip_med_bp
    except Exception:
        toll = float("nan")
    return {
        "symbol": symbol,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "y_mean_bp": round(float(yte.mean()), 3),
        "pred_mean_bp": round(float(pred.mean()), 3),
        "qlike_model": round(qlike(yte, pred), 5),
        "qlike_mean": round(qlike(yte, base), 5),
        "r2_oos": round(r2, 4),
        "toll_bp": round(toll, 3) if np.isfinite(toll) else None,
        "opportunity_mean": round(float(pred.mean() / toll), 3) if np.isfinite(toll) and toll > 0 else None,
        "opportunity_p90": round(float(np.percentile(pred, 90) / toll), 3) if np.isfinite(toll) and toll > 0 else None,
    }


def main() -> None:
    syms = [p.stem.replace("_M1", "") for p in sorted(config.PARQUET_DIR.glob("*_M1.parquet"))]
    print("P11: next-session RANGE prediction - sizing input, not an alpha claim\n")
    print(f"{'symbol':>10} {'range bp':>9} {'pred':>8} {'qlike':>9} {'qlikeMn':>9} {'R2':>7} {'toll':>7} {'opp':>6} {'p90':>6}")
    rows = []
    for s in syms:
        try:
            r = run(s)
        except Exception as e:
            continue
        print(f"{r['symbol']:>10} {r['y_mean_bp']:>9.2f} {r['pred_mean_bp']:>8.2f} "
              f"{r['qlike_model']:>9.4f} {r['qlike_mean']:>9.4f} {r['r2_oos']:>7.3f} "
              f"{(r['toll_bp'] if r['toll_bp'] is not None else float('nan')):>7.2f} "
              f"{(r['opportunity_mean'] if r['opportunity_mean'] is not None else float('nan')):>6.2f} "
              f"{(r['opportunity_p90'] if r['opportunity_p90'] is not None else float('nan')):>6.2f}")
        rows.append(r)
    import json
    (config.ARTIFACTS_DIR / "range_model.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote artifacts/range_model.json")


if __name__ == "__main__":
    main()
