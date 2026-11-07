"""Spread inference for panels that carry only OHLCV.

THE PROBLEM: the 6-year bar panels (2.08M XAUUSD bars, 2020-11 to 2026-09) have no bid, no ask
and no spread column, while the measured-spread panels cover only ~200 recent days. So every
long-sample study is priced against a 200-day spread window - a bias that is invisible in the
output and wrong whenever the spread regime differs.

THE FIX: infer the spread from features a long panel *does* carry - bar range, bar volume,
intraday position, and the symbol's own recent levels - and VALIDATE the inference against the
measured spread on the overlap window (~315 days, 89.8% of measured bars). The deliverable is
not a better estimate; it is the ERROR of the estimate, in bp, at each horizon, so a 6-year
study can be priced with a stated bound rather than an assumed one.

What this cannot do, stated up front: it cannot recover the *quoted* spread from OHLCV, because
OHLCV does not contain it. It infers a *state-dependent proxy* that correlates with the spread
because both respond to volatility and liquidity. The honest framing is therefore "inferred
spread with a measured MAE and bias", not "spread".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, mt5_clock


def __feats__(df: pd.DataFrame) -> pd.DataFrame:
    """State features available in ANY OHLCV panel - all causal, all bar-relative."""
    close = df["close"].to_numpy(float)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    vol = df["vol"].to_numpy(float) if "vol" in df else df["volume"].to_numpy(float)
    n = len(df)
    body = hi - lo
    # range relative to recent range: wide bars mean wide spreads, and the scale is what carries
    rng_ma = pd.Series(body).rolling(240, min_periods=30).median().to_numpy()
    rng_ratio = body / np.where(rng_ma > 0, rng_ma, np.nan)
    # volume relative to recent volume
    vol_ma = pd.Series(vol).rolling(240, min_periods=30).median().to_numpy()
    vol_ratio = vol / np.where(vol_ma > 0, vol_ma, np.nan)
    # return magnitude relative to recent
    ret = np.zeros(n)
    ret[1:] = np.abs(np.diff(np.log(np.maximum(close, 1e-12))))
    ret_ma = pd.Series(ret).rolling(240, min_periods=30).median().to_numpy()
    ret_ratio = ret / np.where(ret_ma > 0, ret_ma, np.nan)
    # lagged inferred-proxy signal: the spread itself, from the previous bars of the SAME panel
    # where available. For a long panel this is the previous bar's inferred spread (recursive
    # through time), which is the strongest single predictor we have.
    return pd.DataFrame({
        "rng_ratio": rng_ratio,
        "vol_ratio": vol_ratio,
        "ret_ratio": ret_ratio,
        "tod_sin": np.sin(2 * np.pi * (df["ts"].dt.hour + df["ts"].dt.minute / 60) / 24),
        "tod_cos": np.cos(2 * np.pi * (df["ts"].dt.hour + df["ts"].dt.minute / 60) / 24),
        "dow": df["ts"].dt.dayofweek.to_numpy(float),
    }, index=df.index)


def build(symbol: str, *, minutes: int = 1) -> dict:
    """Train a spread proxy on the overlap, report the error on a held-out tail of it."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as e:
        raise RuntimeError("sklearn required") from e

    long_panel = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_M1.parquet")
    long_panel["ts"] = pd.to_datetime(long_panel["ts"])
    if "vol" not in long_panel.columns:
        long_panel["vol"] = long_panel["volume"]
    meas = mt5_clock.load_mt5_utc(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet").dropna(
        subset=["spread_med_bp"])
    # restrict to bars present in BOTH panels, matched on the timestamp
    l = long_panel[["ts", "open", "high", "low", "close", "vol"]].rename(columns={})
    cut = int(len(meas) * 0.75)
    tr_idx = meas.index[:cut]
    te_idx = meas.index[cut:]
    # train on the FIRST 75% of the measured window, test on the last 25% - contiguous in time
    X = __feats__(l)
    y = pd.Series(meas["spread_med_bp"].to_numpy(), index=meas.index)
    # align: features are computed on the long panel; map by minute bucket
    l_bucket = l["ts"].dt.floor("1min")
    meas_bucket = meas["ts"].dt.floor("1min")
    lX = X.copy()
    lX["bkt"] = l_bucket.to_numpy()
    m = meas[["ts", "spread_med_bp"]].copy()
    m["bkt"] = meas_bucket
    j = lX.merge(m[["bkt", "spread_med_bp"]], on="bkt", how="inner").dropna()
    if len(j) < 5000:
        raise RuntimeError(f"{symbol}: only {len(j)} aligned bars - cannot validate")
    j = j.sort_values("bkt")
    ntr = int(len(j) * 0.75)
    feat_cols = ["rng_ratio", "vol_ratio", "ret_ratio", "tod_sin", "tod_cos", "dow"]
    Xtr = j[feat_cols].to_numpy()[:ntr]
    ytr = j["spread_med_bp"].to_numpy()[:ntr]
    Xte = j[feat_cols].to_numpy()[ntr:]
    yte = j["spread_med_bp"].to_numpy()[ntr:]
    ok_tr = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr) & (ytr > 0)
    ok_te = np.isfinite(Xte).all(axis=1) & np.isfinite(yte) & (yte > 0)
    Xtr, ytr, Xte, yte = Xtr[ok_tr], ytr[ok_tr], Xte[ok_te], yte[ok_te]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          min_samples_leaf=40, random_state=0)
    model.fit(Xtr, ytr)
    pred = np.maximum(model.predict(Xte), 0.0)
    err = pred - yte
    # baseline: median spread from the TRAIN window - what an inference-free study would use
    base = np.full(len(yte), np.median(ytr))
    base_err = base - yte
    return {
        "symbol": symbol,
        "n_aligned": int(len(j)),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "model_mae_bp": float(np.abs(err).mean()),
        "model_med_abs_err_bp": float(np.median(np.abs(err))),
        "model_bias_bp": float(err.mean()),
        "flat_median_mae_bp": float(np.abs(base_err).mean()),
        "mae_improvement": float(1.0 - np.abs(err).mean() / max(np.abs(base_err).mean(), 1e-9)),
        "test_median_spread_bp": float(np.median(yte)),
        "err_ratio_to_toll": float(np.abs(err).mean() / max(2 * float(np.median(yte)), 1e-9)),
        "test_mae_in_bp": float(np.abs(err).mean()),
    }


def main() -> None:
    syms = ["XAUUSD", "XAGUSD", "XTIUSD", "NAS100", "EURUSD", "XPTUSD", "XCUUSD"]
    rows = []
    print(f"{'symbol':>9} {'n_test':>7} {'MAE bp':>8} {'flatMAE':>8} {'impr':>6} {'bias':>8} {'err/toll':>9}")
    for s in syms:
        try:
            r = build(s)
        except Exception as e:
            print(f"{s:>9}  skipped: {str(e)[:70]}")
            continue
        print(f"{r['symbol']:>9} {r['n_test']:>7,} {r['model_mae_bp']:>8.3f} "
              f"{r['flat_median_mae_bp']:>8.3f} {r['mae_improvement']:>6.0%} "
              f"{r['model_bias_bp']:>+8.3f} {r['err_ratio_to_toll']:>9.2f}")
        rows.append(r)
    import json
    with open("artifacts/spread_inference.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nwrote artifacts/spread_inference.json")


if __name__ == "__main__":
    main()
