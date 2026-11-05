"""Dislocation study: after a large move, what does the forward distribution look like?

Trigger = the bar return exceeding a quantile of the same symbol's |return|
distribution over the whole sample (one volatility-normalized threshold per symbol).
Forward window = H minutes from the trigger bar close.

The question, stated the same shape as the event study: is the forward return after
a dislocation different from the forward return at the same clock time on ordinary
days? The null is therefore time-of-day-matched controls, and the statistic is the
MEDIAN (forward distributions are fat-tailed; mean-based edge estimates on tails are
how the event sweep produced a fake +26bp).

Direction: the trade is counter-trend (fade the dislocation), so the reported edge
is the negative of the forward return.

Usage::

    python -m scripts.dislocation_test --symbol XAUUSD --quantile 0.99 --horizon 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src import mt5_clock  # noqa: E402
from scripts.event_test import window_return  # noqa: E402

TRIM = 0.05
MIN_GAP_MIN = 10


def load_series(symbol: str, timeframe: str = "5min") -> pd.DataFrame:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    df["ts"] = mt5_clock.stored_to_utc(df["ts"])  # parquet clock -> true UTC
    g = df.set_index("ts").resample(timeframe)
    out = pd.DataFrame({
        "close": g["close"].last(),
        "spread": g["spread_med_bp"].median(),
    }).dropna()
    return out


def fwd_return(close: pd.Series, t: pd.Timestamp, horizon_min: int) -> float | None:
    return window_return(close, t, horizon_min)


def trigger_times(close: pd.Series, quantile: float, horizon_min: int) -> list[pd.Timestamp]:
    r = np.log(close).diff() * 1e4
    thr = float(np.quantile(np.abs(r.dropna()), quantile))
    hits = (r.abs() > thr)
    close_index = close.index
    out, last = [], None
    for i in np.flatnonzero(hits.to_numpy()):
        t = close_index[i]
        if last is not None and t - last < pd.Timedelta(minutes=MIN_GAP_MIN):
            continue
        if i + 1 >= len(close_index):
            continue
        if fwd_return(close, t, horizon_min) is None:
            continue
        out.append(t)
        last = t
    return out


def control_times(trigger_times: list[pd.Timestamp], mid_index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Same clock minute, ordinary (non-trigger) day (vectorized)."""
    forbid_days = np.unique([t.floor("D").value for t in trigger_times])
    mid_ns = mid_index.asi8
    day_ns = mid_ns - (mid_ns % 86_400_000_000_000)
    tod_ns = mid_ns % 86_400_000_000_000
    out = []
    for t in trigger_times:
        n = t.value
        cands = mid_ns[(tod_ns == n % 86_400_000_000_000) & (~np.isin(day_ns, forbid_days)) & (mid_ns < n)]
        if len(cands):
            out.append(pd.Timestamp(cands.max(), unit="ns"))
    return out


def collect(symbol: str, quantile: float, horizon_min: int, timeframe: str = "5min") -> dict | None:
    series = load_series(symbol, timeframe)
    close, spread = series["close"], series["spread"]
    tris = trigger_times(close, quantile, horizon_min)
    cts = control_times(tris, close.index)
    ev_r, c_r, ev_sp = [], [], []
    for t in tris:
        r = fwd_return(close, t, horizon_min)
        if r is None:
            continue
        win = spread[(spread.index >= t) & (spread.index <= t + pd.Timedelta(minutes=horizon_min))]
        ev_r.append(-r)  # fade: counter-trend
        ev_sp.append(float(win.median()) if len(win) else np.nan)
    for t in cts:
        r = fwd_return(close, t, horizon_min)
        if r is not None:
            c_r.append(-r)
    if len(ev_r) < 15 or len(c_r) < 15:
        return None
    return {
        "symbol": symbol, "quantile": quantile, "horizon": horizon_min,
        "ev_r": np.asarray(ev_r, float), "c_r": np.asarray(c_r, float),
        "ev_sp": np.asarray(ev_sp, float),
    }


def median_incremental(c: dict) -> float:
    return float(np.median(c["ev_r"]) - np.median(c["c_r"]))


def trimmed_mean(a: np.ndarray) -> float:
    k = int(len(a) * TRIM)
    return float(np.sort(a)[k : len(a) - k].mean()) if k else float(a.mean())


def describe(a: np.ndarray) -> dict:
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {
        "n": int(len(a)), "mean": round(float(a.mean()), 3), "median": round(float(q[2]), 3),
        "p10": round(float(q[0]), 3), "p25": round(float(q[1]), 3),
        "p75": round(float(q[3]), 3), "p90": round(float(q[4]), 3),
        "hit_rate": round(float((a > 0).mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD", "XCUUSD"])
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "dislocation.json")
    args = ap.parse_args()

    combos = []
    for s in args.symbols:
        for h in args.horizons:
            c = collect(s, args.quantile, h)
            if c is not None:
                combos.append(c)
    if not combos:
        raise SystemExit("no usable combinations")
    print(f"family: {len(combos)} candidates | quantile {args.quantile} | permutations {args.perms}")

    real = {i: median_incremental(c) for i, c in enumerate(combos)}
    real_max = max(abs(v) for v in real.values())

    rng = np.random.default_rng(config.SEED)
    null_max = np.empty(args.perms)
    for p in range(args.perms):
        mx = 0.0
        for c in combos:
            pool = np.concatenate([c["ev_r"], c["c_r"]])
            labels = np.array([1] * len(c["ev_r"]) + [0] * len(c["c_r"]))
            rng.shuffle(labels)
            mx = max(mx, abs(float(np.median(pool[labels == 1]) - np.median(pool[labels == 0]))))
        null_max[p] = mx

    rows = []
    for i, c in enumerate(combos):
        sp = float(np.nanmedian(c["ev_sp"]))
        rows.append({
            "symbol": c["symbol"], "quantile": c["quantile"], "horizon": c["horizon"],
            "event_median_bp": float(np.median(c["ev_r"])),
            "control_median_bp": float(np.median(c["c_r"])),
            "median_incremental_bp": round(real[i], 3),
            "trimmed_incremental_bp": round(trimmed_mean(c["ev_r"]) - trimmed_mean(c["c_r"]), 3),
            "spread_median_bp": round(sp, 3) if np.isfinite(sp) else None,
            "net_of_spread_bp": round(float(np.median(c["ev_r"])) - sp, 3) if np.isfinite(sp) else None,
            "null_percentile": round(float((null_max < abs(real[i])).mean()), 4),
            "event_dist": describe(c["ev_r"]), "control_dist": describe(c["c_r"]),
        })
    rows.sort(key=lambda r: -abs(r["median_incremental_bp"]))

    result = {
        "config": {"symbols": args.symbols, "quantile": args.quantile, "horizons": args.horizons,
                   "perms": args.perms, "direction": "counter-trend (fade)", "estimator": "median",
                   "min_gap_min": MIN_GAP_MIN},
        "family": {
            "candidates": len(combos),
            "real_max_abs_incremental_bp": round(real_max, 3),
            "null_max_mean": round(float(null_max.mean()), 3),
            "null_max_p95": round(float(np.percentile(null_max, 95)), 3),
            "real_percentile": round(float((null_max < real_max).mean()), 4),
        },
        "candidates": rows,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fam = result["family"]
    print(f"family max |median incr| = {fam['real_max_abs_incremental_bp']} bp | "
          f"null mean {fam['null_max_mean']} p95 {fam['null_max_p95']} | "
          f"percentile {100 * fam['real_percentile']:.1f}%\n")
    for r in rows:
        net = f"{r['net_of_spread_bp']:.2f}" if r["net_of_spread_bp"] is not None else "n/a"
        print(f"{r['symbol']:>7} q{r['quantile']} h{r['horizon']:>3}: event med {r['event_median_bp']:+.2f} | "
              f"control med {r['control_median_bp']:+.2f} | incr {r['median_incremental_bp']:+.2f} | "
              f"spread {r['spread_median_bp']:.2f} | net {net} | null pct {r['null_percentile']:.3f}")
        e = r["event_dist"]
        print(f"         forward dist: p10 {e['p10']:+.1f} p25 {e['p25']:+.1f} med {e['median']:+.1f} "
              f"p75 {e['p75']:+.1f} p90 {e['p90']:+.1f} | hit {e['hit_rate']:.0%} | n={e['n']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
