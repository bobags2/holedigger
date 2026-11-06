"""Expanded event sweep, median-based.

The mean-based version of this sweep produced a +26.5bp family max that was a
tail-driven artifact (gold's control mean was -22.4bp against a median of -0.4bp:
a few -247bp returns dragged it). This version therefore reports the MEDIAN as the
primary statistic everywhere — real and null use the same estimator.

Significance comes from a label-permutation null: pool each candidate's event and
control returns, shuffle the labels, recompute the median incremental; the family
max of that under permutation is the bar.

Spread is measured per event from the bars themselves (median of the per-bar
spread_med_bp across the window), not borrowed from a session average.

``--swap-hour`` / ``--exclude-swap-crossers`` drop events whose window crosses the
daily swap rollover, which otherwise contaminates longer horizons.

Usage::

    python -m scripts.event_sweep --symbols XAUUSD XAGUSD XTIUSD --currencies USD --horizons 30 60
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
from scripts.event_test import (  # noqa: E402
    control_times,
    event_windows,
    load_events,
    window_return,
)

TRIM = 0.05


def load_series(symbol: str, timeframe: str = "5min") -> pd.DataFrame:
    df = mt5_clock.load_mt5_utc(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    g = df.set_index("ts").resample(timeframe)
    out = pd.DataFrame({
        "close": g["close"].last(),
        "spread": g["spread_med_bp"].median(),
    }).dropna()
    return out


def trimmed_mean(a: np.ndarray) -> float:
    k = int(len(a) * TRIM)
    return float(np.sort(a)[k : len(a) - k].mean()) if k else float(a.mean())


def collect(symbol, currency, impact, horizon, events, series, exclude_swap_crossers, swap_hour,
            min_events=20, min_controls=5):
    """One candidate's event and control forward returns, plus per-event spread.

    ``min_controls`` is 5, not 20, because the control sample is legitimately small once
    each quiet day is used at most once: the distinct-day rule that fixes the reused-control
    defect caps it at the number of quiet days in the panel, which is what the sample size
    really is. The artifact reports the actual n_events/n_controls so a reader can see how
    thin it is rather than trusting a count that was inflated by reuse.
    """
    ev = event_windows(events, currency, impact)
    if not len(ev):
        return None
    close, spread = series["close"], series["spread"]
    swap_cut = pd.Timedelta(hours=swap_hour)
    day_ns = np.int64(86_400 * 10**9)
    ev_r, c_r, ev_sp = [], [], []
    ctrl = control_times(ev.index, close.index, events, currency)
    for t in ev.index:
        if exclude_swap_crossers:
            tod = pd.Timedelta(nanoseconds=t.value % day_ns)
            if tod >= swap_cut or tod + pd.Timedelta(minutes=horizon) > swap_cut + pd.Timedelta(days=1):
                continue
        r = window_return(close, t, horizon)
        if r is None:
            continue
        win = spread[(spread.index >= t) & (spread.index <= t + pd.Timedelta(minutes=horizon))]
        ev_r.append(r)
        ev_sp.append(float(win.median()) if len(win) else np.nan)
    for t in ctrl:
        r = window_return(close, t, horizon)
        if r is not None:
            c_r.append(r)
    if len(ev_r) < min_events or len(c_r) < min_controls:
        return None
    return (np.asarray(ev_r, float), np.asarray(c_r, float), np.asarray(ev_sp, float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD", "XAGUSD", "XTIUSD", "XALUSD", "EURUSD"])
    ap.add_argument("--currencies", nargs="+", default=["USD", "EUR", "GBP"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60])
    ap.add_argument("--impact", default="High importance")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--swap-hour", type=int, default=21)
    ap.add_argument("--exclude-swap-crossers", action="store_true")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "event_sweep.json")
    args = ap.parse_args()

    events = load_events()
    combos = []
    for s in args.symbols:
        series = load_series(s, "5min")
        for ccy in args.currencies:
            for h in args.horizons:
                got = collect(s, ccy, args.impact, h, events, series,
                              args.exclude_swap_crossers, args.swap_hour)
                if got is not None:
                    combos.append({"symbol": s, "currency": ccy, "horizon": h,
                                   "ev_r": got[0], "c_r": got[1], "ev_sp": got[2]})
    if not combos:
        raise SystemExit("no usable combinations")
    print(f"family: {len(combos)} candidates | estimator: median | permutations: {args.perms}"
          + (" | swap-crossers excluded" if args.exclude_swap_crossers else ""))

    def med_inc(c):
        return float(np.median(c["ev_r"]) - np.median(c["c_r"]))

    def median_se(a: np.ndarray) -> float:
        """Bootstrap se of a median (1000 draws, fixed seed) - the scale the raw bp lacked."""
        if len(a) < 5:
            return float("nan")
        r = np.random.default_rng(12345)
        idx = r.integers(0, len(a), size=(1000, len(a)))
        return float(np.median(a[idx], axis=1).std(ddof=1))

    def studentized(c):
        """Median incremental in units of its own bootstrap se.

        A raw-bp median cannot rank across instruments - natgas at a 24bp spread and gold at
        0.2bp are not on one scale, so the family max used to be "noisiest instrument" rather
        than "most statistically extreme" (measured: the old table's top rows were oil and
        natgas, both cost-dead). This statistic puts every candidate on one scale.
        """
        m = med_inc(c)
        se = np.sqrt(median_se(c["ev_r"]) ** 2 + median_se(c["c_r"]) ** 2)
        return float(m / se) if se and np.isfinite(se) and se > 0 else float("nan")

    real = {i: med_inc(c) for i, c in enumerate(combos)}
    real_max = max(abs(v) for v in real.values())

    real_s = {i: studentized(c) for i, c in enumerate(combos)}
    real_max_s = max(abs(v) for v in real_s.values() if np.isfinite(v))
    # per-candidate bootstrap se of the event and control medians: the scale the raw-bp
    # statistic lacked, and the divisor the null worlds use
    real_se = np.array([
        np.sqrt(median_se(c["ev_r"]) ** 2 + median_se(c["c_r"]) ** 2) for c in combos
    ])
    null_max = np.empty(args.perms)
    null_max_s = np.empty(args.perms)
    rng = np.random.default_rng(config.SEED)
    for p in range(args.perms):
        mx = 0.0
        mxs = 0.0
        for i, c in enumerate(combos):
            pool = np.concatenate([c["ev_r"], c["c_r"]])
            labels = np.array([1] * len(c["ev_r"]) + [0] * len(c["c_r"]))
            rng.shuffle(labels)
            a, b = pool[labels == 1], pool[labels == 0]
            d = abs(float(np.median(a) - np.median(b)))
            mx = max(mx, d)
            se = real_se[i]
            if np.isfinite(se) and se > 0:
                mxs = max(mxs, d / se)
        null_max[p] = mx
        null_max_s[p] = mxs

    rows = []
    for i, c in enumerate(combos):
        ev_med = float(np.median(c["ev_r"]))
        sp = float(np.nanmedian(c["ev_sp"]))
        rows.append({
            "symbol": c["symbol"], "currency": c["currency"], "horizon": c["horizon"],
            "n_events": int(len(c["ev_r"])), "n_controls": int(len(c["c_r"])),
            "event_median_bp": round(ev_med, 3),
            "control_median_bp": round(float(np.median(c["c_r"])), 3),
            "median_incremental_bp": round(real[i], 3),
            "studentized_incremental": round(real_s[i], 4) if np.isfinite(real_s[i]) else None,
            "trimmed_incremental_bp": round(trimmed_mean(c["ev_r"]) - trimmed_mean(c["c_r"]), 3),
            "spread_median_bp": round(sp, 3) if np.isfinite(sp) else None,
            "net_of_spread_bp": round(ev_med - sp, 3) if np.isfinite(sp) else None,
            "null_percentile": round(float((null_max < abs(real[i])).mean()), 4),
            "studentized_null_percentile": round(float((null_max_s < abs(real_s[i])).mean()), 4),
        })
    rows.sort(key=lambda r: -abs(r["studentized_incremental"] or 0.0))

    result = {
        "config": {"symbols": args.symbols, "currencies": args.currencies, "statistic": "median + studentized",
                   "horizons": args.horizons, "impact": args.impact,
                   "perms": args.perms, "estimator": "median",
                   "swap_hour": args.swap_hour, "exclude_swap_crossers": args.exclude_swap_crossers},
        "family": {
            "candidates": len(combos),
            "real_max_abs_incremental_bp": round(real_max, 3),
            "null_max_mean": round(float(null_max.mean()), 3),
            "null_max_p95": round(float(np.percentile(null_max, 95)), 3),
            "real_percentile": round(float((null_max < real_max).mean()), 4),
            "family_studentized": {
                "real_max_abs": round(real_max_s, 3),
                "null_mean": round(float(np.nanmean(null_max_s)), 3),
                "null_p95": round(float(np.nanpercentile(null_max_s, 95)), 3),
                "percentile": round(float((np.nan_to_num(null_max_s, nan=-np.inf) < real_max_s).mean()), 4),
            },
        },
        "candidates": rows,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fam = result["family"]
    print(f"family max |median incr| = {fam['real_max_abs_incremental_bp']} bp | "
          f"null mean {fam['null_max_mean']} p95 {fam['null_max_p95']} | "
          f"percentile {100 * fam['real_percentile']:.1f}%\n")
    print(f"{'symbol':>9} {'ccy':>4} {'h':>3} {'n':>4} {'evmed':>7} {'ctrlmed':>8} "
          f"{'incr':>7} {'sprd':>6} {'net':>7} {'null pct':>9}")
    for r in rows[:14]:
        net = f"{r['net_of_spread_bp']:.2f}" if r["net_of_spread_bp"] is not None else "n/a"
        print(f"{r['symbol']:>9} {r['currency']:>4} {r['horizon']:>3} {r['n_events']:>4} "
              f"{r['event_median_bp']:>7.2f} {r['control_median_bp']:>8.2f} "
              f"{r['median_incremental_bp']:>7.2f} {r['spread_median_bp']:>6.2f} {net:>7} "
              f"{r['null_percentile']:>9.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
