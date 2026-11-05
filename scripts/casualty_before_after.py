"""Before/after for the published MT5 hour-of-day windows: stored clock vs corrected UTC.

Recomputes a session window's statistics straight from the MT5 parquet, twice:

  BEFORE  the stamps as stored on disk (``terminal - 3h``, the fixed-offset ingest), which
          is what artifacts/window_sweep.json reports;
  AFTER   the same labelled window after ``mt5_clock.stored_to_utc``, i.e. the hour label
          meaning what it says.

The statistic is the sweep's own (entry = close of the first 5-min-bar-equivalent at/after
the entry hour, exit = close of the last bar at/before entry + hold, same UTC day, exit
capped at 20:00), reproduced standalone here rather than imported so this tool keeps
working while src/window_sweep.py is edited, and so the BEFORE column can be checked
against the published artifact: it reproduces every row checked so far to the last digit.
The artifact cross-check at the bottom says which column the artifact row agrees with —
"stored" means the artifact predates the clock fix, "corrected" means it was regenerated
on the repaired clock.

How to read it: the 'moved' column is the number of sampled days whose entry bar changed
when the clock was repaired — the size of the mistake, ~1/3 of any 350-day window's samples
(74/242 for these windows, all in US standard time). Cost is (spread[entry] +
spread[exit]) / 2 per trade, so the last column says whether the conclusion is still
cost-decided.

Usage::

    python -m scripts.casualty_before_after                                  # copper + lead, hold 2
    python -m scripts.casualty_before_after --symbols XCUUSD --hours 0 1 2 3
    python -m scripts.casualty_before_after --casualties "XCUUSD:3 XCUUSD:6 XPBUSD:9"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, mt5_clock  # noqa: E402

MIN_WIND_BARS = 2


def load(symbol: str, corrected: bool) -> dict:
    """Bar arrays in the sweep's shape (r[0]=0, r[j]=log(close[j]/close[j-1]))."""
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    if corrected:
        df = df.assign(ts=mt5_clock.stored_to_utc(df["ts"]))
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=float)
    ns = pd.DatetimeIndex(df["ts"]).to_numpy().astype("datetime64[ns]").astype("int64")
    r = np.empty(len(close), dtype=float)
    r[0] = 0.0
    r[1:] = np.diff(np.log(close))
    return {"r": r, "ns": ns, "day": ns // (86_400 * 10**9), "sp": df["spread_med_bp"].to_numpy(dtype=float)}


def segments(s: dict, entry_hour: int, hold_hours: int) -> tuple:
    """Per UTC day: entry bar = first at/after the hour, exit = last at/before hour+hold."""
    days = np.unique(s["day"])
    a = np.empty(len(days), dtype=np.int64)
    b = np.empty(len(days), dtype=np.int64)
    starts = np.searchsorted(s["day"], days, side="left")
    ends = np.searchsorted(s["day"], days, side="right")
    entry_cut, exit_cut = entry_hour * 3600 * 10**9, (entry_hour + hold_hours) * 3600 * 10**9
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        times = s["ns"][lo:hi] - days[i] * 86_400 * 10**9
        a[i] = lo + np.searchsorted(times, entry_cut, side="left")
        b[i] = lo + np.searchsorted(times, exit_cut, side="right") - 1
    valid = (b > a) & (b - a >= MIN_WIND_BARS)
    return days, np.clip(a, 0, len(s["ns"]) - 1), np.clip(b, 0, len(s["ns"]) - 1), valid


def stats(s: dict, days: np.ndarray, a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> dict:
    S = np.cumsum(s["r"])
    pnl = np.where(valid, S[b] - S[a], np.nan)
    n = int((~np.isnan(pnl)).sum())
    if n < 5:
        return {"edge": np.nan, "t": np.nan, "n": n, "cost": np.nan, "net": np.nan,
                "e1": np.nan, "e2": np.nan, "t1": np.nan, "t2": np.nan, "entries": {}}
    mean = float(np.nansum(pnl) / n)
    var = float(np.nansum((pnl - mean) ** 2) / (n - 1))
    t = mean / (np.sqrt(max(var, 0.0)) / np.sqrt(n))
    e, x = a[valid], b[valid]
    gross = (S[x] - S[e]) * 1e4
    cost = (s["sp"][e] + s["sp"][x]) / 2.0
    net = np.sign(gross.mean()) * gross - cost
    halves = []
    cut = np.median(days)
    for mask in (days <= cut, days > cut):
        sub = pnl[mask]
        sub = sub[~np.isnan(sub)]
        halves.append((float(sub.mean() * 1e4), float(sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))))))
    return {"edge": mean * 1e4, "t": float(t), "n": n, "cost": float(cost.mean()),
            "net": float(net.mean()), "e1": halves[0][0], "e2": halves[1][0],
            "t1": halves[0][1], "t2": halves[1][1],
            "entries": {int(day): int(bar) for day, bar in zip(days[valid], e)}}


def artifact_row(symbol: str, hour: int, hold: int, path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        art = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    for row in art.get("candidates", []):
        if row.get("symbol") == symbol and row.get("entry_hour_utc") == hour and row.get("hold_hours") == hold:
            return row
    return None


def parse_casualties(spec: list[str]) -> list[tuple]:
    out = []
    for item in spec:
        symbol, hour = item.split(":")
        out.append((symbol, int(hour)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["XCUUSD", "XPBUSD"])
    ap.add_argument("--hold", type=int, default=2)
    ap.add_argument("--hours", nargs="+", type=int, default=list(range(20)))
    ap.add_argument("--casualties", nargs="+", default=["XCUUSD:3", "XPBUSD:6"],
                    help="SYMBOL:HOUR pairs to detail (default: the two published casualties)")
    ap.add_argument("--artifact", type=Path, default=config.ARTIFACTS_DIR / "window_sweep.json")
    args = ap.parse_args()

    d = args.hold
    cache = {sym: (load(sym, False), load(sym, True)) for sym in set(args.symbols) | {c[0] for c in parse_casualties(args.casualties)}}

    for sym in args.symbols:
        before, after = cache[sym]
        print(f"\n================ {sym}, hold {d}h ================")
        print(f"{'h':>2} | {'BEFORE edge':>11} {'t':>6} {'n':>4} {'cost':>6} {'t1':>6} {'t2':>6} "
              f"| {'AFTER edge':>11} {'t':>6} {'n':>4} {'cost':>6} {'t1':>6} {'t2':>6} | moved")
        for h in args.hours:
            sb = stats(before, *segments(before, h, d))
            sa = stats(after, *segments(after, h, d))
            moved = sum(1 for day, bar in sb["entries"].items()
                        if day in sa["entries"] and sa["entries"][day] != bar)
            print(f"{h:>2} | {sb['edge']:>11.3f} {sb['t']:>6.2f} {sb['n']:>4} {sb['cost']:>6.2f} "
                  f"{sb['t1']:>6.2f} {sb['t2']:>6.2f} | {sa['edge']:>11.3f} {sa['t']:>6.2f} {sa['n']:>4} "
                  f"{sa['cost']:>6.2f} {sa['t1']:>6.2f} {sa['t2']:>6.2f} | {moved:>5}")

    print("\n================ casualties in detail ================")
    for sym, h in parse_casualties(args.casualties):
        before, after = cache[sym]
        sb = stats(before, *segments(before, h, d))
        sa = stats(after, *segments(after, h, d))
        days, _a, _b, valid = segments(before, h, d)
        winter = int((pd.to_datetime(days[valid], unit="D") <= pd.Timestamp("2026-03-08")).sum())
        print(f"\n{sym} {h:02d}:00 +{d}h")
        print(f"  BEFORE (stored clock): edge {sb['edge']:+.3f}bp  t {sb['t']:+.2f}  n {sb['n']}  "
              f"cost {sb['cost']:.2f}bp  net {sb['net']:+.2f}bp  halves e {sb['e1']:+.2f}/{sb['e2']:+.2f}")
        print(f"  AFTER  (true UTC)    : edge {sa['edge']:+.3f}bp  t {sa['t']:+.2f}  n {sa['n']}  "
              f"cost {sa['cost']:.2f}bp  net {sa['net']:+.2f}bp  halves e {sa['e1']:+.2f}/{sa['e2']:+.2f}")
        print(f"  {winter}/{sb['n']} sampled days were in US standard time (mislabelled by 1h before the fix)")
        row = artifact_row(sym, h, d, args.artifact)
        if row is None:
            print(f"  artifact: no row for ({sym}, {h}, {d}) in {args.artifact.name} - nothing to cross-check")
            continue
        which = "stored" if np.isclose(row["edge_bp"], sb["edge"], atol=1e-9) else (
            "corrected" if np.isclose(row["edge_bp"], sa["edge"], atol=1e-9) else "neither")
        print(f"  artifact row ({args.artifact.name}): edge {row['edge_bp']:+.3f}bp  t {row['t']:+.2f}  "
              f"n {row['n']}  cost {(row['avg_half_spread_cost_bp'] or float('nan')):.2f}bp  "
              f"net {(row['net_spread_edge_bp'] or float('nan')):+.2f}bp  -> matches the {which} column")


if __name__ == "__main__":
    main()
