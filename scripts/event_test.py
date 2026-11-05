"""Event-driven study: does price behave differently around calendar events,
incremental to the normal time-of-day pattern?

Null design (the load-bearing part): each event is compared against control
windows at the SAME clock time on days with no event for that currency. That
null has the intraday drift baked in, so anything left over is event-specific.

Usage::

    python -m scripts.event_test --symbol EURUSD --impact High --horizon 30
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

MIN_MINUTES = 5   # spacing between two events to call them independent
TOLERANCE_MIN = 2  # control window may sit this far off the event clock time


def load_events() -> pd.DataFrame:
    p = config.DATA_DIR / "events.parquet"
    if not p.exists():
        raise SystemExit("no events.parquet; run `python -m src.events --from ... --to ...` first")
    return pd.read_parquet(p)


def load_mid(symbol: str, timeframe: str = "5min") -> pd.DataFrame:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    m = df.set_index("ts")["close"].resample(timeframe).last().dropna()
    return m.to_frame("close")


def event_windows(events: pd.DataFrame, currency: str, impact: str) -> pd.DataFrame:
    ev = events[(events["currency"] == currency) & (events["importance"] == impact)].copy()
    ev = ev.sort_values("ts")
    keep, last = [], None
    for t in ev["ts"]:
        if last is None or (t - last) > pd.Timedelta(0):
            keep.append(t)
            last = t
    ev = ev[ev["ts"].isin(keep)]
    return ev.drop_duplicates("ts", keep="first").set_index("ts").sort_index()


def control_times(event_times: pd.DatetimeIndex, mid_index: pd.DatetimeIndex,
                  events: pd.DataFrame, currency: str) -> list[pd.Timestamp]:
    """Same clock time, non-event day (vectorized: one candidate per event)."""
    event_days = set(events[events["currency"] == currency]["ts"].dt.floor("D"))
    mid_ns = mid_index.asi8                      # int64 ns since epoch
    day_ns = np.array([n - (n % 86_400_000_000_000) for n in mid_ns], dtype="int64")
    tod_minutes = (mid_ns % 86_400_000_000_000) // 60_000_000_000
    out = []
    for t in event_times:
        n = t.value
        same_tod = tod_minutes == ((n % 86_400_000_000_000) // 60_000_000_000)
        allowed = ~np.isin(day_ns, np.array(list(event_days), dtype="datetime64[ns]").astype(np.int64))
        cands = mid_ns[same_tod & allowed & (mid_ns < n)]
        if len(cands):
            out.append(pd.Timestamp(cands.max(), unit="ns"))
    return out


def window_return(close: pd.Series, t: pd.Timestamp, horizon_min: int) -> float | None:
    end = t + pd.Timedelta(minutes=horizon_min)
    s = close[(close.index >= t) & (close.index <= end)]
    if len(s) < 3 or s.index[-1] < end - pd.Timedelta(minutes=TOLERANCE_MIN):
        return None
    return float(np.log(s.iloc[-1] / s.iloc[0]) * 1e4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--impact", default="High importance")
    ap.add_argument("--horizon", type=int, default=30, help="measurement window minutes")
    ap.add_argument("--timeframe", default="5min")
    args = ap.parse_args()

    events = load_events()
    ev = event_windows(events, args.currency, args.impact)
    mid = load_mid(args.symbol, args.timeframe)["close"]
    mid_index = mid.index

    ev_r = [r for t in ev.index if (r := window_return(mid, t, args.horizon)) is not None]
    ctrl_ts = control_times(ev.index, mid_index, events, args.currency)
    ctrl_r = [r for t in ctrl_ts if (r := window_return(mid, t, args.horizon)) is not None]

    if len(ev_r) < 8:
        raise SystemExit(f"only {len(ev_r)} usable event windows; widen --impact or the calendar range")

    def tstat(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), float(x.mean())

    e_t, e_mean = tstat(ev_r)
    c_t, c_mean = tstat(ctrl_r)
    # event incremental: hypothetical long-at-event win, vs same clock on quiet days
    inc = e_mean - c_mean

    print(f"{args.symbol} | {args.impact} {args.currency} events | {args.horizon}min window")
    print(f"  event windows : n={len(ev_r):4d}  mean={e_mean:+.3f} bp  t={e_t:+.2f}")
    print(f"  clock controls: n={len(ctrl_r):4d}  mean={c_mean:+.3f} bp  t={c_t:+.2f}")
    print(f"  incremental   : {inc:+.3f} bp (positive = event-time drift above the normal hour)")
    print(f"  spread check  : see mt5_audit median at this hour before believing any of it")


if __name__ == "__main__":
    main()
