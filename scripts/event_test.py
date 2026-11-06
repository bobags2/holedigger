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
from src import mt5_clock  # noqa: E402

MIN_MINUTES = 5   # spacing between two events to call them independent
TOLERANCE_MIN = 2  # control window may sit this far off the event clock time


def load_events() -> pd.DataFrame:
    p = config.DATA_DIR / "events.parquet"
    if not p.exists():
        raise SystemExit("no events.parquet; run `python -m src.events --from ... --to ...` first")
    return pd.read_parquet(p)


def load_mid(symbol: str, timeframe: str = "5min") -> pd.DataFrame:
    df = mt5_clock.load_mt5_utc(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    m = df.set_index("ts")["close"].resample(timeframe).last().dropna()
    return m.to_frame("close")


def event_windows(events: pd.DataFrame, currency: str, impact: str) -> pd.DataFrame:
    """Events for one currency/impact, at least ``MIN_MINUTES`` apart.

    Two events inside the same window are not two observations: the second adds nothing
    and quietly halves the spacing the design assumes. Dropping events closer than
    ``MIN_MINUTES`` keeps every window one event's, so the count means what it says.
    """
    ev = events[(events["currency"] == currency) & (events["importance"] == impact)].copy()
    ev = ev.sort_values("ts")
    keep, last = [], None
    for t in ev["ts"]:
        if last is None or (t - last) >= pd.Timedelta(minutes=MIN_MINUTES):
            keep.append(t)
            last = t
    ev = ev[ev["ts"].isin(keep)]
    return ev.drop_duplicates("ts", keep="first").set_index("ts").sort_index()


def control_times(event_times: pd.DatetimeIndex, mid_index: pd.DatetimeIndex,
                  events: pd.DataFrame, currency: str) -> list[pd.Timestamp]:
    """Same clock time, non-event day, ONE control per distinct quiet day.

    Taking the most recent quiet bar at each event's clock time reuses the same quiet day
    for every event in its neighbourhood: measured, 284 event windows drew their controls
    from only 9 distinct days, one of them 72 times. The control mean and its t-stat then
    describe those 9 days, not 284 observations, so the control baseline the incremental is
    measured against is far tighter than it looks - the published "n=284, t=-5.13" was
    really n=9. So: walk events in time order, and never reuse a quiet day that has already
    served as a control. When the supply of quiet days is exhausted the remaining events
    simply get no control, and the sample is honestly smaller.
    """
    event_days = set(events[events["currency"] == currency]["ts"].dt.floor("D"))
    mid_ns = mid_index.asi8                      # int64 ns since epoch
    day_ns = mid_ns - (mid_ns % 86_400_000_000_000)
    tod_minutes = (mid_ns % 86_400_000_000_000) // 60_000_000_000
    event_day_ns = np.array(sorted(d.value for d in event_days), dtype="int64")
    allowed_day = ~np.isin(day_ns, event_day_ns)
    out, used_days = [], set()
    for t in event_times:
        n = t.value
        same_tod = tod_minutes == ((n % 86_400_000_000_000) // 60_000_000_000)
        cands = mid_ns[same_tod & allowed_day & (mid_ns < n)]
        if not len(cands):
            continue
        best = int(cands.max())                  # most recent quiet bar at this clock time
        d = best - (best % 86_400_000_000_000)
        if d in used_days or d == (n - (n % 86_400_000_000_000)):
            continue                             # this day is spent (or IS an event day)
        used_days.add(d)
        out.append(pd.Timestamp(best, unit="ns"))
    return out


def window_return(close: pd.Series, t: pd.Timestamp, horizon_min: int) -> float | None:
    end = t + pd.Timedelta(minutes=horizon_min)
    s = close[(close.index >= t) & (close.index <= end)]
    if len(s) < 3 or s.index[-1] < end - pd.Timedelta(minutes=TOLERANCE_MIN):
        return None
    return float(np.log(s.iloc[-1] / s.iloc[0]) * 1e4)


def test_control_days_are_distinct() -> None:
    """Controls must come from distinct quiet days, or the control t-stat is fabricated.

    The defect: taking the most recent quiet bar at each event's clock time made 284
    controls draw from 9 distinct days (one 72 times). The control baseline therefore
    described 9 days while reporting n=284, and its t-stat was inflated by roughly
    sqrt(284/9) ~ 5.6x - which is what produced the published "control mean -1.84bp,
    t=-5.13" that the +2.0bp event incremental sat on top of. With distinct days the
    control is n=9, t=+0.54, and the incremental vanishes.
    """
    from collections import Counter

    events = load_events()
    mids = load_mid("EURUSD")
    ew = event_windows(events, "USD", "High importance")
    ct = control_times(ew.index, mids.index, events, "USD")
    assert len(ct) >= 5, f"only {len(ct)} controls produced"
    counts = Counter(t.floor("D") for t in ct)
    assert max(counts.values()) == 1, f"a control day was reused: {counts.most_common(3)}"
    # every control must be strictly before its event and never on an event day
    event_day_set = set(events[events["currency"] == "USD"]["ts"].dt.floor("D"))
    for t in ct:
        assert t.floor("D") not in event_day_set


def test_event_windows_are_min_minutes_apart() -> None:
    """``MIN_MINUTES`` must actually separate events, not be declared and unused."""
    events = load_events()
    ew = event_windows(events, "USD", "High importance")
    ts = np.array([int(t.value) for t in ew.index])
    if len(ts) < 2:
        return
    gaps_min = np.diff(ts) / 60_000_000_000
    assert gaps_min.min() >= MIN_MINUTES, f"events {gaps_min.min():.0f}min apart (< {MIN_MINUTES})"


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
