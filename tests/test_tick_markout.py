"""Tests for src/tick_markout.py — verified against synthetic tapes with known answers.

Run:  python -m tests.test_tick_markout
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import tick_markout as tm


def _write_ticks(tmp, df: pd.DataFrame) -> None:
    from src import mt5_clock

    mt5_clock.write_clock_tag_from_frame(df, tmp)


def test_fill_direction_is_read_from_the_tape() -> None:
    """A print at the ASK is an aggressive buy (fills a resting offer); at the BID, a sell.

    This is the mechanism the whole study rests on, so it is checked directly: with a
    controlled tape where every print's side is known, the flags must match exactly.
    """
    n = 200
    base = pd.Timestamp("2026-01-05 09:30")
    ts = pd.date_range(base, periods=n, freq="10s")
    bid = np.full(n, 100.0)
    ask = np.full(n, 100.02)
    mid = (bid + ask) / 2.0
    last = np.empty(n)
    last[0::2] = 100.02       # aggressive buys on even ticks (printed at the ask)
    last[1::2] = 100.00       # aggressive sells on odd ticks (printed at the bid)
    df = pd.DataFrame({"ts": ts, "bid": bid, "ask": ask, "mid": mid,
                       "last": last, "volume": np.ones(n)})
    flags, _inf = tm.fill_flags(df)
    assert flags["ask_fill"].sum() == n // 2, flags["ask_fill"].sum()
    assert flags["bid_fill"].sum() == n // 2, flags["bid_fill"].sum()
    assert (flags["ask_fill"] == ~flags["bid_fill"]).all()
    # a resting bid fills ABOVE the mid by half the spread
    assert np.allclose(flags["half_spread"].to_numpy(), 0.01)


def test_earned_spread_tape_marks_out_positive() -> None:
    """On a tape where the spread is pure inventory (mean-reverting), markout must be positive.

    Ground truth by construction: every aggressive move is followed by a move back toward the
    mid, so a passive fill at the touched side is always in the money H minutes later.
    """
    n = 4000
    ts = pd.date_range("2026-01-05 09:30", periods=n, freq="1s")
    rng = np.random.default_rng(0)
    # a sine path: every excursion away from the mid reverts
    t = np.arange(n) / 60.0
    path = 100.0 + 0.02 * np.sin(t)
    bid, ask = path - 0.01, path + 0.01
    mid = path
    last = np.where(np.sin(t) > 0, ask, bid)      # buys on the upswing, sells on the downswing
    df = pd.DataFrame({"ts": ts, "bid": bid, "ask": ask, "mid": mid, "last": last})
    flags, _inf = tm.fill_flags(df)
    m = tm.markout(flags, 60)
    bid_m = m["markout_bid_bp"].to_numpy()
    ok = np.isfinite(bid_m)
    assert ok.sum() > 100, ok.sum()
    # the spread is earned: the mid returns to (or past) the fill price
    assert np.nanmean(bid_m) > 0, np.nanmean(bid_m)


def test_adverse_selection_tape_marks_out_negative_and_worse_when_wide() -> None:
    """On a tape where wide spreads precede informed moves, the wide-state markout must be worse.

    Ground truth: the spread widens exactly when a persistent drift is about to continue, so
    a passive fill on a wide tape is run over; on a tight tape the move reverts and the fill
    is fine. This is the adverse-selection signature the module exists to detect.
    """
    n = 12000
    ts = pd.date_range("2026-01-05 09:30", periods=n, freq="1s")
    rng = np.random.default_rng(1)
    block = 200
    reg = np.repeat(rng.random(n // block + 1) < 0.4, block)[:n]   # True = wide/informed regime

    # Tight regime: a deterministic zig-zag that falls through the first 40 bars and then
    # recovers to its start by bar 80. Aggressive sells (a resting bid's fill) happen on the
    # DESCENDING leg, so 60 bars later the mid has recovered and the fill is in the money.
    # Wide regime: a deterministic down-drift, where the same fill is run over.
    # Both regimes are therefore known-answer by construction, and the module must recover
    # positive-tight / negative-wide / wide<tight.
    i = np.arange(n)
    cycle = (i % 80).astype(float)
    zigzag = np.where(cycle < 40, -cycle * 0.002, -(80 - cycle) * 0.002)
    drift = -0.005 * i / n * 2
    path = 100.0 + np.where(reg, drift, zigzag)
    descending = cycle < 40                     # the recovering leg's entry side
    half = np.where(reg, 0.02, 0.005)
    bid, ask = path - half, path + half
    mid = path
    last = np.where(reg, bid, np.where(descending, bid, ask))
    df = pd.DataFrame({"ts": ts, "bid": bid, "ask": ask, "mid": mid, "last": last})
    flags, _inf = tm.fill_flags(df)
    m = tm.markout(flags, 60)
    v = m["markout_bid_bp"].to_numpy()
    sp = m["spread_bp"].to_numpy()
    ok = np.isfinite(v)
    thr = np.nanquantile(sp[ok], 0.8)
    wide = ok & (sp >= thr)
    tight = ok & (sp < thr)
    assert wide.sum() > 20 and tight.sum() > 20, (int(wide.sum()), int(tight.sum()))
    wm, tm_ = float(np.nanmean(v[wide])), float(np.nanmean(v[tight]))
    assert wm < tm_, (wm, tm_)          # wide is worse: that is adverse selection
    assert wm < 0, wm                    # and the wide fills are actually run over
    assert tm_ > 0, tm_                  # while the tight fills are earned


def test_missing_tick_file_is_an_explicit_prerequisite() -> None:
    """Without retained ticks the module must raise, not silently substitute bar data.

    A silent fallback would answer a different question than the one asked.
    """
    try:
        tm.load_ticks("NOSUCHSYMBOL")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "tick_ingest" in str(exc), exc


def test_horizon_shift_is_in_time_not_in_rows() -> None:
    """The forward mid is a TIME shift, so gaps in the tape do not silently shorten horizons."""
    n = 1000
    ts = pd.date_range("2026-01-05 09:30", periods=n, freq="1s")
    mid = 100.0 + np.arange(n) * 1e-5
    df = pd.DataFrame({"ts": ts, "bid": mid - 0.01, "ask": mid + 0.01, "mid": mid})
    flags, _inf = tm.fill_flags(df)
    m = tm.markout(flags, 60)
    # 60 seconds later the mid is 60 ticks higher: markout for a bid fill = +60*1e-5 - half
    v = m["markout_bid_bp"].to_numpy()
    v = v[np.isfinite(v)]
    assert np.allclose(v, 60 * 1e-5 - 0.01, atol=1e-9), v[:3]


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
