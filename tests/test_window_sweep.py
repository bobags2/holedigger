"""Tests for src/window_sweep.py and src/costs.py.

Run:  python -m tests.test_window_sweep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import costs
from src.window_sweep import (
    default_windows,
    run_sweep,
    series_from_bars,
    window_segments,
)


def make_synthetic(symbols, n_days, sigma=2e-4, seed=0, planted=None, bars_per_day=288):
    rng = np.random.default_rng(seed)
    out = {}
    for sym in symbols:
        r = rng.standard_normal((n_days, bars_per_day)) * sigma
        if planted and planted["symbol"] == sym:
            h, _d = planted["window"]
            r[:, h * 12 + 6] += planted["size"]  # 30 minutes into the window
        base = np.datetime64("2000-01-03T00:00:00")
        offsets = (np.arange(n_days)[:, None] * 1440 + np.arange(bars_per_day)[None, :] * 5).astype("timedelta64[m]")
        times = base + offsets.ravel()
        closes = 100 * np.exp(np.cumsum(r.ravel()))
        out[sym] = series_from_bars(sym, pd.DatetimeIndex(times), closes)
    return out


def test_segment_pnl_handcrafted() -> None:
    times = pd.date_range("2000-01-03 00:00", periods=13, freq="5min")
    closes = 100 * np.exp(np.cumsum(np.full(13, 1e-4)))
    s = series_from_bars("T", times, closes)
    a, b, valid = window_segments(s, [(0, 1)])
    assert bool(valid[0, 0])
    S = np.cumsum(s.r)
    pnl = S[b[0, 0]] - S[a[0, 0]]
    assert np.isclose(pnl, 12e-4), pnl


def test_planted_effect_detected() -> None:
    series = make_synthetic(
        ["AAA", "BBB", "CCC"], n_days=300, seed=1, planted={"symbol": "AAA", "window": (10, 1), "size": 8e-4}
    )
    windows = [(9, 1), (10, 1), (11, 1), (10, 2)]
    res = run_sweep(series, windows, n_worlds=40, seed=2)
    top = res["candidates"][0]
    assert (top["symbol"], top["entry_hour_utc"], top["hold_hours"]) == ("AAA", 10, 1), top
    assert top["t"] > 10, top["t"]
    assert top["own_null_percentile"] > 0.95, top["own_null_percentile"]


def test_null_calibration_no_effect() -> None:
    series = make_synthetic(["AAA", "BBB", "CCC"], n_days=300, seed=3)
    res = run_sweep(series, default_windows(), n_worlds=60, seed=4)
    assert res["family"]["percentile"] < 0.95, res["family"]
    assert 0.0 <= res["family"]["percentile"] <= 1.0


def test_cost_units() -> None:
    rt = costs.commission_round_trip_bp("EURUSD", 1.1331)
    assert rt is not None and abs(rt - 0.443) < 0.002, rt
    assert costs.commission_round_trip_bp("US500", 6000.0) is None
    be = costs.break_even_spread_bp(1.0, "EURUSD", 1.1331)
    assert be is not None and abs(be - 0.557) < 0.002, be


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
