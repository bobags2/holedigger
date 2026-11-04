"""Tests for the MT5 tick -> M1 mid-bar transform (src/mt5_ingest.py).

Run:  python -m tests.test_mt5_ingest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.mt5_ingest import _p90, mid_minute_bars


def test_mid_minute_bars_known_pattern() -> None:
    ts = pd.to_datetime(
        ["2026-11-04T10:00:00.100", "2026-11-04T10:00:30.000", "2026-11-04T10:00:50.000",
         "2026-11-04T10:01:10.000"]
    )
    bid = [1.10000, 1.10010, 1.10005, 1.10100]
    ask = [1.10000, 1.10015, 1.10005, 1.10105]
    bars = mid_minute_bars(pd.DataFrame({"ts": ts, "bid": bid, "ask": ask}))

    assert len(bars) == 2, bars
    first = bars.iloc[0]
    assert first["tick_n"] == 3
    assert np.isclose(first["open"], 1.10000)      # mid of first tick in the minute
    assert np.isclose(first["close"], 1.10005)     # mid of last tick in the minute
    assert np.isclose(first["high"], 1.100125)     # max mid
    assert np.isclose(first["low"], 1.10000)       # min mid
    assert np.isclose(first["spread_med_bp"], 0.0)  # median spread was zero
    # mean spread 1.6667e-5 price -> bp of mid (1.10005)
    assert np.isclose(first["spread_mean_bp"], 1.6667e-5 / 1.10005 * 1e4, rtol=1e-3)
    # p90 helper: spread values {0, 5e-5, 0} -> 4e-5 price -> bp
    assert np.isclose(first["spread_p90_bp"], 4e-5 / 1.10005 * 1e4, rtol=1e-3)

    second = bars.iloc[1]
    assert second["tick_n"] == 1
    assert np.isclose(second["close"], 1.101025)
    assert np.isclose(second["spread_med_bp"], 5e-5 / 1.101025 * 1e4, rtol=1e-3)


def test_empty_and_zero_spread() -> None:
    empty = mid_minute_bars(pd.DataFrame(columns=["ts", "bid", "ask"]))
    assert len(empty) == 0
    assert list(empty.columns) == ["ts", "open", "high", "low", "close", "tick_n",
                                   "spread_med_bp", "spread_mean_bp", "spread_p90_bp"]

    ts = pd.to_datetime(["2026-11-04T10:00:01.000", "2026-11-04T10:00:02.000"])
    bars = mid_minute_bars(pd.DataFrame({"ts": ts, "bid": [1.05, 1.06], "ask": [1.05, 1.06]}))
    assert len(bars) == 1
    assert bars.iloc[0]["tick_n"] == 2
    assert float(bars.iloc[0]["spread_med_bp"]) == 0.0
    assert float(bars.iloc[0]["spread_mean_bp"]) == 0.0


def test_p90_helper_on_empty() -> None:
    assert np.isnan(_p90([]))


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
