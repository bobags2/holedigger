"""Tests for src/pooled_test.py.

Run:  python -m tests.test_pooled
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pooled_test import ORIENTATION, run_pooled


def make_panel(symbols, n_days=300, sigma=2e-4, seed=0, planted=None, bars=288):
    rng = np.random.default_rng(seed)
    k = len(symbols)
    factor = rng.standard_normal((n_days, bars)) * sigma
    R = np.empty((n_days, bars, k))
    for i in range(k):
        R[:, :, i] = 0.6 * factor + 0.8 * rng.standard_normal((n_days, bars)) * sigma
    if planted is not None:
        h, _hold = planted["window"]
        idx = h * 12 + 6  # 30 minutes into the window
        for i, sym in enumerate(symbols):
            R[:, idx, i] += ORIENTATION[sym] * planted["size"]
    base = np.datetime64("2000-01-03T00:00:00")
    offsets = (np.arange(n_days)[:, None] * 1440 + np.arange(bars)[None, :] * 5).astype("timedelta64[m]")
    times = pd.DatetimeIndex(base + offsets.ravel())
    return times, R.reshape(n_days * bars, k)


def test_planted_dollar_effect_detected() -> None:
    symbols = list(ORIENTATION)
    signs = np.array([ORIENTATION[s] for s in symbols])
    times, R = make_panel(symbols, seed=1, planted={"window": (3, 1), "size": 6e-4})
    windows = [(2, 1), (3, 1), (4, 1), (3, 2)]
    res = run_pooled(times, R, signs, windows, n_worlds=60, seed=2)
    best = int(np.argmax(np.abs(res["real_t"])))
    assert windows[best] == (3, 1), windows[best]
    assert res["candidate"]["t"] > 10, res["candidate"]["t"]
    assert res["candidate"]["own_null_percentile"] > 0.95, res["candidate"]


def test_no_effect_family_calibrated() -> None:
    symbols = list(ORIENTATION)
    signs = np.array([ORIENTATION[s] for s in symbols])
    times, R = make_panel(symbols, seed=3)
    windows = [(2, 1), (3, 1), (4, 1), (3, 2)]
    res = run_pooled(times, R, signs, windows, n_worlds=60, seed=4)
    assert res["family"]["percentile"] < 0.95, res["family"]


def test_direction_matters() -> None:
    """A planted *anti*-dollar effect must be flagged by |t| just the same."""
    symbols = list(ORIENTATION)
    signs = np.array([ORIENTATION[s] for s in symbols])
    times, R = make_panel(symbols, seed=5, planted={"window": (3, 1), "size": -6e-4})
    res = run_pooled(times, R, signs, [(3, 1)], n_worlds=30, seed=6)
    assert res["candidate"]["t"] < -10, res["candidate"]["t"]


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
