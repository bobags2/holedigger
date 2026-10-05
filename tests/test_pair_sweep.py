"""Tests for src/pair_sweep.py.

Run:  python -m tests.test_pair_sweep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pair_sweep import PairData, run_pair_sweep, simulate_pair


def make_pair(name_a="AAA", name_b="BBB", n_days=300, seed=0, planted=True, bars=288) -> PairData:
    rng = np.random.default_rng(seed)
    t = n_days * bars
    rb = rng.standard_normal(t) * 2e-4
    ra = rb.copy()
    if planted:
        rho, sigma_e = 0.85, 6e-5
        s = np.empty(t)
        s[0] = 0.0
        e = rng.standard_normal(t) * sigma_e
        for k in range(1, t):
            s[k] = rho * s[k - 1] + e[k]
        ra[1:] = rb[1:] + np.diff(s)
    else:
        ra = 0.8 * rb + rng.standard_normal(t) * 2e-4  # co-movement, no level reversion
    base = np.datetime64("2000-01-03T00:00:00")
    offsets = (np.arange(n_days)[:, None] * 1440 + np.arange(bars)[None, :] * 5).astype("timedelta64[m]")
    times = pd.DatetimeIndex(base + offsets.ravel())
    la = np.cumsum(ra)
    lb = np.cumsum(rb)
    return PairData(name_a, name_b, times, la, lb, float(np.exp(la[-1])), float(np.exp(lb[-1])))


def test_planted_reversion_detected() -> None:
    p = make_pair(seed=1, planted=True)
    res = run_pair_sweep([p], n_worlds=60, seed=2)
    best = res["candidates"][0]
    assert best["n_trades"] > 20, best
    assert best["t"] > 10, best["t"]
    assert best["own_null_percentile"] > 0.95, best["own_null_percentile"]
    assert res["family"]["percentile"] > 0.95, res["family"]


def test_no_reversion_not_detected() -> None:
    p = make_pair(seed=3, planted=False)
    res = run_pair_sweep([p], n_worlds=60, seed=4)
    assert res["family"]["percentile"] < 0.95, res["family"]


def test_simulate_handcrafted() -> None:
    la = np.array([0.0, 0.0, 0.002, 0.0022, 0.0015, 0.0010, 0.0005])
    lb = np.zeros(7)
    beta = np.ones(7)
    z = np.array([0.0, 0.0, 2.5, 2.0, 0.2, 0.0, 0.0])
    tr = simulate_pair(la, lb, beta, z, z_entry=2.0, z_exit=0.3)
    # entry at bar 2 (z=2.5), exit at first |z|<=0.3 after entry -> bar 4 (z=0.2)
    assert list(tr["entry"]) == [2] and list(tr["exit"]) == [4], (tr["entry"], tr["exit"])
    # short the spread: pnl = -(la[4]-la[2]) = -(0.0015-0.002) = +0.0005
    assert np.isclose(tr["pnl"][0], 5e-4), tr["pnl"]


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
