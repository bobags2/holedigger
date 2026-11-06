"""Tests for src/pair_sweep.py.

Run:  python -m tests.test_pair_sweep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pair_sweep import PairData, parse_pairs, run_pair_sweep, simulate_pair


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
    # signal at bar 2 (z=2.5) but the entry is the NEXT bar's close: bar 3. Exit at the
    # first |z|<=0.3 after the signal -> bar 4 (z=0.2). Entering at the signal bar's own
    # close would book P&L the trade could not have earned.
    assert list(tr["entry"]) == [3] and list(tr["exit"]) == [4], (tr["entry"], tr["exit"])
    # short the spread at the tradable price: -(la[4]-la[3]) = -(0.0015-0.0022) = +0.0007
    assert np.isclose(tr["pnl"][0], 7e-4), tr["pnl"]


def test_entry_is_never_the_signal_bar() -> None:
    """No trade may enter on the bar whose close generated its own signal."""
    rng = np.random.default_rng(0)
    n = 4000
    a = np.cumsum(rng.standard_normal(n) * 1e-4)
    b = np.cumsum(rng.standard_normal(n) * 1e-4)
    beta = np.full(n, 1.0)
    z = (a - b) / 1e-3
    tr = simulate_pair(a, b, beta, z, z_entry=3.0, z_exit=0.5)
    ent, ex = tr["entry"], tr["exit"]
    assert len(ent) > 5, len(ent)
    # every entry is strictly after its signal bar, and every exit after its entry
    sig = np.flatnonzero(np.abs(z) >= 3.0)
    for e in ent:
        assert np.any(sig < e), f"entry {e} is not after any signal bar"
        assert e >= sig.min()
    assert (ex > ent).all()
    assert (np.diff(ent, prepend=-1) > 0).all(), "entries must be strictly increasing"
    # each entry is after ITS signal: the signal bar is the first bar at/with |z|>=3
    # scanning back from the entry. If any entry is itself the first signal bar, the
    # trade would be booking its own signal.
    for e in ent:
        back = np.flatnonzero(np.abs(z[:e]) >= 3.0)
        assert len(back) and back.max() < e, f"entry {e} has no earlier signal bar"


def test_parse_pairs() -> None:
    assert parse_pairs("eurusd/gbpusd, XAUUSD/USDX") == [("EURUSD", "GBPUSD"), ("XAUUSD", "USDX")]
    try:
        parse_pairs("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
