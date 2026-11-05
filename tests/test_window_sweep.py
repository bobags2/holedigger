"""Tests for src/window_sweep.py and src/costs.py.

Run:  python -m tests.test_window_sweep
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src import costs
from src.window_sweep import (
    _shared_day_grid_ok,
    default_windows,
    intraday_slot_null_masks,
    run_sweep,
    series_from_bars,
    window_segments,
)


def make_synthetic(symbols, n_days, sigma=2e-4, seed=0, planted=None, bars_per_day=288):
    rng = np.random.default_rng(seed)
    out = {}
    for sym in symbols:
        r = rng.standard_normal((n_days, bars_per_day)) * sigma
        if planted and planted["symbol"] in (sym, "__all__"):
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


def make_correlated_noise(n_names, n_days, rho=0.5, sigma=2e-4, seed=11, bars_per_day=288):
    """One-factor correlated noise: every name shares the same daily market factor."""
    rng = np.random.default_rng(seed)
    base = np.datetime64("2000-01-03T00:00:00")
    out = {}
    f = rng.standard_normal((n_days, bars_per_day))
    for k in range(n_names):
        r = rng.standard_normal((n_days, bars_per_day)) * sigma
        r = np.sqrt(1 - rho) * r + np.sqrt(rho) * f * sigma
        offsets = (np.arange(n_days)[:, None] * 1440 + np.arange(bars_per_day)[None, :] * 5).astype(
            "timedelta64[m]"
        )
        closes = 100 * np.exp(np.cumsum(r.ravel()))
        out[f"N{k:03d}"] = series_from_bars(
            f"N{k:03d}", pd.DatetimeIndex(base + offsets.ravel()), closes
        )
    return out


def test_sector_null_holds_on_correlated_noise() -> None:
    """Pooled sector z must be calibrated on correlated noise with zero real drift.

    The full-procedure check the earlier tests missed. Permuting each symbol on its
    own index made real t's (correlated through the market factor) face null t's that
    were independent, whose cross-symbol sd was far wider - so a 'significant' sector
    effect appeared in roughly half of pure-noise worlds. The shared-intraday null
    shuffles one day order and one slot order across every name, preserving that
    correlation while destroying clock-slot alignment.

    The scan's headline is its most extreme window, so the reference is the same
    scan-maximum inside each null world (``sector_family``), not the best window's own
    per-window p95: scanning ~20 correlated windows against their marginal 95th
    percentiles rejects a *correct* null ~56% of the time (measured), while the
    scan-max reference holds it at ~12%.
    """
    alpha, trials, discoveries = 0.05, 25, 0
    for t in range(trials):
        series = make_correlated_noise(n_names=20, n_days=60, rho=0.5, seed=1000 + t)
        res = run_sweep(
            series, [(h, 2) for h in range(0, 19)], n_worlds=25, seed=t
        )
        best = max(res["sector"].values(), key=lambda r: abs(r["real_z"]))
        assert best["null_kind"] == "shared-intraday", best
        fam = res["sector_family"]
        if fam["real_max_abs_z"] > fam["null_max_abs_z_p95"]:
            discoveries += 1
    rate = discoveries / trials
    assert rate <= 0.15, f"sector null rejected {rate:.0%} of pure-noise worlds"


def test_shared_intraday_null_masks_invariants() -> None:
    """The shared null is a pure reindexing: permutation, gaps pinned, one world.

    - every mask is a permutation of the bar indices (marginals untouched);
    - each destination day's first bar reads some day's first bar, so the overnight /
      weekend gap return can never leak into an intraday window;
    - all symbols share day_order and slot_order, so they keep co-moving: applying the
      masks to two perfectly correlated symbols reproduces the same permuted returns.
    """
    series = make_synthetic(["AAA", "BBB", "CCC"], n_days=25, seed=21)
    masks = intraday_slot_null_masks(series, np.random.default_rng(7))
    s = series["AAA"]
    starts = np.searchsorted(s.bar_day, np.unique(s.bar_day), side="left")
    for sym, ser in series.items():
        idx = masks[sym]
        assert len(idx) == len(ser.r)
        assert np.array_equal(np.sort(idx), np.arange(len(idx))), sym
        assert not np.array_equal(idx, np.arange(len(idx))), sym  # not the identity
        # destination boundary bars read source boundary bars only
        assert set(idx[starts].tolist()) <= set(starts.tolist()), sym

    # Same world for every name: identical symbols must get identical permuted returns.
    one = make_synthetic(["AAA"], n_days=25, seed=21)["AAA"]
    panel = {"AAA": one, "BBB": dataclasses.replace(one, symbol="BBB")}
    assert _shared_day_grid_ok(panel)
    masks = intraday_slot_null_masks(panel, np.random.default_rng(7))
    ra = panel["AAA"].r[masks["AAA"]]
    rb = panel["BBB"].r[masks["BBB"]]
    assert np.array_equal(ra, rb)

    # A ragged panel (different day set / bar counts) must not take the shared path.
    ragged = dict(panel)
    ragged["CCC"] = make_session_bounded("CCC", n_days=25)
    assert not _shared_day_grid_ok(ragged)


def test_shared_null_fp_and_power() -> None:
    """End-to-end false-positive rate and power of the shared-intraday null.

    FP: on correlated one-factor noise the pooled sector z must reject at <= 15%.
    Power: a genuine clock-slot drift (+3e-4 on one intraday slot, every day, every
    symbol) is destroyed by the slot shuffle, so the real family maximum must still
    clear the null family maximum (percentile > 0.9).
    """
    trials, worlds, discoveries = 25, 25, 0
    for t in range(trials):
        series = make_correlated_noise(n_names=20, n_days=60, rho=0.5, seed=3000 + t)
        res = run_sweep(series, [(h, 2) for h in range(0, 19)], n_worlds=worlds, seed=t)
        best = max(res["sector"].values(), key=lambda r: abs(r["real_z"]))
        assert best["null_kind"] == "shared-intraday", best
        fam = res["sector_family"]
        if fam["real_max_abs_z"] > fam["null_max_abs_z_p95"]:
            discoveries += 1
    rate = discoveries / trials
    assert rate <= 0.15, f"shared null rejected {rate:.0%} of pure-noise worlds"

    planted = make_synthetic(
        [f"S{k:02d}" for k in range(20)],
        n_days=60,
        seed=41,
        planted={"symbol": "__all__", "window": (10, 1), "size": 3e-4},
    )
    res = run_sweep(planted, [(h, 1) for h in range(0, 20)], n_worlds=40, seed=42)
    assert res["null_kind"] == "shared-intraday", res["null_kind"]
    assert res["family"]["percentile"] > 0.9, res["family"]


def test_per_symbol_null_inflates_on_correlated_noise() -> None:
    """Regression evidence for the defect: the old per-symbol null rejects too often."""
    trials, discoveries = 25, 0
    for t in range(trials):
        series = make_correlated_noise(n_names=20, n_days=60, rho=0.5, seed=2000 + t)
        res = run_sweep(
            series, [(h, 2) for h in range(0, 19)], n_worlds=25, seed=t,
            allow_shared_world=False,
        )
        fam = res["sector_family"]
        assert res["null_kind"] == "per-symbol", res["null_kind"]
        if fam["real_max_abs_z"] > fam["null_max_abs_z_p95"]:
            discoveries += 1
    assert discoveries / trials > 0.25, (
        f"per-symbol null rejected only {discoveries / trials:.0%} of noise worlds; "
        "expected >25% (the defect it demonstrates)"
    )


def test_cost_units() -> None:
    rt = costs.commission_round_trip_bp("EURUSD", 1.1331)
    assert rt is not None and abs(rt - 0.443) < 0.002, rt
    assert costs.commission_round_trip_bp("US500", 6000.0) == 0.0
    assert costs.commission_round_trip_bp("FOOBAR", 10.0) is None
    be = costs.break_even_spread_bp(1.0, "EURUSD", 1.1331)
    assert be is not None and abs(be - 0.557) < 0.002, be


def make_session_bounded(name, n_days=300, seed=9, start_hour=12, end_hour=18):
    rng = np.random.default_rng(seed)
    bars = (end_hour - start_hour) * 12
    r = rng.standard_normal(n_days * bars) * 2e-4
    base = np.datetime64("2000-01-03T00:00:00")
    offsets = (
        np.arange(n_days)[:, None] * 1440 + start_hour * 60 + np.arange(bars)[None, :] * 5
    ).astype("timedelta64[m]")
    times = pd.DatetimeIndex(base + offsets.ravel())
    closes = 100 * np.exp(np.cumsum(r))
    return series_from_bars(name, times, closes)


def test_session_bounded_symbol_calibrates() -> None:
    """Session-bounded symbols leave all-NaN candidates; the family must stay calibrated."""
    series = make_synthetic(["AAA"], n_days=300, seed=7)
    series["BBB"] = make_session_bounded("BBB")
    res = run_sweep(series, default_windows(), n_worlds=30, seed=8)
    assert np.isfinite(res["family"]["real_max_abs_t"]), res["family"]
    assert 0.0 <= res["family"]["percentile"] <= 1.0, res["family"]
    assert np.isfinite(res["null"]["family_max_mean"]), res["null"]
    assert any(not np.isfinite(c["t"]) for c in res["candidates"]), "expected NaN candidates"
    assert np.isfinite(res["candidates"][0]["t"]), "NaN candidate sorted first"


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
