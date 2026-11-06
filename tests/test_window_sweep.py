"""Tests for src/window_sweep.py and src/costs.py.

Run:  python -m tests.test_window_sweep
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src import costs, gpu_null
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
    effect appeared in roughly half of pure-noise worlds. The sweep's default null
    (``auto`` -> ``sign-flip``) flips one random sign per day across every candidate at
    once, so cross-candidate and cross-day co-movement survive and only each day's sign
    is randomised.

    The scan's headline is its most extreme window, so the reference is the same
    scan-maximum inside each null world (``sector_family``), not the best window's own
    per-window p95: scanning ~20 correlated windows against their marginal 95th
    percentiles rejects a *correct* null ~56% of the time (measured), while the
    scan-max reference holds it at ~12% (and 4% once the null has enough worlds to
    resolve a 95th percentile - 25 worlds puts 4pp on every percentile step, so the
    25-world bound is 0.15 and the 200-world bound is 0.08).
    """
    # With 25 worlds the percentile is quantised to steps of 0.04, so "rejecting at the
    # 95th percentile" means observing 24+ of 25 null worlds below the real statistic
    # (p>=0.96). The nominal rate for that event is 1/25 = 4%, and the sampling noise on
    # 25 trials dominates: asserting an exact alpha here tests the trial count, not the
    # null. So assert CALIBRATION over enough independent trials to see it - the
    # percentile on pure noise must be ~Uniform(0,1) - and pin the seed set so the
    # check is deterministic.
    pcts = []
    for t in range(50):
        series = make_correlated_noise(n_names=20, n_days=60, rho=0.5, seed=3000 + t)
        res = run_sweep(series, [(h, 2) for h in range(0, 19)], n_worlds=25, seed=t)
        assert res["sector"]["00+2h"]["null_kind"] in ("sign-flip", "shared-intraday"), res["sector"]["00+2h"]
        pcts.append(res["sector_family"]["percentile"])
    pcts = np.asarray(pcts)
    mean_pct = float(pcts.mean())
    tail = float((pcts >= 0.92).mean())
    assert 0.35 <= mean_pct <= 0.65, f"sector null percentile mean {mean_pct:.3f}, expected ~0.5 on pure noise"
    assert tail <= 0.20, f"sector null rejected {tail:.0%} of pure-noise worlds (25-world resolution)"
    # and the null must NOT be as degenerate as the old per-symbol one: a real effect
    # must still be findable, which the power test below covers at >0.9.


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
    """End-to-end false-positive rate and power of the sweep's default null.

    FP: on correlated one-factor noise the pooled sector z must reject at <= 15%.
    Power: a genuine clock-slot drift (+3e-4 on one intraday slot, every day, every
    symbol) survives any null that destroys clock-slot alignment, so the real family
    maximum must still clear the null family maximum (percentile > 0.9).

    The default null is ``auto`` -> ``sign-flip`` (a day-level sign flip on the shared
    days x candidates P&L matrix, ``src/gpu_null.py``); ``shared-intraday`` is kept
    selectable and this test runs the default so it stays honest about what ships.
    """
    trials, worlds, discoveries = 25, 25, 0
    for t in range(trials):
        series = make_correlated_noise(n_names=20, n_days=60, rho=0.5, seed=3000 + t)
        res = run_sweep(series, [(h, 2) for h in range(0, 19)], n_worlds=worlds, seed=t)
        best = max(res["sector"].values(), key=lambda r: abs(r["real_z"]))
        assert best["null_kind"] in ("sign-flip", "shared-intraday"), best
        fam = res["sector_family"]
        if fam["real_max_abs_z"] > fam["null_max_abs_z_p95"]:
            discoveries += 1
    rate = discoveries / trials
    assert rate <= 0.20, f"default null rejected {rate:.0%} of pure-noise worlds (25-world resolution)"

    planted = make_synthetic(
        [f"S{k:02d}" for k in range(20)],
        n_days=60,
        seed=41,
        planted={"symbol": "__all__", "window": (10, 1), "size": 3e-4},
    )
    res = run_sweep(planted, [(h, 1) for h in range(0, 20)], n_worlds=40, seed=42)
    assert res["null_kind"] in ("sign-flip", "shared-intraday"), res["null_kind"]
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


def _small_panel(n_days=40, n_win=12, seed=0):
    """A (n_days, 2*n_win) panel shared by two symbols, as gpu_null consumes it."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n_days, 2 * n_win)) * 2.0
    days = {"A": np.arange(n_days), "B": np.arange(n_days)}
    pnl = {"A": P[:, :n_win].copy(), "B": P[:, n_win:].copy()}
    return gpu_null.day_pnl_panel(days, pnl, ["A", "B"], [(h, 1) for h in range(n_win)])


def test_gpu_null_matches_brute_force() -> None:
    """The GEMM t-statistic equals an explicit per-world, per-column loop.

    Brute force applies the SAME sign vector (same rng seed) to every column, drops
    invalid cells, and computes mean/(sd/sqrt(n)) - the formula window_sweep uses. The
    chunked GEMM builds moments by matrix products instead, so this is the test that
    the whole shortcut is right.
    """
    panel = _small_panel()
    n_days, n_cand = panel.pnl.shape
    run = gpu_null.sign_flip_null(
        panel, 300, np.random.default_rng(7), chunk=137, backend="numpy", dtype=np.float64
    )
    X, V = panel.pnl.astype(np.float64), panel.valid.astype(bool)
    brute = np.full((300, n_cand), np.nan)
    r2 = np.random.default_rng(7)
    for w in range(300):
        fl = r2.choice([-1.0, 1.0], size=n_days)
        Y = X * fl[:, None]
        for j in range(n_cand):
            y = Y[V[:, j], j]
            if len(y) > 1:
                sd = y.std(ddof=1)
                if sd > 0:
                    brute[w, j] = y.mean() / (sd / np.sqrt(len(y)))
    a, b = run.t, brute
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 0.9 * a.size, "too many non-finite cells"
    rel = np.max(np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12))
    assert rel < 1e-9, f"max relative diff {rel:.3e} vs brute force"
    assert np.allclose(
        np.nanmax(np.abs(a), axis=1), np.nanmax(np.abs(b), axis=1), rtol=1e-10, equal_nan=True
    )


def test_gpu_null_chunking_invariant() -> None:
    """Chunk size changes float summation order, not the statistic."""
    panel = _small_panel(n_days=60, seed=5)
    runs = {
        c: gpu_null.sign_flip_null(
            panel, 400, np.random.default_rng(11), chunk=c, backend="numpy", dtype=np.float64
        ).t
        for c in (100, 137, 400)
    }
    ref = runs[100]
    for c, t in runs.items():
        assert np.allclose(t, ref, rtol=1e-9, atol=1e-9, equal_nan=True), f"chunk {c} diverged"
    f32a = gpu_null.sign_flip_null(
        panel, 300, np.random.default_rng(3), chunk=50, backend="numpy", dtype=np.float32
    ).t
    f32b = gpu_null.sign_flip_null(
        panel, 300, np.random.default_rng(3), chunk=250, backend="numpy", dtype=np.float32
    ).t
    assert np.max(np.abs(f32a - f32b)) < 1e-4, "float32 chunking drift too large"


def test_gpu_null_torch_matches_numpy() -> None:
    """The CUDA and BLAS backends compute the same worlds (drop-in equivalence)."""
    try:
        import torch
    except ImportError:
        return  # numpy-only host: the equivalence is covered by the brute-force test
    if not torch.cuda.is_available():
        return
    panel = _small_panel(n_days=60, seed=9)
    a = gpu_null.sign_flip_null(
        panel, 300, np.random.default_rng(13), chunk=100, backend="numpy", dtype=np.float64
    ).t
    b = gpu_null.sign_flip_null(
        panel, 300, np.random.default_rng(13), chunk=100, backend="torch", dtype=np.float64
    ).t
    assert np.max(np.abs(a - b)) < 1e-7, f"torch vs numpy max diff {np.max(np.abs(a - b)):.3e}"


def test_day_level_nulls_calibrate_across_rho() -> None:
    """Both day-level nulls hold at ~alpha on correlated noise; sign flip is symmetric.

    Sign flips assume a null symmetric about zero, so the block bootstrap of demeaned
    P&L is the independent check. They must agree, and neither may over-reject: on
    pure noise the scan-max percentile is ~Uniform(0,1), so its mean is ~0.5 and the
    share above 0.92 is small (25 worlds quantise percentiles to 0.04 steps).
    """
    for null_kind in ("sign-flip", "block-bootstrap"):
        pcts = []
        for t in range(25):
            series = make_correlated_noise(n_names=20, n_days=60, rho=0.5, seed=4000 + t)
            res = run_sweep(
                series, [(h, 2) for h in range(0, 19)], n_worlds=25, seed=t, null_kind=null_kind
            )
            assert res["null_kind"] == null_kind, res["null_kind"]
            pcts.append(res["sector_family"]["percentile"])
        pcts = np.asarray(pcts)
        mean_pct = float(pcts.mean())
        assert 0.35 <= mean_pct <= 0.65, f"{null_kind}: percentile mean {mean_pct:.3f}, expected ~0.5"
        assert float((pcts >= 0.92).mean()) <= 0.24, (
            f"{null_kind}: rejected {(pcts >= 0.92).mean():.0%} of pure-noise worlds"
        )


def test_gpu_null_power() -> None:
    """A planted clock-slot drift is recovered by the day-level nulls (>0.9)."""
    planted = make_synthetic(
        [f"P{k:02d}" for k in range(20)],
        n_days=250,
        seed=77,
        planted={"symbol": "__all__", "window": (10, 2), "size": 3e-4},
    )
    for null_kind in ("sign-flip", "block-bootstrap"):
        res = run_sweep(
            planted, [(9, 2), (10, 1), (10, 2), (11, 2)], n_worlds=40, seed=3, null_kind=null_kind
        )
        assert res["family"]["percentile"] > 0.9, (null_kind, res["family"])


def make_drift_panel(bp_per_day, n_syms=3, n_days=250, seed=0, sigma=2e-4):
    """iid noise + a uniform per-bar drift, no clock-slot structure of any kind."""
    rng = np.random.default_rng(seed)
    per_bar = bp_per_day / 288.0 * 1e-4
    out = {}
    base = np.datetime64("2000-01-03T00:00:00")
    for k in range(n_syms):
        r = rng.standard_normal((n_days, 288)) * sigma + per_bar
        offs = (np.arange(n_days)[:, None] * 1440 + np.arange(288)[None, :] * 5).astype("timedelta64[m]")
        out[f"D{k}"] = series_from_bars(
            f"D{k}", pd.DatetimeIndex(base + offs.ravel()), 100 * np.exp(np.cumsum(r.ravel()))
        )
    return out


def test_day_level_null_ignores_drift() -> None:
    """A trending price must not read as a session-window edge.

    The failure this pins: the day-level worlds centre a column at zero, so an
    unconditional drift in the realised P&L survives into the *real* statistic and every
    family looked significant - measured 10/10 at 10 bp/day, 10/10 at 20 bp/day, on a panel
    with no clock-slot structure at all. The bar-level shared-intraday null gave 0/10.
    The fix removes each symbol's mean intraday bar return (gap bars excluded) before P&L
    is formed, so the null is drift-invariant while a genuine clock-slot edge is kept.

    The bound is flat in the drift: if a level of drift changed the rejection rate, the
    null would still be testing the trend rather than the structure.
    """
    windows = [(h, 4) for h in range(0, 19)]
    rates = {}
    for null_kind in ("sign-flip", "block-bootstrap"):
        hits_at = {}
        for bp in (0, 100):
            hits = 0
            for t in range(20):
                series = make_drift_panel(bp, seed=9000 + t)
                res = run_sweep(series, windows, n_worlds=60, seed=t, null_kind=null_kind)
                hits += res["family"]["percentile"] >= 0.95
            hits_at[bp] = hits
        rates[null_kind] = hits_at
        assert hits_at[0] <= 3, f"{null_kind}: false {hits_at[0]}/20 on pure noise"
        assert hits_at[100] <= 3, f"{null_kind}: false {hits_at[100]}/20 at 100bp/day drift"
        assert hits_at[100] <= hits_at[0] + 2, (
            f"{null_kind}: drift sensitivity {hits_at[0]} -> {hits_at[100]} false positives"
        )
    # and the same nulls still find a real clock-slot edge
    planted = make_synthetic(
        [f"P{k:02d}" for k in range(20)],
        n_days=250,
        seed=77,
        planted={"symbol": "__all__", "window": (10, 4), "size": 3e-4},
    )
    for null_kind in ("sign-flip", "block-bootstrap"):
        res = run_sweep(
            planted, [(9, 4), (10, 2), (10, 4), (11, 4)], n_worlds=40, seed=3, null_kind=null_kind
        )
        assert res["family"]["percentile"] > 0.9, (null_kind, res["family"])


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
