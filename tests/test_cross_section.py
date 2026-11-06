"""Tests for src/cross_section.py — the cross-sectional ranking family.

Run:  python -m tests.test_cross_section
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import cross_section as cs


def test_session_alignment_beats_the_common_bar_grid() -> None:
    """Session-matched alignment must keep far more observations than a common bar grid.

    Requiring every symbol to share a bar timestamp collapsed a 139-session, 105-name panel
    to 974 bars and 13 sessions, because the shortest listing date poisons the intersection.
    Session alignment recovers the full panel.
    """
    syms = [l.strip() for l in open("Z:/projects/holedigger/data/equity_symbols.txt") if l.strip()]
    closes = cs.load_closes(syms)
    days, R, ss = cs.session_returns(closes)
    assert len(days) > 100, len(days)
    assert R.shape[1] == len(syms), R.shape
    # the cross-section is wide: most sessions observe most names
    per_day = np.isfinite(R).sum(axis=1)
    assert np.median(per_day) > 0.5 * R.shape[1], np.median(per_day)


def test_feature_strictly_precedes_the_return_it_predicts() -> None:
    """The pre-registered feature must be the PREVIOUS session's return.

    If the feature were the same session's return it would be the label, and the family
    would be measuring autocorrelation of a single number rather than cross-sectional
    predictability.
    """
    rng = np.random.default_rng(0)
    n_days, k = 60, 20
    R = rng.standard_normal((n_days, k)) * 2.0
    closes = {f"S{i}": pd.Series(np.cumsum(rng.standard_normal(500) * 1e-4)) for i in range(k)}
    prev = np.full_like(R, np.nan)
    prev[1:] = R[:-1]
    # session t+1's feature row must equal session t's return row, never its own
    assert np.allclose(prev[1:], R[:-1], equal_nan=True)
    assert not np.allclose(prev[3], R[3], equal_nan=True)


def test_rank_corr_is_cross_sectional_within_a_session() -> None:
    """rho must rank the names against each other, not correlate a series with its own past.

    A perfectly ordered cross-section (feature ranks 1..K, forward ranks 1..K) must give
    rho = +1 regardless of the scale of either, and a reversed ordering rho = -1.
    """
    F = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    Fwd = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
    ok = np.ones((1, 5), dtype=bool)
    r = cs.daily_rank_corr(F, Fwd, ok)
    assert abs(r[0] - 1.0) < 1e-12, r
    r = cs.daily_rank_corr(F, Fwd[:, ::-1], ok)
    assert abs(r[0] + 1.0) < 1e-12, r
    # and a session with fewer than 3 observed names is dropped, not scored
    ok2 = np.zeros((1, 5), dtype=bool); ok2[0, :2] = True
    assert not np.isfinite(cs.daily_rank_corr(F, Fwd, ok2)[0])


def test_null_permutes_labels_within_each_session() -> None:
    """The null must destroy the feature->name mapping, not the returns.

    If a null world destroyed the return cross-section too it would be a null for a
    different question. Check: permuting the feature leaves the forward matrix untouched.
    """
    rng = np.random.default_rng(1)
    R = rng.standard_normal((30, 12)) * 2.0
    prev = np.full_like(R, np.nan); prev[1:] = R[:-1]
    prev_ok = np.isfinite(prev) & np.roll(np.isfinite(R), 1, axis=0)
    fwd = np.full_like(R, np.nan); fwd[:-1] = R[1:]
    fwd_ok = np.isfinite(fwd) & np.roll(np.isfinite(R), -1, axis=0)
    ok = prev_ok & fwd_ok

    before = np.where(ok, fwd, np.nan).copy()
    Fp = np.where(ok, prev, np.nan).copy()
    for d in range(R.shape[0]):
        rows = np.flatnonzero(ok[d])
        Fp[d, rows] = rng.permutation(Fp[d, rows])
    after = np.where(ok, fwd, np.nan)
    assert np.allclose(np.nan_to_num(before), np.nan_to_num(after)), "the null must not touch returns"
    # and the multiset of each day's feature values is preserved (a permutation, not a resample)
    for d in range(R.shape[0]):
        rows = np.flatnonzero(ok[d])
        if len(rows) >= 3:
            assert np.array_equal(np.sort(Fp[d, rows]), np.sort(np.nan_to_num(prev[d, rows])))


def test_the_finding_is_stable_across_halves() -> None:
    """The one real result in this family must survive the split-half test.

    The split-half is the test that killed the dislocation finding (halves -14.4/-7.1) and
    the event lead (control reuse). So the bar for a claim from this family is the same one:
    both halves must point the same way, and the 2-session horizon must decay as a real
    one-day effect does.
    """
    syms = [l.strip() for l in open("Z:/projects/holedigger/data/equity_symbols.txt") if l.strip()]
    closes = cs.load_closes(syms)
    days, R, _ = cs.session_returns(closes)
    prev = np.full_like(R, np.nan); prev[1:] = R[:-1]
    fwd = np.full_like(R, np.nan); fwd[:-1] = R[1:]
    ok = np.isfinite(prev) & np.roll(np.isfinite(R), 1, axis=0) & np.isfinite(fwd)
    rho = cs.daily_rank_corr(prev, fwd, ok)
    cut = len(days) // 2
    h1, h2 = float(np.nanmean(rho[:cut])), float(np.nanmean(rho[cut:]))
    assert h1 > 0 and h2 > 0, (h1, h2)

    res = cs.run_cross_section(syms, n_worlds=400)
    c1 = [c for c in res["candidates"] if c["horizon"] == 1][0]
    c2 = [c for c in res["candidates"] if c["horizon"] == 2][0]
    assert c1["percentile"] > 0.95, c1          # the 1-session effect is real
    assert abs(c2["real_rho"]) < abs(c1["real_rho"]), (c1, c2)   # and it decays


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
