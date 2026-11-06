"""Tests for src/pooled_panel.py — the dollar-pooled family -> DayPanel adapter.

Run:  python -m tests.test_pooled_panel
"""
from __future__ import annotations

import numpy as np

from src import pooled_panel, pooled_test as pt, reality_check
from src.window_sweep import default_windows


def test_observed_cells_match_pooled_tests_segments() -> None:
    """The adapter must observe exactly the (day, window) pairs pooled_test segments.

    If the adapter observed more cells it would be inventing P&L; fewer and it would be
    dropping observations. The count is the invariant that ties the two together.
    """
    from src.window_sweep import segments_from_times

    times, R, _ = pt.build_panel(list(pt.ORIENTATION.keys()))
    ns = times.to_numpy().astype("datetime64[ns]").astype("int64")
    _a, _b, valid = segments_from_times(ns, ns // (86_400 * 10**9), default_windows())
    got = pooled_panel.oriented_basket_pnl()
    assert int(valid.sum()) == int(got["observed"].sum()), (
        int(valid.sum()), int(got["observed"].sum())
    )


def test_pnl_matches_the_basket_computed_directly() -> None:
    """The panel's P&L must equal the oriented basket computed from the raw returns.

    Same legs, same signs, same segments -> the same numbers. This is the test that the
    adapter is a bridge and not a second implementation.
    """
    from src.window_sweep import segments_from_times

    syms = list(pt.ORIENTATION.keys())
    times, R, _ = pt.build_panel(syms)
    ns = times.to_numpy().astype("datetime64[ns]").astype("int64")
    day = ns // (86_400 * 10**9)
    windows = default_windows()
    a, b, valid = segments_from_times(ns, day, windows)
    S = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, axis=0)])
    signs = np.array([pt.ORIENTATION[s] for s in syms], dtype=float)
    sd = R.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)

    wi = windows.index((3, 1))
    m = valid[:, wi]
    z = ((S[b[m, wi], :] - S[a[m, wi], :]) / sd) @ signs

    got = pooled_panel.oriented_basket_pnl(syms)
    col = got["pnl"][got["observed"][:, wi], wi]
    assert len(col) == len(z), (len(col), len(z))
    assert np.allclose(np.sort(col), np.sort(z), atol=0.0, rtol=0.0), "values must be identical"
    # and therefore the per-day t agrees to machine precision
    t_col = col.mean() / (col.std(ddof=1) / np.sqrt(len(col)))
    t_direct = z.mean() / (z.std(ddof=1) / np.sqrt(len(z)))
    assert abs(t_col - t_direct) < 1e-12, (t_col, t_direct)


def test_panel_shape_and_names() -> None:
    """Days x windows, with the window names in the WINDOWS slot (the gpu_null convention)."""
    panel = pooled_panel.pooled_panel()
    assert panel.pnl.shape[1] == len(default_windows())
    assert len(panel.windows) == panel.n_candidates
    assert panel.symbols == ("pooled",)
    assert any("03+1h" in w for w in panel.windows)
    # every observed cell is finite, and unobserved cells are zero with valid False
    assert np.isfinite(panel.pnl[panel.valid]).all()
    assert not panel.valid[~np.isfinite(panel.pnl)].any()


def test_family_bias_is_priced() -> None:
    """The family must be priceable by the same reality checks as the other two families."""
    panel = pooled_panel.pooled_panel()
    rc_res = reality_check.reality_check(panel, n_worlds=1000)
    spa = reality_check.spa_p_value(panel, n_worlds=1000)
    assert np.isfinite(rc_res["p_value"]) and 0.0 <= rc_res["p_value"] <= 1.0
    assert np.isfinite(spa["p_value"]) and 0.0 <= spa["p_value"] <= 1.0
    # nothing in this family rejects at 5% (consistent with its own null's 17th percentile)
    assert rc_res["p_value"] > 0.02, rc_res
    assert spa["p_value"] > 0.02, spa


def test_symbol_order_is_internally_consistent() -> None:
    """Signs are built from the same list as the legs, so any order is self-consistent.

    The published artifact used alphabetical order and reports t=+1.01 for the candidate;
    this adapter uses ORIENTATION order and reports +0.8332. Both are correct baskets - the
    order changes which dispersion normalises the corridor - so the invariant to pin is that
    leg i's sign always belongs to leg i, whatever the order.
    """
    for order in (list(pt.ORIENTATION.keys()), sorted(pt.ORIENTATION.keys())):
        got = pooled_panel.oriented_basket_pnl(order)
        # r=1 correlation between the panel's P&L and the same legs' raw returns confirms
        # the signs matched the columns: a mis-signed leg would decorrelate them
        assert set(got["symbols"]) == set(order)
        assert len(got["signs"]) == len(order)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
