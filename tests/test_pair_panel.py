"""Tests for src/pair_panel.py — the pair trades -> DayPanel adapter.

Run:  python -m tests.test_pair_panel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import gpu_null, pair_panel, pair_sweep as ps, reality_check as rc


def test_trades_are_booked_to_their_entry_day() -> None:
    """A trade's P&L belongs to the day the decision was taken, not the day it closed.

    Booking by exit day would move P&L across a boundary the trade never saw; booking by
    entry day keeps a day's value equal to the strategy's realised P&L for that day's
    decisions.
    """
    times = pd.date_range("2026-01-05 00:00", periods=20, freq="1h")
    trades = {"entry": np.array([0, 5, 9]), "exit": np.array([4, 12, 19]),
              "pnl": np.array([1e-4, -2e-4, 3e-4])}
    day, pnl = pair_panel.trades_to_daily(trades, times)
    assert len(day) == 3 and len(pnl) == 3
    # entry days are the 5th, the 5th, and the 5th (hours 0, 5, 9 are all Jan 5)
    assert len(np.unique(day)) == 1, day
    # and the values survived the mapping in order
    assert np.allclose(pnl, [1e-4, -2e-4, 3e-4])


def test_multi_trade_days_sum_their_pnl() -> None:
    """A day that took two trades must carry both, not one of them."""
    n = 60 * 24  # a full day of hourly bars
    times = pd.date_range("2026-01-05 00:00", periods=n, freq="1h")
    trades = {"entry": np.array([0, 12]), "exit": np.array([5, 20]),
              "pnl": np.array([1e-4, 2e-4])}
    day, pnl = pair_panel.trades_to_daily(trades, times)
    # both trades entered on Jan 5 -> one day, summed
    assert len(np.unique(day)) == 1
    assert abs(pnl.sum() - 3e-4) < 1e-12


def test_pair_panel_has_one_column_per_candidate() -> None:
    """The panel's columns must be every (pair, z_entry, z_exit) the family scans."""
    pairs = [ps.build_pair("XAUUSD", "XAGUSD")]
    panel = pair_panel.pair_sweep_panel(pairs)
    n_expected = 1 * len(ps.Z_ENTRY) * len(ps.Z_EXIT)
    assert panel.n_candidates == n_expected, (panel.n_candidates, n_expected)
    assert len(panel.symbols) == n_expected
    assert len(panel.days) > 100, len(panel.days)


def test_pair_family_bias_is_priced_like_the_session_family() -> None:
    """The pair family must be priceable by the same reality-check machinery.

    This is the point of the adapter: without it the pair family's selection bias is
    unpriceable, because its per-trade P&L has no calendar axis for a reality check to
    share. With it, both families answer the same question with the same code.
    """
    pairs = [ps.build_pair("EURUSD", "GBPUSD")]
    panel = pair_panel.pair_sweep_panel(pairs)
    rc_res = rc.reality_check(panel, n_worlds=500)
    spa = rc.spa_p_value(panel, n_worlds=500)
    assert np.isfinite(rc_res["p_value"]) and 0.0 <= rc_res["p_value"] <= 1.0
    assert np.isfinite(spa["p_value"]) and 0.0 <= spa["p_value"] <= 1.0
    # on noise-free real data with no planted edge, neither check rejects at 5%
    assert rc_res["p_value"] > 0.02, rc_res
    assert spa["p_value"] > 0.02, spa


def test_panel_sums_are_consistent_with_simulate_pair() -> None:
    """The panel's per-candidate mean P&L must equal the sweep's per-candidate mean.

    The adapter books trades to entry days; the sweep averages over trades. Averaging the
    panel's valid cells must land on the same number, otherwise the adapter has lost or
    duplicated a trade.
    """
    p = ps.build_pair("XAUUSD", "XAGUSD")
    beta, z = ps.rolling_beta_z(p.la, p.lb)
    z0, zx = ps.Z_ENTRY[0], ps.Z_EXIT[0]
    tr = ps.simulate_pair(p.la, p.lb, beta, z, z0, zx)
    direct_mean = float(tr["pnl"].mean())
    panel = pair_panel.pair_sweep_panel([p])
    j = list(panel.symbols).index(f"{p.name}|z{z0:.0f}/{zx:.1f}")
    col = panel.pnl[panel.valid[:, j], j]
    panel_mean = float(col.mean())
    assert abs(panel_mean - direct_mean) < 1e-15, (panel_mean, direct_mean)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
