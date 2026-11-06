"""Turn the pair-sweep's trade series into a DayPanel, so its bias can be priced.

Why this adapter exists
-----------------------
``src/reality_check.py`` prices selection bias on a ``src.gpu_null.DayPanel`` — one shared
days × candidates matrix. The session-window family already produces that shape. The pair
family does not: ``simulate_pair`` returns per-trade P&L with no calendar axis, because a
trade's P&L is realised over a *span* of bars and the family's own null (sequential
simulation) works on trades rather than days.

This module bridges the two: each trade is assigned to the calendar day it ENTERED on, so a
day's value is the P&L of the trades that day's rule took (zero when none). That is the
realised daily P&L of the strategy, which is what a reality check needs. Trades spanning
the day boundary are booked to their entry day on purpose — a trade's P&L belongs to the
decision that opened it, not to the day it happened to close, and the alternative would
book the same P&L twice across two days.

The panel this builds is then priced by ``reality_check.reality_check`` and
``reality_check.spa_p_value`` exactly like the session-window family's.

Usage::

    from src import pair_panel, pair_sweep
    panel = pair_panel.pair_sweep_panel(pairs)
    res = reality_check.reality_check(panel, n_worlds=2000)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gpu_null
from . import pair_sweep as ps


def trades_to_daily(trades: dict, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(day_ns, pnl) per trade: the calendar day each trade entered, and its P&L.

    ``trades`` is ``simulate_pair``'s output; ``times`` is the pair's bar index. A trade that
    closed on a later day is still booked to its entry day (see the module docstring).
    """
    ent = trades["entry"]
    pnl = trades["pnl"]
    if not len(ent):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    ns = times[ent].asi8
    day_ns = ns - (ns % (86_400 * 10**9))   # parenthesised: ns % 86_400 * 1e9 is (ns%86400)*1e9
    return day_ns.astype(np.int64), np.asarray(pnl, dtype=float)


def pair_sweep_panel(pairs: list[ps.PairData], *, z_entry_grid=None, z_exit_grid=None) -> gpu_null.DayPanel:
    """One DayPanel for the whole pair family: days × (pair, z_entry, z_exit).

    Columns are every (pair, z_entry, z_exit) combination the family scans, which is the
    candidate set whose selection bias needs pricing. Days are the union of every trade's
    entry day across all candidates, so a candidate's column is NaN-free on its own days and
    zero elsewhere — the same convention as ``gpu_null.day_pnl_panel``'s valid mask.
    """
    z_in = list(z_entry_grid if z_entry_grid is not None else ps.Z_ENTRY)
    z_out = list(z_exit_grid if z_exit_grid is not None else ps.Z_EXIT)
    columns: list[tuple[str, int, float]] = []
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for p in pairs:
        beta, z = ps.rolling_beta_z(p.la, p.lb)
        for z0 in z_in:
            for zx in z_out:
                tr = ps.simulate_pair(p.la, p.lb, beta, z, z0, zx)
                day, pnl = trades_to_daily(tr, p.times)
                columns.append((p.name, z0, zx))
                series.append((day, pnl))

    all_days = np.unique(np.concatenate([d for d, _ in series if len(d)] if any(len(d) for d, _ in series) else [np.zeros(1, dtype=np.int64)]))
    pnl = np.zeros((len(all_days), len(columns)), dtype=np.float64)
    valid = np.zeros(pnl.shape, dtype=bool)
    for j, (day, values) in enumerate(series):
        if not len(day):
            continue
        rows = np.searchsorted(all_days, day)
        assert np.array_equal(all_days[rows], day), "day axis must lie on the shared grid"
        for r, v in zip(rows, values):
            pnl[r, j] += v          # a day with several trades sums them
            valid[r, j] = True
    windows = tuple(("pair", 0) for _ in columns)
    return gpu_null.DayPanel(
        all_days,
        tuple(f"{c[0]}|z{c[1]:.0f}/{c[2]:.1f}" for c in columns),
        windows,
        pnl,
        valid,
    )
