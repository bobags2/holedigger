"""Map the dollar-pooled family's basket P&L onto a DayPanel, so its bias is priceable.

Why this adapter exists
-----------------------
``pooled_test`` is a *pre-registered single-hypothesis* test: one orientation (USDX long,
the seven dollar crosses short), one candidate window, and a family null over all 69 windows
built from **joint** row permutations of the aligned panel — the same time index applied to
every column, so cross-symbol correlation is preserved.

That null answers the family's own question and answers it properly. What it does not answer
is the project-level question ``src/reality_check`` asks: *is this candidate extreme against
everything else the project tried?* For that the family needs a ``DayPanel`` — a shared
days × candidates matrix — and the pooled family's statistic is not one: it is a basket
t-stat, not a per-day P&L series.

This module builds the panel the reality check can price:

* rows are trading days, as for every other family;
* columns are the **69 session windows**, each column holding the oriented equal-risk
  basket's P&L on that day — i.e. ``z[d] = sum_k sign_k * P[d,k] / sd_k``, the per-day
  contribution ``pooled_test._pooled_t`` averages;
* the sign orientation is the pre-registered one, fixed before the run, so the panel is not
  a search over orientations (that would be a new hypothesis with a new multiplicity).

Because each column is a per-day series, the sign-flip null in ``gpu_null`` is a valid null
for it, and it *preserves cross-symbol co-movement* by construction — a day's whole basket
is flipped together, which is the same protection the joint row permutation gives.

Usage::

    from src import pooled_panel, pooled_test, reality_check
    panel = pooled_panel.pooled_panel(symbols=pooled_test.ORIENTATION)
    res = reality_check.reality_check(panel, n_worlds=2000)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, gpu_null, pooled_test
from .window_sweep import default_windows, segments_from_times


def _oriented_symbols() -> list[str]:
    """The pre-registered orientation's symbols, in the order their signs apply."""
    return list(pooled_test.ORIENTATION.keys())


def oriented_basket_pnl(symbols: list[str] | None = None) -> dict:
    """Per-day basket P&L for every session window, plus the days and the signs.

    Returns ``{"days": ndarray, "pnl": (D, W) float64, "windows": [...], "signs": [...]}``
    where ``pnl[d, w]`` is the oriented equal-risk basket's P&L on day ``d`` for window
    ``w`` — the per-day quantity whose mean over days is ``_pooled_t``'s numerator basis.
    """
    syms = symbols or _oriented_symbols()
    times, R, _prices = pooled_test.build_panel(syms)
    signs = np.array([pooled_test.ORIENTATION[s] for s in syms], dtype=float)

    ns = times.to_numpy().astype("datetime64[ns]").astype(np.int64)
    windows = default_windows()
    # ``pooled_test`` groups bars by CALENDAR DAY (bar_time with ns // 86_400e9 as the day
    # axis), so its segments are per-day, not per-bar. Matching that convention exactly is
    # what keeps this adapter's P&L identical to the original statistic's.
    day_of_bar = ns // (86_400 * 10**9)
    a, b, valid = segments_from_times(ns, day_of_bar, windows)

    S = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, axis=0)])
    k = R.shape[1]
    sd = R.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)          # a dead symbol contributes nothing, not NaN

    days = np.unique(day_of_bar)
    pnl = np.zeros((len(days), len(windows)), dtype=np.float64)
    mask = np.zeros(pnl.shape, dtype=bool)
    for w, (h, d) in enumerate(windows):
        m = valid[:, w]
        if not m.any():
            continue
        # one segment per calendar day for this window, so rows are unique per day
        P = S[b[m, w], :] - S[a[m, w], :]
        z = (P / sd) @ signs                  # the oriented, equal-risk basket, per day
        rows = np.searchsorted(days, day_of_bar[a[m, w]])
        assert np.array_equal(days[rows], day_of_bar[a[m, w]]), "day axis must lie on the grid"
        pnl[rows, w] = z
        mask[rows, w] = True
    return {
        "days": days,
        "pnl": pnl,
        "observed": mask,      # which (day, window) cells carry a real observation
        "windows": windows,
        "signs": signs,
        "symbols": syms,
    }


def pooled_panel(symbols: list[str] | None = None) -> gpu_null.DayPanel:
    """The dollar-pooled family as a DayPanel: days × session windows.

    ``symbols`` carries the family label and ``windows`` carries one entry per candidate
    window, which is the convention ``gpu_null``/``window_sweep`` expect (columns are
    symbol-major, then window-major, so a sector reshape stays valid). The pre-registered
    orientation is fixed (see the module docstring) — no orientation search, because that
    would be a new hypothesis with its own multiplicity.

    Note on the symbol order: ``build_panel`` concatenates the legs in the order it is
    given, and the signs are built from the same list, so any order is internally
    consistent. The published ``pooled_dollar.json`` was run with the legs in ALPHABETICAL
    order and gives a candidate t of +1.01; this adapter uses ``ORIENTATION`` order and
    gives +0.833 on identical P&L values. Both are right; they are different baskets
    (different legs' dispersions normalise the corridor differently), which is exactly why
    the order is stated rather than left implicit.
    """
    got = oriented_basket_pnl(symbols)
    names = tuple(f"pooled|{h:02d}+{d}h" for h, d in got["windows"])
    ok = np.isfinite(got["pnl"]) & got["observed"]
    return gpu_null.DayPanel(got["days"], ("pooled",), names, got["pnl"], ok)
