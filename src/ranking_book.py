"""Price the cross-sectional ranking strategy as a real book, net of measured cost.

Why this module exists
----------------------
``cross_section`` established that the previous session's return predicts the next session's
cross-sectional ordering: rho +0.035, percentile 1.0000 over 4000 permutation worlds, stable
across halves. That is a *statistical* result. It is not yet a trading result, and the
difference is the whole discipline this project runs on: a ranking correlation says nothing
about what a long/short book earns after spread.

So this module turns the ranking into a book and prices it:

* each session, rank the names by the feature (previous session's return);
* go long the top quintile, short the bottom quintile, equal-weight;
* the book's P&L is the *spread* between the quintiles' forward returns, which is already a
  market-neutral construction — the market's own drift cancels;
* subtract the measured round-trip cost: the entry and exit spreads of every leg, from the
  per-bar data, so the cost is what a trade at those moments would actually pay.

The result is a daily P&L series, which is exactly what ``src.oos_eval`` prices: append the
column to the panel and the reality checks, the split-half, and the step-down adjusted
p-values all apply to the *strategy* rather than to the correlation.

Usage::

    from src import ranking_book
    res = ranking_book.run(symbols, quintile=0.2)
    print(res["gross_bp_per_day"], res["net_bp_per_day"], res["net_t"])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, cross_section as cs


def session_bounds(closes: dict[str, pd.Series]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(session_days, {symbol: entry/exit index range per session}) for the session grid."""
    out = {}
    for sym, s in closes.items():
        m = (s.index.hour >= cs.SESSION_START_HOUR) & (s.index.hour < cs.SESSION_END_HOUR)
        ss = s[m]
        if not len(ss):
            out[sym] = np.zeros((0, 2), dtype=np.int64)
            continue
        day = ss.index.normalize().asi8
        ud = np.unique(day)
        first = np.searchsorted(day, ud, side="left")
        last = np.searchsorted(day, ud, side="right") - 1
        out[sym] = np.column_stack([first, last])
    all_days = np.unique(np.concatenate(
        [closes[s].index.normalize().asi8 for s in closes] if closes else [np.zeros(1, np.int64)]
    ))
    return all_days, out


def spread_at_bars(closes: dict[str, pd.Series], bounds: dict[str, np.ndarray],
                   syms: list[str]) -> dict[str, np.ndarray]:
    """Per-symbol (entry spread, exit spread) in bp at each session's first/last bar."""
    out = {}
    for sym in syms:
        b = bounds.get(sym)
        if b is None or not len(b):
            out[sym] = np.zeros((0, 2))
            continue
        df = pd.read_parquet(config.PARQUET_DIR / f"{sym}_MT5_M1.parquet")
        sp = df.set_index("ts")["spread_med_bp"].resample("5min").last().dropna()
        # the same session mask the close series was built with
        m = (sp.index.hour >= cs.SESSION_START_HOUR) & (sp.index.hour < cs.SESSION_END_HOUR)
        ss = sp[m]
        if not len(ss):
            out[sym] = np.zeros((len(bounds[sym]), 2))
            continue
        e = ss.to_numpy()[bounds[sym][:, 0]]
        x = ss.to_numpy()[bounds[sym][:, 1]]
        out[sym] = np.column_stack([e, x])
    return out


def run(
    symbols: list[str],
    *,
    quintile: float = 0.2,
    n_worlds: int = 4000,
    seed: int = config.SEED,
) -> dict:
    """The ranking book: gross and net of measured spread, per session and in aggregate.

    Long the top ``quintile`` of the previous-session return ranking, short the bottom
    ``quintile``, equal-weight, held for one session. Cost is the measured entry + exit
    spread of every leg, so the net number is what the book would have earned.
    """
    closes = cs.load_closes(symbols)
    days, R, syms = cs.session_returns(closes)
    _all_days, bounds = session_bounds(closes)
    spreads = spread_at_bars(closes, bounds, syms)

    # ALIGNED, not per-symbol. spread_at_bars returns ONE ROW PER SYMBOL SESSION (48-139 rows
    # for a 139-day union: a symbol that missed sessions has fewer). The cost loop indexed it
    # with the UNION day index, so sp[d-1] read a DIFFERENT SESSION's spread - silently, since
    # 48 rows fits inside 139. The fix maps each symbol's own session days to their positions
    # in the union, so sp_by_col[leg][d] is that leg's entry/exit spread on session d.
    sp_by_col = []
    for s in syms:
        arr = spreads.get(s)
        if arr is None or not len(arr):
            sp_by_col.append(np.full((len(days), 2), np.nan))
            continue
        own_days = bounds.get(s)[:, 1] if bounds.get(s) is not None and len(bounds.get(s)) else np.zeros(0, dtype=np.int64)
        # bounds row i's bar index -> normalize to the day it belongs to
        sp = pd.read_parquet(config.PARQUET_DIR / f"{s}_MT5_M1.parquet").set_index("ts")
        m = (sp.index.hour >= cs.SESSION_START_HOUR) & (sp.index.hour < cs.SESSION_END_HOUR)
        own_ts = sp.index[m][own_days].normalize().asi8
        pos = np.searchsorted(days, own_ts)
        keep = (pos < len(days)) & (days[np.clip(pos, 0, len(days) - 1)] == own_ts)
        out = np.full((len(days), 2), np.nan)
        if keep.any():
            out[pos[keep]] = arr[keep]
        sp_by_col.append(out)
    # `prev` indexes the SAME union-day rows as the returns matrix, so d-1 is now correct
    spread_by_col = sp_by_col

    prev = np.full_like(R, np.nan)
    prev[1:] = R[:-1]
    fwd = np.full_like(R, np.nan)
    fwd[:-1] = R[1:]

    gross, cost, n_long, n_short = [], [], [], []
    k = R.shape[1]
    n_legs = max(2, int(round(k * quintile)))
    for d in range(1, len(days) - 1):
        f, r = prev[d], fwd[d]
        ok = np.isfinite(f) & np.isfinite(r)
        # rank ONLY the names observed that session. Sorting the unobserved names to the
        # bottom of the feature would make the short book "the names that did not trade",
        # which is not a position - and it NaN'd every session (measured: 6 of 21 shorts).
        obs = np.flatnonzero(ok)
        if len(obs) < max(3 * n_legs, 10):
            continue
        order = obs[np.argsort(f[obs], kind="stable")]     # ascending feature, observed only
        shorts, longs = order[:n_legs], order[-n_legs:]
        g = float(r[longs].mean() - r[shorts].mean())      # market-neutral long/short spread
        # THE COST, DEFINED BEFORE THE ENTRY. A leg costs its ENTRY spread plus its EXIT
        # spread, both measured on the SESSION THE TRADE IS HELD OVER (session d), for every
        # leg the book actually holds. Charging the average across legs makes cost_bp a
        # per-leg figure on the same scale as gross_bp, which is what makes
        # ``net = gross - cost`` a statement about the book.
        #
        # The version before this fix summed ``sp[d-1]`` - the PREVIOUS session's entry+exit,
        # both columns - and spread_at_bars returned per-symbol session rows indexed with the
        # union-day array, so it also read the wrong session's spreads. That produced 42.98 bp
        # where the median equity entry+exit is ~10 bp.
        c, legs = 0.0, 0
        for leg in list(longs) + list(shorts):
            sp = spread_by_col[leg]
            if len(sp) and 0 <= d < len(sp) and np.isfinite(sp[d]).all():
                c += float(sp[d][0]) + float(sp[d][1])
                legs += 1
        gross.append(g)
        cost.append(c / max(legs, 1))
        n_long.append(int(len(longs)))
        n_short.append(int(len(shorts)))

    gross = np.asarray(gross, float)
    cost = np.asarray(cost, float)
    net = gross - cost

    def tstat(a: np.ndarray) -> float:
        if len(a) < 5:
            return float("nan")
        sd = a.std(ddof=1)
        return float(a.mean() / (sd / np.sqrt(len(a)))) if sd > 0 else float("nan")

    # the null: the same book built on a world where the feature carries no information.
    # Permuting the feature within each session destroys the ranking while leaving every
    # session's return cross-section intact, so the null prices the BOOK, not the correlation.
    rng = np.random.default_rng(seed)
    null_net = np.empty(n_worlds)
    for w in range(n_worlds):
        Pp = prev.copy()
        for d in range(1, len(days) - 1):
            ok = np.isfinite(prev[d]) & np.isfinite(fwd[d])
            rows = np.flatnonzero(ok)
            if len(rows) >= max(3 * n_legs, 10):
                Pp[d, rows] = rng.permutation(prev[d, rows])
        g, c = [], []
        for d in range(1, len(days) - 1):
            f, r = Pp[d], fwd[d]
            obs = np.flatnonzero(np.isfinite(f) & np.isfinite(r))
            if len(obs) < max(3 * n_legs, 10):
                continue
            order = obs[np.argsort(f[obs], kind="stable")]
            shorts, longs = order[:n_legs], order[-n_legs:]
            g.append(float(r[longs].mean() - r[shorts].mean()))
            cst, nlegs = 0.0, 0
            for leg in list(longs) + list(shorts):
                sp = spread_by_col[leg]
                if len(sp) and 0 <= d < len(sp) and np.isfinite(sp[d]).all():
                    cst += float(sp[d][0]) + float(sp[d][1])
                    nlegs += 1
            c.append(cst / max(nlegs, 1))
        null_net[w] = float(np.nanmean(np.asarray(g) - np.asarray(c)))

    return {
        "config": {
            "symbols": len(syms), "quintile": quintile, "legs_per_side": n_legs,
            "worlds": n_worlds, "seed": seed, "session_utc": [cs.SESSION_START_HOUR, cs.SESSION_END_HOUR],
        },
        "n_sessions": int(len(gross)),
        "gross_bp_per_session": round(float(gross.mean()), 5),
        "cost_bp_per_session": round(float(cost.mean()), 5),
        "net_bp_per_session": round(float(net.mean()), 5),
        "gross_t": round(tstat(gross), 3),
        "net_t": round(tstat(net), 3),
        "null_net_mean": round(float(np.nanmean(null_net)), 5),
        "null_net_p95": round(float(np.nanpercentile(null_net, 95)), 5),
        "net_percentile": round(float((null_net < net.mean()).mean()), 4),
        "hit_rate": round(float((net > 0).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--quintile", type=float, default=0.2)
    ap.add_argument("--worlds", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "ranking_book.json")
    args = ap.parse_args()
    config.ensure_dirs()

    src = Path(args.symbols_file) if args.symbols_file else (config.DATA_DIR / "equity_symbols.txt")
    syms = [l.strip() for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    res = run(syms, quintile=args.quintile, n_worlds=args.worlds)

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    r = res
    print(f"ranking book (long/short top-vs-bottom {args.quintile:.0%}, "
          f"{r['config']['legs_per_side']} legs/side, {r['n_sessions']} sessions)")
    print(f"  gross   {r['gross_bp_per_session']:+.4f} bp/session  t={r['gross_t']:+.2f}")
    print(f"  cost    {r['cost_bp_per_session']:+.4f} bp/session")
    print(f"  NET     {r['net_bp_per_session']:+.4f} bp/session  t={r['net_t']:+.2f}")
    print(f"  null    mean {r['null_net_mean']:+.4f}  p95 {r['null_net_p95']:+.4f}"
          f"  -> percentile {r['net_percentile']:.3f}")
    print(f"  hit rate {r['hit_rate']:.0%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
