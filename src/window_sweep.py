"""First registered real sweep: fixed session-window drift.

A candidate is (symbol, entry_hour, hold_hours): enter at the close of the first 5-min bar
at/after ``entry_hour`` UTC, exit at the close of the last bar at/before
``entry_hour + hold_hours``, same calendar day, exit capped at 20:00 UTC so no trade
crosses the daily break (probed: gaps start at 21:00+ UTC on this feed).

Statistic per candidate: |t| of the mean per-trade log return. The family max is calibrated
against per-symbol permutation surrogates (mean_block=1): the same procedure run on worlds
with no time-of-day structure. Costs come from src/costs.py; the headline number is
``break_even_spread_bp`` = gross edge - commission, i.e. how much spread the candidate can
pay per round trip and still break even.

Discipline: every candidate is registered in the hypothesis registry before evaluation.

Usage::

    python -m src.window_sweep [--worlds 200] [--symbols EURUSD ...]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, costs, noise_floor
from .registry import Registry
from .surrogates import stationary_bootstrap_indices

HOLDS = [1, 2, 4, 8]
MAX_EXIT_HOUR = 20  # UTC


@dataclass
class Series:
    symbol: str
    r: np.ndarray  # length n; r[0] = 0, r[j] = log(close[j] / close[j-1])
    bar_time: np.ndarray  # int64 ns, per bar
    bar_day: np.ndarray  # int64 days-since-epoch, per bar
    price: float  # last close
    spread_bp: np.ndarray | None = None  # per-bar spread in bp; None for mid-only sources


def series_from_bars(symbol: str, times, closes, spread_bp=None) -> Series:
    t = pd.DatetimeIndex(times)
    close = np.asarray(closes, dtype=float)
    r = np.empty(len(close), dtype=float)
    r[0] = 0.0
    r[1:] = np.diff(np.log(close))
    ns = t.to_numpy().astype("datetime64[ns]").astype("int64")
    spread = None if spread_bp is None else np.asarray(spread_bp, dtype=float)
    return Series(symbol, r, ns, ns // (86_400 * 10**9), float(close[-1]), spread)


def build_series(symbol: str) -> Series:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_M1.parquet").set_index("ts")
    m5 = df.resample("5min").last().dropna()
    return series_from_bars(symbol, m5.index, m5["close"].to_numpy())


def build_mt5_series(symbol: str) -> Series:
    """M1 mid bars from MT5 ticks, carrying per-bar spread for per-trade costing."""
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    return series_from_bars(symbol, df["ts"], df["close"].to_numpy(), df["spread_med_bp"].to_numpy())


def default_windows(holds: list[int] = HOLDS, max_exit: int = MAX_EXIT_HOUR) -> list[tuple[int, int]]:
    return [(h, d) for d in holds for h in range(0, max_exit - d + 1)]


def segments_from_times(
    bar_time: np.ndarray, bar_day: np.ndarray, windows: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, b, valid): bar indices of entry/exit per (day, window). Invalid entries are junk."""
    days = np.unique(bar_day)
    n_days, n_win = len(days), len(windows)
    a = np.zeros((n_days, n_win), dtype=np.int64)
    b = np.zeros((n_days, n_win), dtype=np.int64)
    entry_cut = np.array([h * 3600 for h, _ in windows], dtype=np.int64) * 10**9
    exit_cut = np.array([(h + d) * 3600 for h, d in windows], dtype=np.int64) * 10**9
    starts = np.searchsorted(bar_day, days, side="left")
    ends = np.searchsorted(bar_day, days, side="right")
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        times = bar_time[lo:hi] - days[i] * 86_400 * 10**9
        a[i] = lo + np.searchsorted(times, entry_cut, side="left")
        b[i] = lo + np.searchsorted(times, exit_cut, side="right") - 1
    valid = (b > a) & (b - a >= 2)
    a = np.clip(a, 0, len(bar_time) - 1)
    b = np.clip(b, 0, len(bar_time) - 1)
    return a, b, valid


def window_segments(s: Series, windows: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return segments_from_times(s.bar_time, s.bar_day, windows)


def pnl_matrix(S: np.ndarray, a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pnl = S[b] - S[a]
    return np.where(valid, pnl, np.nan)


def window_stats(pnl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, t, n) per column; NaN-safe."""
    ok = ~np.isnan(pnl)
    n = ok.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(pnl, axis=0) / n
        var = np.nansum((pnl - mean) ** 2, axis=0) / (n - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        t = mean / (std / np.sqrt(n))
    return mean, t, n


def _tstat_of(x: np.ndarray) -> float:
    """t-stat of a per-trade P&L series; NaN for degenerate inputs."""
    if len(x) < 5:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(len(x))))


def _window_stats_of(
    s: "Series", seg_sym: tuple, idx: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, t, n) per window under bar arrangement ``idx`` (``None`` = the real series).

    The real path and every null world are built through this one function, so a
    candidate's null statistic cannot drift from its real statistic.
    """
    r = s.r if idx is None else s.r[idx]
    return window_stats(pnl_matrix(np.cumsum(r), *seg_sym))


def _t_of(s: "Series", seg_sym: tuple, idx: np.ndarray | None = None) -> np.ndarray:
    """Per-window t only; the t-view of ``_window_stats_of`` used by the null loops."""
    return _window_stats_of(s, seg_sym, idx)[1]


def _day_structure(s: "Series") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = np.unique(s.bar_day)
    starts = np.searchsorted(s.bar_day, days, side="left")
    lengths = np.diff(np.append(starts, len(s.bar_day)))
    return days, starts, lengths


def _shared_day_grid_ok(series: dict[str, "Series"]) -> bool:
    """All series must present the same days with the same bar count per day."""
    ref = next(iter(series.values()))
    d0, _s0, l0 = _day_structure(ref)
    # ``intraday_slot_null_masks`` writes exactly L = lengths[0] bars per day, so the
    # counts must also be uniform across days: a series whose days disagree with each
    # other would get masks that index the wrong bars.
    if len(l0) == 0 or not np.all(l0 == l0[0]):
        return False
    for s in series.values():
        d, _s, l = _day_structure(s)
        if len(d) != len(d0) or not np.array_equal(d, d0):
            return False
        if not np.array_equal(l, l0):
            return False
    return True


def intraday_slot_null_masks(
    series: dict[str, "Series"], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """One null world shared by every symbol: shuffle day order and intraday slot usage.

    Destroyed: clock-time alignment. Destination slot j receives the source day's
    ``slot_order[j]`` bar, so a window at UTC hour h samples an arbitrary set of intraday
    returns rather than the same clock hour every day.

    Preserved: each symbol's marginal return distribution (a pure reindexing, never
    resampling); day-boundary bars — ``slot_order[0] == 0`` always, so the overnight /
    weekend gap return sitting in each day's first bar can never be moved into an intraday
    window that cannot contain it; and the cross-sectional composition of every bar row —
    all names apply the same ``day_order`` and ``slot_order``, so they keep co-moving through
    the market factor. That last property is what the sector null needs: the old per-symbol
    permutation made real t's (correlated) face null t's (independent), whose cross-symbol
    sd is much wider, so pooled ``mean(t_i)/(sd(t_i)/sqrt(N))`` rejected 50-70% of pure
    noise worlds instead of ~5%.

    Callers must align universes first: see ``_shared_day_grid_ok``.
    """
    ref = next(iter(series.values()))
    days, _starts, lengths = _day_structure(ref)
    L = int(lengths[0])
    day_order = stationary_bootstrap_indices(len(days), 1.0, rng)
    # slot 0 pinned: gaps stay at day boundaries; the rest are freely reshuffled.
    slot_order = np.concatenate(([0], 1 + rng.permutation(L - 1))) if L > 2 else np.arange(L)
    masks = {}
    for sym, s in series.items():
        starts = np.searchsorted(s.bar_day, days, side="left")
        idx = np.empty(len(s.bar_day), dtype=np.int64)
        for dest_day, src_day in enumerate(day_order):
            lo = starts[dest_day]
            idx[lo : lo + L] = starts[src_day] + slot_order
        masks[sym] = idx
    return masks


def run_sweep(
    series: dict[str, Series],
    windows: list[tuple[int, int]],
    n_worlds: int,
    seed: int = config.SEED,
    allow_shared_world: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    syms = list(series)
    n_win = len(windows)

    seg = {sym: window_segments(s, windows) for sym, s in series.items()}
    real_mean, real_t, real_n = {}, {}, {}
    for sym, s in series.items():
        real_mean[sym], real_t[sym], real_n[sym] = _window_stats_of(s, seg[sym])

    # Per-trade spread costing: a trade crosses half a spread on each side, measured at
    # the actual entry and exit bars. Only sources with a spread column (MT5 ticks) get it.
    net_by_key: dict[tuple, dict] = {}
    for sym, s in series.items():
        if s.spread_bp is None:
            continue
        a, b, valid = seg[sym]
        S = np.cumsum(s.r)
        for wi, (h, d) in enumerate(windows):
            m = valid[:, wi]
            e, x = a[m, wi], b[m, wi]
            if not len(e):
                continue
            gross = (S[x] - S[e]) * 1e4
            cost = (s.spread_bp[e] + s.spread_bp[x]) / 2.0
            sgn = np.sign(gross.mean())  # trade the profitable direction
            net = sgn * gross - cost
            net_by_key[(sym, h, d)] = {
                "net_edge_bp": float(net.mean()),
                "net_t": _tstat_of(net),
                "cost_bp": float(cost.mean()),
            }

    # One null world must keep the cross-section co-moving (shared day AND slot order) or
    # the pooled sector z compares correlated real t's against independent null t's and
    # is significant on pure noise. The shared path needs every symbol on one bar grid;
    # on ragged panels (different histories, missing minutes) fall back to the per-symbol
    # permutation and say so in the result.
    shared_ok = allow_shared_world and _shared_day_grid_ok(series)
    null_kind = "shared-intraday" if shared_ok else "per-symbol"

    null_t = np.empty((n_worlds, len(syms) * n_win))
    for w in range(n_worlds):
        if shared_ok:
            masks = intraday_slot_null_masks(series, rng)
            for i, sym in enumerate(syms):
                null_t[w, i * n_win : (i + 1) * n_win] = _t_of(series[sym], seg[sym], masks[sym])
        else:
            col = 0
            for sym, s in series.items():
                r_sh = s.r.copy()
                r_sh[1:] = s.r[1:][stationary_bootstrap_indices(len(s.r) - 1, 1.0, rng)]
                pnl = pnl_matrix(np.cumsum(r_sh), *seg[sym])
                _, t, _ = window_stats(pnl)
                null_t[w, col : col + n_win] = t
                col += n_win

    edge_bp = np.concatenate([real_mean[sym] for sym in syms]) * 1e4
    t_real = np.concatenate([real_t[sym] for sym in syms])
    n_real = np.concatenate([real_n[sym] for sym in syms])
    cand_sym = [sym for sym in syms for _ in windows]
    cand_win = [win for _ in syms for win in windows]

    abs_null = np.abs(null_t)
    abs_real = np.abs(t_real)
    # Session-bounded symbols leave all-NaN candidates (windows outside their session);
    # they must not poison the family maximum or the percentiles.
    null_max = np.nanmax(abs_null, axis=1)
    real_max = float(np.nanmax(abs_real))
    percentile = float((null_max < real_max).mean())
    z = float((real_max - null_max.mean()) / null_max.std(ddof=1)) if n_worlds > 1 else float("nan")
    per_candidate_pct = (abs_null < abs_real[None, :]).mean(axis=0)
    per_candidate_pct[~np.isfinite(abs_real)] = np.nan

    # Sector-level test: for each window, the cross-sectional mean t (equal-weight basket),
    # calibrated against the same statistic computed inside every null world. The worlds
    # must share one day/slot index across symbols (null_kind), otherwise the real t's —
    # correlated through the market factor — face independent null t's and the pooled z
    # is significant on pure noise.
    sector = {}
    sector_null_z = []
    sector_real_z = []
    if len(syms) > 1:
        null_grid = null_t.reshape(n_worlds, len(syms), n_win)
        for wi, (h, d) in enumerate(windows):
            reals = np.array([real_t[sym][wi] for sym in syms])
            reals = reals[np.isfinite(reals)]
            if len(reals) < 10:
                continue
            real_z = float(reals.mean() / (reals.std(ddof=1) / np.sqrt(len(reals))))
            null_z = np.empty(n_worlds)
            for w in range(n_worlds):
                vals = null_grid[w, :, wi]
                vals = vals[np.isfinite(vals)]
                null_z[w] = vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)))
            sector[f"{h:02d}+{d}h"] = {
                "real_z": real_z,
                "null_z_p95": float(np.percentile(np.abs(null_z), 95)),
                "percentile": float((np.abs(null_z) < abs(real_z)).mean()),
                "n_symbols": int(len(reals)),
                "null_kind": null_kind,
            }
            sector_null_z.append(np.abs(null_z))
            sector_real_z.append(abs(real_z))

    # The scan's headline is its most extreme window, so the null it must clear is the
    # same maximum computed inside each world. A per-window p95 is the wrong reference
    # for the best window: scanning ~20 correlated windows against ~20 marginal 95th
    # percentiles rejects a correct null roughly half the time (measured 56% at N=25
    # worlds vs 12% for the max-calibrated p95 below).
    sector_family = {}
    if sector_null_z:
        null_max_z = np.column_stack(sector_null_z).max(axis=1)  # per world
        real_max_z = float(np.max(sector_real_z))
        sector_family = {
            "real_max_abs_z": real_max_z,
            "null_max_abs_z_mean": float(null_max_z.mean()),
            "null_max_abs_z_p95": float(np.percentile(null_max_z, 95)),
            "percentile": float((null_max_z < real_max_z).mean()),
            "n_windows": int(len(sector_null_z)),
        }

    # Half-sample split (by day) for the top candidates later
    half_t, half_edge = {}, {}
    for sym, s in series.items():
        days = np.unique(s.bar_day)
        cut = np.median(days)
        pnl = pnl_matrix(np.cumsum(s.r), *seg[sym])
        for k, mask in enumerate([days <= cut, days > cut]):
            m, t, _ = window_stats(pnl[mask])
            half_edge[(sym, k)] = m * 1e4
            half_t[(sym, k)] = t

    rows = []
    for j in range(len(cand_sym)):
        sym = cand_sym[j]
        h, d = cand_win[j]
        nb = net_by_key.get((sym, h, d))
        comm = costs.commission_round_trip_bp(sym, series[sym].price)
        spread = costs.ADOPTED_SPREAD_BP.get(sym, 0.0)
        gross = abs(edge_bp[j])  # the tradeable direction is sign(edge)
        be = gross if comm is None else gross - comm
        net = None if comm is None else gross - comm
        net_tape = None if comm is None else gross - comm - spread
        rows.append(
            {
                "symbol": sym,
                "entry_hour_utc": h,
                "hold_hours": d,
                "edge_bp": float(edge_bp[j]),
                "t": float(t_real[j]),
                "n": int(n_real[j]),
                "own_null_percentile": float(per_candidate_pct[j]),
                "commission_rt_bp": comm,
                "spread_rt_bp": spread,
                "break_even_spread_bp": be,
                "net_bp_spread0": net,
                "net_bp_tape": net_tape,
                "net_spread_edge_bp": None if nb is None else nb["net_edge_bp"],
                "net_spread_t": None if nb is None else nb["net_t"],
                "avg_half_spread_cost_bp": None if nb is None else nb["cost_bp"],
                "t_first_half": float(half_t[(sym, 0)][windows.index((h, d))]),
                "t_second_half": float(half_t[(sym, 1)][windows.index((h, d))]),
            }
        )
    rows.sort(key=lambda r: -abs(r["t"]) if np.isfinite(r["t"]) else 1.0)

    replication = {}
    for row in rows[:3]:
        key = f"{row['entry_hour_utc']:02d}:00 +{row['hold_hours']}h"
        replication[key] = {}
        for sym in syms:
            j = (syms.index(sym)) * n_win + windows.index((row["entry_hour_utc"], row["hold_hours"]))
            replication[key][sym] = {"t": float(t_real[j]), "edge_bp": float(edge_bp[j])}

    floor = noise_floor.expected_max_t(len(cand_sym), n_sims=400, rng=rng)["mean"]
    return {
        "config": {
            "family": "session_window_v1",
            "symbols": syms,
            "windows": len(windows),
            "candidates": len(cand_sym),
            "worlds": n_worlds,
            "max_exit_hour_utc": MAX_EXIT_HOUR,
            "seed": seed,
        },
        "null": {
            "family_max_mean": float(null_max.mean()),
            "family_max_p95": float(np.percentile(null_max, 95)),
            "family_max_max": float(null_max.max()),
            "analytic_expected_max": float(floor),
        },
        "null_kind": null_kind,
        "family": {
            "real_max_abs_t": real_max,
            "percentile": percentile,
            "z": z,
            "top_real": rows[0],
        },
        "candidates": rows,
        "replication": replication,
        "sector": sector,
        "sector_family": sector_family,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    ap.add_argument("--symbols-file", default=None, help="path to a newline-separated symbol list")
    ap.add_argument("--mt5-done", action="store_true",
                    help="run on every symbol whose MT5 ingest state is complete")
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--source", choices=["ctrader", "mt5"], default="ctrader",
                    help="ctrader mid parquet, or MT5 tick-derived parquet with per-bar spreads")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "window_sweep.json")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.symbols_file:
        args.symbols = [
            line.strip()
            for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.mt5_done:
        done, empty = [], []
        for p in sorted(config.DATA_DIR.glob("mt5_state_*.json")):
            state = json.loads(p.read_text(encoding="utf-8"))
            if not state.get("done") or state.get("bars", 0) < 5000:
                continue
            # coverage from the parquet itself: a done state with a short span is a stale pull
            pq = config.PARQUET_DIR / f"{state['symbol']}_MT5_M1.parquet"
            ok = False
            if pq.exists():
                try:
                    ts = pd.read_parquet(pq, columns=["ts"])["ts"]
                    have = (ts.iloc[-1] - ts.iloc[0]).days
                    want = (pd.Timestamp(state["until"]) - pd.Timestamp(state["start"])).days
                    ok = have >= 0.8 * want
                except (KeyError, ValueError, FileNotFoundError):
                    ok = False
            (done if ok else empty).append(state["symbol"])
        if not done:
            raise SystemExit("no completed MT5 symbols; run src.mt5_ingest first")
        args.symbols = done
        print(f"running on {len(done)} completed MT5 symbols"
              + (f" (skipped {len(empty)} with no data: {', '.join(empty)})" if empty else ""))
    windows = default_windows()
    build = build_mt5_series if args.source == "mt5" else build_series
    series = {sym: build(sym) for sym in args.symbols}

    # Register every candidate before anything is evaluated.
    reg = Registry(config.REGISTRY_PATH)
    for sym in series:
        for h, d in windows:
            reg.register(
                {"family": "session_window_v1", "symbol": sym, "entry_hour_utc": h, "hold_hours": d},
                generator="window_sweep",
                generator_version="1",
            )

    result = run_sweep(series, windows, args.worlds)

    for row in result["candidates"]:
        hid = reg.register(
            {
                "family": "session_window_v1",
                "symbol": row["symbol"],
                "entry_hour_utc": row["entry_hour_utc"],
                "hold_hours": row["hold_hours"],
            },
            generator="window_sweep",
            generator_version="1",
        )
        reg.record_evaluation(
            hid,
            {
                "edge_bp": round(row["edge_bp"], 5),
                "t": round(row["t"], 3),
                "n": row["n"],
                "own_null_percentile": round(row["own_null_percentile"], 4),
                "break_even_spread_bp": None if row["break_even_spread_bp"] is None else round(row["break_even_spread_bp"], 5),
                "net_bp_spread0": None if row["net_bp_spread0"] is None else round(row["net_bp_spread0"], 5),
                "net_bp_tape": None if row["net_bp_tape"] is None else round(row["net_bp_tape"], 5),
                "net_spread_edge_bp": None if row["net_spread_edge_bp"] is None else round(row["net_spread_edge_bp"], 5),
                "net_spread_t": None if row["net_spread_t"] is None else round(row["net_spread_t"], 3),
            },
            {"null": {"mean_block": 1.0, "worlds": args.worlds, "kind": result["null_kind"]},
             "source": args.source, "symbols": len(series),
             "spread": "adopted per symbol (tape 2026-10-05)"},
        )

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    c = result["config"]
    print(f"family {c['family']}: {c['candidates']} candidates, {c['windows']} windows x {len(c['symbols'])} symbols")
    print(f"\nnull ({result['null_kind']}, {c['worlds']} worlds): family max |t| mean {result['null']['family_max_mean']:.2f}"
          f" | p95 {result['null']['family_max_p95']:.2f} | max {result['null']['family_max_max']:.2f}"
          f" | analytic {result['null']['analytic_expected_max']:.2f}")
    print(f"real family max |t| = {result['family']['real_max_abs_t']:.2f}"
          f" -> percentile {100 * result['family']['percentile']:.1f}%, z = {result['family']['z']:.2f}\n")

    hdr = (f"{'symbol':>7} {'window':>12} {'edge bp':>9} {'t':>7} {'n':>6} {'own%':>6}"
           f" {'BE spr':>7} {'net@tape':>9} {'t1':>6} {'t2':>6}")
    print(hdr)
    print("-" * len(hdr))
    for row in result["candidates"][:15]:
        be = "n/a" if row["break_even_spread_bp"] is None else f"{row['break_even_spread_bp']:.3f}"
        net = "n/a" if row["net_bp_tape"] is None else f"{row['net_bp_tape']:.3f}"
        win = f"{row['entry_hour_utc']:02d}+{row['hold_hours']}h"
        print(
            f"{row['symbol']:>7} {win:>12} {row['edge_bp']:>9.4f} {row['t']:>7.2f} {row['n']:>6}"
            f" {row['own_null_percentile']:>6.3f} {be:>7} {net:>9} {row['t_first_half']:>6.2f} {row['t_second_half']:>6.2f}"
        )

    netrows = [r for r in result["candidates"] if r["net_spread_edge_bp"] is not None]
    if netrows:
        print("\ntop candidates net of per-trade spread (half-spread each side, at the actual bars):")
        for row in sorted(netrows, key=lambda r: -abs(r["net_spread_t"]))[:10]:
            print(f"  {row['symbol']:>7} {row['entry_hour_utc']:02d}+{row['hold_hours']}h"
                  f"  net={row['net_spread_edge_bp']:+.3f}bp  t={row['net_spread_t']:+6.2f}"
                  f"  cost={row['avg_half_spread_cost_bp']:.3f}bp  gross_t={row['t']:+.2f}")

    if result["sector"]:
        sf = result["sector_family"]
        print("\nsector-level test (cross-sectional mean t, calibrated inside the same null):")
        print(f"  scan max |z| = {sf['real_max_abs_z']:.2f} vs null scan-max p95 {sf['null_max_abs_z_p95']:.2f}"
              f" ({100 * sf['percentile']:.1f}% of worlds, {sf['n_windows']} windows)")
        for key, st in sorted(result["sector"].items(), key=lambda kv: -abs(kv[1]["real_z"]))[:6]:
            print(f"  {key:>7}  z={st['real_z']:+7.2f}  null p95={st['null_z_p95']:.2f}"
                  f"  percentile={100 * st['percentile']:5.1f}%  n={st['n_symbols']}")

    print("\nreplication of top windows across symbols (t-stat):")
    for key, per_sym in result["replication"].items():
        parts = "  ".join(f"{s}:{v['t']:+.1f}" for s, v in per_sym.items())
        print(f"  {key:>12}  {parts}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
