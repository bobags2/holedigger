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


def series_from_bars(symbol: str, times, closes) -> Series:
    t = pd.DatetimeIndex(times)
    close = np.asarray(closes, dtype=float)
    r = np.empty(len(close), dtype=float)
    r[0] = 0.0
    r[1:] = np.diff(np.log(close))
    ns = t.to_numpy().astype("datetime64[ns]").astype("int64")
    return Series(symbol, r, ns, ns // (86_400 * 10**9), float(close[-1]))


def build_series(symbol: str) -> Series:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_M1.parquet").set_index("ts")
    m5 = df.resample("5min").last().dropna()
    return series_from_bars(symbol, m5.index, m5["close"].to_numpy())


def default_windows(holds: list[int] = HOLDS, max_exit: int = MAX_EXIT_HOUR) -> list[tuple[int, int]]:
    return [(h, d) for d in holds for h in range(0, max_exit - d + 1)]


def window_segments(s: Series, windows: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, b, valid): bar indices of entry/exit per (day, window). Invalid entries are junk."""
    days = np.unique(s.bar_day)
    n_days, n_win = len(days), len(windows)
    a = np.zeros((n_days, n_win), dtype=np.int64)
    b = np.zeros((n_days, n_win), dtype=np.int64)
    entry_cut = np.array([h * 3600 for h, _ in windows], dtype=np.int64) * 10**9
    exit_cut = np.array([(h + d) * 3600 for h, d in windows], dtype=np.int64) * 10**9
    starts = np.searchsorted(s.bar_day, days, side="left")
    ends = np.searchsorted(s.bar_day, days, side="right")
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        times = s.bar_time[lo:hi] - days[i] * 86_400 * 10**9
        a[i] = lo + np.searchsorted(times, entry_cut, side="left")
        b[i] = lo + np.searchsorted(times, exit_cut, side="right") - 1
    valid = (b > a) & (b - a >= 2)
    a = np.clip(a, 0, len(s.r) - 1)
    b = np.clip(b, 0, len(s.r) - 1)
    return a, b, valid


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


def run_sweep(
    series: dict[str, Series],
    windows: list[tuple[int, int]],
    n_worlds: int,
    seed: int = config.SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    syms = list(series)
    n_win = len(windows)

    seg = {sym: window_segments(s, windows) for sym, s in series.items()}
    real_mean, real_t, real_n = {}, {}, {}
    for sym, s in series.items():
        pnl = pnl_matrix(np.cumsum(s.r), *seg[sym])
        real_mean[sym], real_t[sym], real_n[sym] = window_stats(pnl)

    null_t = np.empty((n_worlds, len(syms) * n_win))
    for w in range(n_worlds):
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
    null_max = abs_null.max(axis=1)
    real_max = float(abs_real.max())
    percentile = float((null_max < real_max).mean())
    z = float((real_max - null_max.mean()) / null_max.std(ddof=1)) if n_worlds > 1 else float("nan")
    per_candidate_pct = (abs_null < abs_real[None, :]).mean(axis=0)

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
        comm = costs.commission_round_trip_bp(sym, series[sym].price)
        be = edge_bp[j] if comm is None else edge_bp[j] - comm
        net = None if comm is None else edge_bp[j] - comm
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
                "break_even_spread_bp": be,
                "net_bp_spread0": net,
                "t_first_half": float(half_t[(sym, 0)][windows.index((h, d))]),
                "t_second_half": float(half_t[(sym, 1)][windows.index((h, d))]),
            }
        )
    rows.sort(key=lambda r: -abs(r["t"]))

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
        "family": {
            "real_max_abs_t": real_max,
            "percentile": percentile,
            "z": z,
            "top_real": rows[0],
        },
        "candidates": rows,
        "replication": replication,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "window_sweep.json")
    args = ap.parse_args()

    config.ensure_dirs()
    windows = default_windows()
    series = {sym: build_series(sym) for sym in args.symbols}

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
            },
            {"null": {"mean_block": 1.0, "worlds": args.worlds}, "spread_bp": 0.0},
        )

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    c = result["config"]
    print(f"family {c['family']}: {c['candidates']} candidates, {c['windows']} windows x {len(c['symbols'])} symbols")
    print(f"\nnull (permutation, {c['worlds']} worlds): family max |t| mean {result['null']['family_max_mean']:.2f}"
          f" | p95 {result['null']['family_max_p95']:.2f} | max {result['null']['family_max_max']:.2f}"
          f" | analytic {result['null']['analytic_expected_max']:.2f}")
    print(f"real family max |t| = {result['family']['real_max_abs_t']:.2f}"
          f" -> percentile {100 * result['family']['percentile']:.1f}%, z = {result['family']['z']:.2f}\n")

    hdr = f"{'symbol':>7} {'window':>12} {'edge bp':>9} {'t':>7} {'n':>6} {'own%':>6} {'BE spr':>8} {'net@0':>7} {'t1':>7} {'t2':>7}"
    print(hdr)
    print("-" * len(hdr))
    for row in result["candidates"][:15]:
        be = "n/a" if row["break_even_spread_bp"] is None else f"{row['break_even_spread_bp']:.3f}"
        net = "n/a" if row["net_bp_spread0"] is None else f"{row['net_bp_spread0']:.3f}"
        win = f"{row['entry_hour_utc']:02d}+{row['hold_hours']}h"
        print(
            f"{row['symbol']:>7} {win:>12} {row['edge_bp']:>9.4f} {row['t']:>7.2f} {row['n']:>6}"
            f" {row['own_null_percentile']:>6.3f} {be:>8} {net:>7} {row['t_first_half']:>7.2f} {row['t_second_half']:>7.2f}"
        )

    print("\nreplication of top windows across symbols (t-stat):")
    for key, per_sym in result["replication"].items():
        parts = "  ".join(f"{s}:{v['t']:+.1f}" for s, v in per_sym.items())
        print(f"  {key:>12}  {parts}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
