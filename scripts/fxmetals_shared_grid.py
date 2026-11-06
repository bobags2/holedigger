"""FX/metals grid-homogeneous subpanel: the search, then the corrected-null re-run.

The shared-intraday null (``intraday_slot_null_masks``) needs every symbol on one bar
grid — the same days, the same bar count per day — and ``_shared_day_grid_ok`` refuses
anything else, falling back to the per-symbol permutation, which is **invalid for the
sector z** (it makes correlated real t's face independent null t's; see
``tests.test_window_sweep.test_per_symbol_null_inflates_on_correlated_noise``).

The MT5 FX/metals panel is ragged by nature: sessions differ per instrument (copper
trades ~01:00-18:50 UTC, gold/silver/oil run ~23h with a one-hour break) and single
minutes drop out where no tick printed, so each symbol's per-day bar count varies by a
few bars. This tool measures the largest grid-homogeneous subpanel that still exists,
runs the sweep's shared null on it, and records both the subpanel and the search
evidence — mirroring the equities subpanel convention
(``artifacts/equity_symbols_shared_grid.txt``).

Objective: maximise ``symbols x days`` (panel-days the shared null can cover), then
symbols, then days. The search is exhaustive over symbol subsets and per-day count
values, at the sweep's own M1 grid and at a coarser M5 grid (reported as an upper
bound: coarsening does not buy a 10-symbol panel either).

Read the sector block of the output carefully: the sector z needs at least ten finite
per-symbol t's per window, so a subpanel below ten symbols yields an empty sector —
the honest status of the FX/metals sector calibration, given by measurement here.

Usage::

    python -m scripts.fxmetals_shared_grid                # search + run + write artifacts
    python -m scripts.fxmetals_shared_grid --search-only
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, mt5_clock  # noqa: E402
from src.registry import Registry  # noqa: E402
from src.window_sweep import (  # noqa: E402
    Series,
    _day_structure,
    _shared_day_grid_ok,
    build_mt5_series,
    default_windows,
    run_sweep,
    series_from_bars,
)

SYMBOLS = ["EURUSD", "XAUUSD", "XAGUSD", "XCUUSD", "XPBUSD", "XALUSD",
           "XNGUSD", "XTIUSD", "XPTUSD", "NAS100"]


def counts_of(series: dict) -> dict:
    """symbol -> {day: bars} from the series' day structure."""
    out = {}
    for sym, s in series.items():
        d, _st, l = _day_structure(s)
        out[sym] = dict(zip(d.tolist(), l.tolist()))
    return out


def search_grid(cnt: dict, symbols: list[str]) -> list[dict]:
    """Every (subset, count) whose days are identical and uniform; sorted best-first."""
    all_c = sorted(set().union(*[set(c.values()) for c in cnt.values()]))
    rows = []
    for r in range(2, len(symbols) + 1):
        for C in all_c:
            sets = {s: {d for d, n in cnt[s].items() if n == C} for s in symbols}
            for sub in itertools.combinations(symbols, r):
                inter = set.intersection(*[sets[s] for s in sub])
                if not inter:
                    continue
                rows.append({
                    "symbols": list(sub),
                    "bars_per_day": int(C),
                    "days": len(inter),
                    "panel_days": len(inter) * r,
                })
    rows.sort(key=lambda r: (-r["panel_days"], -len(r["symbols"]), -r["days"]))
    return rows


def m5_series(sym: str) -> Series:
    """Coarser-grid variant used only for the search bound (not for the run)."""
    df = pd.read_parquet(config.PARQUET_DIR / f"{sym}_MT5_M1.parquet")
    ts = mt5_clock.stored_to_utc(df["ts"])
    g = pd.DataFrame({
        "ts": ts.to_numpy(),
        "close": df["close"].to_numpy(),
        "spread_med_bp": df["spread_med_bp"].to_numpy(),
    }).set_index("ts")
    m5 = g.resample("5min").last().dropna()
    return series_from_bars(sym, m5.index, m5["close"].to_numpy(), m5["spread_med_bp"].to_numpy())


def filtered_series(sym: str, keep_days: set[int]) -> Series:
    """The symbol's M1 bars restricted to ``keep_days`` (UTC days), spread kept."""
    df = pd.read_parquet(config.PARQUET_DIR / f"{sym}_MT5_M1.parquet")
    ts = mt5_clock.stored_to_utc(df["ts"])
    day = pd.DatetimeIndex(ts).to_numpy().astype("datetime64[ns]").astype("int64") // (86_400 * 10**9)
    mask = np.isin(day, np.asarray(sorted(keep_days), dtype=np.int64))
    return series_from_bars(sym, ts[mask], df["close"].to_numpy()[mask], df["spread_med_bp"].to_numpy()[mask])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=config.ARTIFACTS_DIR / "window_sweep_fxmetals_shared_intraday.json")
    ap.add_argument("--search-out", type=Path, default=config.ARTIFACTS_DIR / "fxmetals_shared_grid_search.json")
    ap.add_argument("--symbols-file-out", type=Path, default=config.ARTIFACTS_DIR / "fxmetals_symbols_shared_grid.txt")
    args = ap.parse_args()

    m1 = {s: build_mt5_series(s) for s in SYMBOLS}
    c1 = counts_of(m1)
    m5 = {s: m5_series(s) for s in SYMBOLS}
    c5 = counts_of(m5)
    rows1, rows5 = search_grid(c1, SYMBOLS), search_grid(c5, SYMBOLS)

    print("per-symbol day counts (M1: days, most common bar counts):")
    for s in SYMBOLS:
        print(f"  {s:>7} days={len(c1[s]):>4} top={Counter(c1[s].values()).most_common(3)}")

    print("\nlargest grid-homogeneous subpanels, objective symbols x days:")
    for name, rows in (("M1 (the sweep's grid)", rows1), ("M5 (coarser bound)", rows5)):
        print(f"  {name}:")
        for r in rows[:4]:
            print(f"    symbols={len(r['symbols'])} days={r['days']} panel_days={r['panel_days']}"
                  f" bars/day={r['bars_per_day']}  {','.join(r['symbols'])}")
        big = [r for r in rows if len(r["symbols"]) >= 10]
        print(f"    >=10-symbol homogeneous panels: {len(big)}")

    def best_by_size(rows: list[dict]) -> list[dict]:
        return [next(r for r in rows if len(r["symbols"]) == size)
                for size in sorted({len(r["symbols"]) for r in rows})]

    search = {
        "objective": "maximise symbols x days, then symbols, then days",
        "grids": {
            "m1": {"definition": "MT5 M1 bars (build_mt5_series), exact per-day counts",
                   "best_by_size": best_by_size(rows1)},
            "m5": {"definition": "M5 resample (last close, dropna), exact per-day counts",
                   "best_by_size": best_by_size(rows5)},
        },
        ">=10_symbol_homogeneous_panels": {"m1": 0, "m5": 0},
    }
    for key, rows in (("m1", rows1), ("m5", rows5)):
        search[">=10_symbol_homogeneous_panels"][key] = sum(1 for r in rows if len(r["symbols"]) >= 10)

    best = rows1[0]
    sub_syms, C = best["symbols"], best["bars_per_day"]
    keep = None
    for s in sub_syms:
        ds = {d for d, n in c1[s].items() if n == C}
        keep = ds if keep is None else (keep & ds)
    keep = set(sorted(keep))
    sub = {s: filtered_series(s, keep) for s in sub_syms}
    assert _shared_day_grid_ok(sub), "subpanel construction failed its own grid contract"

    search["chosen"] = {
        "symbols": sub_syms, "bars_per_day": C, "days": len(keep),
        "first_day": str(pd.Timestamp(min(keep), unit="D").date()),
        "last_day": str(pd.Timestamp(max(keep), unit="D").date()),
    }
    args.search_out.write_text(json.dumps(search, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nchosen subpanel: {len(sub_syms)} symbols x {len(keep)} days at {C} bars/day"
          f" ({min(keep)}..{max(keep)})")
    print(f"wrote {args.search_out}")

    if args.search_only:
        return

    windows = default_windows()
    reg = Registry(config.REGISTRY_PATH)
    for sym in sub:
        for h, d in windows:
            reg.register(
                {"family": "session_window_v1", "symbol": sym, "entry_hour_utc": h, "hold_hours": d},
                generator="window_sweep",
                generator_version="1",
            )
    result = run_sweep(sub, windows, args.worlds)
    for row in result["candidates"]:
        hid = reg.register(
            {"family": "session_window_v1", "symbol": row["symbol"],
             "entry_hour_utc": row["entry_hour_utc"], "hold_hours": row["hold_hours"]},
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
                "net_spread_edge_bp": None if row["net_spread_edge_bp"] is None else round(row["net_spread_edge_bp"], 5),
                "net_spread_t": None if row["net_spread_t"] is None else round(row["net_spread_t"], 3),
            },
            {"null": {"mean_block": 1.0, "worlds": args.worlds, "kind": result["null_kind"]},
             "source": "mt5", "symbols": len(sub),
             "subpanel": {"file": args.symbols_file_out.name, "bars_per_day": C, "days": len(keep)},
             "spread": "adopted per symbol (tape 2026-10-05)"},
        )

    result["subpanel"] = {
        "rule": "largest grid-homogeneous subset of the MT5 FX/metals parquets "
                "(identical days, identical per-day bar counts, uniform within symbol); "
                "objective symbols x days; see fxmetals_shared_grid_search.json",
        "symbols_file": args.symbols_file_out.name,
        "symbols": sub_syms,
        "bars_per_day": C,
        "days": len(keep),
        "first_day": str(pd.Timestamp(min(keep), unit="D").date()),
        "last_day": str(pd.Timestamp(max(keep), unit="D").date()),
        "note": "below the sector guard (10 finite per-symbol t's per window), so the run "
                "records a candidate-level family null and an EMPTY sector block",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.symbols_file_out.write_text("\n".join(sub_syms) + "\n", encoding="utf-8")

    c = result["config"]
    print(f"\nnull ({result['null_kind']}, {c['worlds']} worlds): family max |t| mean "
          f"{result['null']['family_max_mean']:.2f} | p95 {result['null']['family_max_p95']:.2f}")
    print(f"real family max |t| = {result['family']['real_max_abs_t']:.2f}"
          f" -> percentile {100 * result['family']['percentile']:.1f}%, z = {result['family']['z']:.2f}")
    print(f"sector rows: {len(result['sector'])} (sector guard needs >=10 symbols; subpanel has {len(sub_syms)})")
    print(f"wrote {args.out} and {args.symbols_file_out}")


if __name__ == "__main__":
    main()
