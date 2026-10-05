"""Analyze a live spread tape (``ctbridge.tape`` output).

Produces per-symbol spread distributions (median / p90 / p99, share of zero and crossed
quotes) and a median-spread-by-hour table — the numbers that replace the snapshot in
``src/costs.py`` once the tape exists.

Usage::

    python -m scripts.tape_analysis                     # newest ../ctrader/tape/*.csv
    python -m scripts.tape_analysis --tape PATH [--out PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow direct script run

from src import config  # noqa: E402

TAPE_DIR = config.REPO_ROOT.parent / "ctrader" / "tape"


def find_newest_tape() -> Path:
    candidates = sorted(glob.glob(str(TAPE_DIR / "*.csv")))
    if not candidates:
        raise SystemExit(f"no tape CSVs found under {TAPE_DIR}")
    return Path(candidates[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tape", type=Path, default=None, help="tape CSV (default: newest)")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "tape_summary.json")
    args = ap.parse_args()

    path = args.tape or find_newest_tape()
    df = pd.read_csv(
        path,
        usecols=["recv_utc", "symbol", "bid", "ask", "spread_bp"],
        dtype={"symbol": "category", "bid": "float64", "ask": "float64", "spread_bp": "float64"},
    )
    df["hour"] = df["recv_utc"].str.slice(11, 13)
    df["spread_bp"] = pd.to_numeric(df["spread_bp"], errors="coerce")

    per_symbol = {}
    for sym, g in df.groupby("symbol", observed=True):
        s = g["spread_bp"].dropna()
        per_symbol[str(sym)] = {
            "events": int(len(g)),
            "first": str(g["recv_utc"].iloc[0]),
            "last": str(g["recv_utc"].iloc[-1]),
            "median_spread_bp": float(s.median()) if len(s) else None,
            "p90_spread_bp": float(s.quantile(0.90)) if len(s) else None,
            "p99_spread_bp": float(s.quantile(0.99)) if len(s) else None,
            "zero_share": float((s <= 0).mean()) if len(s) else None,
            "crossed_share": float((s < 0).mean()) if len(s) else None,
            "missing_side_share": float(g["spread_bp"].isna().mean()),
        }

    hourly = (
        df.dropna(subset=["spread_bp"])
        .groupby(["symbol", "hour"], observed=True)["spread_bp"]
        .median()
        .unstack()
        .round(4)
    )

    out = {"tape": str(path), "rows": int(len(df)), "per_symbol": per_symbol,
           "hourly_median_spread_bp": {str(s): {str(h): v for h, v in row.dropna().items()}
                                       for s, row in hourly.iterrows()}}
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"tape {path}  rows {len(df):,}")
    print(f"\n{'symbol':>8} {'events':>10} {'med bp':>8} {'p90 bp':>8} {'p99 bp':>8} {'zero%':>7} {'crossed%':>9}")
    print("-" * 62)
    for sym, st in per_symbol.items():
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"{sym:>8} {st['events']:>10,} {fmt(st['median_spread_bp']):>8} {fmt(st['p90_spread_bp']):>8}"
              f" {fmt(st['p99_spread_bp']):>8} {100 * (st['zero_share'] or 0):>6.1f}%"
              f" {100 * (st['crossed_share'] or 0):>8.1f}%")

    print(f"\nmedian spread bp by UTC hour ({len(hourly.columns)} hours seen):")
    print(hourly.to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
