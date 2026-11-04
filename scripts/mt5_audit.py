"""Market-quality audit of MT5 tick-derived bars.

For each symbol: coverage, feed activity, and spread statistics by UTC hour —
including the specific hours that matter for trading decisions (the 03:00-05:00
fade-window question can be answered directly from this, historically).

Usage::

    python -m scripts.mt5_audit [--symbols EURUSD XCUUSD ...] [--hours 3 5 6 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def audit_symbol(sym: str, hours: list[int]) -> dict:
    path = config.PARQUET_DIR / f"{sym}_MT5_M1.parquet"
    if not path.exists():
        return {"symbol": sym, "error": "missing parquet"}
    df = pd.read_parquet(path)
    if not len(df):
        return {"symbol": sym, "error": "empty"}
    df["hour"] = df["ts"].dt.hour
    span_days = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / 86400
    by_hour = df.groupby("hour").agg(
        bars=("ts", "size"),
        med_sp=("spread_med_bp", "median"),
        p90_sp=("spread_p90_bp", "median"),
        ticks=("tick_n", "mean"),
    )
    out = {
        "symbol": sym,
        "bars": int(len(df)),
        "days": round(span_days, 1),
        "first": str(df["ts"].iloc[0]),
        "last": str(df["ts"].iloc[-1]),
        "median_spread_bp": float(df["spread_med_bp"].median()),
        "p90_spread_bp": float(df["spread_p90_bp"].quantile(0.9)),
        "median_ticks_per_bar": float(df["tick_n"].median()),
        "hours_covered": int(len(by_hour)),
        "hour_table": {
            int(h): {"bars": int(r["bars"]), "med_spread_bp": round(float(r["med_sp"]), 3),
                     "p90_spread_bp": round(float(r["p90_sp"]), 3),
                     "ticks_per_bar": round(float(r["ticks"]), 1)}
            for h, r in by_hour.iterrows()
        },
    }
    for h in hours:
        out[f"spread_at_{h:02d}_bp"] = (
            round(float(by_hour.loc[h, "med_sp"]), 3) if h in by_hour.index else None
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    ap.add_argument("--hours", nargs="+", type=int, default=[3, 5, 6, 7, 8, 15, 16])
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "mt5_audit.json")
    args = ap.parse_args()

    results = [audit_symbol(s, args.hours) for s in args.symbols]
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"{'symbol':>7} {'bars':>8} {'days':>6} {'med_bp':>7} {'p90_bp':>7} {'hrs':>4} "
          + " ".join(f"{h:02d}h" for h in args.hours))
    print("-" * (38 + 4 * len(args.hours)))
    for r in results:
        if "error" in r:
            print(f"{r['symbol']:>7} {r['error']}")
            continue
        cells = " ".join(
            "  n/a" if r.get(f"spread_at_{h:02d}_bp") is None else f"{r[f'spread_at_{h:02d}_bp']:>4.2f}"
            for h in args.hours
        )
        print(f"{r['symbol']:>7} {r['bars']:>8,} {r['days']:>6.0f} {r['median_spread_bp']:>7.3f} "
              f"{r['p90_spread_bp']:>7.3f} {r['hours_covered']:>4} {cells}")
    print(f"\n(columns {', '.join(f'{h:02d}h' for h in args.hours)} = median spread bp at that UTC hour)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
