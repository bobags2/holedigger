"""Cost probe: is the lag-1 reversion family beatable at this broker's costs?

Answers three questions with measurements, not priors:
  1. what do the quoted $/lot commissions mean in basis points per round trip
  2. how big is the reversion edge at each timeframe (M1 -> H1)
  3. does the edge concentrate in large-move bars (where a limit-order variant would
     have to live to matter)

Spread is NOT included: it has not been measured yet. The bridge can log live bid/ask;
that is the next step. Broker cost inputs below are the user's quoted figures — edit
BROKER_COST_PER_LOT when they change.

Usage: python -m scripts.cost_probe
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow direct script run

from src import config  # noqa: E402

BROKER_COST_PER_LOT = {"XAUUSD": 4.96, "EURUSD": 2.51, "GBPUSD": 2.97}
LOT_UNITS = {"XAUUSD": 100.0, "EURUSD": 100_000.0, "GBPUSD": 100_000.0}
TIMEFRAMES = [(1, "1min"), (5, "5min"), (15, "15min"), (30, "30min"), (60, "60min"), (240, "240min")]
BUCKET_EDGES = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]


def load_close(symbol: str) -> pd.Series:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_M1.parquet")
    return df.set_index("ts")["close"]


def adjacency(close: pd.Series, tf_min: int, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """Log returns of exact-adjacency bars + boolean mask aligned with the returns."""
    x = close.resample(rule).last().dropna()
    r = np.log(x).diff().dropna().to_numpy()
    gap = x.index.to_series().diff().dt.total_seconds().div(60).to_numpy()[1:]
    ok = gap == tf_min
    good = ok[1:] & ok[:-1]
    return r, good


def edge_stats(prev: np.ndarray, nxt: np.ndarray) -> dict:
    """Reversion edge in bp/trade and its t-stat. Positive edge = reversion pays."""
    pnl = np.sign(prev) * nxt
    edge_bp = -float(pnl.mean()) * 1e4
    t = -float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))))
    return {"edge_bp": edge_bp, "t": t, "n": int(len(pnl))}


def main() -> None:
    out: dict = {"costs": {}, "edge_by_timeframe": {}, "buckets_m5": {}}

    print("== commission per 1.00 lot, converted to bp (round trip) ==")
    print(f"{'symbol':>7} {'last':>10} {'notional/lot':>14} {'per side bp':>12} {'round trip bp*':>14}")
    cost_rt = {}
    for sym in BROKER_COST_PER_LOT:
        close = load_close(sym)
        last = float(close.iloc[-1])
        notional = LOT_UNITS[sym] * last
        per_side_bp = BROKER_COST_PER_LOT[sym] / notional * 1e4
        cost_rt[sym] = 2 * per_side_bp
        out["costs"][sym] = {
            "last": last,
            "notional_per_lot": notional,
            "usd_per_lot": BROKER_COST_PER_LOT[sym],
            "per_side_bp": per_side_bp,
            "round_trip_bp_if_per_side": 2 * per_side_bp,
            "round_trip_bp_if_round_turn": per_side_bp,
        }
        print(f"{sym:>7} {last:>10.4f} {notional:>14,.0f} {per_side_bp:>12.3f} {2 * per_side_bp:>14.3f}")
    print("* assuming your quotes are per side; halve if they are round-turn\n")

    print("== lag-1 reversion edge by timeframe (exact-adjacency rows only) ==")
    print(
        f"{'symbol':>7} {'tf':>6} {'edge bp':>8} {'t':>7} {'mean|move| bp':>13} {'edge/move':>10}"
        f" {'n':>10} {'bp/day':>8}"
    )
    for sym in BROKER_COST_PER_LOT:
        close = load_close(sym)
        span_days = max((close.index[-1] - close.index[0]).days, 1)
        for tf_min, rule in TIMEFRAMES:
            r, good = adjacency(close, tf_min, rule)
            st = edge_stats(r[:-1][good], r[1:][good])
            mean_move = float(np.abs(r[1:][good]).mean()) * 1e4
            gross_bp_day = st["edge_bp"] * st["n"] / span_days
            key = f"{sym}|{rule}"
            out["edge_by_timeframe"][key] = {**st, "mean_abs_move_bp": mean_move, "gross_bp_per_day": gross_bp_day}
            print(
                f"{sym:>7} {rule:>6} {st['edge_bp']:>8.4f} {st['t']:>7.2f} {mean_move:>13.2f}"
                f" {st['edge_bp'] / mean_move * 100:>9.2f}% {st['n']:>10,} {gross_bp_day:>8.2f}"
            )

    print("\n== reversion edge by prior-move size, M5 (edge bp vs round-trip cost) ==")
    print(f"{'symbol':>7} {'bucket':>12} {'|move| bp':>10} {'edge bp':>8} {'t':>7} {'n':>9} {'net*(per-side)':>15} {'net*(rt-turn)':>14}")
    for sym in BROKER_COST_PER_LOT:
        close = load_close(sym)
        r, good = adjacency(close, 5, "5min")
        prev, nxt = r[:-1][good], r[1:][good]
        mag = np.abs(prev)
        qs = np.quantile(mag, BUCKET_EDGES)
        sym_buckets = []
        for i in range(len(BUCKET_EDGES) - 1):
            lo, hi = qs[i], qs[i + 1]
            m = (mag >= lo) & (mag < hi) if i < len(BUCKET_EDGES) - 2 else (mag >= lo)
            st = edge_stats(prev[m], nxt[m])
            mean_move = float(mag[m].mean()) * 1e4
            net_per_side = st["edge_bp"] - cost_rt[sym]
            net_rt = st["edge_bp"] - cost_rt[sym] / 2
            label = f"{int(BUCKET_EDGES[i] * 100)}-{int(BUCKET_EDGES[i + 1] * 100)}%"
            sym_buckets.append(
                {
                    "bucket": label,
                    "mean_abs_move_bp": mean_move,
                    **st,
                    "net_bp_if_per_side": net_per_side,
                    "net_bp_if_round_turn": net_rt,
                }
            )
            print(
                f"{sym:>7} {label:>12} {mean_move:>10.2f} {st['edge_bp']:>8.4f} {st['t']:>7.2f}"
                f" {st['n']:>9,} {net_per_side:>15.4f} {net_rt:>14.4f}"
            )
        out["buckets_m5"][sym] = sym_buckets

    with (config.ARTIFACTS_DIR / "cost_probe.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nwrote {config.ARTIFACTS_DIR / 'cost_probe.json'}")


if __name__ == "__main__":
    main()
