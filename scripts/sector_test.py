"""Price a discovered sector-level window pattern as a real equal-weight portfolio.

The sweep's sector test says a window drifts sector-wide; this script answers whether
the drift survives *costs*: per symbol, per day, with the spread measured at the
trade's own entry and exit bars, then aggregated into one equal-weight portfolio P&L
series (which is where cross-sectional correlation is dealt with honestly).

Usage::

    python -m scripts.sector_test --entry 18 --hold 2 [--symbols-file ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.window_sweep import build_mt5_series, window_segments  # noqa: E402


def load_panel(symbols: list[str], entry: int, hold: int) -> tuple[dict, dict]:
    """Per-day gross and cost series for each symbol for this window."""
    gross, cost, days_of = {}, {}, {}
    for sym in symbols:
        try:
            s = build_mt5_series(sym)
        except (FileNotFoundError, KeyError):
            continue
        if s.spread_bp is None or len(s.r) < 5000:
            continue
        a, b, v = window_segments(s, [(entry, hold)])
        m = v[:, 0]
        if m.sum() < 30:
            continue
        S = np.cumsum(s.r)
        e, x = a[m, 0], b[m, 0]
        days = pd.to_datetime(np.unique(s.bar_day)[m], unit="D")
        gross[sym] = (S[x] - S[e]) * 1e4            # mid-to-mid window return, bp
        cost[sym] = (s.spread_bp[e] + s.spread_bp[x]) / 2.0
        days_of[sym] = days
    return gross, cost, days_of


def align(gross: dict, cost: dict, days_of: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames_g, frames_c = [], []
    for sym, d in days_of.items():
        frames_g.append(pd.Series(gross[sym], index=d, name=sym))
        frames_c.append(pd.Series(cost[sym], index=d, name=sym))
    G = pd.concat(frames_g, axis=1).sort_index()
    C = pd.concat(frames_c, axis=1).sort_index()
    common = G.notna().sum(axis=1) >= 0.5 * G.shape[1]
    return G[common].fillna(0.0), C[common].fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, required=True, help="entry hour UTC")
    ap.add_argument("--hold", type=int, required=True, help="hold hours")
    ap.add_argument("--symbols-file", default=None, help="newline-separated symbol list")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "sector_test.json"))
    args = ap.parse_args()

    if args.symbols_file:
        symbols = [l.strip() for l in Path(args.symbols_file).read_text().splitlines() if l.strip()]
    else:
        symbols = args.symbols or config.SYMBOLS

    gross, cost, days_of = load_panel(symbols, args.entry, args.hold)
    if not gross:
        raise SystemExit("no usable symbols")
    G, C = align(gross, cost, days_of)
    sgn = -1.0 if G.mean().mean() < 0 else 1.0     # trade the drift's direction
    pnl_gross = sgn * G.mean(axis=1)
    pnl_net = pnl_gross - C.mean(axis=1)

    def stats(s: pd.Series) -> dict:
        t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
        return {
            "n_days": int(len(s)),
            "mean_bp": round(float(s.mean()), 4),
            "median_bp": round(float(s.median()), 4),
            "t": round(float(t), 2),
            "sharpe_ann": round(float(s.mean() / s.std(ddof=1)) * float(np.sqrt(252)), 2),
            "hit_rate": round(float((s > 0).mean()), 3),
        }

    per_symbol = []
    for sym in G.columns:
        g = sgn * G[sym] - C[sym]
        per_symbol.append({
            "symbol": sym,
            "gross_bp": round(float(sgn * G[sym].mean()), 3),
            "gross_t": round(float((sgn * G[sym]).mean() / ((sgn * G[sym]).std(ddof=1) / np.sqrt(len(G)))), 2),
            "cost_bp": round(float(C[sym].mean()), 3),
            "net_bp": round(float(g.mean()), 3),
            "net_t": round(float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), 2),
        })
    per_symbol.sort(key=lambda r: -r["net_bp"])

    # half-sample split: does the portfolio edge survive in both halves?
    mid = pnl_net.index[len(pnl_net) // 2]
    halves = {"first": stats(pnl_net.loc[:mid]), "second": stats(pnl_net.loc[mid:])}
    gross_halves = {"first": stats(pnl_gross.loc[:mid]), "second": stats(pnl_gross.loc[mid:])}

    result = {
        "config": {"entry_hour_utc": args.entry, "hold_hours": args.hold,
                   "symbols": len(G.columns), "direction": "short" if sgn < 0 else "long"},
        "gross": stats(pnl_gross),
        "net_of_spread": stats(pnl_net),
        "median_symbol": {
            "gross_bp": round(float(np.median([r["gross_bp"] for r in per_symbol])), 3),
            "cost_bp": round(float(np.median([r["cost_bp"] for r in per_symbol])), 3),
            "net_bp": round(float(np.median([r["net_bp"] for r in per_symbol])), 3),
            "share_profitable": round(float(np.mean([r["net_bp"] > 0 for r in per_symbol])), 3),
        },
        "per_symbol": per_symbol,
        "series_tail": {str(k.date()): round(float(v), 3) for k, v in pnl_net.tail(10).items()},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    c = result["config"]
    print(f"window {c['entry_hour_utc']:02d}:00 +{c['hold_hours']}h UTC | {c['symbols']} symbols | "
          f"direction: {c['direction']}")
    for name, key in (("gross (mid-to-mid)", "gross"), ("net of measured spread", "net_of_spread")):
        st = result[key]
        print(f"  {name}: mean {st['mean_bp']:+.3f} bp/day  t={st['t']:+.2f}  "
              f"Sharpe {st['sharpe_ann']:+.2f}  hit {st['hit_rate']:.0%}  n={st['n_days']}")
    ms = result["median_symbol"]
    print(f"  median symbol: gross {ms['gross_bp']:+.2f} - cost {ms['cost_bp']:.2f} = net {ms['net_bp']:+.2f} bp/trade "
          f"({ms['share_profitable']:.0%} of names profitable)")
    print(f"\n  best 8 net: {', '.join(f'{r['symbol']} {r['net_bp']:+.1f}' for r in per_symbol[:8])}")
    print(f"  worst 4 net: {', '.join(f'{r['symbol']} {r['net_bp']:+.1f}' for r in per_symbol[-4:])}")
    print("  half-sample (gross | net):  "
          + "  ".join(f"{k}: {gross_halves[k]['mean_bp']:+.2f}/{halves[k]['mean_bp']:+.2f} bp/day "
                      f"(t {gross_halves[k]['t']:+.1f}/{halves[k]['t']:+.1f})" for k in ("first", "second")))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
