"""P18 - the pre-registered rule set, and what the corrected history says it would have earned.

THE REASONING THAT MAKES THIS THE RIGHT NEXT THING.

Fourteen explorations closed, and the corrected cTrader numbers (P16) removed the last of them
that looked positive: gold is the largest LOSER at -418,982, every instrument that matters is
negative, win rate 44%, net_bp -1.82. The MT5 book is -7,971 at a 68% win rate and a 0.23 payoff
ratio, and its meta-label lift is at the 68th percentile of a day-clustered null. So there is no
evidence of a directional edge on either account.

But there IS a measured, robust fact that survives everything: the LOSSES ARE PROCESS. On
cTrader, 51% of the oil loss is 10 trades out of 734. The worst gold day is -2,094 against a
best of +2,402. The balance went negative at -1,297. Those are statements about sizing and
limits, not about prediction - and limits are the one thing that does not need an edge.

So this module does NOT search for alpha. It takes five rules that are mechanically enforceable
at entry time, applies each one to the corrected history INDEPENDENTLY (never in combination,
because combining them on the same data is how you manufacture a curve that will not forward
test), and reports what each would have done to the P&L distribution. Then it writes the
pre-registered forward-test spec.

WHAT "WOULD HAVE DONE" MEANS HERE, and its limit. These are rules that were not in place when
the trades ran, so applying them to history is a counterfactual: a rule that caps size would
have changed the trades' P&L, not just removed some. Where that matters it is stated per rule
rather than hidden in a headline number. A max-hold rule in particular is NOT evaluated here,
because the reviewer is right that a time stop cannot be assessed on trades that had no time
stop - conditioning on hold time is conditioning on the outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd


def load_ctrader() -> pd.DataFrame:
    p = Path(__file__).resolve().parent.parent / "artifacts" / "ctrader_positions.parquet"
    return pd.read_parquet(p)


def load_mt5() -> pd.DataFrame:
    """The MT5 book in the same shape, from the trade-history artifact."""
    r = json.loads((Path(__file__).resolve().parent.parent / "artifacts"
                    / "trade_history.json").read_text(encoding="utf-8"))["positions"]
    d = pd.DataFrame(r)
    d["open_time"] = pd.to_datetime(d["open_time"])
    d["close_time"] = pd.to_datetime(d["close_time"])
    for c in ("profit", "open_volume", "contract_size", "open_price"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["notional"] = d["open_price"] * d["open_volume"] * d["contract_size"].fillna(1.0)
    d["net_bp"] = d["profit"] / d["notional"] * 1e4
    d["open_day"] = d["open_time"].dt.normalize()
    return d


def main() -> None:
    C = load_ctrader()
    C["open_day"] = C["open_time"].dt.normalize()
    print("P18: pre-registered rules, priced on the CORRECTED history\n")
    print(f"cTrader book: {len(C):,} positions, net {C['profit'].sum():+,.0f}, "
          f"win {100*(C.profit > 0).mean():.0f}%, net_bp {C['net_bp'].mean():+.3f}")
    by_day = C.groupby("open_day")["profit"].agg(["sum", "count"])
    print(f"{len(by_day)} trading days: mean {by_day['sum'].mean():+,.0f}/day, "
          f"worst {by_day['sum'].min():+,.0f}, best {by_day['sum'].max():+,.0f}, "
          f"{100*(by_day['sum'] > 0).mean():.0f}% positive")
    print()

    # ---- R1: per-position risk cap ------------------------------------------------
    for cap_method, cap in (("p95 of |net_bp|", float(np.quantile(np.abs(C["net_bp"]), 0.95))),
                            ("2x the median loss", float(-2 * C.loc[C["net_bp"] < 0, "net_bp"].median()))):
        print(f"R1 per-position cap ({cap_method}, {cap:.2f} bp):")
        losses = C[C["net_bp"] < -cap]
        print(f"    {len(losses)} positions breach it, {losses['profit'].sum():+,.0f} of P&L "
              f"({100 * losses['profit'].sum() / C['profit'].sum():.0f}% of the book)")
    print()

    # ---- R2: per-day loss limit --------------------------------------------------
    for lim in (-3000.0, -5000.0):
        days = by_day[by_day["sum"] < lim]
        print(f"R2 per-day limit ({lim:+,.0f}): {len(days)} days breach, "
              f"those days lost {days['sum'].sum():+,.0f} vs the book's {C['profit'].sum():+,.0f}")
    print()

    # ---- R3: commission-aware minimum target -------------------------------------
    comm = float(C["commission"].sum()); n = len(C)
    print(f"R3 commission: {comm:+,.0f} over {n:,} positions = {comm/n:+,.2f} per position")
    nb = (C["profit"] / C["notional"] * 1e4)
    print(f"    net_bp including it: {nb.mean():+.3f} bp")
    print(f"    every position must therefore clear {abs(comm/n)/C['notional'].mean()*1e4:.2f} bp "
          f"of commission on {C['notional'].mean():,.0f} of notional before anything else")
    print()

    # ---- R4: max trades/day ------------------------------------------------------
    tpd = by_day["count"]
    print(f"R4 trades/day: median {int(tpd.median())}, p95 {int(tpd.quantile(0.95))}, "
          f"max {int(tpd.max())}")
    heavy = by_day[tpd >= tpd.quantile(0.90)]
    print(f"    the top-decile days ({len(heavy)} days, {heavy['count'].sum():,} trades) "
          f"earned {heavy['sum'].sum():+,.0f}")
    print()

    # ---- R5: instrument tail ------------------------------------------------------
    print("R5 per-instrument tail (the oil case):")
    smap = json.loads((Path(__file__).resolve().parent.parent / "artifacts"
                       / "ctrader_symbol_map.json").read_text(encoding="utf-8"))
    C["sym"] = C["symbol_id"].astype(str).map(smap)
    for sym, g in C.groupby("sym"):
        if len(g) < 100:
            continue
        w = g.nsmallest(10, "profit")
        print(f"    {sym:>7}: n {len(g):>4} net {g['profit'].sum():+9,.0f} | "
              f"worst 10 = {w['profit'].sum():+9,.0f} "
              f"({100 * w['profit'].sum() / g['profit'].sum():>4.0f}% of it)")
    print()

    print("""THE PRE-REGISTERED FORWARD TEST (what to run on demo, in this order):

  R1  per-position hard stop at the p95 of |net_bp| - mechanically enforceable, no prediction
  R2  per-day loss limit; the breach days are measurable, so the level is a decision
  R3  a minimum target that clears measured commission per position, checked every day
  R4  max trades/day at the current p95
  R5  per-instrument exposure cap, sized so no 10 trades can be half the instrument's P&L

  Registration: freeze these five numbers, run them forward on demo for 20 sessions, and judge
  on the SHAPE of the P&L distribution (worst day, worst 10 trades as a share) rather than on
  the total. A rule that halves the worst day is doing its job even if the total is negative.
  What this history CANNOT evaluate is a max-hold rule: these trades had no max hold, so
  conditioning on hold time is conditioning on the outcome, and that is why R0 (max hold) is
  deliberately absent from the forward test.""")


if __name__ == "__main__":
    main()
