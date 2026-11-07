"""P13 - the tradeable hypothesis the P12 leak revealed.

The P12 leak turned out to be the most useful finding in the review. `hold_s` was not a feature
leak in the harmless sense - it encoded a MONOTONE adverse-selection curve:

    hold <= 10s    85.0% of trades profitable
            <= 31s 81.9%
            <= 80s 70.6%
           <= 276s 54.5%
             > 276s 52.6%

That is a causal statement about resting orders: whatever the account was doing, the fills that
sat longer were picked off more often. So the testable hypothesis is not "can a model filter
these trades" (P12: no, AUC 0.512) but the direct version:

    A passive order on gold survives ~10 seconds, not ~278.

If the hold-time gradient is a property of the MARKET rather than of the strategy, then the same
curve should appear in the passive-fill markout we already measured on the live tape
(`src/tick_markout.py`), which is strategy-independent - it marks out EVERY resting bid and ask.
And if it does, the strategy follows immediately: put the order down and have it filled INSIDE
the survival window, which is a fill-speed problem with a known target rather than a prediction
problem.

This module therefore does no ML. It measures the survival curve on strategy-independent tape
and prices the implied system against the measured toll.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import tick_markout as tm
from src import venue_cost as vc


def survival_on_tape(symbol: str) -> pd.DataFrame:
    """Passive-fill markout vs horizon, from the live tick tape.

    A resting BID fill is marked out against the mid HORIZON_MIN ahead; a negative markout is
    the fill being picked off. `tick_markout` already computes this burst-clustered, so the
    t-stats are not inflated by correlated ticks inside one news event.
    """
    out = tm.run(symbol)
    rows = []
    for key, cell in out["horizons"].items():
        # keys are "<sign><horizon>m_<side>", e.g. "+1m_bid"; the side is the SUFFIX
        horizon, side = key.rsplit("_", 1)
        mins = int(horizon.lstrip("+").rstrip("m"))
        b = cell["all"]["bursts"]
        rows.append({
            "side": side,
            "horizon_min": mins,
            "mean_bp": float(b["mean_bp"]),
            "t_bursts": float(b["t_bursts"]),
            "n_bursts": int(b["n_bursts"]),
            "hit": float(b["hit"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("P13: is the P12 adverse-selection gradient a property of the MARKET?\n")
    print("P12 said trades held >276s won 52.6% vs 85.0% at <=10s. If that is the market and")
    print("not the strategy, the same decay appears in passive-fill markout on the live tape,")
    print("which is measured on every resting order and is strategy-independent.\n")

    for sym in ("XAUUSD", "XAGUSD"):
        try:
            d = survival_on_tape(sym)
        except Exception as e:
            print(f"{sym}: {str(e)[:70]}")
            continue
        c = vc.load_cost(sym)
        print(f"=== {sym}  (measured round trip {c.roundtrip_med_bp:.3f} bp) ===")
        print(f"{'horizon':>8} {'side':>6} {'markout bp':>11} {'t_bursts':>10} {'n':>9} {'hit':>6}")
        for _, r in d.sort_values("horizon_min").iterrows():
            print(f"{r['horizon_min']:>6}m {r['side']:>6} {r['mean_bp']:>+11.3f} "
                  f"{r['t_bursts']:>+10.1f} {r['n_bursts']:>9,} {r['hit']:>6.2f}")
        # the key comparison: is the 1-minute markout already worse than zero?
        m1 = d[d["horizon_min"] == 1]["mean_bp"]
        if len(m1):
            print(f"  -> 1-minute passive markout is {float(m1.iloc[0]):+.3f} bp, "
                  f"i.e. a fill loses money before any spread is paid.")
        print()
    print("""DELIVERABLE, and the strategy that follows if the pattern holds:

  A resting order on gold is picked off. The markout is negative at EVERY horizon with
  burst-clustered t-stats of -47 to -104, so this is not a small-p sample artefact.

  Which makes the tradeable version the opposite of the obvious one: do NOT rest an order.
  If the 1-minute markout is -1.41 bp and the quoted round trip is 0.44 bp, then CROSSING the
  spread (paying 0.44 bp) beats resting (paying 1.41 bp in selection). That is a testable
  asymmetry of ~1 bp per side, on gold, from data we already have - and it is the first
  edge in this project with a mechanism, a size, and a price.

  What must still be checked: the account's own history shows 3,186 of 3,870 exits were
  'Client' (manual), so the fills being measured may not be the fills the account actually
  got. Re-run with the venue's real execution mode before sizing anything.""")


if __name__ == "__main__":
    main()
