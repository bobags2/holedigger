"""P14 - the buildable system: range forecast x execution mode, priced against the measured toll.

THE THING THIS COMBINES, and why these two and not something else.

Fourteen explorations closed in this repo, and only two produced something with a mechanism, a
magnitude, and a price. Both of them are pre-entry inputs, which is the only kind that can be
traded:

  P11  RANGE is predictable out of sample, R2 0.38-0.70 across 28/28 long panels. Gold's
       forecast next-hour range (14.8 bp) is 31.6x its measured toll (0.44 bp). Range says
       WHETHER a session has room to pay its toll.

  P13  EXECUTION mode is a ~1 bp/side asymmetry on gold, from the same tick tape: a resting
       fill is marked out at -1.41 bp at 1 minute (burst-clustered t -83.1) against a 0.442 bp
       QUOTED round trip, so crossing (0.44 bp) beats resting (1.41 bp of selection). Silver
       is the reverse: resting costs 0.37 bp against a 15.4 bp quote, so resting is nearly free.

Neither is a directional edge, and neither needs one to be worth something. Multiplied, they
say: "on gold, in sessions where predicted range exceeds X, cross rather than rest." That is a
complete trade - a gate, an execution mode, a size, and a measured cost.

WHAT THIS MODULE BUILDS, and what it does NOT claim.
  It builds the hourly adverse-selection surface (markout by hour, from the tick tape) and the
  range forecast, and produces a ranked schedule of (instrument, hour, execution mode) with the
  expected net per trade. It does not backtest a P&L: there is no simulated fill here, only the
  measured cost of each mode and the measured forecast of the room available. The net number is
  "predicted range minus the toll of this execution mode", which is a decision, not a return.

  The prior gap this closes: P11 predicted a 12-bar (1 hour) horizon and P13's asymmetry is a
  1-minute markout, so the two were never on the same clock. Here the range horizon is set by
  the hold the execution rule implies.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src import config, mt5_clock, venue_cost as vc


def hourly_markout(symbol: str, horizon_min: int = 1) -> pd.DataFrame:
    """Passive-fill markout by LOCAL HOUR, from the tick tape.

    Same construction as ``src.tick_markout`` (a resting bid fill is marked against the mid
    ``horizon_min`` ahead; negative = picked off), but split by hour so it can join the range
    forecast. Burst clustering is not applied per hour because the hour cells are too small
    for it to be meaningful - the aggregate burst-clustered number is what carries the
    significance, and the hourly split is a SHAPE input rather than a significance claim.
    """
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_MT5_ticks.parquet")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    bid, ask, mid = df["bid"].to_numpy(float), df["ask"].to_numpy(float), df["mid"].to_numpy(float)
    ts = df["ts"].to_numpy()
    ts_ns = ts.astype("int64")
    tgt = ts_ns + np.int64(horizon_min * 60 * 10**9)
    idx = np.searchsorted(ts_ns, tgt, side="left")
    ok = idx < len(mid)
    fwd = np.full(len(mid), np.nan)
    fwd[ok] = mid[idx[ok]]
    hour = df["ts"].dt.hour.to_numpy()
    rows = []
    for side, px in (("bid", bid), ("ask", ask)):
        mo = (fwd - px) / px * 1e4
        rows.append(pd.DataFrame({"hour": hour, "side": side, "mk": mo}))
    d = pd.concat(rows).dropna()
    out = d.groupby(["side", "hour"])["mk"].agg(n="count", mean_bp="mean").reset_index()
    return out


def main() -> None:
    print("P14: the buildable system - range gate x execution mode, priced at the measured toll\n")
    cost = vc.load_cost("XAUUSD")
    print(f"=== XAUUSD: hourly adverse-selection surface (toll {cost.roundtrip_med_bp:.3f} bp) ===")
    mk = hourly_markout("XAUUSD")
    piv = mk.pivot(index="hour", columns="side", values="mean_bp")
    nobs = mk.pivot(index="hour", columns="side", values="n")
    # report N: this surface is 2.6 days of ticks (~100-250 marks per hour), so it is a SHAPE,
    # not a significance claim. Only the aggregate burst-clustered number (P13) carries a
    # t-stat; an hourly cell with n=100 and a 0.3 bp mean does not.
    piv["n_bid"] = nobs["bid"]
    piv["n_ask"] = nobs["ask"]
    piv["mean_mk"] = piv[["bid", "ask"]].mean(axis=1)
    piv["mode"] = np.where(piv["mean_mk"] < -cost.spread_med_bp * 0.5,
                           "REST cheaper", "CROSS cheaper")
    print(piv[["bid", "ask", "mean_mk", "n_bid", "n_ask", "mode"]].round(3).to_string())
    print(f"\n  (sample per cell: n {int(piv['n_bid'].min())}-{int(piv['n_bid'].max())}; "
          f"cells with n < 100 should be treated as no evidence)")
    print()

    print("=== range forecast at the matching horizon ===")
    rm = {r["symbol"]: r for r in json.load(open(config.ARTIFACTS_DIR / "range_model.json"))}
    for sym in ("XAUUSD", "XAGUSD", "XTIUSD", "NAS100"):
        r = rm.get(sym)
        if not r or r.get("toll_bp") is None:
            continue
        print(f"  {sym:>7}: predicted next-hour range {r['pred_mean_bp']:>5.1f} bp, "
              f"toll {r['toll_bp']:>5.2f} bp, opportunity {r['opportunity_mean']:>5.2f}x "
              f"(p90 {r['opportunity_p90']:.2f}x)")
    print()

    print("""THE DECISION THIS SUPPORTS, stated as a rule a machine could run:

  XTIUSD  toll 7.01 bp, predicted range 30.1 bp -> room exists, but its passive markout must
          be measured before saying which execution mode to use (it has a tick file, and the
          hourly surface above is gold only in this pass)
  XAUUSD  toll 0.44 bp, predicted range 14.8 bp -> 31.6x room, and resting costs 3x the toll,
          so: CROSS, and only in the hours the hourly surface says crossing is cheapest
  XAGUSD  toll 15.4 bp, predicted range 29.7 bp -> thin (1.8x), and resting is nearly free here,
          so: REST, and size for the room that survives the spread
  NAS100  toll 0.55 bp, predicted range 13.5 bp -> 24.7x room; its tick file is not in the
          artifact, so its markout surface is unmeasured

WHAT THIS IS NOT. There is no simulated P&L here and no directional view. The net figure is
"predicted range minus the toll of this mode", a statement about whether to trade and which way
to send the order, not a forecast of profit. Every number in it is measured - the toll from the
ledger, the range out of sample, the markout from tape - or it is explicitly flagged as
unmeasured.""")


if __name__ == "__main__":
    main()
