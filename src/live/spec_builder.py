"""Build the frozen forward spec from the measured cost ledger.

The spec is the contract the live monitor runs against, so every number in it is either a
MEASURED venue toll or the frozen multiplier - never a fit.

WHY IT WAS REBUILT. The first version carried a single symbol (``XAUUSD: 1.3269``) and
``equity_ref: 0.0``, because ``forward_test`` wrote out exactly the dict it was handed without
checking that the instruments you plan to monitor are in it. The monitor's per-symbol cap was
therefore inert for every second symbol - it read ``cap None bp`` and the cost cap never bound.
Nothing failed loudly, which is the failure mode this repo exists to catch.

WHAT IT WRITES NOW. One entry per symbol that has a measured toll on ANY of the two venues, with
both round trips recorded so the cheaper one is visible in the artifact rather than inferred.
Symbols with no measured toll anywhere are listed under ``unpriced`` so their absence is a
recorded gap rather than a silent omission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import venue_cost as vc

DEFAULT_SYMBOLS = ["XAUUSD", "XAGUSD", "XTIUSD", "NAS100", "US500", "XPTUSD", "EURUSD", "USDX"]


def build(symbols: list[str] | None = None, *, cap_multiple: float = 3.0,
          out=None, verbose: bool = True) -> dict:
    syms = symbols or DEFAULT_SYMBOLS
    cap_bp, detail, unpriced = {}, {}, []
    for s in syms:
        try:
            mt5 = vc.load_cost(s)
        except Exception:
            mt5 = None
        try:
            ctr = vc.load_tape_cost(s)
        except Exception:
            ctr = None
        if mt5 is None and ctr is None:
            unpriced.append(s)
            continue
        # the venue you would actually trade on: the cheaper measured round trip
        venue_toll = min(
            (c.roundtrip_med_bp for c in (mt5, ctr) if c is not None), default=float("nan"))
        cap_bp[s] = round(cap_multiple * venue_toll, 4)
        detail[s] = {
            "venue_toll_median_bp": round(venue_toll, 4),
            "cap_bp": round(cap_multiple * venue_toll, 4),
            "mt5_rt_med_bp": round(mt5.roundtrip_med_bp, 4) if mt5 else None,
            "ctrader_rt_med_bp": round(ctr.roundtrip_med_bp, 4) if ctr else None,
            "mt5_n_bars": mt5.n_bars if mt5 else None,
            "ctrader_n_quotes": ctr.n_bars if ctr else None,
        }
        if verbose:
            which = "ctrader" if (ctr is not None and (mt5 is None or ctr.roundtrip_med_bp < mt5.roundtrip_med_bp)) else "mt5"
            print(f"  {s:>7}: toll {venue_toll:>7.3f} bp ({which}) -> cap "
                  f"{cap_multiple:g}x = {cap_bp[s]:>7.3f} bp")
    if unpriced and verbose:
        print(f"  unpriced (no measured spread on either venue): {', '.join(unpriced)}")

    spec = {
        "version": 2,
        "cap_multiple": cap_multiple,
        "cap_bp": cap_bp,
        "cap_detail": detail,
        "unpriced": unpriced,
        # the judging criteria, fixed before the run
        "judge_on": ["net_expectancy_after_costs_ci", "max_drawdown", "ruin_prob",
                     "worst10_share", "trades_executed_frac"],
        "target": {
            "net_expectancy_ci_low_min": 0.0,     # the lower bound must clear zero
            "max_drawdown_pct_max": 25.0,
            "ruin_prob_max": 0.05,
            "worst10_share_max": 0.5,
        },
        "null": "day-clustered sign-flip, per day, 200 worlds",
        "drawdown_stop_frac": 0.25,             # the kill switch, same number the monitor uses
        "derivation": (
            "cap_bp = cap_multiple x the instrument's MEASURED round-trip toll, taken from the "
            "cheaper of the two venues in src/venue_cost. Cost is measured, not fitted. "
            "NOTE: this is the spec's definition; the earlier claim that a 3x-toll cap improves "
            "a book was WITHDRAWN as path-unaware (research/p19). The cap is now a sizing "
            "constraint, not a return claim."
        ),
    }
    if out:
        Path(out).write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return spec


def main() -> None:
    spec_path = Path(__file__).resolve().parents[2] / "artifacts" / "forward_spec.json"
    print("building the frozen forward spec from the measured cost ledger")
    spec = build(out=spec_path)
    print(f"\nwrote {spec_path} with {len(spec['cap_bp'])} priced symbols "
          f"and {len(spec['unpriced'])} unpriced")


if __name__ == "__main__":
    main()
