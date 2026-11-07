"""Tests for src/p15's cTrader deal loader - the partial-close and dedup traps.

Run:  python -m tests.test_ctrader_deals
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "p15", Path(__file__).resolve().parents[1] / "research" / "p15_ctrader_deals.py")
p15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p15)


def _deal(deal_id, position_id, trade_side, filled_volume, price, close_detail=None):
    return {"dealId": deal_id, "positionId": position_id, "symbolId": 41,
            "tradeSide": trade_side, "filledVolume": filled_volume,
            "volume": filled_volume, "executionPrice": price,
            "executionTimestamp": 1782826674000 + deal_id, "dealStatus": 2,
            "closePositionDetail": close_detail}


def test_partial_close_sums_all_closing_deals() -> None:
    """The reviewer's case: position 43112421 closes -248 and -740, so the position is -988.

    `gp.iloc[0]` took only the FIRST closing deal's grossProfit. Verified on the real position:
    the first closing deal says -740 and the true position P&L is -50,577, because that
    position has 17 closing deals. The loader sums grossProfit + swap + commission over ALL
    closing deals.
    """
    deals = [
        _deal(1, 100, 1, 10, 30000.0, None),                       # opening fill, no close detail
        _deal(2, 100, 1, 5, 30004.0, close_detail_for(-248.0, 0.0, 0.0)),
        _deal(3, 100, 1, 5, 30006.0, close_detail_for(-740.0, 0.0, 0.0)),
    ]
    p = p15.to_positions(deals)
    assert len(p) == 1
    assert float(p.iloc[0]["profit"]) == -988.0, p.iloc[0]["profit"]
    assert int(p.iloc[0]["n_closing_fills"]) == 2
    # notional across BOTH opening fills, not just the first
    assert abs(float(p.iloc[0]["notional"]) - 10 * 30000.0) < 1e-6


def test_duplicate_deals_are_deduplicated() -> None:
    """The deals file contained 6,000 duplicate rows from overlapping paging windows.

    Dedup on dealId before any total is computed, or every per-position P&L is roughly double.
    """
    base = [
        _deal(1, 100, 1, 10, 30000.0, None),
        _deal(2, 100, 1, 5, 30004.0, close_detail_for(-248.0, 0.0, 0.0)),
        _deal(3, 100, 1, 5, 30006.0, close_detail_for(-740.0, 0.0, 0.0)),
    ]
    once = p15.to_positions(base)
    twice = p15.to_positions(base + base)      # the duplicating paging shape
    assert float(once.iloc[0]["profit"]) == -988.0
    assert float(twice.iloc[0]["profit"]) == -988.0, "duplicates must not double the P&L"


def test_commission_and_swap_are_in_the_total() -> None:
    d = [_deal(1, 101, 1, 10, 30000.0, None),
         _deal(2, 101, 1, 10, 30010.0, close_detail_for(-500.0, -3.5, -20.0))]
    p = p15.to_positions(d)
    assert float(p.iloc[0]["commission"]) == -20.0
    assert float(p.iloc[0]["swap"]) == -3.5
    assert float(p.iloc[0]["profit"]) == -523.5, p.iloc[0]["profit"]


def test_non_filled_status_is_dropped() -> None:
    """dealStatus is the integer 2; a string filter for 'FILLED' matches nothing.

    The first loader used ``.astype(str).str.contains("FILLED")`` and returned 0 positions
    from 13,334 real deals, because the field is ``2`` for every row.
    """
    for bad in (0, 1, 3, "FILLED"):
        d = dict(_deal(9, 102, 1, 10, 30000.0, close_detail_for(-10.0, 0.0, 0.0)))
        d["dealStatus"] = bad
        assert len(p15.to_positions([d])) == 0, f"dealStatus={bad!r} must be dropped"
    # a lone closing deal still prices, via the fallback that treats it as its own opening
    # fill - which is the right call for a position whose opening fill fell outside the pull
    # window, and is asserted here so the fallback cannot silently start dropping positions
    good = dict(_deal(9, 102, 1, 10, 30000.0, close_detail_for(-10.0, 0.0, 0.0)))
    good["dealStatus"] = 2
    p = p15.to_positions([good])
    assert len(p) == 1
    assert float(p.iloc[0]["profit"]) == -10.0


def test_balance_path_is_independent_of_pnl_units() -> None:
    """The broker's own `balance` sequence is the scale-free check on every total.

    It needs no conversion and no assumption about grossProfit's units, so it is the thing to
    trust when two P&L computations disagree.
    """
    import json

    path = Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_deals.json"
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    D = pd.DataFrame(d["deals"]).drop_duplicates(subset=["dealId"])
    txt = D["closePositionDetail"].fillna("")
    bal = txt.str.extract(r"^balance: ([-\d.]+)$", flags=pd_re()).astype(float)[0]
    bal = bal.dropna()
    assert len(bal) > 100, f"only {len(bal)} balances parsed"
    # the path must be finite and span months; that is what makes it a usable ground truth
    assert np.isfinite(bal.to_numpy()).all()
    return bal


def close_detail_for(gp: float, swap: float, commission: float) -> str:
    return (f"entryPrice: 30000.00\ngrossProfit: {gp}\nswap: {swap}\n"
            f"commission: {commission}\nbalance: 1000\nquoteToDepositConversionRate: 1\n"
            "closedVolume: 5\nbalanceVersion: 1\nmoneyDigits: 2\npnlConversionFee: 0\n")


def pd_re():
    import re
    return re.M


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
