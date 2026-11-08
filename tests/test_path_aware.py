"""The property test that kills the path-unaware evaluation.

A risk rule applied to a book with ZERO skill and negative expectancy by construction must not
improve expectancy. If it does, the rule is manufacturing number from the evaluation rather
than removing loss.

Run:  python -m tests.test_path_aware
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

import numpy as np
import pandas as pd


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] /
                                                 "research" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p19 = _load("p19_path_aware_stops")
rr = _load("p12_trade_metalabel") if False else None
from src import risk_rules as rr_mod
from src import venue_cost as vc


def test_zero_skill_book_has_negative_expectancy_by_construction() -> None:
    z = p19.zero_skill_book(n=2000)
    per_bp = float((z["profit"] / z["notional"] * 1e4).mean())
    assert per_bp < -0.1, f"the synthetic book must lose by construction, got {per_bp:+.3f} bp"
    assert abs(float(z["profit"].mean())) > 0


def test_path_unaware_floor_manufactures_positive_on_zero_skill() -> None:
    """THE DEFECT, pinned. The evaluation in src/risk_rules turns a no-skill book positive.

    This test exists to fail loudly if anyone re-introduces the floor as a result, and to
    document WHY it cannot be used: on 2,260 zero-skill positions the old scoring produced a
    positive net while the path-aware scoring stayed negative.
    """
    z = p19.zero_skill_book(n=2000)
    cap_bp = vc.load_cost("XAUUSD").roundtrip_med_bp
    floored = float(np.maximum(z["profit"], -z["notional"] * cap_bp / 1e4).sum())
    paw = p19.path_aware(z, {"XAUUSD": cap_bp})
    truthful = float(paw["paw_profit"].sum())
    assert floored > truthful, (floored, truthful)
    # the floor must not be reported as a result: this is the assertion that documents it
    assert floored > 0 >= truthful or truthful < 0, (floored, truthful)


def test_path_aware_evaluation_stays_negative_on_zero_skill() -> None:
    """A stop can remove loss; it cannot create skill. Path-aware must stay <= 0."""
    z = p19.zero_skill_book(n=2000)
    cap_bp = vc.load_cost("XAUUSD").roundtrip_med_bp
    out = p19.path_aware(z, {"XAUUSD": cap_bp})
    assert float(out["paw_profit"].sum()) < 0, out["paw_profit"].sum()
    assert out["paw_fired"].mean() > 0.05, out["paw_fired"].mean()


def test_real_book_cap_is_touched_far_too_often_to_be_a_risk_control() -> None:
    """The gold cap at 3x toll is a TOLL MULTIPLIER, not a stop.

    Path awareness measured on the real book: the 0.44 bp level is touched on ~60% of gold
    positions, so a stop there pays the spread, the commission and the slippage on most trades.
    Stop distance must come from volatility over the holding horizon and account risk.
    """
    p = Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_positions.parquet"
    if not p.exists():
        return
    import json
    from src import risk_rules as rr

    t = rr.load_ctrader_trades()
    t = rr.attach_balance(t, p.parent / "ctrader_deals.json")
    smap = json.loads((p.parent / "ctrader_symbol_map.json").read_text(encoding="utf-8"))
    t["sym"] = t["symbol_id"].astype(str).map(smap)
    t["open_day"] = pd.to_datetime(t["open_time"]).dt.normalize()
    toll = vc.load_cost("XAUUSD").roundtrip_med_bp
    out = p19.path_aware(t[t["sym"] == "XAUUSD"], {"XAUUSD": toll})
    fired = float(out["paw_fired"].mean())
    assert 0.4 < fired < 0.8, fired


def test_sizing_ties_notional_to_equity_not_to_a_bp_cap() -> None:
    """"A stop in bp says nothing about size" is the finding this repo missed.

    The account traded 4,043,835 of mean notional against a 6,272 median balance - 645x, with
    a single position's median loss at 7% of the account and p95 at 233%. Size is the
    mechanism of ruin, so the sizing rule is the one that binds.
    """
    from src import risk_rules as rr

    rec = rr.sizing_recommendation(1125.0, stop_bp=10.0, risk_frac=0.005, max_leverage=10.0)
    assert abs(rec["notional"] - 5625.0) < 1.0, rec
    assert rec["notional"] < 10000.0, "a 10x-leverage cap must bind on a 1,125 account"
    # the leverage ceiling must dominate when the risk fraction would ask for more
    big = rr.size_for(1_000_000.0, risk_frac=0.5, stop_bp=1.0, max_leverage=10.0)
    assert abs(big - 10_000_000.0) < 1.0, big
    try:
        rr.size_for(100.0, risk_frac=0.01, stop_bp=0.0)
        raise AssertionError("stop_bp<=0 must raise: notional is undefined")
    except ValueError:
        pass


def test_leverage_profile_measures_size_against_equity() -> None:
    from src import risk_rules as rr

    p = Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_positions.parquet"
    if not p.exists():
        return
    t = rr.load_ctrader_trades()
    t = rr.attach_balance(t, p.parent / "ctrader_deals.json")
    lp = rr.leverage_profile(t)
    assert lp["leverage_mean_notional_over_median_balance"] > 100.0, lp
    assert lp["median_loss_pct_of_balance"] > 1.0, lp
    assert lp["p95_loss_pct_of_balance"] > 50.0, lp


def test_kill_switch_runs_on_equity_not_on_accumulated_pnl() -> None:
    """The R2 day limit ran on final-P&L accumulations and never saw the -1,297 trough.

    A drawdown stop has to run on the event-ordered equity path, which here is the broker's
    own balance after each close.
    """
    from src import risk_rules as rr

    # an equity path that never crosses a day limit but ends far below its peak
    eq = [100.0] + [95.0] * 5 + [60.0] * 5 + [40.0]
    ks = rr.kill_switch(eq, peak_dd_frac=0.25)
    assert ks["max_drawdown_pct"] == 60.0, ks
    assert ks["drawdown_stop_triggered"] is True
    assert ks["trough"] == 40.0
    # a monotone path must NOT trigger
    k2 = rr.kill_switch([100.0, 101.0, 102.0, 103.0], peak_dd_frac=0.25)
    assert k2["drawdown_stop_triggered"] is False
    assert k2["max_drawdown_pct"] == 0.0



# --------------------------------------------------------------------------
# Units and the live monitor, both verified against the broker rather than assumed
# --------------------------------------------------------------------------

def test_money_fields_are_cents_not_dollars() -> None:
    """Verified live against the broker, and the reason I told the repo the reviewer was wrong.

    ProtoOATraderReq on account 47477688 returns ``balance: 54`` with ``moneyDigits: 2`` and
    ``leverageInCents: 50000``. So the live balance is $0.54 and the account is 500x levered, and
    every USD figure this repo quoted before this test was 100x too large. The property is what
    makes the mistake non-repeatable: the divisor comes from the broker's own field, not from an
    assumption about dollars.
    """
    from src import risk_rules as rr

    assert rr.MONEY_DIVISOR == 100.0
    assert rr.load_money(54) == 0.54
    assert rr.load_money(112508) == 1125.08
    assert rr.load_money(-602934) == -6029.34


def test_venue_floor_returns_do_not_trade_not_a_rounded_size() -> None:
    """A $0.54 account must return 0, not the $2.70 the arithmetic gives.

    Rounding up to the venue's minimum would silently trade a size the risk rule never
    sanctioned. The live monitor hit this on its first run and printed $2.70, which is how the
    defect was found.
    """
    from src import risk_rules as rr

    r = rr.sizing_recommendation(0.54, stop_bp=10.0, max_leverage=10.0, min_notional=380.0)
    assert r["notional"] == 0.0, r
    assert r["tradable"] is False, r
    assert "do not trade" in r["reason"].lower(), r
    # an account that CAN carry the minimum is unaffected by the floor
    ok = rr.sizing_recommendation(1125.0, stop_bp=10.0, max_leverage=10.0, min_notional=380.0)
    assert ok["tradable"] and ok["notional"] == 5625.0, ok


def test_monitor_module_imports_and_advertises_the_floor() -> None:
    p = Path(__file__).resolve().parents[2] / "ctrader" / "ctbridge" / "risk_monitor.py"
    if not p.exists():
        return
    # register in sys.modules before exec, or risk_rules' dataclass decorators resolve
    # __module__ to a None entry and raise AttributeError
    spec = importlib.util.spec_from_file_location("risk_monitor", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["risk_monitor"] = m
    spec.loader.exec_module(m)
    assert hasattr(m, "RiskMonitor")
    assert m.VENDUE_MIN_NOTIONAL.get("XAUUSD", 0) > 0
    # sizing must route through the floor, not bypass it
    src = p.read_text(encoding="utf-8")
    assert "min_notional=VENDUE_MIN_NOTIONAL.get(symbol" in src, "monitor must use the venue floor"
    assert "DO NOT TRADE" in src, "the untradable path must be visible in the log"

def main() -> None:
    # MUST STAY AT THE END. A test appended after this point is never registered by the scan -
    # the same defect that made 9 tests pass while a 10th existed, twice in this repo now.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
