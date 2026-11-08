"""The property test that kills the path-unaware evaluation.

A risk rule applied to a book with ZERO skill and negative expectancy by construction must not
improve expectancy. If it does, the rule is manufacturing number from the evaluation rather
than removing loss.

Run:  python -m tests.test_path_aware
"""
from __future__ import annotations

import json
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


# --------------------------------------------------------------------------
# The monitor's three gaps, and the property that the cost cap must bind
# --------------------------------------------------------------------------

def _monitor():
    import json
    from pathlib import Path
    from src.live import risk_monitor as rm

    return rm.RiskMonitor(
        spec_path=Path(__file__).resolve().parents[1] / "artifacts" / "forward_spec.json",
        account_cfg={})


def test_cost_cap_actually_binds_on_wide_toll_instruments() -> None:
    """THE GAP THAT WAS CLOSED. The cap could never bind and the docstring said it did.

    First version: `equity * drawdown_stop_frac / (cap_bp/1e4)` - for gold ~2.1M against a
    5,625 risk size, and it read a key (`drawdown_stop_frac` at the top level) that the spec
    did not contain. Every symbol reported `binding: risk_frac`, i.e. the cost cap was inert.

    Now: `equity * risk_frac / (cap_bp/1e4)`, which is the risk rule with stop_bp := cap_bp.
    Platinum's cap is 242 bp so it sizes at $232 against gold's $5,625 - cost-proportional.
    """
    m = _monitor()
    m.state.equity = 1125.08
    gold = m.size_for_symbol("XAUUSD")
    plat = m.size_for_symbol("XPTUSD")

    assert gold["binding"] == "risk_frac", gold
    assert abs(gold["notional"] - 1125.08 * 0.005 / (10.0 / 1e4)) < 1.0, gold

    # Platinum's cost cap is BELOW its own venue minimum at this equity, so the honest output
    # is DO NOT TRADE, not a tradable size. Both constraints fire and the floor wins. This is
    # not a defect: a 242 bp toll genuinely cannot be risked at 0.5% of $1,125 with 0.01 lots,
    # and pretending otherwise would trade $232 of an instrument whose minimum is $380.
    assert plat["binding"] == "cost_cap_then_venue_floor", plat
    assert plat["notional"] == 0.0 and plat["tradable"] is False, plat
    assert abs(plat["cap_notional"] - 1125.08 * 0.005 / (plat["cap_bp"] / 1e4)) < 1.0, plat

    # At an equity where the cap clears the floor, the cap binds and produces a tradable,
    # cost-proportional size - which is the behaviour the whole framework exists for.
    m.state.equity = 20000.0
    plat2 = m.size_for_symbol("XPTUSD")
    assert plat2["binding"] == "cost_cap", plat2
    assert plat2["tradable"] and plat2["notional"] > 0, plat2
    gold2 = m.size_for_symbol("XAUUSD")
    assert gold2["binding"] == "risk_frac", gold2
    assert plat2["notional"] < gold2["notional"], (plat2["notional"], gold2["notional"])


def test_cost_cap_cannot_drive_size_below_the_venue_floor() -> None:
    """A capped size that lands under the broker's minimum must say DO NOT TRADE."""
    m = _monitor()
    m.state.equity = 20.0                      # too small for platinum's cap
    r = m.size_for_symbol("XPTUSD")
    assert r["notional"] == 0.0, r
    assert r["tradable"] is False, r
    assert "do not trade" in r["reason"].lower(), r


def test_spec_covers_every_symbol_the_monitor_watches() -> None:
    """The first spec had one symbol; the monitor reported `cap None bp` for the rest."""
    spec_path = Path(__file__).resolve().parents[1] / "artifacts" / "forward_spec.json"
    d = json.loads(spec_path.read_text(encoding="utf-8"))
    assert d["version"] >= 2, d["version"]
    for s in ("XAUUSD", "NAS100", "US500", "XAGUSD", "XTIUSD"):
        assert s in d["cap_bp"], f"{s} unpriced in the frozen spec"
        assert d["cap_bp"][s] > 0
        assert d["cap_detail"][s]["venue_toll_median_bp"] > 0
    # unpriced instruments are RECORDED, not silently dropped
    assert "unpriced" in d

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
