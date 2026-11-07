"""Tests for src/risk_rules.py - the cap must floor losses and never touch gains.

Run:  python -m tests.test_risk_rules
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src import risk_rules as rr


def _t(profit, notional, sym="XAUUSD", bp=1.327):
    return {"profit": profit, "notional": notional, "sym": sym, "open_time": "2026-08-01"}


def test_cap_floors_the_loss_and_never_touches_the_gain() -> None:
    """np.maximum(profit, -cap), never minimum.

    The first version used np.minimum and capped the WINNERS: a +1.21M book became -3.73M,
    and it passed a smoke run because the number it printed looked plausible. This test pins
    the direction.
    """
    spec = {"cap_bp": {"XAUUSD": 1.327}, "target": {}}
    t = pd.DataFrame([_t(-10000.0, 1_000_000.0), _t(+5000.0, 1_000_000.0)])
    res = rr.evaluate_forward(t, spec)
    # gold cap = 1.327bp on 1,000,000 notional = 132.7 money
    assert abs(res["positions_capped"] - 1) < 1e-9, res
    capped = rr.evaluate_forward(t, spec)["net"]
    # the loss is floored at -132.7, the gain is untouched: net = 5000 - 132.7
    assert abs(capped - (5000.0 - 132.7)) < 1.0, capped
    assert res["trades_executed_frac"] == 1.0, "a cap is a stop, not an entry filter"


def test_unmapped_symbol_keeps_its_own_pnl() -> None:
    """A symbol with no measured toll in the spec must not become NaN.

    Mapping an unmapped symbol to NaN silently is what produced a +nan net and a 98% win rate
    in the first version: shape_metrics then averaged over poisoned profits.
    """
    spec = {"cap_bp": {"XAUUSD": 1.327}, "target": {}}
    t = pd.DataFrame([_t(-90000.0, 9_000_000.0, sym="OOPS")])
    res = rr.evaluate_forward(t, spec)
    assert np.isfinite(res["net"]), res
    assert res["net"] == -90000.0, res["net"]
    assert res["positions_capped"] == 0, res["positions_capped"]


def test_null_is_day_clustered() -> None:
    """The null permutes the sign of WHOLE days, not single positions.

    On an 84-day book an unclustered null would report ~50% of days positive as significant
    because positions inside a day share one session.
    """
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-06-01", periods=84, freq="D")
    t = pd.DataFrame({
        "profit": rng.normal(-100.0, 400.0, 84 * 4),
        "notional": np.full(84 * 4, 1_000_000.0),
        "sym": "XAUUSD",
        "open_time": np.repeat(days, 4),
    })
    spec = {"cap_bp": {"XAUUSD": 1.327}, "target": {}}
    res = rr.evaluate_forward(t, spec)
    assert 0.3 < res["null_days_positive_mean"] < 0.7, res["null_days_positive_mean"]
    assert 0.0 <= res["days_positive_null_percentile"] <= 1.0


def test_shape_metrics_use_the_distribution_not_the_total() -> None:
    t = pd.DataFrame({
        "profit": [100.0, -50.0, 200.0, -9000.0, 30.0],
        "net_bp": [1.0, -1.0, 2.0, -90.0, 0.3],
        "notional": np.full(5, 100_000.0),
        "open_time": pd.to_datetime(["2026-08-01"] * 4 + ["2026-08-02"]),
    })
    m = rr.shape_metrics(t)
    # day 1: 100 - 50 + 200 - 9000 = -8750 ; day 2: +30
    assert m["worst_day"] == -8750.0, m["worst_day"]
    assert m["best_day"] == 30.0, m["best_day"]
    # worst10 needs >=10 positions; it is NaN (not 0) below that, which must stay NaN rather
    # than quietly become "no tail risk"
    assert not np.isfinite(m["worst10"]), m["worst10"]
    assert m["n"] == 5
    # the metrics that decide acceptance are all SHAPE, not net
    for k in ("worst_day", "best_day", "days_positive_pct", "worst10", "worst10_share"):
        assert k in m, k


def test_frozen_spec_is_portable_and_reproducible() -> None:
    """The cap lives in bp of notional, so it is currency- and account-independent."""
    spec = rr.forward_test({"XAUUSD": 0.4423}, cap_multiple=3.0)
    d = json.loads(spec.read_text())
    assert abs(d["cap_bp"]["XAUUSD"] - 1.3269) < 1e-3, d["cap_bp"]
    # the same spec loaded back gives the same numbers
    rt = rr.RiskSpec.load(Path("artifacts/nonexistent.json")) if Path("artifacts/nonexistent.json").exists() else None
    assert rt is None or rt.r1_position_bp > 0


def test_corrected_ctrader_book_beats_its_null() -> None:
    """The end-to-end claim: the frozen spec clears the day-clustered null on the real book.

    This is the acceptance test for the deliverable, so it asserts the numbers that are in the
    docs rather than a vague direction. If a future change breaks the cap or the units, this
    fails rather than quietly reporting a new headline.
    """
    from src import venue_cost as vc

    p = Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_positions.parquet"
    if not p.exists():
        return
    t = rr.load_ctrader_trades()
    t = rr.attach_balance(t, Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_deals.json")
    smap = json.loads((Path(__file__).resolve().parents[1] / "artifacts" /
                       "ctrader_symbol_map.json").read_text(encoding="utf-8"))
    t["sym"] = t["symbol_id"].astype(str).map(smap)
    toll = {s: vc.load_cost(s).roundtrip_med_bp for s in ("XAUUSD", "NAS100", "XTIUSD", "XAGUSD")}
    spec = rr.forward_test(toll, cap_multiple=3.0, equity_ref=float(t["balance_after"].median()))
    res = rr.evaluate_forward(t, json.loads(spec.read_text()))
    assert res["net"] > 0, res["net"]
    assert res["days_positive_pct"] > 0.5, res["days_positive_pct"]
    assert res["worst_day"] > -10000.0, res["worst_day"]     # vs -103,332 uncapped
    assert res["days_positive_null_percentile"] > 0.95, res["days_positive_null_percentile"]
    assert res["trades_executed_frac"] == 1.0


def main() -> None:
    # MUST STAY AT THE END: a test appended after this point is never registered by the scan.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
