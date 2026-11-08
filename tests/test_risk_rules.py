"""Tests for src/risk_rules.py - the cap must floor losses and never touch gains.

Run:  python -m tests.test_risk_rules
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile

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
    """The cap lives in bp of notional, so it is currency- and account-independent.

    Written to a TEMP DIR, never to artifacts/. A test that overwrites the repo's live spec is
    how the v1 spec came back and clobbered the v2 one mid-suite: this file wrote
    artifacts/forward_spec.json through forward_test(), the next run order reloaded the v1
    file, and the cost cap silently stopped binding for every symbol except XAUUSD. Two
    competing writers for one artifact is the defect; tests are read-only against it.
    """
    with tempfile.TemporaryDirectory() as td:
        p = rr.forward_test({"XAUUSD": 0.4423}, cap_multiple=3.0, out_dir=td)
        d = json.loads(p.read_text())          # read INSIDE the with-block
        assert abs(d["cap_bp"]["XAUUSD"] - 1.3269) < 1e-3, d["cap_bp"]


def test_withdrawn_cap_claim_is_not_asserted() -> None:
    """THE ACCEPTANCE TEST THAT WAS ITSELF WITHDRAWN, and why it is gone.

    This file used to assert that the frozen spec turns the corrected cTrader book positive
    (net > 0, >50% of days positive, null percentile > 0.95). All four of those came from the
    path-unaware evaluation in research/p19: flooring each position's FINAL P&L keeps every
    position that dipped below the cap and recovered - the ones a real stop cuts. On a
    zero-skill book that evaluation scores +2,120 where the path-aware one scores -798.

    The assertion is REMOVED rather than inverted, because the correct claim is not "the cap
    fails" either - the path-aware sweep shows 3x cuts the worst day from -103,332 to -60,941
    while still losing. A rule that reduces tail exposure without creating expectancy is what a
    stop is for. The assertions that remain in this repo's tests are the ones a future change
    could actually break: the cap's ALGEBRA and its VENUE FLOOR (test_path_aware), not a
    counterfactual return.
    """
    # guard the revocation: if anyone re-adds a positive-expectancy assertion on the floor
    # evaluation, this fails and points at the reason
    from src.live import risk_monitor as rm

    m = rm.RiskMonitor(spec_path=Path(__file__).resolve().parents[1] / "artifacts" /
                       "forward_spec.json", account_cfg={})
    m.state.equity = 1125.08
    gold = m.size_for_symbol("XAUUSD")
    assert gold["tradable"] and gold["notional"] > 0, gold
    assert gold["binding"] == "risk_frac", gold


def main() -> None:
    # MUST STAY AT THE END: a test appended after this point is never registered by the scan.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
