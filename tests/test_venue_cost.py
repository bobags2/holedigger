"""Tests for src/venue_cost.py — Layer 0 must not be able to lie.

Run:  python -m tests.test_venue_cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import venue_cost as vc


def test_financing_is_per_year_not_per_day() -> None:
    """A hold of N hours costs annual x N/(365x24), in bp.

    Verified defect in the first version: dividing by 24 instead of 365x24 overstated
    financing by 365x, which priced a one-hour FX hold at 8.3bp instead of 0.023bp and would
    have marked every overnight strategy unviable.
    """
    c = vc.load_cost("EURUSD")
    assert c.financing_annual == 0.02
    # a full year must cost exactly 200bp = 2% of notional
    assert abs(c.cost_for_hold(365 * 24) - (c.roundtrip_med_bp + 200.0)) < 1e-6
    # one hour must be 200/8760 of that
    assert abs(c.cost_for_hold(1) - (c.roundtrip_med_bp + 200.0 / 8760)) < 1e-6
    # financing must be monotone in the hold and never negative
    assert c.cost_for_hold(1) < c.cost_for_hold(24) < c.cost_for_hold(24 * 7)


def test_zero_quote_defect_is_repaired_not_silently_priced() -> None:
    """95% of EURUSD bars quote a zero spread; the median must not be that zero."""
    c = vc.load_cost("EURUSD")
    assert c.zero_quote_frac > 0.5, "expected the defect to be visible on EURUSD"
    # the repaired estimate is the 75th percentile of NONZERO quotes
    assert 0.4 < c.spread_med_bp < 0.7, c.spread_med_bp
    assert c.roundtrip_med_bp > 1.0, "a repaired instrument must not look free"


def test_instrument_that_is_nearly_free_is_priced_as_such() -> None:
    """The cheapest real instruments must land at their true toll."""
    for sym, lo, hi in (("XAUUSD", 0.1, 0.8), ("NAS100", 0.2, 1.0)):
        c = vc.load_cost(sym)
        assert lo < c.spread_med_bp < hi, (sym, c.spread_med_bp)


def test_expensive_instruments_are_flagged_not_averaged_away() -> None:
    """Platinum's 35bp spread must survive into the ledger; no clumping into a mean."""
    c = vc.load_cost("XPTUSD")
    assert c.spread_med_bp > 20.0
    assert c.asset_class == "metal", c.asset_class


def test_ledger_ranks_by_taken_toll() -> None:
    L = vc.ledger()
    assert not L.empty
    assert list(L.columns)[:4] == ["symbol", "asset_class", "spread_med_bp", "spread_p90_bp"]
    # sorted ascending by round-trip
    rts = L["roundtrip_med_bp"].tolist()
    assert rts == sorted(rts)
    # XPTUSD must appear near the expensive end, XAUUSD near the cheap end
    assert L.iloc[0]["symbol"] in ("XAUUSD", "NAS100", "EURUSD")
    assert "XPTUSD" in L.tail(10)["symbol"].tolist()


def test_zero_quote_instrument_raises_when_unusable() -> None:
    """If every spread quote is zero there is nothing to price; that must raise, not infer."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "FAKE_MT5_M1.parquet"
        pd.DataFrame({
            "spread_med_bp": np.zeros(500),
            "spread_mean_bp": np.zeros(500),
            "spread_p90_bp": np.zeros(500),
        }).to_parquet(p)
        try:
            vc.load_cost("FAKE", parquet=p)
            raise AssertionError("expected ValueError: no nonzero spread quotes")
        except ValueError:
            pass


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
