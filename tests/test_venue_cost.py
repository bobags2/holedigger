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


def test_tape_ledger_prices_symbols_mt5_cannot() -> None:
    """US500/USDX have no MT5 tick history but ARE on the cTrader tape."""
    for sym in ("US500", "USDX", "XPDUSD"):
        c = vc.load_tape_cost(sym)
        assert c.n_bars >= 200, (sym, c.n_bars)
        assert c.spread_med_bp > 0, sym
        assert c.roundtrip_med_bp == 2 * c.spread_med_bp


def test_venue_ratio_is_systematic_by_instrument_class() -> None:
    """cTrader is ~0.5x MT5 on base metals and silver, ~0.9x on gold/indices/energy.

    This is the load-bearing claim of the cross-venue ledger: a strategy papered on one
    venue's numbers can be wrong by 2x on another, and the sign of the error is predictable
    by asset class. If this ever stops holding, the cross-venue corrections in this repo
    stop being valid and must be recomputed.
    """
    v = vc.venue_comparison(["XAUUSD", "XAGUSD", "XCUUSD", "XALUSD", "NAS100", "XTIUSD"])
    v = v.set_index("symbol")
    for s in ("XAGUSD", "XCUUSD", "XALUSD"):
        assert 0.3 < v.loc[s, "ratio_ct_vs_mt5"] < 0.7, (s, v.loc[s, "ratio_ct_vs_mt5"])
    for s in ("XAUUSD", "NAS100", "XTIUSD"):
        assert 0.75 < v.loc[s, "ratio_ct_vs_mt5"] < 1.1, (s, v.loc[s, "ratio_ct_vs_mt5"])


def test_fx_zero_quote_defect_raises_on_the_tape_too() -> None:
    """EURUSD/GBPUSD report exactly 0bp on the tape: same defect, must not be priced as free."""
    for sym in ("EURUSD", "GBPUSD"):
        try:
            vc.load_tape_cost(sym)
            raise AssertionError(f"{sym} should be unusable on the tape (zero quotes)")
        except ValueError:
            pass


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


def test_crypto_costs_are_the_venues_most_expensive_by_a_wide_margin() -> None:
    """BTC/ETH are 15x/67x gold's round trip. This is why weekend crypto is an execution input.

    Without this test the ledger would treat crypto like any other asset, and a 'weekend
    activity is half weekday' finding could be read as an opportunity rather than as a
    warning about fill depth.
    """
    gold = vc.load_cost("XAUUSD")
    for sym in ("BTCUSD", "ETHUSD"):
        c = vc.load_crypto_cost(sym)
        assert c.asset_class == "crypto"
        assert c.roundtrip_med_bp > gold.roundtrip_med_bp * 5, (sym, c.roundtrip_med_bp)

def main() -> None:
    # NOTE: this entry point must stay at the END of the file. An earlier version had tests
    # appended after `if __name__` so they were never registered by this scan: 9 tests passed
    # while a 10th existed and was silently never executed. A test file that reports a
    # smaller count than it has tests is the defect, not a curiosity.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
