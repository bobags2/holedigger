"""Broker cost model.

Commission inputs: Fusion Markets, quoted by the user as USD per 1.00 lot. The table
below is *per side* (the cTrader convention); round trip doubles it. If your quotes are
actually round-turn figures, set ``PER_SIDE = False`` and everything halves.

Spread: measured live by the user 2026-10-05 (liquid session): EURUSD 0.0 pips,
GBPUSD 0.0 pips, XAUUSD up to ~0.4 pips. Adopted spread for evaluation is 0.0 bp pending
the live tape (``ctbridge`` quote logging) — break_even_spread_bp() is the metric that
matters until then: it is the spread a candidate could pay and still break even.
"""
from __future__ import annotations

# USD per 1.00 lot, per side (see PER_SIDE note above). Fusion Markets, 2026-10-05.
# Zero-commission symbols still pay the spread; their round-trip cost is spread-only.
COMMISSION_PER_LOT = {
    "XAUUSD": 4.96,
    "EURUSD": 2.51,
    "GBPUSD": 2.97,
    "XAGUSD": 3.69,
    "XTIUSD": 0.0,
    "US500": 0.0,
    "NAS100": 0.0,
    "USDX": 0.0,
}
PER_SIDE = True

# Units of the base asset (or ounces) per 1.00 lot — standard cTrader contract sizes.
# Only needed for symbols with non-zero commission.
LOT_UNITS = {"XAUUSD": 100.0, "XAGUSD": 5000.0, "EURUSD": 100_000.0, "GBPUSD": 100_000.0}

# Measured from the live tape (ctbridge.tape, 2026-10-05 16:21–17:21 UTC; medians;
# full distribution in artifacts/tape_summary.json). A round trip pays the spread once.
LIVE_SPREAD_OBSERVATION = {
    "observed_utc": "2026-10-05T16:21/17:21",
    "broker": "Fusion Markets",
    "source": "tape_20261005_162141.csv (44,369 quotes)",
    "note": "EURUSD/GBPUSD 99.7% zero-spread; XAUUSD 0.19 bp; indices 0.26-0.39 bp; "
            "silver 3.28, oil 3.23, USDX 1.86 bp",
}
ADOPTED_SPREAD_BP = {
    "EURUSD": 0.000,
    "GBPUSD": 0.000,
    "XAUUSD": 0.194,
    "XAGUSD": 3.282,
    "XTIUSD": 3.233,
    "US500": 0.386,
    "NAS100": 0.258,
    "USDX": 1.858,
}


def known_symbol(symbol: str) -> bool:
    return symbol in COMMISSION_PER_LOT


def commission_round_trip_bp(symbol: str, price: float) -> float | None:
    """Round-trip commission in bp of notional. None when costs are unknown."""
    if symbol not in COMMISSION_PER_LOT:
        return None
    if COMMISSION_PER_LOT[symbol] == 0.0:
        return 0.0
    if symbol not in LOT_UNITS:
        return None
    notional = LOT_UNITS[symbol] * price
    per_side_bp = COMMISSION_PER_LOT[symbol] / notional * 1e4
    return per_side_bp * (2.0 if PER_SIDE else 1.0)


def total_round_trip_bp(symbol: str, price: float, spread_bp: float | None = None) -> float | None:
    """Commission + spread, in bp of notional. None when costs are unknown."""
    comm = commission_round_trip_bp(symbol, price)
    if comm is None:
        return None
    if spread_bp is None:
        spread_bp = ADOPTED_SPREAD_BP.get(symbol, 0.0)
    return comm + spread_bp


def net_edge_bp(edge_bp: float, symbol: str, price: float, spread_bp: float | None = None) -> float | None:
    cost = total_round_trip_bp(symbol, price, spread_bp)
    return None if cost is None else edge_bp - cost


def break_even_spread_bp(edge_bp: float, symbol: str, price: float) -> float | None:
    """How much spread the candidate can pay per round trip and still break even.

    For unknown-cost symbols this equals the gross edge (commission assumed zero).
    """
    comm = commission_round_trip_bp(symbol, price)
    return edge_bp - (comm or 0.0)
