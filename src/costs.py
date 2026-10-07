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

# Measured from live tapes (ctbridge.tape; medians; full distribution in
# artifacts/tape_summary.json). A round trip pays the spread once.
LIVE_SPREAD_OBSERVATION = {
    "observed_utc": ["2026-10-05T16:21/17:21 (majors)", "2026-10-06T15:53/16:53 (expanded universe)"],
    "broker": "Fusion Markets",
    "source": "tape_20261005_162141.csv; tape_20261006_155337.csv",
    "note": "Majors: EURUSD/GBPUSD 99.7% zero-spread, XAUUSD 0.19bp, indices 0.26-0.39bp. "
            "Expanded (London hours): copper 2.50, Brent 2.89, aluminium 7.01, zinc 7.22, "
            "coffee ~7.9, soybean 5.81, wheat 8.06, sugar 9.46, cocoa 11-19, nickel 11.45, "
            "lead 13.09, natgas 23.36, platinum 35.24, palladium 59.42 bp.",
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
    # Expanded universe (tape 2026-10-06, London hours).
    "COFARA": 7.878,
    "COFROB": 7.859,
    "CORN": 11.158,
    "COTTON": 11.230,
    "OJ": 20.190,
    "SOYBEAN": 5.810,
    "SUGAR": 9.460,
    "SUGARRAW": 15.004,
    "UKCOCOA": 19.142,
    "USCOCOA": 11.474,
    "WHEAT": 8.055,
    "XALUSD": 7.010,
    "XBRUSD": 2.889,
    "XCUUSD": 2.495,
    "XNGUSD": 23.361,
    "XNIUSD": 11.450,
    "XPBUSD": 13.087,
    "XPDUSD": 59.418,
    "XPTUSD": 35.235,
    "XZNUSD": 7.216,
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
