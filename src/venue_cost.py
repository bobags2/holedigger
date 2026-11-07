"""Venue cost ledger — Layer 0 of the edge engine.

Every strategy claim in this repo is denominated in basis points. A basis point of gross edge
is worth nothing if the instrument's spread is three basis points, so no candidate enters the
register until it has been priced against this ledger at its own instrument, its own hour, and
its own holding period.

The ledger is MEASURED, not assumed: it reads the spread columns of the MT5 bar panels, which
carry the broker's own quoted spread per bar. The one parameter that is not measured is
financing, which is not in the bar data; it is taken from published 2026 schedules and
declared as an assumption at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, mt5_clock

# Published 2026 overnight financing on CFD notionals, annualised, as a fraction of notional.
# Sources: 2026 broker fee analyses. NOT measured - declared assumption, and the one number in
# this module that is not from the tape. A strategy holding N nights pays N times this.
FINANCING_ANNUAL_FRAC = {
    "index": 0.045,     # S&P/Nasdaq/Dow CFDs, SOFR + markup 3-6%
    "metal": 0.030,     # precious metals 2-4%
    "energy": 0.045,
    "fx": 0.020,        # rate differential + markup; 2026 majors are debit-heavy both ways
    "equity": 0.035,
    "soft_ag": 0.045,
}

PREFIX_CLASS = {
    "NAS": "index", "US5": "index", "US30": "index", "GER": "index", "UK": "index",
    "XAU": "metal", "XAG": "metal", "XPT": "metal", "XPD": "metal", "COP": "metal",
    "XTI": "energy", "XBR": "energy", "XNG": "energy",
    "EUR": "fx", "GBP": "fx", "USD": "fx", "AUD": "fx", "NZD": "fx", "CAD": "fx", "CHF": "fx", "JPY": "fx",
}
DEFAULT_CLASS = "equity"


def asset_class(symbol: str) -> str:
    """Classify an instrument for the financing lookup."""
    s = symbol.upper()
    for pfx in sorted(PREFIX_CLASS, key=len, reverse=True):
        if s.startswith(pfx):
            return PREFIX_CLASS[pfx]
    return DEFAULT_CLASS


@dataclass(frozen=True)
class VenueCost:
    """The toll a single trade pays at this venue, in basis points."""

    symbol: str
    asset_class: str
    n_bars: int
    spread_med_bp: float       # median quoted spread: the typical toll
    spread_mean_bp: float
    spread_p90_bp: float       # the toll when the book is thin
    roundtrip_med_bp: float    # entry + exit at the typical quote
    roundtrip_p90_bp: float    # entry + exit at a thin-book quote - the honest planning number
    financing_annual: float
    zero_quote_frac: float = 0.0   # share of bars reporting a zero spread (defect signal)

    def cost_for_hold(self, hours: float, p90: bool = False) -> float:
        """Total toll in bp for a trade held ``hours`` — spread plus financing.

        Financing is a FRACTION OF NOTIONAL PER YEAR, so a hold of ``hours`` costs
        ``annual × hours / (365 × 24)``. Verified defect: the first version divided by 24
        only, which overstated financing by 365× (a one-hour FX hold priced at 8.3 bp instead
        of 0.023 bp) and would have marked every overnight strategy unviable. The 365 is not
        a seasonality assumption — it is what "annual" means.
        """
        rt = self.roundtrip_p90_bp if p90 else self.roundtrip_med_bp
        fin_bp = self.financing_annual * (hours / (365.0 * 24.0)) * 1e4
        return rt + fin_bp

    @property
    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "spread_med_bp": round(self.spread_med_bp, 4),
            "spread_p90_bp": round(self.spread_p90_bp, 4),
            "roundtrip_med_bp": round(self.roundtrip_med_bp, 4),
            "roundtrip_p90_bp": round(self.roundtrip_p90_bp, 4),
            "financing_annual": round(self.financing_annual, 4),
            "zero_quote_frac": round(self.zero_quote_frac, 4),
        }


def load_cost(symbol: str, *, parquet=None) -> VenueCost:
    """Measured venue cost for ``symbol`` from its MT5 M1 panel.

    The spread estimator is a robust percentile of the NONZERO quotes, not a median of all of
    them. Verified defect: EURUSD reports a spread of exactly 0 on 95% of its bars (337,690 of
    355,388), which is a quote-processing artifact rather than a real ECN spread. Taking the
    median of that distribution yields a "free" instrument and any strategy priced against it
    has a free edge that does not exist. Percentiles over the nonzero quotes do not have that
    failure mode, so the ledger uses the 90th percentile as the planning number and the 75th
    as the typical one.
    """
    path = parquet or (config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    df = pd.read_parquet(path, columns=["spread_med_bp", "spread_mean_bp", "spread_p90_bp"])
    df = df.dropna(subset=["spread_med_bp"])
    if df.empty:
        raise ValueError(f"{symbol}: no measured spread in {path}")
    svals = df["spread_med_bp"].to_numpy(dtype=float)
    nonzero = svals[svals > 0]
    if len(nonzero) < 100:
        raise ValueError(
            f"{symbol}: only {len(nonzero)} nonzero spread quotes of {len(svals)} - "
            "this instrument's spread is not usable, its median is dominated by zero quotes"
        )
    zero_frac = float((svals == 0).mean())
    med = float(np.percentile(nonzero, 75))
    mean = float(df["spread_mean_bp"].median())
    p90 = float(np.percentile(nonzero, 90))
    cls = asset_class(symbol)
    return VenueCost(
        symbol=symbol, asset_class=cls, n_bars=len(df),
        spread_med_bp=med, spread_mean_bp=mean, spread_p90_bp=p90,
        roundtrip_med_bp=2 * med, roundtrip_p90_bp=(2 * p90 if p90 > 0 else 2 * med),
        financing_annual=FINANCING_ANNUAL_FRAC.get(cls, 0.045),
        zero_quote_frac=zero_frac,
    )


def load_panel_spread_by_hour(symbol: str, *, parquet=None) -> pd.DataFrame:
    """Median/p90 spread by local hour — the intraday shape of the toll.

    The research record is unanimous that spread varies several-fold through the session, so a
    single per-instrument median overprices quiet hours and underprices thin ones. Any strategy
    that fires at a fixed hour must be priced at that hour.
    """
    path = parquet or (config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    df = mt5_clock.load_mt5_utc(path)
    df = df.dropna(subset=["spread_med_bp"])
    if df.empty:
        raise ValueError(f"{symbol}: no measured spread in {path}")
    df["hour"] = df["ts"].dt.hour
    out = df.groupby("hour")["spread_med_bp"].agg(
        n="count", med="median", p90=lambda s: float(np.percentile(s, 90)),
        mean="mean",
    ).reset_index()
    return out


def ledger(parquet_dir=None, min_bars: int = 10_000) -> pd.DataFrame:
    """The full venue cost ledger, cheapest round trip first."""
    from pathlib import Path

    d = parquet_dir or config.PARQUET_DIR
    rows = []
    for path in sorted(Path(d).glob("*_MT5_M1.parquet")):
        sym = path.name.replace("_MT5_M1.parquet", "")
        try:
            c = load_cost(sym, parquet=path)
        except Exception:
            continue
        if c.n_bars < min_bars:
            continue
        rows.append(c.summary)
    return pd.DataFrame(rows).sort_values("roundtrip_med_bp").reset_index(drop=True)
