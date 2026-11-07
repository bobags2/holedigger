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
    "crypto": 0.060,      # highest average on the venue: 4-8% after markup
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


def load_tape_cost(symbol: str, *, tape_dir=None, min_quotes: int = 200) -> VenueCost:
    """Measured venue cost from the cTrader live quote tape instead of the MT5 panels.

    The tape is the second of two independent sources and it changes answers. It records
    bid/ask/spread_bp per spot event from the cTrader Open API feed, so it carries the quoted
    spread for symbols the MT5 panels don't cover (US500, USDX, and the agricultural CFDs), and
    it prices a venue we can actually choose between.

    Same zero-quote repair as ``load_cost``: FX majors resolve to a single price on this feed
    (EURUSD and GBPUSD report exactly 0.000 bp), so the estimate is a percentile of the
    nonzero quotes and ``zero_quote_frac`` records that the instrument is unusable on it.
    """
    from pathlib import Path

    d = tape_dir or Path(__file__).resolve().parents[2] / "ctrader" / "tape"
    frames = [Path(p) for p in sorted(d.glob("tape_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no cTrader tape files in {d}")
    rows = []
    for p in frames:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        df = df[df["symbol"] == symbol]
        if not df.empty:
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"{symbol}: not present in the cTrader tape ({len(frames)} files scanned)")
    df = pd.concat(rows)
    sv = df["spread_bp"].dropna()
    sv = sv[np.isfinite(sv)]
    nonzero = sv[sv > 0].to_numpy()
    if len(nonzero) < min_quotes:
        raise ValueError(
            f"{symbol}: only {len(nonzero)} usable nonzero spread quotes on the tape "
            f"({len(sv)} total) - unusable on this venue, same zero-quote defect as MT5's EURUSD"
        )
    zero_frac = float((sv == 0).mean())
    med = float(np.percentile(nonzero, 75))
    p90 = float(np.percentile(nonzero, 90))
    cls = asset_class(symbol)
    return VenueCost(
        symbol=symbol, asset_class=cls, n_bars=int(len(sv)),
        spread_med_bp=med, spread_mean_bp=float(nonzero.mean()), spread_p90_bp=p90,
        roundtrip_med_bp=2 * med, roundtrip_p90_bp=2 * p90,
        financing_annual=FINANCING_ANNUAL_FRAC.get(cls, 0.045),
        zero_quote_frac=zero_frac,
    )


def load_crypto_cost(symbol: str, *, parquet=None) -> VenueCost:
    """Measured venue cost for BTCUSD/ETHUSD from the 5-minute mirror of the broker's tape.

    Crypto is the only market that runs on a weekend, so it is the only instrument class whose
    spread can be added when everything else is closed. It is in the ledger precisely so nobody
    has to guess: **BTCUSD costs 6.6 bp and ETHUSD 29.4 bp round trip, i.e. 15x and 67x gold's
    0.44 bp.** That is the number that decides whether any weekend crypto idea is worth
    attempting at all, and it is why the weekend activity finding (books at 0.37-0.51x weekday
    tick rate) is an execution input and not the basis of a strategy.
    """
    from pathlib import Path

    d = parquet or (Path(__file__).resolve().parents[1] / "data" / "parquet" / f"{symbol}_crypto_5m.parquet")
    if not Path(d).exists():
        raise FileNotFoundError(f"{symbol}: no crypto panel at {d} — regenerate with research/p6_weekend_crypto.py")
    df = pd.read_parquet(d)
    sv = df["spread_med_bp"].to_numpy(dtype=float)
    nonzero = sv[sv > 0]
    if len(nonzero) < 30:
        raise ValueError(f"{symbol}: {len(nonzero)} usable nonzero spread bars")
    med = float(np.percentile(nonzero, 75))
    p90 = float(np.percentile(nonzero, 90))
    return VenueCost(
        symbol=symbol, asset_class="crypto", n_bars=int(len(df)),
        spread_med_bp=med, spread_mean_bp=float(nonzero.mean()), spread_p90_bp=p90,
        roundtrip_med_bp=2 * med, roundtrip_p90_bp=2 * p90,
        financing_annual=FINANCING_ANNUAL_FRAC["crypto"], zero_quote_frac=float((sv == 0).mean()),
    )


def venue_comparison(symbols: list[str], *, min_quotes: int = 200) -> pd.DataFrame:
    """MT5 vs cTrader measured spread, per symbol, with the ratio that decides the verdict.

    The ratio is the point: cTrader is 0.84-0.94x MT5 on NAS100, gold, energy, platinum and
    natural gas, but 0.43-0.53x on silver, aluminium, copper and lead. Base-metal and silver
    edges that are cost-submerged on MT5 can clear on cTrader, and a strategy papered on one
    venue's numbers can be wrong by 2x on another.
    """
    rows = []
    for s in symbols:
        rec = {"symbol": s}
        for label, fn in (("mt5", load_cost), ("ctrader", load_tape_cost)):
            try:
                c = fn(s) if label == "mt5" else load_tape_cost(s, min_quotes=min_quotes)
                rec[f"{label}_med_bp"] = round(c.spread_med_bp, 4)
                rec[f"{label}_p90_bp"] = round(c.spread_p90_bp, 4)
                rec[f"{label}_rt_bp"] = round(c.roundtrip_med_bp, 4)
                rec[f"{label}_n"] = c.n_bars
            except Exception:
                rec[f"{label}_med_bp"] = None
                rec[f"{label}_rt_bp"] = None
        if rec.get("mt5_med_bp") and rec.get("ctrader_med_bp"):
            rec["ratio_ct_vs_mt5"] = round(rec["ctrader_med_bp"] / rec["mt5_med_bp"], 3)
        rows.append(rec)
    return pd.DataFrame(rows)


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
