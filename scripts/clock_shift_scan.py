"""Cross-source clock scan: which integer-hour shift aligns the MT5 panel with cTrader?

For every month of the overlap (and optionally every day of a range) this moves the MT5
5-minute close series by each integer shift k in −6..+6 hours and scores it against the
cTrader 5-minute series, which is the true-UTC reference (built from the bridge's ASCII
history; `--since` keeps the scan inside that file's genuinely-M1 era — the first rows of
the `*_M1.csv` files are one bar per trading day). Three diagnostics per shift:

  match    fraction of common bars with |log p_mt5 − log p_ct| < 1e-5
  amatch   the same after removing the aligned set's median log difference — the two feeds
           can quote one instrument at a constant level offset (copper does), which makes
           the raw match useless there while amatch still reads the alignment
  rcorr    Pearson correlation of 5-min log returns, each series on its own grid, at the
           aligned timestamps (near 1.00 when the clocks agree, ~0.00 when they do not)

How to read it: k* is the shift the MT5 panel needs to sit on the cTrader clock. The panels
were ingested as ``terminal − 3h`` with the offset assumed constant, so on disk k* must be
**0 through US summer time and +1h through US standard time** — a month table whose k*
switches between those two values at the US DST Sundays is the bug, and a constant k* would
have meant a constant offset. Look at the margin too: at a wrong k the other season scores
0.00–0.03, so a 0.5–0.95 winner is not a coin flip. (The fix and its evidence: src/mt5_clock.py.)

Usage::

    python -m scripts.clock_shift_scan                                  # all months
    python -m scripts.clock_shift_scan --symbols XAUUSD XCUUSD
    python -m scripts.clock_shift_scan --days 2026-03-01 2026-03-14     # day resolution
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

TOL = 1e-5
SHIFTS = list(range(-6, 7))
DEFAULT_SYMBOLS = ["EURUSD", "XAUUSD", "XCUUSD"]
MIN_BARS = 50


def five_min_close(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.resample("5min").last().dropna()


def load_mt5(symbol: str) -> pd.Series:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet", columns=["ts", "close"])
    return five_min_close(df.set_index("ts")["close"])


def load_ct(symbol: str, since: str) -> pd.Series:
    path = config.HISTORY_DIR / f"{symbol}_M1.csv"
    if not path.exists():
        raise FileNotFoundError(f"no cTrader reference at {path}")
    raw = pd.read_csv(path, usecols=["Date", "Time", "Close"], dtype={"Date": str, "Time": str})
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%Y/%m/%d %H:%M:%S")
    out = five_min_close(pd.Series(raw["Close"].to_numpy(dtype=float), index=ts))
    return out[out.index >= since]


def scores(mt5: pd.Series, ct: pd.Series, k: int, lo=None, hi=None) -> tuple:
    """(match, amatch, rcorr, n) for one shift, restricted to a timestamp range by lo/hi.

    Returns come from the full series, so a slice boundary neither invents nor drops a
    return; only the score window moves.
    """
    sh = pd.Series(mt5.to_numpy(dtype=float), index=mt5.index + pd.Timedelta(hours=k))
    j = sh.index.intersection(ct.index)
    if lo is not None:
        j = j[j >= lo]
    if hi is not None:
        j = j[j <= hi]
    j = j[~j.duplicated()]
    if len(j) < MIN_BARS:
        return np.nan, np.nan, np.nan, len(j)
    a = np.log(sh.reindex(j).to_numpy(dtype=float))
    b = np.log(ct.reindex(j).to_numpy(dtype=float))
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < MIN_BARS:
        return np.nan, np.nan, np.nan, int(ok.sum())
    d = a[ok] - b[ok]
    ra = sh.diff().reindex(j).to_numpy(dtype=float)
    rb = ct.diff().reindex(j).to_numpy(dtype=float)
    good = ~(np.isnan(ra) | np.isnan(rb))
    return (
        float((np.abs(d) < TOL).mean()),
        float((np.abs(d - np.median(d)) < TOL).mean()),
        float(np.corrcoef(ra[good], rb[good])[0, 1]) if good.sum() > MIN_BARS else np.nan,
        int(ok.sum()),
    )


def ranked(res: dict) -> tuple:
    order = sorted(SHIFTS, key=lambda k: (-np.inf if np.isnan(res[k][2]) else res[k][2]), reverse=True)
    return order[0], order[1]


def month_table(symbol: str, mt5: pd.Series, ct: pd.Series) -> None:
    months = sorted(set(mt5.index.to_period("M").astype(str)) & set(ct.index.to_period("M").astype(str)))
    print(f"\nrequested shift k* is read top-down per month: 'best k' needs a large lead over the runner-up.")
    print(f"{'month':>8} {'best k':>6} {'match':>6} {'amatch':>7} {'rcorr':>6} {'n':>6} "
          f"{'2nd k':>6} {'rcorr@2nd':>9} | rcorr at k=-1 +0 +1")
    for mo in months:
        mm = mt5.index[mt5.index.to_period("M").astype(str) == mo]
        cc = ct.index[ct.index.to_period("M").astype(str) == mo]
        lo, hi = max(mm.min(), cc.min()), min(mm.max(), cc.max())
        res = {k: scores(mt5, ct, k, lo=lo, hi=hi) for k in SHIFTS}
        kstar, k2 = ranked(res)
        m, am, rc, n = res[kstar]
        rmin1, r0, rp1 = (res[k][2] for k in (-1, 0, 1))
        print(f"{mo:>8} {kstar:>6} {m:>6.3f} {am:>7.3f} {rc:>6.3f} {n:>6} {k2:>6} {res[k2][2]:>9.3f} "
              f"| {rmin1:>+7.3f} {r0:>+7.3f} {rp1:>+7.3f}")


def day_table(symbol: str, mt5: pd.Series, ct: pd.Series, lo: str, hi: str) -> None:
    sel = mt5[(mt5.index >= lo) & (mt5.index <= hi)]
    ct_sel = ct[(ct.index >= lo) & (ct.index <= hi)]
    days = sorted(set(sel.index.floor("D")) & set(ct_sel.index.floor("D")))
    print(f"\nday resolution {lo}..{hi} — the k* column is where the changeover shows up:")
    print(f"{'day':>12} {'k*':>3} {'match':>6} {'amatch':>7} {'rcorr':>6} {'rcorr@k0':>9} {'rcorr@k+1':>10} {'n':>6}")
    for d in days:
        mm = sel.index[sel.index.floor("D") == d]
        cc = ct_sel.index[ct_sel.index.floor("D") == d]
        res = {k: scores(mt5, ct, k, lo=max(mm.min(), cc.min()), hi=min(mm.max(), cc.max()))
               for k in SHIFTS}
        kstar, _ = ranked(res)
        m, am, rc, n = res[kstar]
        print(f"{str(d)[:10]:>12} {kstar:>3} {m:>6.3f} {am:>7.3f} {rc:>6.3f} "
              f"{res[0][2]:>+9.3f} {res[1][2]:>+10.3f} {n:>6}")


def pooled_table(symbol: str, mt5: pd.Series, ct: pd.Series) -> None:
    print(f"\n{symbol} pooled over the whole overlap:")
    print(f"{'k':>3} {'n_common':>9} {'match':>6} {'amatch':>7} {'rcorr':>6}")
    for k in SHIFTS:
        m, am, rc, n = scores(mt5, ct, k)
        print(f"{k:>3} {n:>9} {m:>6.3f} {am:>7.3f} {rc:>6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--since", default="2025-11-01",
                    help="drop cTrader rows before this date (the M1 files open with daily bars)")
    ap.add_argument("--days", nargs=2, metavar=("LO", "HI"), default=None,
                    help="also scan day by day inside this window (e.g. --days 2026-03-01 2026-03-14)")
    args = ap.parse_args()

    for symbol in args.symbols:
        mt5 = load_mt5(symbol)
        ct = load_ct(symbol, args.since)
        print(f"\n=== {symbol}: MT5 {len(mt5)} 5-min bars {mt5.index[0]} -> {mt5.index[-1]}; "
              f"cTrader {len(ct)} bars {ct.index[0]} -> {ct.index[-1]}")
        month_table(symbol, mt5, ct)
        if args.days:
            day_table(symbol, mt5, ct, args.days[0], args.days[1])
        pooled_table(symbol, mt5, ct)


if __name__ == "__main__":
    main()
