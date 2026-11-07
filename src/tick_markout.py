"""Tick-level markout and adverse selection: does a passive fill earn the spread?

The question this answers
---------------------------
Every cost-dead verdict in this project charges a spread and assumes a passive fill earns
it. That assumption is testable, and the test is about *order flow within a bar*: who crossed
the spread, in which direction, and what happened to the mid afterwards. If a wide spread is
wide because of informed flow, a resting bid gets run over and the markout is negative; if it
is wide because of inventory or uncertainty, the filler earns it.

At the BAR level (``src/markout``) the answer was already unambiguous and negative on every
symbol tested: passive fills lose at every horizon, and the wide-spread state makes it WORSE,
not better. That study approximates a fill with "the bar ticked down" because the raw tape
is not retained. This module measures it properly, on ticks:

* a **bid-side fill** is an ask-side print that trades through our resting bid — the tape's
  aggressive seller hit us;
* the **markout** is ``mid[t+H] - fill_price``, and since a passive bid fills at the bid
  (i.e. above the mid by half a spread at that moment), a positive markout means the spread
  was earned;
* the **adverse-selection test** splits the markout by the spread state at the fill: if wide
  spreads are informed, the wide-state markout must be worse than the tight-state one.

What it needs
-------------
Raw ticks with bid and ask. ``src.tick_ingest`` pulls and retains them
(``data/parquet/{SYMBOL}_MT5_ticks.parquet``, true UTC, clock-tagged through the parquet's own
metadata so the file declares its clock). Until a symbol has a tick file, ``run`` raises a
clear prerequisite error rather than substituting the bar approximation silently.

Usage::

    from src import tick_markout
    res = tick_markout.run("XAGUSD", horizons=(1, 5, 15, 30))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

HORIZONS_MIN = (1, 5, 15, 30)
SPREAD_QUANTILE = 0.8     # "wide" = the top 20% of spread states at that symbol/hour


def load_ticks(symbol: str) -> pd.DataFrame:
    """Retained raw ticks for one symbol, or a clear prerequisite error."""
    p = config.PARQUET_DIR / f"{symbol}_MT5_ticks.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist - run `python -m src.tick_ingest --symbols {symbol} "
            f"--days 14` first (needs HOLEDIGGER_MT5_URL and HOLEDIGGER_MT5_KEY set). "
            f"Tick-level markout cannot be answered from bar summaries."
        )
    df = pd.read_parquet(p).sort_values("ts").reset_index(drop=True)
    for c in ("bid", "ask", "mid"):
        if c not in df.columns:
            raise ValueError(f"{p} has no {c!r} column")
    return df


def fill_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tick: which side did the tape move, and would a resting order have been filled.

    **What this tape can and cannot tell us.** MT5's tick history carries ``time_ms``,
    ``bid`` and ``ask`` only — no last price and no volume, so there is no print-level
    aggressor to observe. The convention "a print at the ask is an aggressive buy" cannot be
    evaluated here, and any module claiming to measure markout at print level on this tape
    would be measuring something else.

    What IS observable is which side the quote changed on, and that carries information:

    * the **bid rises** (the ask unchanged or behind) → buyers lifted the offer, so a
      resting OFFER would have been filled;
    * the **bid falls** → sellers hit the bid, so a resting BID would have been filled.

    That is a *quote-move* inference, weaker than a print-level one: a resting order is also
    filled when the quote moves *toward* it without a trade, so this measures a lower bound
    on fills and is conservative about the markout. The direction convention is the honest
    one and it is stated rather than assumed.

    ``last`` is still honoured when a tape does carry it (a different provider), so the
    print-level path remains available for a tape that has one.
    """
    bid = df["bid"].to_numpy(float)
    ask = df["ask"].to_numpy(float)
    mid = df["mid"].to_numpy(float) if "mid" in df else (bid + ask) / 2.0
    last = df["last"].to_numpy(float) if "last" in df else np.full(len(bid), np.nan)
    has_last = np.isfinite(last)

    up = np.zeros(len(bid), dtype=bool)     # the quote moved up -> a resting offer filled
    down = np.zeros(len(bid), dtype=bool)   # the quote moved down -> a resting bid filled
    up[1:] = np.diff(bid) > 0
    down[1:] = np.diff(bid) < 0
    # non-positive spread ticks carry no information about direction
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)

    if has_last.any():
        # print-level path: a print at the ask is an aggressive buy, at the bid a sell
        d_bid = np.abs(last - bid)
        d_ask = np.abs(last - ask)
        both = np.isfinite(d_bid) & np.isfinite(d_ask)
        at_bid = both & (d_bid < d_ask)
        at_ask = both & (d_ask < d_bid)
        bid_fill = (at_bid & ok) | (down & ok & ~has_last)
        ask_fill = (at_ask & ok) | (up & ok & ~has_last)
        inference = "print-level where last is available, quote-move otherwise"
    else:
        bid_fill = down & ok                  # resting bid filled (tape moved down)
        ask_fill = up & ok                    # resting offer filled (tape moved up)
        inference = "quote-move (no last price on this tape)"

    spread = np.where(ok, ask - bid, np.nan)
    return pd.DataFrame({
        "ts": df["ts"].to_numpy(),
        "mid": mid, "bid": bid, "ask": ask, "spread": spread,
        "bid_fill": bid_fill,
        "ask_fill": ask_fill,
        "half_spread": spread / 2.0,
    }), inference


def markout(flags: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Mark out each passive fill against the mid ``horizon_min`` later.

    A bid-side fill prices at ``mid + half_spread`` (above the mid at fill time), so its
    markout is ``mid[t+H] - (mid + half_spread)``. Positive means the passive side EARNED the
    spread; negative means it was run over.
    """
    ts = pd.DatetimeIndex(flags["ts"])
    mid_v = flags["mid"].to_numpy()
    # A POSITION shift, not a time shift. ``mid.shift(freq=...)`` is a no-op on a regular
    # index (verified: it returns the series unchanged), so the earlier version marked every
    # fill out against the fill's own mid. The horizon is expressed in rows of the tape, and
    # the tape's cadence is recorded so a reader can convert rows to minutes.
    cadence = ts.to_series().diff().median()
    rows = int(round(pd.Timedelta(minutes=horizon_min) / cadence)) if pd.notna(cadence) and cadence else 0
    fwd = np.full(len(mid_v), np.nan)
    if rows > 0 and rows < len(mid_v):
        fwd[: len(mid_v) - rows] = mid_v[rows:]
    # A resting BID fills at the bid = mid - half_spread (below the mid at fill time), so a
    # bid-side markout is mid[t+H] - bid_fill. A resting OFFER fills at the ask = mid +
    # half_spread, so its markout is ask_fill - mid[t+H]. The markout is in PRICE units, so
    # it is converted to bp of the fill-time mid - the same units every other number in this
    # project uses. (An earlier version signed the bid fill the wrong way round, which turned
    # a real loss into a gain.)
    fill_bid = mid_v - flags["half_spread"].to_numpy()   # at the bid
    fill_ask = mid_v + flags["half_spread"].to_numpy()   # at the ask
    px = np.maximum(mid_v, 1e-12)
    out = pd.DataFrame({
        "markout_bid_bp": np.where(flags["bid_fill"], (fwd - fill_bid) / px * 1e4, np.nan),
        "markout_offer_bp": np.where(flags["ask_fill"], (fill_ask - fwd) / px * 1e4, np.nan),
        "spread_bp": flags["spread"].to_numpy() / px * 1e4,
        "hour": ts.hour.to_numpy(),
    })
    # DRIFT-NET markout: what a passive fill actually earned, = the half spread it captured
    # minus what the tape's subsequent move gave back. This is NOT the raw markout minus the
    # drift (that subtraction leaves the half-spread itself and is horizon-flat by
    # construction - it was tried and discarded). Instead:
    #     captured  = half_spread            (paid nothing extra to get filled)
    #     gave back = adverse move over the horizon, measured from the fill mid
    #     earned    = captured - gave back
    # For a resting BID the adverse move is the mid FALLING (fwd < mid); for a resting OFFER
    # it is the mid RISING. So:
    #     earned_bid   =  half_spread - max(0, mid - fwd)
    #     earned_offer =  half_spread - max(0, fwd - mid)
    # which is horizon-dependent, sign-correct, and answers the question that matters: does
    # resting earn the spread after the tape moves against you?
    half_bp = flags["half_spread"].to_numpy() / px * 1e4
    move = (fwd - mid_v) / px * 1e4                      # positive = mid rose
    out["earned_bid_bp"] = np.where(
        flags["bid_fill"], half_bp - np.maximum(-move, 0.0), np.nan
    )
    out["earned_offer_bp"] = np.where(
        flags["ask_fill"], half_bp - np.maximum(move, 0.0), np.nan
    )
    return out


def _stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if len(a) < 30:
        return {"n": int(len(a)), "mean_bp": None, "t": None, "hit": None}
    sd = a.std(ddof=1)
    return {"n": int(len(a)), "mean_bp": round(float(a.mean()), 5),
            "t": round(float(a.mean() / (sd / np.sqrt(len(a)))), 3) if sd > 0 else None,
            "hit": round(float((a > 0).mean()), 4)}


def run(symbol: str, *, horizons: tuple[int, ...] = HORIZONS_MIN) -> dict:
    df = load_ticks(symbol)
    flags, inference = fill_flags(df)
    rows = {}
    for H in horizons:
        m = markout(flags, H)
        for side in ("markout_bid_bp", "earned_bid_bp", "earned_offer_bp"):
            v = m[side].to_numpy()
            sp = m["spread_bp"].to_numpy()
            ok = np.isfinite(v) & np.isfinite(sp)
            wide = ok & (sp >= np.nanquantile(sp[ok], SPREAD_QUANTILE))
            key = f"+{H}m_{'bid' if 'bid' in side else 'offer'}"
            rows[key] = {
                "all": _stats(v),
                "wide_spread": _stats(v[wide]),
                "tight_spread": _stats(v[ok & ~wide]),
            }
    return {
        "symbol": symbol,
        "ticks": int(len(df)),
        "span": [str(df["ts"].iloc[0]), str(df["ts"].iloc[-1])],
        "bid_fills": int(flags["bid_fill"].sum()),
        "ask_fills": int(flags["ask_fill"].sum()),
        "horizons": rows,
        "inference": inference,
        "interpretation": (
            "mean_bp > 0 means the passive side EARNS the spread even after the tape moves; "
            "<= 0 means it is eaten. A wide-spread markout WORSE than tight is adverse "
            "selection: the width is informed flow, not inventory. On an MT5 tape the "
            "aggressor is not observable, so a fill is inferred from which side the QUOTE "
            "moved on - a lower bound on fills, hence a conservative markout."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["XAGUSD"])
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "tick_markout.json")
    args = ap.parse_args()
    config.ensure_dirs()

    results = {}
    for sym in args.symbols:
        try:
            r = run(sym)
        except FileNotFoundError as exc:
            print(f"{sym}: {exc}")
            continue
        results[sym] = r
        print(f"{sym}: {r['ticks']:,} ticks, {r['bid_fills']:,} bid fills, "
              f"{r['ask_fills']:,} ask fills")
        for k, v in r["horizons"].items():
            a, w, t = v["all"], v["wide_spread"], v["tight_spread"]
            if a["mean_bp"] is None:
                continue
            wm = f"{w['mean_bp']:+.4f}" if w["mean_bp"] is not None else "n/a"
            tm = f"{t['mean_bp']:+.4f}" if t["mean_bp"] is not None else "n/a"
            print(f"  {k:>12}: {a['mean_bp']:+.4f}bp (t {a['t']:+.2f}, hit {a['hit']:.0%})"
                  f"  | wide {wm} / tight {tm}")

    if results:
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
