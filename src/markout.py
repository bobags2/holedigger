"""Bar-level markout and adverse selection: does a wide spread get earned or eaten?

The question this answers
--------------------------
Every cost-dead verdict in this project has charged a *spread* and moved on. That assumes a
passive fill (a limit order resting at the bid) earns the spread. It might not: a wide spread
is wide for a reason, and if the reason is informed flow, the passive filler gets run over —
the classic adverse-selection result. If the reason is just inventory/uncertainty, the
filler earns it.

That distinction decides whether "dead on cost" verdicts are dead at all, and no module in
this repo has measured it. So: enter passively at the bid (or offer) on a chosen tape state,
hold for H minutes, and mark out against the mid. The markout is

    gross markout = mid[t+H] - fill_price          (a passive bid fill: fill = mid + half-spread)
    net markout    = gross markout                 (the passive side already captured the
                                                    spread by definition of the fill)

which is the number that says whether the spread is *earned*. Conditioning the entry on the
tape's spread state — and on whether the bar's price was rising or falling — is what tests
whether a state-aware passive strategy can pay for itself.

Honest scope note: the raw tick tape is not retained (the MT5 ingest writes per-minute bars
with spread summaries), so this is a BAR-level study, not a tick-level one. That limits the
adverse-selection claim to the bar scale; a tick-level markout needs a re-pull with tick
retention, and that is recorded as a prerequisite rather than simulated.

Usage::

    from src import markout
    res = markout.run("XAUUSD", horizons=(1, 5, 15, 30))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, mt5_clock

HORIZONS = (1, 5, 15, 30)
SPREAD_QUANTILES = (0.5, 0.8, 0.95)   # tape states: median, 80th, 95th percentile spread


def load_bars(symbol: str) -> pd.DataFrame:
    """True-UTC M1 bars with close, spread, and the forward returns a markout needs."""
    df = mt5_clock.load_mt5_utc(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def markout_table(df: pd.DataFrame, horizons: tuple[int, ...]) -> dict[int, dict[str, np.ndarray]]:
    """Per-horizon pass of the passive-bid entry, for several tape states.

    For each horizon H:

    * ``gross`` — the mid's H-minute forward move, unconditional. On a random tape this is
      ~0, so any nonzero mean is the drift the earlier families measured.
    * ``passive_bid`` — a limit bid filled at ``mid + half_spread`` when the bar ticks DOWN
      (a seller crossed you), marked out against the mid at t+H. This is the number that
      says whether resting a bid earns the half-spread.
    * ``passive_offer`` — the mirror.
    * ``wide`` / ``tight`` — the passive-bid markout split by whether the bar's spread was
      above or below its session median, which is the test of "wide spread = earned or
      eaten".
    """
    close = df["close"].to_numpy(dtype=float)
    sp = df["spread_med_bp"].to_numpy(dtype=float)
    logmid = np.log(close)
    ret = np.zeros(len(close))
    ret[1:] = np.diff(logmid)
    bar_ret_bp = ret * 1e4

    out: dict[int, dict[str, np.ndarray]] = {}
    for H in horizons:
        fwd = np.full(len(close), np.nan)
        fwd[: len(close) - H] = (logmid[H:] - logmid[: len(close) - H]) * 1e4
        half = sp / 2.0
        # a passive bid is filled when a seller crosses the spread: the bar prints down
        down = bar_ret_bp < 0
        up = bar_ret_bp > 0
        pb = np.where(down, fwd - half, np.nan)      # paid the half-spread to get in, marked to mid
        po = np.where(up, -fwd - half, np.nan)       # mirror: short at the offer, marked to mid
        sess_med = pd.Series(sp).groupby(df["ts"].dt.hour).transform("median").to_numpy()
        wide = sp > sess_med
        pb_wide = np.where(down & wide, fwd - half, np.nan)
        pb_tight = np.where(down & ~wide, fwd - half, np.nan)
        out[H] = {
            "gross": fwd,
            "passive_bid": pb,
            "passive_offer": po,
            "passive_bid_wide": pb_wide,
            "passive_bid_tight": pb_tight,
        }
    return out


def _stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if len(a) < 30:
        return {"n": int(len(a)), "mean_bp": None, "t": None, "hit": None}
    sd = a.std(ddof=1)
    return {
        "n": int(len(a)),
        "mean_bp": round(float(a.mean()), 5),
        "t": round(float(a.mean() / (sd / np.sqrt(len(a)))), 3) if sd > 0 else None,
        "hit": round(float((a > 0).mean()), 4),
    }


def run(symbol: str, *, horizons: tuple[int, ...] = HORIZONS) -> dict:
    df = load_bars(symbol)
    tab = markout_table(df, horizons)
    rows = {}
    for H, cols in tab.items():
        rows[f"+{H}m"] = {k: _stats(v) for k, v in cols.items()}
    return {
        "symbol": symbol,
        "bars": int(len(df)),
        "span": [str(df["ts"].iloc[0]), str(df["ts"].iloc[-1])],
        "median_spread_bp": round(float(np.nanmedian(df["spread_med_bp"])), 4),
        "horizons": rows,
        "note": (
            "passive_bid > 0 means resting a bid EARNS the half-spread even after the tape "
            "moves against you; <= 0 means it is eaten. The wide/tight split is the "
            "state-aware version."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD", "XAGUSD", "XTIUSD"])
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "markout.json")
    args = ap.parse_args()
    config.ensure_dirs()

    results = {}
    for sym in args.symbols:
        p = config.PARQUET_DIR / f"{sym}_MT5_M1.parquet"
        if not p.exists():
            continue
        results[sym] = run(sym)
        print(f"{sym}: median spread {results[sym]['median_spread_bp']:.2f}bp")
        for H, cols in results[sym]["horizons"].items():
            pb, pw, pt = cols["passive_bid"], cols["passive_bid_wide"], cols["passive_bid_tight"]
            g = cols["gross"]
            print(f"  {H:>5}: gross {g['mean_bp']:+.4f} (t {g['t']:+.2f}) | "
                  f"passive bid {pb['mean_bp']:+.4f} (t {pb['t']:+.2f}, hit {pb['hit']:.0%}) | "
                  f"wide {pw['mean_bp']:+.4f} / tight {pt['mean_bp']:+.4f}")

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
