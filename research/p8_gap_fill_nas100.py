"""P8 — gap fill on our own NAS100 panel: is a 70% fill rate worth anything at OUR toll?

FROZEN SPEC, written 2026-11-07 before computing anything.

The document's Tier-3 #10 says small gaps fill 70%+. The published work is far more precise about
the same statistic and about its value:

  - ripbacktest (ES, 1-min, 1,737 sessions, $30.76 rt): gaps of 0.10-0.25% filled 73.8% same-day,
    and ALL THREE trade versions lost money (-$14 to -$22 per trade, cause Costs). Their kill:
    the target IS the gap, median 13.75 points, and the round trip is 0.6 of a point.
  - the independent 6,984-gap study adds the control that matters: a mirror level the same
    distance on the OTHER side is reached 77.5% of the time, so small gaps fill ~2.9 points
    above chance - a lean, not a magnet.

So the question at our venue has two parts, in this order:
  1. Do our NAS100 gaps fill at the published rate? (the statistic)
  2. Is the MIRROR level reached at a similar rate? (the control - this decides everything)
  3. Is the fill payoff minus Nasdaq's 0.55 bp toll and its financing positive?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, venue_cost as vc


def build_daily(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_M1.parquet")
    df["ts"] = pd.to_datetime(df["ts"])
    d = df.set_index("ts").resample("1D").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"),
    ).dropna(subset=["open"])
    d["prev_close"] = d["close"].shift(1)
    return d


def main() -> None:
    symbol = "NAS100"
    d = build_daily(symbol)
    lvl = float(d["close"].mean())
    cost = vc.load_cost(symbol)
    toll = cost.roundtrip_med_bp
    print(f"P8: gap fill on {symbol}, {len(d):,} sessions")
    print(f"  venue toll: {toll:.3f} bp round trip (measured); financing "
          f"{cost.financing_annual / 365 * 1e4:.4f} bp for one overnight")
    print()
    s = d.dropna(subset=["prev_close"]).copy()
    s["gap_pct"] = (s["open"] - s["prev_close"]) / s["prev_close"]
    s["gap_bp"] = s["gap_pct"] * 1e4
    s["up"] = s["gap_pct"] > 0
    # how often the gap fills: price trades back to prev_close at any point in the session
    s["filled"] = np.where(s["up"], s["low"] <= s["prev_close"], s["high"] >= s["prev_close"])
    # the control: a MIRROR level the same distance on the opposite side of the open
    s["mirror"] = s["open"] - s["gap_bp"] / 1e4 * s["prev_close"] if False else s["open"] + (s["gap_bp"] / 1e4) * s["prev_close"] * np.where(s["up"], -1, 1)
    s["mirror_hit"] = np.where(s["up"], s["low"] <= s["mirror"], s["high"] >= s["mirror"])
    # median time-to-fill is not available on daily bars; report the same-day fill only

    print(f"{'gap size (bp)':>14} {'n':>5} {'filled':>8} {'mirror':>8} {'lean vs mirror':>14}")
    for lo, hi in ((-1e9, 10), (10, 25), (25, 50), (50, 100), (100, 1e9)):
        m = (s["gap_bp"].abs() >= lo) & (s["gap_bp"].abs() < hi)
        if m.sum() < 30:
            continue
        print(f"{f'{lo:.0f} to {hi:.0f}':>14} {int(m.sum()):>5} {s[m]['filled'].mean():>8.1%} "
              f"{s[m]['mirror_hit'].mean():>8.1%} {(s[m]['filled'].mean()-s[m]['mirror_hit'].mean())*100:>+13.1f} pts")
    print()
    # the tradeable version: fade every gap over 10bp at the open, target the fill, stop 1 gap
    tr = s[s["gap_bp"].abs() >= 10].copy()
    tr["side"] = np.where(tr["up"], -1, 1)          # fade
    tr["target"] = tr["prev_close"]
    tr["stop"] = np.where(tr["up"], tr["open"] + (tr["open"] - tr["prev_close"]).abs(),
                          tr["open"] - (tr["open"] - tr["prev_close"]).abs())
    hit_t = np.where(tr["up"], tr["low"] <= tr["target"], tr["high"] >= tr["target"])
    hit_s = np.where(tr["up"], tr["high"] >= tr["stop"], tr["low"] <= tr["stop"])
    # conservative: if both in one session, the stop wins
    outcome = np.where(hit_s & hit_t, "stop", np.where(hit_t, "target",
                       np.where(hit_s, "stop", "none")))
    px = np.where(outcome == "target", tr["target"],
         np.where(outcome == "stop", tr["stop"], tr["close"]))
    pts = (px - tr["open"].to_numpy()) * tr["side"].to_numpy()
    gross_bp = pts / tr["close"].to_numpy() * 1e4
    net_bp = gross_bp - toll
    print(f"--- trade the fade (all gaps >=10bp, target = prev close, stop = 1 gap beyond entry) ---")
    print(f"  trades {len(tr):>5}  gross {gross_bp.mean():+7.3f} bp  toll {toll:.3f}  "
          f"NET {net_bp.mean():+7.3f} bp  hit {hit_t.mean():.1%}")
    yr = pd.DatetimeIndex(tr.index).year.to_numpy()
    det = {int(y): float(net_bp[yr == y].mean()) for y in sorted(set(yr)) if (yr == y).sum() > 10}
    print("  per year: " + " ".join(f"{k}:{v:+.2f}" for k, v in det.items())
          + f"   -> {sum(1 for v in det.values() if v>0)}/{len(det)} years positive")
    print()
    print("DELIVERABLE: does a genuine fill rate minus OUR toll pay - the question the")
    print("document never asked because it quoted the fill rate and stopped there.")


if __name__ == "__main__":
    main()
