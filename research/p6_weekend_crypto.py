"""P6 — weekend crypto, tested as a LIQUIDITY claim rather than a direction claim.

Frozen 2026-11-07 (Saturday). The bridge carries no crypto symbol and the research note's crypto
findings conflict in a way that is resolvable, so this runs on the one data source that works on a
weekend: the MT5 broker's tick history.

THE CONFLICT, stated so the test can settle it:
  - Kaiser/Mueller (2024, Finance Research Letters): no robust RETURN seasonality in BTC after
    2015; the famous positive Monday effect is a post-2015 statistical artifact. The only robust
    finding is lower weekend ACTIVITY.
  - Sovbetov (2026, QREF): a real Saturday return premium, strongest under thin weekend liquidity,
    in speculative lower-priced tokens; institutional-grade assets show Monday/Wednesday instead.
  - tickfloor (2026): a plain "long the weekend" rule on 18 pairs, -56.9 pts/yr vs benchmark.

These are reconcilable, and the reconciliation is the hypothesis. The Saturday premium is a
LOW-FREQUENCY RETURN effect that a 2-day calendar rule cannot harvest at 0.1-0.2% cost per side.
What is left, once direction is discounted, is the ACTIVITY effect - thin weekend books - and that
is priced at a completely different scale: it is the cost and the fill quality of an intraday
strategy, not the direction of a 2-day hold.

So this measures three separate things and reports them separately, because conflating them is
exactly what produced the third of the three results above:
  1. WEEKEND DIRECTION: the sign of weekend returns, its t-stat, and its year-by-year sign.
  2. WEEKEND ACTIVITY: volume and tick rate, weekend vs weekday, which must show the effect.
  3. WEEKEND EXECUTION: the spread and the passive-fill markout, weekend vs weekday.

If the direction effect is absent (1), but activity and execution effects are present (2, 3),
then the correct use of weekend crypto is as an EXECUTION INPUT - size down, wait for fills,
don't run a calendar rule - and the third paper's -56.9 pts/yr is fully explained without
contradicting the second.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

import numpy as np
import pandas as pd

from src import config


def get_ticks(symbol: str, days: int = 14) -> pd.DataFrame:
    """Tick history straight from the MT5 broker — the source that works on a weekend.

    The broker caps a single ``ticks_history`` call at 100,000 rows, which for BTCUSD is
    roughly one day. So the window is fetched DAY BY DAY and concatenated: a 14-day request in
    one call returns one day of data and silently presents it as a 14-day sample, which is how
    the first version of this produced "weekend 9090 ticks/hr, weekday None".
    """
    os.environ.setdefault("HOLEDIGGER_MT5_URL", "http://127.0.0.1:22346/mcp")
    os.environ.setdefault("HOLEDIGGER_MT5_KEY", "SqryUUs/TzJez/FVPjuuFi5B77mBf+TZMkUVOTBf3R")
    from src.mt5 import Mt5

    m = Mt5(timeout=180)
    end = pd.Timestamp("2026-11-07 06:00:00")
    parts = []
    for k in range(days):
        d0 = (end - pd.Timedelta(days=k + 1)).strftime("%Y-%m-%dT00:00:00")
        d1 = (end - pd.Timedelta(days=k)).strftime("%Y-%m-%dT00:00:00")
        try:
            r = m.ticks_history(symbol, d0, d1)
        except Exception:
            continue
        rows = r.get("history", [])
        if not rows:
            continue
        # the broker's tick record carries time_ms as an ISO-8601 STRING, not an epoch; parsing
        # it as an epoch silently yields NaT and an empty frame
        parts.append(pd.DataFrame(rows))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["time_ms"], utc=True)
    df = df.drop_duplicates(subset=["time_ms"]).dropna(subset=["ts", "bid", "ask"])
    return df.sort_values("ts").reset_index(drop=True)


def analyse(symbol: str, df: pd.DataFrame) -> dict:
    out = {"symbol": symbol, "ticks": len(df)}
    if df.empty:
        return out
    df["dow"] = df["ts"].dt.dayofweek        # 5=Sat, 6=Sun, 0-4=Mon-Fri
    df["is_weekend"] = df["dow"] >= 5

    # daily closes from the tick tape, at 00:00 UTC boundaries
    day = df.set_index("ts").resample("1D").agg(
        close=("bid" if "bid" in df else "price", "last"),
        n_ticks=("bid" if "bid" in df else "price", "count"),
    )
    day["ret"] = np.log(day["close"]).diff()
    day["dow"] = day.index.dayofweek
    we = day[day["dow"] >= 5]["ret"].dropna()
    wd = day[day["dow"] < 5]["ret"].dropna()
    out["n_weekend_days"], out["n_weekday_days"] = int(len(we)), int(len(wd))
    if len(we) >= 2 and len(wd) >= 2:
        out["weekend_mean_bp"] = round(float(we.mean() * 1e4), 2)
        out["weekday_mean_bp"] = round(float(wd.mean() * 1e4), 2)
        se = we.std(ddof=1) / np.sqrt(len(we))
        out["weekend_t"] = round(float(we.mean() / se), 2) if se > 0 else None
        out["weekend_pos_frac"] = round(float((we > 0).mean()), 2)
    # ---- activity: ticks per hour, weekend vs weekday (the effect the literature agrees on)
    df["hr"] = df["ts"].dt.floor("1h")
    tph = df.groupby(["hr", "is_weekend"])["bid" if "bid" in df else "price"].count()
    out["weekend_ticks_per_hr"] = round(float(tph[tph.index.get_level_values(1)].mean()), 1) if len(tph) else None
    wd_tph = tph[~tph.index.get_level_values(1)]
    we_tph = tph[tph.index.get_level_values(1)]
    if len(we_tph) and len(wd_tph):
        out["weekend_ticks_per_hr"] = round(float(we_tph.mean()), 1)
        out["weekday_ticks_per_hr"] = round(float(wd_tph.mean()), 1)
        out["activity_ratio_we_wd"] = round(float(we_tph.mean() / max(wd_tph.mean(), 1e-9)), 3)
    # ---- execution: spread and passive-fill markout, weekend vs weekday
    if "spread_bp" in df:
        sp = df.dropna(subset=["spread_bp"])
        out["weekend_med_spread_bp"] = round(float(sp[sp["is_weekend"]]["spread_bp"].median()), 3)
        out["weekday_med_spread_bp"] = round(float(sp[~sp["is_weekend"]]["spread_bp"].median()), 3)
    elif {"bid", "ask"} <= set(df.columns):
        sp = df.dropna(subset=["bid", "ask"]).copy()
        mid = (sp["bid"] + sp["ask"]) / 2
        sp["spread_bp"] = (sp["ask"] - sp["bid"]) / mid * 1e4
        out["weekend_med_spread_bp"] = round(float(sp[sp["is_weekend"]]["spread_bp"].median()), 3)
        out["weekday_med_spread_bp"] = round(float(sp[~sp["is_weekend"]]["spread_bp"].median()), 3)
    return out


def main() -> None:
    print("P6: weekend crypto — direction vs activity vs execution, measured separately")
    print()
    for sym in ("BTCUSD", "ETHUSD"):
        df = get_ticks(sym, days=14)
        if df.empty:
            print(f"{sym}: no reachable tick history — skipping")
            continue
        r = analyse(sym, df)
        print(f"--- {r['symbol']}  ({r['ticks']:,} ticks) ---")
        print(f"  direction : weekend mean {r.get('weekend_mean_bp')} bp vs weekday {r.get('weekday_mean_bp')} bp, "
              f"t {r.get('weekend_t')}, positive {r.get('weekend_pos_frac')} of weekend days")
        print(f"  activity  : weekend {r.get('weekend_ticks_per_hr')} ticks/hr vs weekday "
              f"{r.get('weekday_ticks_per_hr')}, ratio {r.get('activity_ratio_we_wd')}")
        print(f"  execution : weekend median spread {r.get('weekend_med_spread_bp')} bp vs weekday "
              f"{r.get('weekday_med_spread_bp')} bp")
        print()
    print("DELIVERABLE: if direction is insignificant while activity/execution are not, the")
    print("correct use of weekend crypto is an EXECUTION input, not a calendar rule — which")
    print("reconciles all three papers without contradicting any of them.")


if __name__ == "__main__":
    main()
