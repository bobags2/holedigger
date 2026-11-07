"""Hourly seasonality screen across the own-panel instruments, priced against the ledger.

WHY THIS TEST: the published record is both encouraging and damning in the same breath - 214
of 5,086 time-of-day rules survived an honest out-of-sample (4.2%), and the kill mechanism was
uniformly the spread: only 12.5% of FX weekday/hour buckets move more than a 1-pip round trip.
So the interesting question is not "which hours lean up", it is "which instruments have hours
whose average move exceeds their own round-trip toll". That is a measurement, not a strategy,
and it is answerable for every instrument at once.

The screen deliberately does NOT trade. It measures, per instrument per hour:
  - mean signed return of the hour (direction lean)
  - mean absolute move of the hour (the representable opportunity)
  - the ratio of that move to the instrument's own measured round-trip toll, median and p90
  - the out-of-sample t-stat of the direction lean, on held-out years

Everything here is gross direction and gross range; the report is the cost ratio, because that
is what decides whether a direction lean is worth anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import venue_cost as vc


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = pd.Grouper(freq=f"{minutes}min", key="ts")
    out = df.groupby(g).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).reset_index()
    out["r"] = np.log(out["close"]).diff()
    return out.dropna(subset=["open"]).reset_index(drop=True)


def screen(symbol: str, *, minutes: int = 60, holdout_year: int = 2024,
           parquet=None) -> dict:
    """Per-hour direction lean, range, and cost ratio for one instrument."""
    from src import config
    panel = config.PARQUET_DIR / f"{symbol}_M1.parquet"
    if not panel.exists():
        raise FileNotFoundError(f"{symbol}: no long bar panel to screen")
    cost_panel = config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet"
    if not cost_panel.exists():
        raise FileNotFoundError(f"{symbol}: no measured spread; it cannot be priced")
    df = pd.read_parquet(panel)
    df["ts"] = pd.to_datetime(df["ts"])
    bars = resample(df, minutes)
    cost = vc.load_cost(symbol)

    bars["h"] = bars["ts"].dt.hour
    bars["yr"] = bars["ts"].dt.year
    dev = bars[bars["yr"] < holdout_year]
    oos = bars[bars["yr"] >= holdout_year]

    rows = []
    rt_med, rt_p90 = cost.roundtrip_med_bp, cost.roundtrip_p90_bp
    for h, g in dev.groupby("h"):
        if len(g) < 200:
            continue
        lean = float(g["r"].mean()) * 1e4
        move = float(g["r"].abs().mean()) * 1e4
        sd = float(g["r"].std(ddof=1)) * 1e4   # SAME units as lean/move, else t is off by 1e4
        t_lean = lean / (sd / np.sqrt(len(g))) if sd > 0 else np.nan
        rows.append(dict(hour=h, n=len(g), lean_bp=lean, move_bp=move, sd_bp=sd,
                         t_lean=t_lean,
                         move_over_rt_p90=move / rt_p90 if rt_p90 > 0 else np.inf,
                         lean_over_rt_p90=abs(lean) / rt_p90 if rt_p90 > 0 else np.inf))
    per = pd.DataFrame(rows)
    # out-of-sample confirmation of the strongest dev lean
    oos_lean = oos.groupby("h")["r"].mean() * 1e4
    per["oos_lean_bp"] = per["hour"].map(oos_lean)
    per["oos_same_sign"] = np.sign(per["lean_bp"]) == np.sign(per["oos_lean_bp"].fillna(0))
    per["oos_same_sign"] = per["oos_same_sign"] & per["oos_lean_bp"].notna()

    return {
        "symbol": symbol,
        "rt_med_bp": rt_med,
        "rt_p90_bp": rt_p90,
        "per_hour": per,
        "n_hours_beat_p90": int((per["move_over_rt_p90"] > 1).sum()),
        "n_hours": int(len(per)),
    }


def main() -> None:
    from src import config
    import os
    syms = []
    for f in sorted(os.listdir(config.PARQUET_DIR)):
        if f.endswith("_MT5_M1.parquet"):
            syms.append(f.replace("_MT5_M1.parquet", ""))
    print(f"screening {len(syms)} instrument(s) with a measured spread, 1h bars")
    print(f"{'symbol':>10} {'rt_p90':>8} {'hrs':>4} {'hrs>toll':>8}  best hours by move/toll")
    out_rows = []
    for s in syms:
        try:
            r = screen(s)
        except Exception as e:
            continue
        top = r["per_hour"].sort_values("move_over_rt_p90", ascending=False).head(3)
        pretty = ", ".join(f"{int(r_['hour']):02d}h:{r_['move_over_rt_p90']:.1f}x" for _, r_ in top.iterrows())
        print(f"{s:>10} {r['rt_p90_bp']:>8.2f} {r['n_hours']:>4} {r['n_hours_beat_p90']:>8}  {pretty}")
        out_rows.append(dict(symbol=s, rt_p90_bp=r["rt_p90_bp"], n_hours=r["n_hours"],
                             hours_beating_toll=r["n_hours_beat_p90"],
                             frac_beating=r["n_hours_beat_p90"] / max(r["n_hours"], 1)))
    pd.DataFrame(out_rows).to_csv("artifacts/hourly_cost_screen.csv", index=False)


if __name__ == "__main__":
    main()
