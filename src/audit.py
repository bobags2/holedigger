"""Artifact audit: what the data actually is, before any model touches it.

Per symbol it measures: row/duplicate/order integrity, OHLC legality, zero-volume
share, per-year and per-day bar counts, inter-bar gap distribution, monthly bar
density, and ``uniform_start`` — the first month after which density is stable,
i.e. where the series is honestly M1 rather than server-side decimated.

Usage::

    python -m src.audit [--symbols EURUSD XAUUSD ...] [--history-dir PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FULL_DAY_FRACTION = 0.5  # a day counts as "full" above this fraction of the recent median


def load_history_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(
        path,
        dtype={
            "Date": "string",
            "Time": "string",
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Volume": "float64",
        },
    )
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y/%m/%d %H:%M:%S")
    return df, pd.Series(ts, name="ts")


def audit_symbol(path: Path) -> dict:
    df, ts = load_history_csv(path)
    n = len(df)

    duplicates = int(ts.duplicated().sum())
    out_of_order = int((ts.diff() < pd.Timedelta(0)).sum())

    oc_max = df[["Open", "Close"]].max(axis=1)
    oc_min = df[["Open", "Close"]].min(axis=1)
    violations = {
        "high_lt_low": int((df["High"] < df["Low"]).sum()),
        "high_lt_body": int((df["High"] < oc_max).sum()),
        "low_gt_body": int((df["Low"] > oc_min).sum()),
        "nonpositive_price": int((df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()),
        "negative_volume": int((df["Volume"] < 0).sum()),
    }

    mins = ts.diff().dt.total_seconds().div(60.0).dropna()
    gaps = {
        "gt_5min": int((mins > 5).sum()),
        "gt_1h": int((mins > 60).sum()),
        "gt_1d": int((mins > 1440).sum()),
    }

    per_year = {str(k): int(v) for k, v in ts.dt.year.value_counts().sort_index().items()}

    day = ts.dt.floor("D")
    daily = day.value_counts().sort_index()
    typical = float(daily.iloc[-60:].median()) if len(daily) else 0.0
    day_density_share = float((daily >= FULL_DAY_FRACTION * typical).mean())

    # A month counts as honest M1 when its bars are 1-minute spaced AND its bar and
    # active-day counts are in line with the recent norm. Partial sessions (Sunday
    # opens) depress single-day counts but not the monthly total, so per-day
    # thresholds are deliberately not used here.
    month_bars = ts.dt.to_period("M").value_counts().sort_index()
    month_days = daily.index.to_period("M").value_counts().sort_index()
    mins_labelled = pd.Series(mins.to_numpy(), index=ts.iloc[1:].to_numpy())
    monthly_gap = mins_labelled.groupby(mins_labelled.index.to_period("M")).median()
    ref_bars = float(month_bars.iloc[-12:].median())
    ref_days = float(month_days.iloc[-12:].median())
    uniform = (
        (monthly_gap.reindex(month_bars.index, fill_value=np.inf) <= 2.0)
        & (month_bars >= 0.6 * ref_bars)
        & (month_days.reindex(month_bars.index, fill_value=0) >= 0.6 * ref_days)
    )
    months = list(month_bars.index)
    flags = uniform.to_numpy()
    n_months = len(flags)
    tail_false = np.cumsum((~flags)[::-1])[::-1]
    run = np.zeros(n_months + 1, dtype=int)
    maxrun = np.zeros(n_months + 1, dtype=int)
    for i in range(n_months - 1, -1, -1):
        run[i] = 0 if flags[i] else run[i + 1] + 1
        maxrun[i] = max(run[i], maxrun[i + 1])

    # Earliest month whose suffix tolerates documented outages (<=5% non-uniform
    # months, never 3 consecutive) instead of demanding a flawless run. The chosen
    # boundary month may be a transition month with mixed resolution, so the
    # curated start is refined to day granularity using weekday densities.
    uniform_start_month = None
    uniform_start_ts = None
    for i in range(n_months):
        if not flags[i]:
            continue
        tail_len = n_months - i
        if tail_false[i] > max(1, 0.05 * tail_len) or maxrun[i] > 2:
            continue
        period = months[i]
        wk = daily[(daily.index >= period.start_time) & (daily.index <= period.end_time)]
        wk = wk[wk.index.dayofweek < 5]
        ok = wk >= FULL_DAY_FRACTION * typical
        pos = next((k for k in range(len(ok)) if bool(ok.iloc[k:].all())), None)
        if pos is None:
            if i + 1 < n_months and flags[i + 1]:
                nxt = period + 1
                uniform_start_month, uniform_start_ts = str(nxt), str(nxt.start_time)
                break
            continue
        uniform_start_month, uniform_start_ts = str(period), str(wk.index[pos])
        break

    if uniform_start_month is not None:
        start_period = pd.Period(uniform_start_month, freq="M")
        anomalous_months = [str(m) for m in months if m >= start_period and not uniform[m]]
    else:
        anomalous_months = []

    return {
        "symbol": path.name.split("_")[0],
        "source": str(path),
        "rows": n,
        "first": str(ts.min()),
        "last": str(ts.max()),
        "span_years": round((ts.max() - ts.min()).total_seconds() / (365.25 * 86400), 3),
        "duplicates": duplicates,
        "out_of_order": out_of_order,
        "zero_volume_share": round(float((df["Volume"] == 0).mean()), 6),
        "violations": violations,
        "gaps": gaps,
        "bars_per_year": per_year,
        "typical_bars_per_day": typical,
        "day_density_share": round(day_density_share, 4),
        "uniform_start_month": uniform_start_month,
        "uniform_start_ts": uniform_start_ts,
        "anomalous_months": anomalous_months,
        "monthly_bars": {str(m): int(v) for m, v in month_bars.items()},
        "monthly_days": {str(m): int(v) for m, v in month_days.items()},
        "monthly_uniform": {str(m): int(v) for m, v in uniform.items()},
        "monthly_median_gap_minutes": {str(m): round(float(v), 3) for m, v in monthly_gap.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    ap.add_argument("--history-dir", type=Path, default=config.HISTORY_DIR)
    args = ap.parse_args()

    config.ensure_dirs()
    reports = []
    for symbol in args.symbols:
        path = args.history_dir / f"{symbol}_M1.csv"
        print(f"[audit] {symbol} <- {path}", flush=True)
        report = audit_symbol(path)
        reports.append(report)
        with (config.AUDIT_DIR / f"{symbol}.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    out = config.ARTIFACTS_DIR / "audit_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, sort_keys=True)

    summary = pd.DataFrame(
        [
            {
                "symbol": r["symbol"],
                "rows": r["rows"],
                "years": r["span_years"],
                "first": r["first"][:10],
                "last": r["last"][:10],
                "uniform_start": (r["uniform_start_ts"] or "-")[:10],
                "anomalous": len(r["anomalous_months"]),
                "dup": r["duplicates"],
                "ooo": r["out_of_order"],
                "zero_vol%": round(100 * r["zero_volume_share"], 2),
                "viol": sum(r["violations"].values()),
                "gap_gt_1d": r["gaps"]["gt_1d"],
            }
            for r in reports
        ]
    )
    print()
    print(summary.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
