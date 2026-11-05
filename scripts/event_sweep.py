"""Expanded event sweep: does the +2bp incremental survive across the family?

Family = symbol x currency x horizon. Each candidate's evidence is the incremental
(event-window mean minus matched clock-control mean). The family verdict comes from
a label-permutation null: pool each candidate's event and control returns, randomly
reassign the labels, recompute the incremental; the max |incremental| over the
family under that permutation is the bar a real candidate must clear.

Usage::

    python -m scripts.event_sweep --symbols EURUSD GBPUSD XAUUSD --currencies USD EUR --horizons 15 30 60 --impact "High importance"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from scripts.event_test import (  # noqa: E402
    control_times,
    event_windows,
    load_events,
    load_mid,
    window_return,
)


def collect(symbol, currency, impact, horizon, events, mids):
    ev = event_windows(events, currency, impact)
    if not len(ev):
        return None
    mid = mids[symbol]
    ev_r = [r for t in ev.index if (r := window_return(mid, t, horizon)) is not None]
    ctrl = control_times(ev.index, mid.index, events, currency)
    c_r = [r for t in ctrl if (r := window_return(mid, t, horizon)) is not None]
    if len(ev_r) < 20 or len(c_r) < 20:
        return None
    return np.asarray(ev_r, float), np.asarray(c_r, float)


def incremental(ev_r, c_r):
    return float(ev_r.mean() - c_r.mean())


def t_of(ev_r, c_r):
    diff = ev_r.mean() - c_r.mean()
    se = np.sqrt(ev_r.var(ddof=1) / len(ev_r) + c_r.var(ddof=1) / len(c_r))
    return float(diff / se) if se > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "XAUUSD"])
    ap.add_argument("--currencies", nargs="+", default=["USD", "EUR", "GBP"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--impact", default="High importance")
    ap.add_argument("--perms", type=int, default=200, help="label permutations for the family null")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "event_sweep.json")
    args = ap.parse_args()

    events = load_events()
    mids = {s: load_mid(s, "5min")["close"] for s in args.symbols}

    combos = []
    for s in args.symbols:
        for ccy in args.currencies:
            for h in args.horizons:
                pair = collect(s, ccy, args.impact, h, events, mids)
                if pair is not None:
                    combos.append({"symbol": s, "currency": ccy, "horizon": h,
                                   "ev_r": pair[0], "c_r": pair[1]})
    if not combos:
        raise SystemExit("no usable combinations; widen the inputs")
    print(f"family: {len(combos)} candidates | permutations: {args.perms}")

    real = {i: incremental(c["ev_r"], c["c_r"]) for i, c in enumerate(combos)}
    real_max = max(abs(v) for v in real.values())

    rng = np.random.default_rng(config.SEED)
    null_max = np.empty(args.perms)
    for p in range(args.perms):
        mx = 0.0
        for c in combos:
            pool = np.concatenate([c["ev_r"], c["c_r"]])
            labels = np.array([1] * len(c["ev_r"]) + [0] * len(c["c_r"]))
            rng.shuffle(labels)
            mx = max(mx, abs(pool[labels == 1].mean() - pool[labels == 0].mean()))
        null_max[p] = mx

    rows = []
    for i, c in enumerate(combos):
        rows.append({
            "symbol": c["symbol"], "currency": c["currency"], "horizon": c["horizon"],
            "n_events": int(len(c["ev_r"])), "n_controls": int(len(c["c_r"])),
            "incremental_bp": round(real[i], 4),
            "t": round(t_of(c["ev_r"], c["c_r"]), 3),
            "null_percentile": round(float((null_max < abs(real[i])).mean()), 4),
        })
    rows.sort(key=lambda r: -r["incremental_bp"])

    result = {
        "config": {"symbols": args.symbols, "currencies": args.currencies,
                   "horizons": args.horizons, "impact": args.impact, "perms": args.perms},
        "family": {
            "candidates": len(combos),
            "real_max_abs_incremental_bp": round(real_max, 4),
            "null_max_mean": round(float(null_max.mean()), 4),
            "null_max_p95": round(float(np.percentile(null_max, 95)), 4),
            "real_percentile": round(float((null_max < real_max).mean()), 4),
        },
        "candidates": rows,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fam = result["family"]
    print(f"family max |incr| = {fam['real_max_abs_incremental_bp']} bp | null max mean "
          f"{fam['null_max_mean']} p95 {fam['null_max_p95']} | percentile {100 * fam['real_percentile']:.1f}%")
    print()
    print(f"{'symbol':>10} {'ccy':>4} {'h':>3} {'events':>7} {'incr bp':>9} {'t':>7} {'null pct':>9}")
    for r in rows[:12]:
        print(f"{r['symbol']:>10} {r['currency']:>4} {r['horizon']:>3} {r['n_events']:>7} "
              f"{r['incremental_bp']:>9.3f} {r['t']:>7.2f} {r['null_percentile']:>9.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
