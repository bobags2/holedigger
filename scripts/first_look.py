"""First calibration look — plumbing demonstration, not research.

Toy family: sign of the lagged M5 log return, momentum (+) and reversion (-), over a
small lag grid, on the curated M1 parquet resampled to M5. Real t-stats are compared
against the distribution of the *same family's* best t on surrogate worlds
(``mean_block=1``: no serial structure of any horizon; lag-0 cross-correlation and
marginals preserved exactly).

What this does not do: costs, holdouts, purged CV, or any economic mechanism. A
candidate that looks good here is a queue entry, not a finding.

Usage::

    python -m scripts.first_look [--symbols EURUSD GBPUSD XAUUSD] [--worlds 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow direct script run

from src import config, noise_floor  # noqa: E402
from src.surrogates import surrogate_returns  # noqa: E402

LAGS = [1, 2, 3, 5, 10, 15, 30, 60]
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD"]


def load_m5_closes(symbols: list[str]) -> pd.DataFrame:
    closes = {}
    for s in symbols:
        df = pd.read_parquet(config.PARQUET_DIR / f"{s}_M1.parquet").set_index("ts")
        m5 = (
            df.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        closes[s] = m5["close"]
    return pd.concat(closes, axis=1).dropna()


def family_tstats(r: np.ndarray, lags: list[int]) -> dict[str, np.ndarray]:
    """t-stat per rule per column. r: (T, K) log returns."""
    out: dict[str, np.ndarray] = {}
    for lag in lags:
        pnl = np.sign(r[:-lag]) * r[lag:]
        t = pnl.mean(axis=0) / (pnl.std(axis=0, ddof=1) / np.sqrt(pnl.shape[0]))
        out[f"mom_{lag}"] = t
        out[f"rev_{lag}"] = -t
    return out


def flatten(tstats: dict[str, np.ndarray], syms: list[str]) -> dict[str, float]:
    return {f"{name}|{s}": float(tstats[name][i]) for name in tstats for i, s in enumerate(syms)}


def lag1_probe(r: np.ndarray, spacing_ok: np.ndarray) -> dict:
    """Lag-1 reversion edge, and how much of it survives exact 5-minute adjacency.

    ``spacing_ok[i]`` is True when bar i+1 is exactly 5 minutes after bar i. Rows
    across session boundaries are kept in the main family; this probe isolates how
    much of the lag-1 effect is genuine bar-to-bar structure rather than gap-fill.
    """
    pnl = np.sign(r[:-1]) * r[1:]
    good = spacing_ok[1:] & spacing_ok[:-1]

    def tstat(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=0) / (x.std(axis=0, ddof=1) / np.sqrt(x.shape[0]))

    return {
        "share_clean": float(good.mean()),
        "t_all": tstat(pnl),
        "t_clean": tstat(pnl[good]),
        "momentum_edge_bp_all": pnl.mean(axis=0) * 1e4,
        "momentum_edge_bp_clean": pnl[good].mean(axis=0) * 1e4,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--lags", nargs="+", type=int, default=LAGS)
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--mean-block", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "first_look.json")
    args = ap.parse_args()

    config.ensure_dirs()
    wide = load_m5_closes(args.symbols)
    ret = np.log(wide).diff().dropna()
    R = ret.to_numpy()
    syms = list(ret.columns)
    years = (wide.index[-1] - wide.index[0]).total_seconds() / (365.25 * 86400)
    gap_min = wide.index.to_series().diff().dt.total_seconds().div(60.0).to_numpy()
    spacing_ok = np.isfinite(gap_min[1:]) & (gap_min[1:] == 5.0)

    real = flatten(family_tstats(R, args.lags), syms)
    real_max = max(abs(v) for v in real.values())

    probe = lag1_probe(R, spacing_ok)
    probe_by_symbol = {
        s: {k: (float(v[i]) if isinstance(v, np.ndarray) else float(v)) for k, v in probe.items()}
        for i, s in enumerate(syms)
    }

    rng = np.random.default_rng(config.SEED)
    null_max = np.empty(args.worlds)
    null_rule = {k: 0.0 for k in real}
    for w in range(args.worlds):
        flat = flatten(family_tstats(surrogate_returns(R, args.mean_block, rng), args.lags), syms)
        for k in null_rule:
            null_rule[k] += flat[k]
        null_max[w] = max(abs(v) for v in flat.values())
    for k in null_rule:
        null_rule[k] /= args.worlds

    n_family = len(real)
    floor = noise_floor.expected_max_t(n_family, n_sims=400, rng=rng)
    percentile = float((null_max < real_max).mean())
    z = float((real_max - null_max.mean()) / null_max.std(ddof=1))

    order = sorted(real, key=lambda k: -abs(real[k]))
    result = {
        "config": {
            "symbols": syms,
            "lags": args.lags,
            "mean_block": args.mean_block,
            "worlds": args.worlds,
            "n_bars": int(R.shape[0]),
            "years": round(years, 3),
            "family_size": n_family,
        },
        "real_tstats": real,
        "real_top": {k: round(real[k], 3) for k in order[:8]},
        "lag1_probe": probe_by_symbol,
        "null": {
            "family_max_mean": float(null_max.mean()),
            "family_max_sd": float(null_max.std(ddof=1)),
            "family_max_p95": float(np.percentile(null_max, 95)),
            "family_max_max": float(null_max.max()),
            "analytic_floor_mean": float(floor["mean"]),
            "per_rule_mean_abs_should_be_zero": {k: round(v, 4) for k, v in null_rule.items()},
        },
        "verdict": {
            "real_max_abs_t": round(real_max, 3),
            "real_max_percentile_in_null": round(percentile, 4),
            "real_max_z_vs_null": round(z, 3),
            "real_max_over_analytic_floor": round(real_max / floor["mean"], 3),
        },
        "note": (
            "Toy family on mid-price M5 bars; no costs, no holdout, no mechanism. "
            "A high percentile here is a queue entry, not a finding."
        ),
    }

    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    names = sorted({k.split("|")[0] for k in real})
    table = pd.DataFrame(
        [{"rule": name, **{s: real[f"{name}|{s}"] for s in syms}} for name in names]
    ).set_index("rule")
    print(f"M5 bars: {R.shape[0]:,}  symbols: {', '.join(syms)}  span: {years:.2f}y  family: {n_family}")
    print("\nreal t-stats:")
    print(table.round(2).to_string())
    probe_df = pd.DataFrame(probe_by_symbol).T[
        ["share_clean", "t_all", "t_clean", "momentum_edge_bp_all", "momentum_edge_bp_clean"]
    ]
    print("\nlag-1 probe (momentum edge in bp/trade; negative = reversion pays):")
    print(probe_df.round(4).to_string())
    print("\nnull distribution of the family max |t|  (mean_block=%.1f, %d worlds):" % (args.mean_block, args.worlds))
    print(f"  mean {null_max.mean():.2f} | sd {null_max.std(ddof=1):.2f} | p95 {np.percentile(null_max, 95):.2f} | max {null_max.max():.2f}")
    print(f"  analytic E[max t] for N={n_family}: {floor['mean']:.2f}")
    print("\nreal family max |t| = %.2f  -> percentile in null: %.1f%%  (z = %.2f)" % (real_max, 100 * percentile, z))
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
