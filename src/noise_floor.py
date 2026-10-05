"""The selection-bias floor: what the best of N searched rules looks like when
there is nothing to find.

Under the null a rule's t-stat is ~N(0,1) once the sample is long enough
(validated against direct simulation in ``tests/test_noise_floor.py``). The best of
N *independent* rules therefore has expected max ~ sqrt(2 ln N), so

    SR_ann(best of N noise rules) ~ E[max Z_N] / sqrt(years)

Correlated rule families buy part of this back. Do not lean on that analytically —
measure the effective family with the surrogate factory and use this as the
model-free cross-check.

Usage::

    python -m src.noise_floor [--trials 1000 10000 100000 1000000] [--years 5.7]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import config


def expected_max_t(n_trials: int, n_sims: int = 1000, rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(config.SEED)
    mx = np.array([rng.standard_normal(n_trials).max() for _ in range(n_sims)])
    return {
        "n_trials": int(n_trials),
        "n_sims": int(n_sims),
        "mean": float(mx.mean()),
        "p05": float(np.percentile(mx, 5)),
        "p50": float(np.percentile(mx, 50)),
        "p95": float(np.percentile(mx, 95)),
    }


def floor_sharpe(n_trials: int, years: float, n_sims: int = 1000, rng=None) -> dict:
    stats = expected_max_t(n_trials, n_sims=n_sims, rng=rng)
    scale = float(np.sqrt(years))
    return {
        "n_trials": int(n_trials),
        "years": float(years),
        "sharpe_mean": stats["mean"] / scale,
        "sharpe_p05": stats["p05"] / scale,
        "sharpe_p95": stats["p95"] / scale,
    }


def validate_direct(n_rules: int, n_days: int, n_trials: int, rng=None) -> dict:
    """Direct simulation: N rule return series of length n_days, iid normal. No shortcuts."""
    rng = rng or np.random.default_rng(config.SEED)
    vals = np.empty(n_trials)
    for i in range(n_trials):
        x = rng.standard_normal((n_rules, n_days))
        t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n_days))
        vals[i] = t.max()
    return {
        "n_rules": int(n_rules),
        "n_days": int(n_days),
        "n_trials": int(n_trials),
        "mean_direct": float(vals.mean()),
        "sd_direct": float(vals.std(ddof=1)),
        "analytic": expected_max_t(n_rules, n_sims=400, rng=rng)["mean"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", nargs="+", type=int, default=[1_000, 10_000, 100_000, 1_000_000])
    ap.add_argument("--years", type=float, default=5.7, help="effective data span in years")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "noise_floor.json"))
    args = ap.parse_args()

    config.ensure_dirs()
    rng = np.random.default_rng(config.SEED)

    print(f"selection-bias floor @ {args.years:.2f} years effective span\n")
    print(f"{'rules tried':>12} {'E[max t]':>9} {'p05':>6} {'p95':>6} {'SR mean':>8} {'SR p95':>7}")
    out = {}
    for n in args.trials:
        stats = expected_max_t(n, n_sims=args.sims, rng=rng)
        scale = np.sqrt(args.years)
        row = floor_sharpe(n, args.years, n_sims=args.sims, rng=rng)
        out[str(n)] = {"t": stats, "sharpe": row}
        print(
            f"{n:>12,} {stats['mean']:>9.2f} {stats['p05']:>6.2f} {stats['p95']:>6.2f}"
            f" {row['sharpe_mean']:>8.2f} {row['sharpe_p95']:>7.2f}"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"years": args.years, "table": out}, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
