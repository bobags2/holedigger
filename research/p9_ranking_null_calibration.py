"""Calibration of the SHIPPED ranking null on synthetic zero-predictability data.

The reviewer's central claim, tested directly:

    run_cross_section computes the real statistic on RAW returns, while
    session_shift_null(beta_neutral=True) computes it on RESIDUALISED returns.
    Those are different functions, so the percentile is not a p-value.

This generates one-factor panels with ZERO idiosyncratic predictability at several market
autocorrelations, applies the real statistic and the shipped null exactly as the production
code does, and reports how often the real statistic lands above the null's own p95.
Nominal alpha for a p95 test is 5%.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.cross_section import beta_neutralise, daily_rank_corr, session_shift_null


def make_panel(n_days: int, k: int, rho_mkt: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R, prev): one-factor, market AR(rho_mkt), no idio predictability."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(1.0, 0.5, k)
    sd = 0.01 * np.sqrt(1 - rho_mkt**2)
    e = np.zeros(n_days)
    e[0] = rng.normal(0, 0.01)
    for t in range(1, n_days):
        e[t] = rho_mkt * e[t - 1] + rng.normal(0, sd)
    R = e[:, None] * beta[None, :] + rng.normal(0, 0.01, (n_days, k))
    prev = np.full_like(R, np.nan)
    prev[1:] = R[:-1]
    return R, prev


def one_trial(n_days, k, rho_mkt, seed, n_worlds=200):
    R, prev = make_panel(n_days, k, rho_mkt, seed)
    ok = np.isfinite(prev) & np.roll(np.isfinite(R), 1, axis=0)
    # THE REAL STATISTIC, verbatim from run_cross_section AFTER the fix: beta-neutralised
    Rn = beta_neutralise(R)
    ok_n = np.isfinite(prev) & np.roll(np.isfinite(Rn), 1, axis=0)
    rho = daily_rank_corr(prev, Rn, ok_n)
    real = float(np.nanmean(rho))
    # THE NULL on the SAME panel, same transform
    rng = np.random.default_rng(seed * 7 + 1)
    null = session_shift_null(Rn, prev, n_worlds=n_worlds, rng=rng)
    return real, float(np.nanpercentile(null, 95)), float(np.nanmean(null))


def main() -> None:
    print("Calibration of the SHIPPED ranking null on zero-predictability data")
    print("real statistic on RAW returns  vs  null on RESIDUALISED returns\n")
    n_days, k = 400, 42
    trials = 40
    print(f"{'market AR':>10} {'FP rate':>9} {'real':>8} {'null p95':>9} {'null mean':>10}")
    for rho_mkt in (0.0, 0.3, 0.6):
        hits = 0
        acc_real, acc_p95, acc_mean = [], [], []
        for t in range(trials):
            real, p95, nmean = one_trial(n_days, k, rho_mkt, seed=t + 1)
            hits += 1 if real > p95 else 0
            acc_real.append(real); acc_p95.append(p95); acc_mean.append(nmean)
        print(f"{rho_mkt:>10.1f} {hits / trials:>9.1%} {np.mean(acc_real):>8.4f} "
              f"{np.mean(acc_p95):>9.4f} {np.mean(acc_mean):>10.4f}")
    print(f"\nNominal alpha (real above its own null's p95) = 5%. trials={trials}, n_worlds=200")
    print("Doc claim under test: 8% at AR 0, and 8% at AR 0.3 and 0.6.")


if __name__ == "__main__":
    main()
