"""Tests for src/noise_floor.py.

Run:  python -m tests.test_noise_floor
"""
from __future__ import annotations

import numpy as np

from src import noise_floor


def test_monotone_in_n() -> None:
    rng = np.random.default_rng(0)
    small = noise_floor.expected_max_t(10, n_sims=200, rng=rng)["mean"]
    large = noise_floor.expected_max_t(100_000, n_sims=200, rng=rng)["mean"]
    assert small < large
    assert 1.35 < small < 1.75, small  # E[max of 10 N(0,1)] ~ 1.54


def test_against_direct_simulation() -> None:
    """The N(0,1) approximation must match simulated backtests of iid rules."""
    rng = np.random.default_rng(1)
    res = noise_floor.validate_direct(n_rules=1500, n_days=1000, n_trials=40, rng=rng)
    assert abs(res["mean_direct"] - res["analytic"]) < 0.2, res
    assert res["sd_direct"] < 0.6, res


def test_headline_floor_number() -> None:
    """1M independent rules over ~5.7 years: the lucky-noise Sharpe is ~2."""
    rng = np.random.default_rng(2)
    sr = noise_floor.floor_sharpe(1_000_000, years=5.7, n_sims=200, rng=rng)
    assert 1.8 < sr["sharpe_mean"] < 2.25, sr
    assert sr["sharpe_mean"] < sr["sharpe_p95"]


def test_floor_scales_with_years() -> None:
    rng = np.random.default_rng(3)
    a = noise_floor.floor_sharpe(10_000, years=4.0, n_sims=300, rng=rng)["sharpe_mean"]
    b = noise_floor.floor_sharpe(10_000, years=16.0, n_sims=300, rng=rng)["sharpe_mean"]
    assert abs(a / b - 2.0) < 0.15, (a, b)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
