"""Contract tests for the null factory (src/surrogates.py).

Run:  python -m tests.test_surrogates
"""
from __future__ import annotations

import numpy as np

from src.surrogates import returns_to_prices, stationary_bootstrap_indices, surrogate_returns


def _tstat(pnl: np.ndarray) -> float:
    return float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))))


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.dot(x[1:], x[:-1]) / np.dot(x, x))


def _lag_rule_t(r: np.ndarray, lag: int) -> float:
    pnl = np.sign(r[:-lag]) * r[lag:]
    return _tstat(pnl)


def test_index_vector_shape_and_run_length() -> None:
    rng = np.random.default_rng(0)
    n, mean_block = 200_000, 20.0
    idx = stationary_bootstrap_indices(n, mean_block, rng)
    assert idx.shape == (n,) and idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < n
    adjacency = float((idx[1:] == idx[:-1] + 1).mean())
    expected = 1.0 - 1.0 / mean_block
    assert abs(adjacency - expected) < 0.01, (adjacency, expected)


def test_permutation_null_destroys_all_horizons() -> None:
    """mean_block=1: a planted lag-20 effect must vanish."""
    rng = np.random.default_rng(1)
    T, lag = 50_000, 20
    e = rng.standard_normal(T)
    r = e.copy()
    r[lag:] += 0.4 * e[:-lag]

    t_real = _lag_rule_t(r, lag)
    assert abs(t_real) > 20, t_real

    ts = [abs(_lag_rule_t(surrogate_returns(r[:, None], 1.0, rng)[:, 0], lag)) for _ in range(5)]
    assert max(ts) < 5, ts


def test_block_null_destroys_signal_beyond_block() -> None:
    """mean_block=2: structure at lag 20 must still vanish (blocks are far shorter)."""
    rng = np.random.default_rng(2)
    T, lag = 50_000, 20
    e = rng.standard_normal(T)
    r = e.copy()
    r[lag:] += 0.4 * e[:-lag]

    ts = [abs(_lag_rule_t(surrogate_returns(r[:, None], 2.0, rng)[:, 0], lag)) for _ in range(5)]
    assert max(ts) < 5, ts


def test_block_length_semantics_partial_retention() -> None:
    """mean_block=L keeps ~(1 - 1/L) of adjacent-bar structure and nothing else.

    This is the documented trade-off, not a bug: put the block length between the
    noise scale and the candidate horizon.
    """
    rng = np.random.default_rng(3)
    T = 50_000
    e = rng.standard_normal(T)
    r = np.empty(T)
    r[0] = e[0]
    for t in range(1, T):
        r[t] = 0.2 * r[t - 1] + e[t]

    t_real = _lag_rule_t(r, 1)
    assert t_real > 20, t_real

    ratios = []
    for _ in range(10):
        s = surrogate_returns(r[:, None], 2.0, rng)[:, 0]
        ratios.append(_lag_rule_t(s, 1) / t_real)
    ratio = float(np.mean(ratios))
    assert 0.35 < ratio < 0.65, ratio


def test_preserves_vol_clustering() -> None:
    """mean_block=50 keeps |r| autocorrelation (GARCH-driven) intact."""
    rng = np.random.default_rng(4)
    T = 50_000
    w, a, b = 0.1, 0.1, 0.85
    e = rng.standard_normal(T)
    s2 = np.empty(T)
    r = np.empty(T)
    s2[0] = w / (1 - a - b)
    for t in range(T):
        if t:
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * e[t]

    acf_real = _acf1(np.abs(r))
    surrogates = [surrogate_returns(r[:, None], 50.0, rng)[:, 0] for _ in range(5)]
    acf_s = float(np.mean([_acf1(np.abs(s)) for s in surrogates]))
    assert abs(acf_s - acf_real) < 0.03, (acf_real, acf_s)
    assert acf_real > 0.05, acf_real


def test_preserves_lag0_cross_correlation() -> None:
    rng = np.random.default_rng(5)
    T, rho = 50_000, 0.6
    x = rng.standard_normal((T, 2))
    x[:, 1] = rho * x[:, 0] + np.sqrt(1 - rho**2) * x[:, 1]
    c_real = float(np.corrcoef(x.T)[0, 1])
    for mean_block in (1.0, 50.0):
        y = surrogate_returns(x, mean_block, rng)
        c = float(np.corrcoef(y.T)[0, 1])
        assert abs(c - c_real) < 0.02, (mean_block, c_real, c)


def test_returns_to_prices_roundtrip() -> None:
    rng = np.random.default_rng(6)
    r = rng.standard_normal((100, 3)) * 0.01
    p = returns_to_prices(r, s0=100.0)
    assert p.shape == (101, 3)
    assert np.allclose(p[0], 100.0)
    assert np.allclose(np.diff(np.log(p), axis=0), r)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
