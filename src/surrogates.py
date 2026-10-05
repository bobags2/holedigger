"""Null factory: surrogate worlds with tradable structure removed.

This is the calibration core. A candidate's evidence is measured against the
distribution of what the *same search procedure* produces on worlds that contain
no exploitable structure — not against zero.

Contract, enforced by ``tests/test_surrogates.py``:

* ``mean_block == 1`` — joint time-index permutation. Destroys serial structure of
  every horizon; preserves the exact marginal distribution and the cross-sectional
  row composition (lag-0 cross-correlation is untouched). Use as the null for
  "no serial predictability".
* ``mean_block  > 1`` — Politis-Romano stationary bootstrap over a joint time index
  shared by all columns. Serial dependence survives *within* blocks only, so a
  signal whose memory is shorter than the block scale leaks through. Put the block
  length between the noise scale and the candidate horizon on purpose, and record
  which null was used in the candidate's evidence packet.

What the factory cannot do: preserve volatility clustering while also killing
predictability at the same horizon — those are the same autocorrelation at
different lags. That trade-off is why the block length is an explicit parameter
and why the analytic floor (``src/noise_floor.py``) is kept as a second, model-free
check.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Index vector of length n. Mean run length ~ mean_block (circular blocks)."""
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    if mean_block == 1:
        return rng.permutation(n)
    p = 1.0 / float(mean_block)
    out = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        length = int(rng.geometric(p))
        start = int(rng.integers(0, n))
        take = min(length, n - filled)
        out[filled : filled + take] = (start + np.arange(take, dtype=np.int64)) % n
        filled += take
    return out


def surrogate_returns(r: np.ndarray, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One surrogate world for a (T, K) return panel; same time index for all columns."""
    idx = stationary_bootstrap_indices(r.shape[0], mean_block, rng)
    return r[idx]


def iter_surrogates(
    r: np.ndarray, mean_block: float, n_worlds: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    for _ in range(n_worlds):
        yield surrogate_returns(r, mean_block, rng)


def returns_to_prices(r: np.ndarray, s0: float = 1.0) -> np.ndarray:
    """Notional price levels from log returns: (T, K) returns -> (T+1, K) prices."""
    pad = np.zeros((1, r.shape[1]), dtype=r.dtype)
    return s0 * np.exp(np.cumsum(np.concatenate([pad, r], axis=0), axis=0))
