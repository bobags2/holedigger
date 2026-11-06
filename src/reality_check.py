"""Data-snooping reality checks: White's RC and Hansen's SPA, on the shared panel.

Why this exists
---------------
A candidate's own null asks "is *this* candidate extreme under worlds with no structure".
It does not ask the question the whole project needs answered: **is this candidate extreme
against everything else the project tried?** With 9,271 registered hypotheses and 8,649 of
them evaluated more than once, a t of 3.0 means one thing if it is the only thing ever tried
and something else entirely if 7,000 siblings were scanned for a maximum.

White's Reality Check and Hansen's Superior Predictive Ability are the two standard answers,
and they differ in exactly the way that matters here:

* **White's RC** tests the *equal-weighted average* of all candidates against the null of
  "every candidate has zero performance". It is the right test for "does the search as a
  whole find anything", and it is *conservative* — it can reject when no single candidate
  survives on its own.
* **Hansen's SPA** tests the *best* candidate, but studentizes it by the family's own
  dispersion and truncates the bootstrap distribution to remove the "a lucky null world has
  a huge max" tail that makes the naive max-t anticonservative. That truncation is what
  makes SPA the sharper tool for "is the best candidate real".

Both run on ``src.gpu_null.DayPanel`` — one days x candidates matrix — and both reuse the
sign-flip/block-bootstrap worlds, so 1e5 worlds cost one GEMM per chunk.

Usage::

    from src import reality_check
    res = reality_check.reality_check(panel, n_worlds=10_000)
    spa = reality_check.spa_p_value(panel, n_worlds=10_000, candidate=res["best"])
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import gpu_null

# Hansen's truncation: the fraction of the bootstrap distribution beyond which the max is
# replaced by the threshold. Hansen (2005) shows the normalized statistic is well behaved
# at this value; it is deliberately not tuned to the data.
SPA_TRUNCATION = 0.95


def family_statistics(panel: gpu_null.DayPanel) -> dict:
    """The three per-candidate views a reality check needs, from the panel.

    ``t``      — per-candidate studentized t (what the sweep reports).
    ``se``     — per-candidate standard error, the SPA studentizer.
    ``mean``   — per-candidate mean per-day P&L, White's performance measure.
    ``n``      — per-candidate day count.
    """
    mean, t, n = gpu_null.real_moments(panel)
    p = panel.pnl
    with np.errstate(invalid="ignore", divide="ignore"):
        ss = np.einsum("ij,ij->j", p, p) - (p.sum(axis=0) ** 2) / n
        var = ss / np.maximum(n - 1.0, 1.0)
        se = np.sqrt(np.maximum(var, 0.0)) / np.sqrt(np.maximum(n, 1.0))
    return {"t": t, "se": se, "mean": mean, "n": n}


def reality_check(
    panel: gpu_null.DayPanel,
    *,
    n_worlds: int = 10_000,
    rng: np.random.Generator | None = None,
    chunk: int = gpu_null.DEFAULT_CHUNK,
    backend: str = "auto",
    bootstrap: bool = False,
) -> dict:
    """White's Reality Check: is the *equal-weight average* of all candidates real?

    The portfolio is constructed as one combined candidate — each day's P&L is the mean of
    that day's per-candidate P&L — and its t is computed on both the real panel and every
    null world. Both sides therefore live on the same dimensionless scale, which a raw-mean
    comparison would not (an earlier version compared a raw mean against a distribution of
    t-statistics and reported meaningless p-values).

    The null asks "does this portfolio have zero performance"; the check asks how often the
    null worlds beat it. ``bootstrap=True`` uses a day-block bootstrap of demeaned P&L
    instead of sign flips, for panels whose columns are serially dependent.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    # the equal-weight portfolio: one column, each day's value the mean across the
    # candidates observed THAT day (a day with no observation carries no portfolio value,
    # so it must not be averaged as a zero)
    with np.errstate(invalid="ignore"):
        with np.warnings.catch_warnings():
            # a day with no observed candidate is expected (weekends, holidays); the NaN it
            # produces is handled by port_ok below, so the warning is noise
            np.warnings.simplefilter("ignore", RuntimeWarning)
            row_mean = np.where(panel.valid, panel.pnl, np.nan)
            port_val = np.nanmean(row_mean, axis=1, keepdims=True)
    port_ok = np.isfinite(port_val)
    port = gpu_null.DayPanel(
        panel.days[port_ok[:, 0]], ("portfolio",), ((0, 1),),
        np.where(port_ok, port_val, 0.0), port_ok, panel.demean,
    )
    real_t = float(gpu_null.real_moments(port)[1][0])

    worlds = _worlds(port, n_worlds, rng, chunk, backend, bootstrap)
    null = np.asarray(worlds)[:, 0]
    p = float((null >= real_t).mean())
    return {
        "test": "white-rc",
        "real_t": real_t,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "n_candidates": panel.n_candidates,
        "n_worlds": n_worlds,
        "note": (
            "conservative: the equal-weight portfolio over the whole search, so it rejects "
            "when the search as a whole finds nothing even if one candidate alone survives"
        ),
    }


def spa_p_value(
    panel: gpu_null.DayPanel,
    *,
    n_worlds: int = 10_000,
    rng: np.random.Generator | None = None,
    chunk: int = gpu_null.DEFAULT_CHUNK,
    backend: str = "auto",
    bootstrap: bool = False,
    candidate: int | None = None,
    truncation: float = SPA_TRUNCATION,
) -> dict:
    """Hansen's SPA p-value for one candidate, against the whole family.

    The naive max-t is anticonservative: taking the best of 7,245 candidates against a
    per-candidate 95th percentile rejects far more than 5% of noise worlds, because a lucky
    null world can produce a huge maximum no real search would match. Hansen's fix works on
    **mean performance** (not t) and needs two things:

    1. *studentize by the candidate's own standard error*, so candidates measured on very
       different dispersion are comparable - the statistic is ``mean_j / se_j``;
    2. *truncate* the bootstrap distribution at the ``truncation`` quantile and replace the
       tail with that threshold, which removes the anticonservative tail.

    The p-value is the share of null worlds whose truncated studentized max reaches the real
    candidate's studentized mean. Degenerate candidates (no valid day, se = 0) cannot be
    tested and cannot be the null's max, so they are excluded rather than divided by zero -
    an earlier version divided by se directly and produced a null p95 of 33,578, i.e. a
    studentizer so large that nothing could ever reject.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    stats = family_statistics(panel)
    t, se, mean = stats["t"], stats["se"], stats["mean"]
    live = np.isfinite(mean) & np.isfinite(se) & (se > 0)
    if not live.any():
        raise ValueError("no candidate with a usable standard error")
    abs_mean_live = np.where(live, np.abs(np.nan_to_num(mean, nan=-np.inf)), -np.inf)
    j = int(np.argmax(abs_mean_live)) if candidate is None else int(candidate)
    if not (0 <= j < len(mean)):
        raise ValueError(f"candidate {j} outside 0..{len(mean) - 1}")
    if not live[j]:
        raise ValueError(f"candidate {j} has no usable standard error")

    worlds = _world_means(panel, n_worlds, rng, chunk, backend, bootstrap)
    # wm: (n_worlds, n_candidates) per-candidate MEAN performance in each world. Studentize
    # by the real se - the same dispersion the candidate was measured on - so the comparison
    # is on one scale for every candidate.
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(
            live[None, :],
            np.nan_to_num(worlds, nan=0.0) / np.maximum(se[None, :], 1e-300),
            -np.inf,
        )
    v = np.max(a, axis=1)                       # each world's best studentized mean
    thr = float(np.quantile(v, truncation))
    v_trunc = np.minimum(v, thr)                # remove the anticonservative tail
    real = float(mean[j]) / float(se[j])        # the real candidate's studentized mean
    p = float((v_trunc >= real).mean())
    return {
        "test": "hansen-spa",
        "candidate": j,
        "real_t": float(t[j]),
        "real_studentized_mean": real,
        "candidate_se_bp": float(se[j]) * 1e4,
        "null_studentized_p95": float(np.quantile(v, 0.95)),
        "truncation": truncation,
        "threshold": thr,
        "p_value": p,
        "n_candidates": int(live.sum()),
        "n_candidates_total": panel.n_candidates,
        "n_worlds": n_worlds,
    }


def _world_means(panel, n_worlds, rng, chunk, backend, bootstrap):
    """Per-world per-candidate MEAN P&L, not t: SPA studentizes means.

    ``sign_flip_null`` returns t-statistics, so this reconstructs the mean from the same
    moment identity: under a flip vector f, ``mean_j(f) = (f @ x_j)/n_j``. Reusing the null's
    own GEMM is not possible here (it computes t directly), so the means are computed in one
    batched product over the flips - which is one GEMM and keeps 1e5 worlds cheap.
    """
    n = panel.n_valid.astype(np.float64)
    x = panel.pnl
    n_days, n_cand = x.shape
    if bootstrap:
        out = np.empty((n_worlds, n_cand), dtype=np.float64)
        for w in range(n_worlds):
            i = gpu_null.stationary_bootstrap_indices(
                n_days, gpu_null.DEFAULT_BOOTSTRAP_DAYS, rng
            )
            out[w] = x[i].sum(axis=0) / np.maximum(n, 1.0)
        return out
    # sign flips: one block of (n_worlds, n_days) +-1 vectors, one GEMM
    flips = rng.integers(0, 2, size=(n_worlds, n_days)).astype(np.float64) * 2.0 - 1.0
    return flips @ x / np.maximum(n, 1.0)


def _worlds(panel, n_worlds, rng, chunk, backend, bootstrap):
    """One batch of null worlds on the same panel, sign-flip or day-block bootstrap."""
    if bootstrap:
        return gpu_null.block_bootstrap_null(
            panel, n_worlds, rng, chunk=chunk, backend=backend
        ).t
    return gpu_null.sign_flip_null(
        panel, n_worlds, rng, chunk=chunk, backend=backend
    ).t
