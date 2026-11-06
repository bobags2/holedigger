"""``P`` as the universal interface: any strategy's OOS daily P&L, priced by one null.

What this module is for
-----------------------
Everything upstream of ``src/gpu_null.py`` answers one question per run: *does this
candidate clear its null?* A learned strategy — a model, a hyper-parameter set, a feature
set, a horizon — produces a **daily P&L column**, and this module makes that column
first-class: append it to the same days x candidates matrix ``P`` the sign-flip null
already prices, and the best of everything is calibrated at 1e5+ worlds for the cost of
one GEMM per chunk.

Two disciplines live here, and both are load-bearing:

* **purged walk-forward.** A column is only ever computed out-of-sample: fit on days
  ``[0, k)``, realise days ``[k, k+h)``, purge the ``embargo`` days adjacent to the split.
  Otherwise the column's own lookahead is priced as edge, which is the single most common
  way a "learned strategy" beats a null.
* **selection inside the test.** Selecting the winner on one part of the sample and
  reporting it on another is not a test. ``falsify_selection`` runs the *whole selection
  rule* inside every null world: the same rule that picked the real winner picks a null
  winner, so the reported p-value already carries the cost of having looked.

Design note on what "a strategy" means: it is anything with
``fit(days) -> predictor`` and ``pnl(predictor, days) -> daily P&L``. That is a
gradient-boosted model, a linear model, a hand-written rule, or a candidate column already
in ``P`` — the null does not care, which is the point.

Usage::

    from src import oos_eval
    cols = oos_eval.walk_forward_columns(bars, strategies, n_folds=5, embargo=2)
    panel = oos_eval.merge_columns(panel, cols)
    res = oos_eval.falsify_selection(panel, n_worlds=10_000)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from . import gpu_null

DEFAULT_FOLDS = 5
DEFAULT_EMBARGO = 2  # days after a training window whose labels can still see it


# --------------------------------------------------------------------- strategies


@dataclass(frozen=True)
class Strategy:
    """Anything with ``fit`` and ``pnl``. The null does not know which kind it is.

    ``fit(day_index, returns) -> predictor`` sees ONLY training days.
    ``pnl(predictor, day_index, returns) -> array`` produces the realised daily P&L for
    the days it is given. Both are pure functions of their arguments, which is what makes
    the whole thing reproducible and the purging meaningful.
    """

    name: str
    fit: Callable
    pnl: Callable
    family: str = "learned"


def walk_forward_columns(
    days: Sequence,
    strategies: Sequence[Strategy],
    *,
    n_folds: int = DEFAULT_FOLDS,
    embargo: int = DEFAULT_EMBARGO,
    ret_fn: Callable | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Out-of-sample daily P&L for every strategy, plus its valid mask.

    Returns ``{strategy_name: {"day": [...], "pnl": [...], "oob": [...]}}`` where ``oob``
    marks the fold's test days. Days inside the embargo band (or before the first test
    fold / after the last) carry ``pnl == None`` and ``valid == False``: they are never
    realised, never scored, and the null's day count excludes them.

    ``ret_fn(day) -> per-symbol return vector`` is how a strategy reads the market; the
    default is a flat-zero so a caller that supplies ``pnl`` directly still works.
    """
    n = len(days)
    if n_folds < 2:
        raise ValueError("n_folds >= 2: one fold leaves nothing to test against")
    if embargo < 0:
        raise ValueError("embargo >= 0")
    n_folds = min(n_folds, n - 1)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    out: dict[str, dict[str, np.ndarray]] = {}

    for st in strategies:
        pnl = np.full(n, np.nan)
        oob = np.zeros(n, dtype=bool)
        for k in range(n_folds - 1):
            a, b = int(edges[k]), int(edges[k + 1])
            # Purge, on both sides of the boundary. A label inside [b, b+embargo) can still
            # have seen a signal from before b, so the first ``embargo`` test days are
            # dropped; symmetrically the last ``embargo`` training days are dropped. Without
            # this the boundary leaks and the column prices its own lookahead as edge.
            train_hi = a - embargo
            test_lo = b + embargo
            if train_hi <= 0:
                # keep the opening fold honest: train on nothing is not a fold
                train_hi = max(a, 1)
            train = np.arange(0, train_hi)
            test = np.arange(test_lo, int(min(edges[k + 2], n)))
            if not len(train) or not len(test):
                continue
            predictor = st.fit(days[train], ret_fn)
            realised = np.asarray(st.pnl(predictor, days[test], ret_fn), dtype=float)
            if realised.shape != (len(test),):
                raise ValueError(f"{st.name}: pnl shape {realised.shape} != {(len(test),)}")
            pnl[test] = realised
            oob[test] = True
        out[st.name] = {"day": np.asarray(days), "pnl": pnl, "oob": oob}
    return out


def merge_columns(panel: gpu_null.DayPanel, cols: Mapping[str, Mapping[str, np.ndarray]]) -> gpu_null.DayPanel:
    """Append OOS columns to an existing panel. The extended ``P`` is one panel again."""
    if not cols:
        return panel
    n_days = len(panel.days)
    pnl = [panel.pnl]
    valid = [panel.valid]
    symbols = list(panel.symbols)
    windows = list(panel.windows)
    for name, c in cols.items():
        p = np.asarray(c["pnl"], dtype=float)
        if len(p) != n_days:
            raise ValueError(f"{name}: {len(p)} days, panel has {n_days}")
        ok = np.isfinite(p)
        pnl.append(np.where(ok, p, 0.0)[:, None])
        valid.append(ok[:, None])
        symbols.append(name)
        windows.append(("oos", 0))
    return gpu_null.DayPanel(
        panel.days,
        tuple(symbols),
        tuple(windows),
        np.hstack(pnl),
        np.hstack(valid),
        panel.demean,
    )


# ------------------------------------------------------- selection inside the null


def select_top_k(t: np.ndarray, k: int = 1) -> np.ndarray:
    """Indices of the ``k`` most extreme candidates (the selection rule itself).

    Idiosyncratic details (ties broken by index) are fixed so the rule is a *function*:
    the same rule must pick a winner inside a null world, or the test is not a test.
    """
    if k < 1:
        raise ValueError("k >= 1")
    a = np.abs(np.nan_to_num(t, nan=-np.inf))
    order = np.argsort(-a, kind="stable")
    return order[:k]


def falsify_selection(
    panel: gpu_null.DayPanel,
    *,
    k: int = 1,
    n_worlds: int = 10_000,
    rng: np.random.Generator | None = None,
    chunk: int = gpu_null.DEFAULT_CHUNK,
    backend: str = "auto",
    max_bytes: int = gpu_null.DEFAULT_MAX_BYTES,
    null_kind: str = "sign-flip",
    block_days: float | None = None,
) -> dict:
    """Price a *selection rule*, not a candidate: the whole rule runs inside every world.

    The reported number is the fraction of null worlds whose selected candidate is at
    least as extreme as the real selection. A strategy that genuinely carries structure
    has its selection standing far outside that distribution; a strategy that merely won
    a search sits inside it, which is the whole point of running the rule inside.

    ``block_days`` switches to the block-bootstrap worlds, which is the right choice when
    the columns are serially dependent (a rolling-fit model's columns are, by construction).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if null_kind not in ("sign-flip", "block-bootstrap"):
        raise ValueError(f"null_kind must be sign-flip|block-bootstrap, got {null_kind!r}")
    _real_mean, real_t, _real_n = gpu_null.real_moments(panel)  # (mean, t, n): rank on t
    picked = select_top_k(real_t, k)
    real_stat = float(np.abs(real_t[picked]).max())

    if null_kind == "sign-flip":
        run = gpu_null.sign_flip_null(
            panel, n_worlds, rng, chunk=chunk, backend=backend,
            max_bytes=max_bytes, stat="studentized",
        )
    else:
        run = gpu_null.block_bootstrap_null(
            panel, n_worlds, rng, chunk=chunk, backend=backend,
            max_bytes=max_bytes, mean_block=block_days or gpu_null.DEFAULT_BOOTSTRAP_DAYS,
        )
    # The rule, run inside every world: best candidate per world, chosen by the same
    # |t| ranking the real selection used. Broadcasting the ranking over all candidates
    # at once is what keeps 10k worlds cheap (one argsort, no Python loop).
    abs_t = np.abs(np.nan_to_num(run.t, nan=-np.inf))            # (worlds, candidates)
    null_pick = np.argsort(-abs_t, axis=1, kind="stable")[:, :k]  # per-world winner ids
    null_stat = np.abs(np.take_along_axis(run.t, null_pick, axis=1)).max(axis=1)
    p_vals = candidate_p_values(panel, run.t)

    return {
        "selected": picked,
        "real_stat": real_stat,
        "null": {
            "mean": float(null_stat.mean()),
            "p95": float(np.percentile(null_stat, 95)),
            "max": float(null_stat.max()),
            "percentile": float((null_stat < real_stat).mean()),
        },
        # how much of the WHOLE family is non-null, not just whether the winner survives:
        # one null sample answers both, and pi0 is what says whether searching here is
        # worth the multiplicity at all. Built on per-candidate p-values, because Storey's
        # estimator is defined on p-values and a best-of-family sample has no per-candidate
        # tail to fit.
        "p_values": p_vals,
        "pi0": storey_pi0(p_vals),
        # family-wise-adjusted: the fraction of null worlds in which ANY candidate reaches
        # this one's extremeness, i.e. the price of having searched the whole family
        "adjusted_p": stepdown_from_null(run.t, real_t),
        "n_worlds": n_worlds,
        "null_kind": null_kind,
        "info": run.info(),
    }


def candidate_p_values(panel: gpu_null.DayPanel, null_t: np.ndarray) -> np.ndarray:
    """Per-candidate two-sided p-values, from a null (worlds x candidates) t matrix.

    ``p_j = (worlds with |t| >= |t_real_j| + 1) / (worlds + 1)`` — the add-one avoids
    reporting exactly zero, which is what makes the Storey tail fit and FDR arithmetic
    well defined. This is the per-candidate counterpart of the family scan-max, and it is
    the input every multiplicity correction needs.
    """
    real_t = gpu_null.real_moments(panel)[1]
    abs_r = np.abs(np.nan_to_num(real_t, nan=-np.inf))
    ge = (np.abs(np.nan_to_num(null_t, nan=-np.inf)) >= abs_r[None, :]).sum(axis=0)
    return (ge + 1.0) / (null_t.shape[0] + 1.0)


def storey_pi0(p: np.ndarray, lam: float = 0.5) -> dict:
    """Storey's estimate of the share of the candidate family that is non-null.

    ``pi0 = #{p > lam} / ((1 - lam) * m)``: with a uniform null, the fraction of
    p-values above any lambda is ``1 - lam``, so the observed share above lambda divided
    by that expectation estimates the null-proportion, and the excess over 1 is real
    structure. ``pi0`` near 1 means searching this family found nothing its multiplicity
    did not already buy; below 1 means structure exists somewhere in the family.
    """
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    m = len(p)
    if m == 0 or not (0.0 < lam < 1.0):
        return {"pi0": float("nan"), "m": m, "lambda": lam, "note": "degenerate input"}
    frac_above = float((p > lam).mean())
    pi0 = min(1.0, frac_above / (1.0 - lam))
    return {
        "pi0": pi0,
        "m": m,
        "lambda": lam,
        "share_above_lambda": frac_above,
        "note": "share of candidates consistent with a uniform null; 1.0 = nothing",
    }


def stepdown_from_null(null_t: np.ndarray, real_t: np.ndarray) -> np.ndarray:
    """Family-wise-adjusted p-values from the null, using each world's own family max.

    ``adj_j = mean over worlds of [ max_i |t_world,i| >= |t_real,j| ]``: the fraction of
    null worlds in which *some* candidate reaches this candidate's extremeness. That is the
    step-down distribution of the family max, and it is what controls family-wise error —
    a candidate is significant only if being this extreme is rare for the search as a
    whole, not merely for itself. The whole thing is one broadcast comparison against a
    per-world max, so 1e5 worlds cost one pass.
    """
    a = np.abs(np.nan_to_num(null_t, nan=-np.inf))
    world_max = a.max(axis=1)                      # the null's own family max, per world
    r = np.abs(np.nan_to_num(real_t, nan=-np.inf))
    return (world_max[None, :] >= r[:, None]).mean(axis=1)
