"""Day-level sign-flip permutation null for the session-window sweep (GPU GEMM).

Why this null
-------------
``run_sweep``'s shared-intraday null permutes whole days *and* intraday slots. It keeps
the cross-section co-moving, but it is a permutation of **bars**: overnight / weekend gap
returns can land in windows that never see them (slot 0 is pinned, which mitigates it, not
eliminates it), every null world has exactly zero serial dependence, and the shared path
needs one bar grid for every symbol — on the ragged 105-name equities panel
(``_shared_day_grid_ok`` is False) the sweep falls back to the invalid per-symbol null.

This module acts on **one shared matrix P of realised per-day window P&L**, shape
``(n_days, n_candidates)``, rows trading days, columns ``(symbol, entry_hour, hold)``.
A null world is one random ±1 per day applied to *every* candidate at once: a day's whole
cross-section is either itself or its mirror. Nothing else changes, so cross-candidate and
cross-day co-movement, volatility clustering and fat tails all survive — only each day's
sign is randomised. Nothing needs a common bar grid: a window with no bars that day is
``0.0`` with ``valid`` False, the flip moves neither, and the day count ``n_j`` keeps the
statistic honest. The ragged panel runs as-is.

The statistic, and why it is two matrix products
------------------------------------------------
The day-level t of candidate ``j`` under a flip vector ``f`` is exactly what
``window_sweep.window_stats`` computes on the flipped values,
``t_j(f) = mean_j(f) / (sd_j(f) / sqrt(n_j))`` with ``mean_j(f) = (f @ x_j) / n_j``;
with ``ddof=1`` and the sample mean that is

    t_j(f) = s_j * sqrt(n_j - 1) / sqrt(n_j * Q_j - s_j**2),
    s_j = f @ x_j,  Q_j = sum_i x_ij**2,  n_j = valid days of candidate j.

``Q_j`` and ``n_j`` are identical in every world (``(f**2) @ (x**2) == x**2`` because
``f**2 == 1``), so the second moment is precomputed once and a whole chunk of worlds costs
one GEMM ``F @ P`` plus one cheap elementwise pass. Real and null statistics are the same
function of the same panel, so a candidate's null statistic cannot drift from its real t
(the repo's contract in ``_window_stats_of``). ``stat="studentized"`` (default) is the
formula above; ``stat="demeaned"`` is the spec's centred variant, which flips
``x - xbar`` and divides by its *fixed* scale, ``t = (f @ (x - xbar)) * sqrt(n-1) /
sqrt(n * sum (x - xbar)**2)``. They differ only in the tails:
``scripts/sign_flip_calibration.py`` measures both, and the default is the one that holds
its size at least as well on correlated noise.

The reference null
------------------
Sign flips assume the null is symmetric about zero. ``block_bootstrap_null`` builds the
alternative: centre each column, resample whole day blocks with replacement keeping every
row's cross-section intact, and compute the same t from the resampled values. Serial
dependence shorter than the block scale survives inside the blocks, so where the two
nulls disagree on real data the block bootstrap is the wider, more conservative reference.
It is the same GEMM with day counts instead of signs.

Backends
--------
``backend="auto"`` uses torch/CUDA when torch imports and a CUDA device exists, otherwise
float32 numpy BLAS; ``"numpy"``/``"torch"`` force one. Both expose the same one-method
seam (``dot``), so the torch backend is a drop-in. The chunk size is clamped so one
chunk's transient tiles stay under ``max_bytes`` (default 2 GiB); ``NullRun.info()``
reports the backend, the peak device allocation and the measured GEMM throughput.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .surrogates import stationary_bootstrap_indices

DEFAULT_CHUNK = 10_000
DEFAULT_DTYPE = np.float32
DEFAULT_MAX_BYTES = 2 * 1024**3
DEFAULT_BOOTSTRAP_DAYS = 5.0  # Politis-Romano mean block length, in days


@dataclass(frozen=True)
class DayPanel:
    """The shared days x candidates matrix a day-level null acts on.

    ``pnl[i, j]`` is window ``j``'s log-return P&L on day ``i`` (0.0 where invalid) and
    ``valid[i, j]`` says whether day ``i`` carries an observation for candidate ``j``.
    Zeros are the choice for invalid cells: a sign flip cannot move them, they are the
    additive identity for the per-candidate moments, and ``valid`` keeps the day count
    the statistic divides by — so raggedness stops mattering and the GEMM never sees NaN.
    """

    days: np.ndarray  # (n_days,) int64 day numbers
    symbols: tuple[str, ...]
    windows: tuple[tuple[int, int], ...]
    pnl: np.ndarray  # (n_days, n_candidates) float64, 0.0 where invalid
    valid: np.ndarray  # (n_days, n_candidates) bool
    demean: tuple[tuple[str, float], ...] = ()  # what was subtracted, per symbol, and why

    @property
    def n_valid(self) -> np.ndarray:
        return self.valid.sum(axis=0)

    @property
    def n_candidates(self) -> int:
        return self.pnl.shape[1]

    def column(self, j: int) -> np.ndarray:
        """The valid per-day values of candidate ``j`` (the brute-force reference view)."""
        return self.pnl[self.valid[:, j], j]


def day_pnl_panel(
    days_by_symbol: Mapping[str, np.ndarray],
    pnl_by_symbol: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    windows: Sequence[tuple[int, int]],
    demean_by: Mapping[str, float] | None = None,
) -> DayPanel:
    """Assemble the shared panel from per-symbol ``(days, pnl)`` blocks.

    ``pnl_by_symbol[sym]`` is ``(len(days_by_symbol[sym]), len(windows))`` with NaN where
    the window is invalid on that day. Days are unioned across symbols, columns are
    symbol-major so ``reshape(n_worlds, n_symbols, n_windows)`` stays valid for the sector
    test, and a symbol missing a day leaves that row's cells invalid for that symbol only.

    ``demean_by[sym]`` subtracts a constant from every valid cell of that symbol before the
    panel is built. ``run_sweep`` passes each symbol's mean intraday bar return (gap bars
    excluded) so a plain buy-and-hold drift cannot read as a session-window edge: the
    sign-flip and block-bootstrap worlds centre a column at zero, so any unconditional
    drift in the realised P&L survives into the null-free *real* statistic and makes it
    look extreme. Removing it first means the null tests "time-of-day structure", which is
    what the hypothesis claims. ``None`` (the default) keeps the raw P&L, and the value
    used is recorded in the artifact via ``DayPanel.demean``.
    """
    syms = list(symbols)
    n_win = len(windows)
    all_days = np.unique(np.concatenate([np.asarray(days_by_symbol[s]) for s in syms]))
    pnl = np.zeros((len(all_days), len(syms) * n_win), dtype=np.float64)
    valid = np.zeros(pnl.shape, dtype=bool)
    demean = dict(demean_by or {})
    for i, sym in enumerate(syms):
        d = np.asarray(days_by_symbol[sym])
        p = np.asarray(pnl_by_symbol[sym], dtype=np.float64)
        if p.shape != (len(d), n_win):
            raise ValueError(f"{sym}: pnl shape {p.shape} != {(len(d), n_win)}")
        rows = np.searchsorted(all_days, d)
        if not np.array_equal(all_days[rows], d):
            raise ValueError(f"{sym}: day axis does not lie on the shared day grid")
        ok = np.isfinite(p)
        if not ok.any():
            continue
        p = np.where(ok, p - float(demean.get(sym, 0.0)), np.nan)
        sl = slice(i * n_win, (i + 1) * n_win)
        pnl[rows, sl] = np.where(ok, p, 0.0)
        valid[rows, sl] = ok
    return DayPanel(all_days, tuple(syms), tuple(windows), pnl, valid, tuple(sorted(demean.items())))


def t_of_moments(total: np.ndarray, sumsq: np.ndarray, n: np.ndarray) -> np.ndarray:
    """t = mean/(sd/sqrt n) from (sum x, sum x**2, n) — the formula in ``window_stats``.

    Mirrors ``window_sweep.window_stats`` step for step: sd around the sample mean with
    ``ddof=1``, variance clipped at 0, NaN for degenerate columns (``n < 2`` or zero
    spread) exactly where that function gives NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean = total / n
        var = (sumsq - total * total / n) / (n - 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        return mean / (sd / np.sqrt(n))


def real_moments(panel: DayPanel, centre: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, t, n) per candidate, from the panel the null worlds are built from.

    With ``studentized`` worlds this is the same function as ``window_stats`` on the
    bar-level P&L (the measured deviation is recorded in ``null_info``), so real and null
    live on one scale by construction.

    ``centre=True`` subtracts each column's own mean first, and is **required** whenever
    the hypothesis is "time-of-day structure" rather than "the window has positive
    expected value": an unconditional drift (a trending price) puts a nonzero mean into
    every window's P&L, and a sign-flip world built from that same panel reproduces it,
    so the real t looks extreme against the null and pure drift reads as a session-window
    edge (measured: 10/10 false family rejections at 10bp/day). Centring the column makes
    the real statistic and the null worlds one function of one centred panel again. Callers
    that centre must use ``stat="demeaned"`` for the null too, or the two disagree.
    """
    n = panel.n_valid.astype(np.float64)
    p = panel.pnl
    if centre:
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(panel.valid, p - p.sum(axis=0) / n, 0.0)
    total = p.sum(axis=0)
    sumsq = np.einsum("ij,ij->j", p, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / n
    return mean, t_of_moments(total, sumsq, n), n


def family_max(t: np.ndarray) -> np.ndarray:
    """Per-world max |t| over candidates, NaN-safe — ``run_sweep``'s family statistic."""
    return np.nanmax(np.abs(t), axis=1)


@dataclass
class NullRun:
    """One batch of null worlds: the (worlds x candidates) t matrix plus where it ran."""

    t: np.ndarray  # (n_worlds, n_candidates) float64
    kind: str
    backend: str
    n_worlds: int
    n_days: int
    n_candidates: int
    chunk: int
    dtype: str
    seconds: float
    gemm_flops: int
    peak_bytes: int | None

    def info(self) -> dict:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "worlds": self.n_worlds,
            "n_days": self.n_days,
            "candidates": self.n_candidates,
            "chunk": self.chunk,
            "dtype": self.dtype,
            "seconds": round(self.seconds, 4),
            "gemm_flops": int(self.gemm_flops),
            "gemm_gflop_per_s": round(self.gemm_flops / 1e9 / max(self.seconds, 1e-9), 3),
            "peak_bytes": self.peak_bytes,
        }


def _clamp_chunk(chunk: int, n_candidates: int, n_matrices: int, max_bytes: int) -> int:
    """Cap the chunk so one chunk's transient tiles stay under ``max_bytes``.

    Per world the driver holds one (chunk, K) product per backend matrix plus ~3 more
    tiles (numerator, denominator, result) at 4 bytes per element.
    """
    per_world = (n_matrices + 3) * 4 * int(n_candidates)
    return int(max(1, min(chunk, max_bytes // max(per_world, 1))))


class _NumpyBackend:
    """float32 (or float64) BLAS GEMMs on the host; the drop-in when torch is absent."""

    name = "numpy-blas"

    def __init__(self, mats: list[np.ndarray], dtype):
        self.dtype = np.dtype(dtype)
        self._mats = [np.ascontiguousarray(m, dtype=self.dtype) for m in mats]
        self.peak_bytes: int | None = None
        self.gemm_flops = 0

    def dot(self, a: np.ndarray) -> list[np.ndarray]:
        a = np.ascontiguousarray(a, dtype=self.dtype)
        self.gemm_flops += 2 * a.size * self._mats[0].shape[1]
        return [a @ m for m in self._mats]


class _TorchBackend:
    """cuBLAS (or torch-CPU) GEMMs behind the same seam."""

    def __init__(self, mats: list[np.ndarray], dtype, device: str | None = None):
        import torch

        self._torch = torch
        self.dtype = np.dtype(dtype)
        tdt = torch.float32 if self.dtype == np.float32 else torch.float64
        self._tdt = tdt
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._mats = [
            torch.as_tensor(np.ascontiguousarray(m, dtype=self.dtype), dtype=tdt, device=self.device)
            for m in mats
        ]
        self.name = f"torch-{self.device.type}"
        self.peak_bytes: int | None = None
        self.gemm_flops = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def dot(self, a: np.ndarray) -> list[np.ndarray]:
        a = np.ascontiguousarray(a, dtype=self.dtype)
        t = self._torch.as_tensor(a, dtype=self._tdt, device=self.device)
        out = [t @ m for m in self._mats]
        self.gemm_flops += 2 * a.size * self._mats[0].shape[1]
        if self.device.type == "cuda":
            self.peak_bytes = int(self._torch.cuda.max_memory_allocated(self.device))
        return [o.detach().to("cpu").numpy() for o in out]


def make_backend(mats: list[np.ndarray], dtype, backend: str = "auto"):
    """``auto`` -> torch/CUDA when available, else numpy BLAS; ``torch``/``numpy`` force."""
    if backend not in ("auto", "torch", "numpy"):
        raise ValueError(f"backend must be auto|torch|numpy, got {backend!r}")
    if backend in ("auto", "torch"):
        try:
            import torch
        except ImportError:
            if backend == "torch":
                raise
        else:
            if torch.cuda.is_available():
                return _TorchBackend(mats, dtype)
            if backend == "torch":
                return _TorchBackend(mats, dtype)
    return _NumpyBackend(mats, dtype)


def sign_flip_null(
    panel: DayPanel,
    n_worlds: int,
    rng: np.random.Generator,
    *,
    chunk: int = DEFAULT_CHUNK,
    dtype=DEFAULT_DTYPE,
    backend: str = "auto",
    stat: str = "studentized",
    flips: np.ndarray | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> NullRun:
    """``n_worlds`` day-level sign-flip worlds over ``panel`` -> per-world t statistics.

    One ±1 per day, applied to every candidate at once. ``flips`` may be supplied
    explicitly (tests, reproducible worlds); otherwise every world is drawn in one block
    *before* chunking, so the chunk size cannot change which worlds are computed.
    """
    if stat not in ("studentized", "demeaned"):
        raise ValueError(f"stat must be studentized|demeaned, got {stat!r}")
    n_days, n_cand = panel.pnl.shape
    chunk = _clamp_chunk(chunk, n_cand, 1, max_bytes)
    n = panel.n_valid.astype(np.float64)
    if flips is None:
        flips = rng.integers(0, 2, size=(n_worlds, n_days)).astype(np.float32) * 2.0 - 1.0
    flips = np.asarray(flips, dtype=np.float32)
    if flips.shape != (n_worlds, n_days):
        raise ValueError(f"flips must be {(n_worlds, n_days)}, got {flips.shape}")

    scale = None
    if stat == "studentized":
        mat = panel.pnl
        fixed = np.einsum("ij,ij->j", panel.pnl, panel.pnl)  # flip-invariant: (f**2) @ x**2
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = panel.pnl.sum(axis=0) / n
        mat = np.where(panel.valid, panel.pnl - mean, 0.0)
        c2 = np.einsum("ij,ij->j", mat, mat)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt((n - 1.0) / (n * np.maximum(c2, 0.0)))
        fixed = np.zeros(0)

    be = make_backend([mat], dtype, backend)
    out = np.empty((n_worlds, n_cand), dtype=np.float64)
    fixed_d = fixed.astype(be.dtype, copy=False)
    n_d = n.astype(be.dtype, copy=False)
    scale_d = None if scale is None else scale.astype(be.dtype, copy=False)
    t0 = time.perf_counter()
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for lo in range(0, n_worlds, chunk):
            hi = min(lo + chunk, n_worlds)
            s = be.dot(flips[lo:hi])[0]
            out[lo:hi] = t_of_moments(s, fixed_d, n_d) if scale_d is None else s * scale_d
    return NullRun(
        t=out,
        kind=f"sign-flip:{stat}",
        backend=be.name,
        n_worlds=n_worlds,
        n_days=n_days,
        n_candidates=n_cand,
        chunk=chunk,
        dtype=str(np.dtype(dtype)),
        seconds=time.perf_counter() - t0,
        gemm_flops=be.gemm_flops,
        peak_bytes=be.peak_bytes,
    )


def block_bootstrap_null(
    panel: DayPanel,
    n_worlds: int,
    rng: np.random.Generator,
    *,
    mean_block: float = DEFAULT_BOOTSTRAP_DAYS,
    chunk: int = DEFAULT_CHUNK,
    dtype=DEFAULT_DTYPE,
    backend: str = "auto",
    index: np.ndarray | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> NullRun:
    """Reference null: whole day *blocks* resampled with replacement, cross-section intact.

    Columns are centred first, so the world distribution is a null for the *mean* (without
    it every world would carry the realised edge and nothing could reject). Blocks keep
    serial dependence shorter than the block scale — the volatility clustering the sign
    flip must assume away — inside the null. Weights are day counts ``A[w, i]``, so
    ``A @ X``, ``A @ X**2`` and ``A @ valid`` give the resampled first moment, second
    moment and day count, and the same ``t_of_moments`` closes it out.
    """
    n_days, n_cand = panel.pnl.shape
    chunk = _clamp_chunk(chunk, n_cand, 3, max_bytes)
    n = panel.n_valid.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = panel.pnl.sum(axis=0) / n
    centred = np.where(panel.valid, panel.pnl - mean, 0.0)
    if index is None:
        index = np.empty((n_worlds, n_days), dtype=np.int64)
        for w in range(n_worlds):
            index[w] = stationary_bootstrap_indices(n_days, mean_block, rng)
    index = np.asarray(index, dtype=np.int64)
    if index.shape != (n_worlds, n_days):
        raise ValueError(f"index must be {(n_worlds, n_days)}, got {index.shape}")
    counts = np.zeros((n_worlds, n_days), dtype=np.float32)
    for w in range(n_worlds):
        counts[w] = np.bincount(index[w], minlength=n_days)

    be = make_backend([centred, centred * centred, panel.valid.astype(np.float64)], dtype, backend)
    out = np.empty((n_worlds, n_cand), dtype=np.float64)
    t0 = time.perf_counter()
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for lo in range(0, n_worlds, chunk):
            hi = min(lo + chunk, n_worlds)
            s, q, nn = be.dot(counts[lo:hi])
            out[lo:hi] = t_of_moments(s, q, nn)
    return NullRun(
        t=out,
        kind=f"block-bootstrap:{mean_block:g}d",
        backend=be.name,
        n_worlds=n_worlds,
        n_days=n_days,
        n_candidates=n_cand,
        chunk=chunk,
        dtype=str(np.dtype(dtype)),
        seconds=time.perf_counter() - t0,
        gemm_flops=be.gemm_flops,
        peak_bytes=be.peak_bytes,
    )
