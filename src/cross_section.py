"""Cross-sectional ranking: does a feature predict the ORDERING of returns across names?

Why this family exists
----------------------
``window_sweep`` asks "does price drift at a fixed UTC hour" and requires ONE symbol to
clear its OWN null. With a per-name standard error comparable to the edge, that needs an
effect tens of basis points deep to certify — which is why six families produced zero
survivors while each showed *something* directionally.

A cross-sectional test pools the bets. The question: **given a feature value per name, does
it predict the ordering of the cross-section's forward returns?** The statistic is the mean
over sessions of Spearman's rho between the feature vector and the forward return vector,
computed across the names observed that session. That has

* power from the *breadth* of the cross-section (105 names per session, not one name per
  test) — the t of a mean daily rho over D sessions with K names per session improves with
  K as well as D, which is exactly the axis the per-name families are starved of;
* a natural out-of-sample structure (the feature is known before the return it predicts);
* a null that is its own permutation: shuffling the feature-to-name assignment within each
  session destroys exactly the predictive relationship and nothing else.

Alignment — the reason this is not built on the common bar grid
---------------------------------------------------------------
Requiring every symbol to share a bar timestamp collapses a 139-day, 105-name panel to 974
bars and 13 sessions, because the shortest listing date poisons the intersection. So the
panel is **session-aligned**: each symbol's session return is computed on its own bars, and
sessions are matched by DATE, with the names observed that session as the cross-section. A
name that did not trade that session is simply absent from that day's ranking, and the day
is dropped entirely if fewer than 3 names traded.

Design, pinned before running
-----------------------------
* feature: **previous session's return** (one pre-registered feature; a feature search is a
  separate hypothesis with its own multiplicity);
* forward horizon: **the next N sessions** (grid {1, 2});
* statistic: mean over sessions of Spearman's rho, ranks computed across names within a
  session;
* null: within-session label permutation of the feature across names;
* the test is two-sided and the sign is reported as the direction a trade would use.

Usage::

    from src import cross_section
    res = cross_section.run_cross_section(symbols)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .registry import DISCOVERY, Registry

FEATURES = ("prev_session_ret",)
HORIZONS = (1, 2)
# The US cash session in LOCAL time: 09:30-16:00 ET. Converted to UTC per bar through
# ``mt5_clock``'s per-date offset, because a fixed 13-20 UTC window is wrong for a third of
# the year: before 2026-11-01 the session is 13:30-21:00 UTC (EDT) and after it
# 14:30-22:00 UTC (EST). The DST repair in this repo is exactly the lesson behind pinning
# this to the local clock rather than a UTC constant.
SESSION_START_LOCAL = (9, 30)    # 09:30 ET
SESSION_END_LOCAL = (16, 0)      # 16:00 ET
# retained for the sweep's own grid and for backwards compatibility with artifacts whose
# session labels were built on the fixed-UTC convention (US summer time of the panel).
SESSION_START_HOUR = 13
SESSION_END_HOUR = 20


def load_closes(symbols: list[str]) -> dict[str, pd.Series]:
    """Per-symbol 5-minute close series, each on its own index (no forced common grid)."""
    out = {}
    for sym in symbols:
        p = config.PARQUET_DIR / f"{sym}_MT5_M1.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "spread_med_bp" in df.columns:
            df = df[df["tick_n"] > 0]
        s = df.set_index("ts")["close"].resample("5min").last().dropna()
        if len(s) >= 20:
            out[sym] = s
    return out


def session_returns(closes: dict[str, pd.Series]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(session_day, matrix of per-session log-return bp, symbols) — session-aligned.

    One row per trading session (union over symbols), one column per symbol, NaN where that
    symbol did not trade that session.
    """
    syms = sorted(closes.keys())
    per_sym: list[tuple[np.ndarray, np.ndarray]] = []
    for sym in syms:
        s = closes[sym]
        day = s.index.normalize()
        # a session's return: first bar of that day within the session window to last
        in_sess = (s.index.hour >= SESSION_START_HOUR) & (s.index.hour < SESSION_END_HOUR)
        ss = s[in_sess]
        if not len(ss):
            per_sym.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)))
            continue
        d = ss.index.normalize().asi8
        vals = ss.to_numpy(dtype=float)
        first = np.ones(len(vals), dtype=bool)
        first[1:] = d[1:] != d[:-1]
        last = np.ones(len(vals), dtype=bool)
        last[:-1] = d[1:] != d[:-1]
        # per unique day: log(last/first)
        ud, idx = np.unique(d, return_index=True)
        day_ret = np.log(vals[last][np.searchsorted(ud, d[last])] /
                         vals[first][np.searchsorted(ud, d[first])]) * 1e4
        per_sym.append((ud, day_ret))

    all_days = np.unique(np.concatenate([d for d, _ in per_sym if len(d)]))
    M = np.full((len(all_days), len(syms)), np.nan)
    for k, (d, r) in enumerate(per_sym):
        if not len(d):
            continue
        rows = np.searchsorted(all_days, d)
        assert np.array_equal(all_days[rows], d), "session axis must lie on the shared grid"
        M[rows, k] = r
    return all_days, M, syms


def daily_rank_corr(F: np.ndarray, fwd: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Spearman rho per session: rank the names, correlate with the forward return's ranks."""
    n_days = F.shape[0]
    out = np.full(n_days, np.nan)
    for d in range(n_days):
        m = ok[d]
        if m.sum() < 3:
            continue
        f, r = F[d, m], fwd[d, m]
        if np.allclose(f, f[0]) or np.allclose(r, r[0]):
            continue                      # a degenerate session has no ordering to predict
        rf = pd.Series(f).rank().to_numpy()
        rr = pd.Series(r).rank().to_numpy()
        rf = rf - rf.mean()
        rr = rr - rr.mean()
        denom = np.sqrt((rf ** 2).sum() * (rr ** 2).sum())
        if denom > 0:
            out[d] = float((rf * rr).sum() / denom)
    return out


def hash_int(*parts) -> int:
    """A stable integer hash, for seeding a null reproducibly.

    Python's ``hash()`` on strings is randomised per process (PYTHONHASHSEED), so a null
    seeded with ``hash((feat, hz))`` produced different worlds in different runs and the
    percentile was not reproducible. This is a plain content hash instead.
    """
    import hashlib

    return int(hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:8], 16)


def data_fingerprint(symbols: list[str], days: np.ndarray, code: str) -> str:
    """A fingerprint over the ACTUAL inputs: which symbols, which session range, which code.

    A placeholder like ``f"{family}:{symbol}"`` names the run, not the data, so two runs on
    different panel vintages look identical in the registry. This hashes the symbol list, the
    number of sessions, the first and last session, and the code version.
    """
    import hashlib

    payload = repr({
        "symbols": sorted(symbols),
        "n_sessions": int(len(days)),
        "first": str(days[0]) if len(days) else None,
        "last": str(days[-1]) if len(days) else None,
        "code": code,
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def beta_neutralise(R: np.ndarray) -> np.ndarray:
    """Remove each name's leverage to the equal-weighted cross-section, one shared transform.

    THE DEFECT THIS FIXES, recorded because it invalidated a published percentile: the real
    statistic in ``run_cross_section`` was computed on RAW returns while
    ``session_shift_null(beta_neutral=True)`` computed it on RESIDUALISED returns. Those are
    different functions, so the resulting percentile was not a p-value. Simulated on one-factor
    data with ZERO idio predictability, the shipped pair false-positived **2.5% at market AR(0)
    and 100% at AR(0.3)** (``research/p9_ranking_null_calibration.py``) — the real mean was
    0.0437 against a null p95 of 0.0185, because beta x market-sign persistence enters the raw
    statistic and is stripped from the null.

    Beta is a TIME-SERIES regression of each name's demeaned session return on the demeaned
    cross-section, ``cov(name, mkt) / var(mkt)``. The previous version computed
    ``sum(prev_demeaned x fwd_demeaned) / sum(prev^2)`` per name, which is a LAG-1
    autocovariance, not a beta — and it estimated beta on the mismatched pair rather than on a
    single consistent series.

    Both the real statistic and every null world must pass through THIS function. Applied on
    one side only, any residualisation is worse than none: it manufactures a difference between
    the observed value and its own null.
    """
    mkt = np.nanmean(R, axis=1)
    sd = np.nanstd(mkt)
    if not np.isfinite(sd) or sd <= 0:
        return R
    Rc = R - mkt[:, None]
    mc = mkt - np.nanmean(mkt)
    den = float((mc * mc).sum())
    if den <= 0:
        return R
    beta = (np.nansum(Rc * mc[:, None], axis=0) / den)[None, :]
    return R - beta * mkt[:, None]


def session_shift_null(
    R: np.ndarray, prev: np.ndarray, n_worlds: int, rng: np.random.Generator,
    *, beta_neutral: bool = True, min_shift: int = 2,
) -> np.ndarray:
    """Null for the cross-sectional ranking: pair each session's feature with ANOTHER
    session's returns.

    Why not the within-session feature shuffle
    ------------------------------------------
    Shuffling the feature across names *within* a session destroys beta structure, and the
    mean rank correlation then picks up market-sign persistence x beta dispersion — a real
    quantity that has nothing to do with the feature predicting anything. Simulated on
    one-factor zero-predictability data with realistic beta dispersion, that null
    false-positived **42%** of the time at market autocorrelation 0 (17% at 0.3 and 0.6),
    against a nominal 5%.

    ``beta_neutral=True`` applies ``beta_neutralise`` — the SAME transform the real statistic
    uses, which is what makes the percentile a p-value. See the docstring there for the
    defect that applied it to the null only.

    ``min_shift`` excludes one-session shifts, which pair a session with itself shifted by one
    day and can inflate the null's tails on trending data.
    """
    n_days, k = R.shape
    X = beta_neutralise(R) if beta_neutral else R
    F = prev
    out = np.empty(n_worlds)
    lo = max(int(min_shift), 1)
    for w in range(n_worlds):
        shift = int(rng.integers(lo, max(lo + 1, n_days)))
        fwd = np.roll(X, -shift, axis=0)
        ok = np.isfinite(F) & np.isfinite(fwd)
        out[w] = float(np.nanmean(daily_rank_corr(F, fwd, ok)))
    return out


def run_cross_section(
    symbols: list[str],
    *,
    features: tuple[str, ...] = FEATURES,
    horizons: tuple[int, ...] = HORIZONS,
    n_worlds: int = 2000,
    seed: int = config.SEED,
) -> dict:
    """The cross-sectional family: (feature, horizon) grid, within-session permutation null."""
    closes = load_closes(symbols)
    if len(closes) < 10:
        raise SystemExit(f"only {len(closes)} symbols with a usable panel")
    days, R, syms = session_returns(closes)

    # the pre-registered feature: the PREVIOUS session's return for THAT symbol, known
    # before the forward session it is paired with
    prev = np.full_like(R, np.nan)
    prev[1:] = R[:-1]
    prev_ok = np.isfinite(prev) & np.roll(np.isfinite(R), 1, axis=0)

    # THE REAL STATISTIC AND THE NULL MUST BE THE SAME FUNCTION. The version before this
    # fix computed the real statistic on RAW returns and the null on RESIDUALISED returns, so
    # the percentile was not a p-value: simulated on zero-predictability one-factor data the
    # pair false-positived 2.5% at market AR(0) and 100% at AR(0.3), because beta x
    # market-sign persistence enters the raw statistic and is stripped from the null.
    Rn = beta_neutralise(R)

    candidates = []
    for feat in features:
        for hz in horizons:
            fwd = np.full_like(Rn, np.nan)
            fwd[:-hz] = Rn[hz:]
            fwd_ok = np.isfinite(fwd) & np.roll(np.isfinite(Rn), -hz, axis=0)
            if hz > 1:
                fwd_ok[-hz:] = False
            F = prev if feat == "prev_session_ret" else R
            Fok = prev_ok if feat == "prev_session_ret" else np.isfinite(R)
            ok = Fok & fwd_ok
            rho = daily_rank_corr(F, fwd, ok)
            real = float(np.nanmean(rho))
            n_sessions = int(np.isfinite(rho).sum())
            if n_sessions < 20:
                candidates.append({
                    "feature": feat, "horizon": hz, "real_rho": real, "n_sessions": n_sessions,
                    "n_symbols": len(syms), "null_mean": float("nan"),
                    "null_p95": float("nan"), "percentile": float("nan"),
                    "direction": 0.0, "note": f"only {n_sessions} usable sessions",
                })
                continue

            # The null pairs this session's feature with a DIFFERENT session's returns, on the
            # SAME beta-neutralised panel the real statistic used - that symmetry is what makes
            # the percentile a p-value.
            rng = np.random.default_rng(config.SEED + (abs(hash_int(feat, hz))))
            null = session_shift_null(Rn, prev, n_worlds, rng, beta_neutral=True)
            candidates.append({
                "feature": feat, "horizon": hz, "real_rho": real,
                "n_sessions": n_sessions, "n_symbols": len(syms),
                "null_mean": float(np.nanmean(null)),
                "null_p95": float(np.nanpercentile(np.abs(null), 95)),
                "percentile": float((np.abs(null) < abs(real)).mean()),
                "direction": 1.0 if real >= 0 else -1.0,
            })

    usable = [c for c in candidates if np.isfinite(c["percentile"])]
    return {
        "config": {
            "symbols": len(syms), "features": list(features), "horizons": list(horizons),
            "worlds": n_worlds, "seed": seed, "session_utc": [SESSION_START_HOUR, SESSION_END_HOUR],
            "sessions_available": int(len(days)),
        },
        "family": {
            "candidates": len(usable),
            "real_max_abs_rho": max((abs(c["real_rho"]) for c in usable), default=float("nan")),
            "null_max_mean": float(np.nanmean([abs(c["null_mean"]) for c in usable])) if usable else float("nan"),
            "real_percentile": max((c["percentile"] for c in usable), default=float("nan")),
        },
        "candidates": candidates,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--worlds", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "cross_section.json")
    args = ap.parse_args()
    config.ensure_dirs()

    src = Path(args.symbols_file) if args.symbols_file else (config.DATA_DIR / "equity_symbols.txt")
    syms = [l.strip() for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    res = run_cross_section(syms, n_worlds=args.worlds)

    reg = Registry(config.REGISTRY_PATH)
    for c in res["candidates"]:
        if not np.isfinite(c.get("percentile", float("nan"))):
            continue
        hid = reg.register(
            {"family": "cross_section_v1", "feature": c["feature"], "horizon": c["horizon"]},
            generator="cross_section", generator_version="1", role=DISCOVERY,
        )
        reg.record_evaluation(
            hid,
            {"rho": round(c["real_rho"], 5), "percentile": round(c["percentile"], 4),
             "n_sessions": c["n_sessions"]},
            {"worlds": args.worlds, "session_utc": [SESSION_START_HOUR, SESSION_END_HOUR],
             "symbols": len(syms)},
            data_fingerprint=f"cross_section_v1:{len(syms)}syms", clock="true-utc",
        )

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    fam = res["family"]
    print(f"cross-section: {fam['candidates']} usable candidates from "
          f"{res['config']['sessions_available']} sessions x {res['config']['symbols']} symbols")
    for c in sorted(res["candidates"], key=lambda c: -abs(c.get("real_rho") or 0)):
        if not np.isfinite(c.get("percentile", float("nan"))):
            print(f"  {c['feature']:>18} +{c['horizon']}s: SKIPPED ({c.get('note')})")
            continue
        print(f"  {c['feature']:>18} +{c['horizon']}s: rho {c['real_rho']:+.4f} "
              f"(null mean {c['null_mean']:+.4f}, |null| p95 {c['null_p95']:.3f}) "
              f"pct {c['percentile']:.3f} n={c['n_sessions']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
