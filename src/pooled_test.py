"""Pre-registered pooled test: dollar-strength drift at fixed session windows.

Motivation: the first per-symbol sweep showed a coherent-looking cluster at 03:00 UTC —
all eight symbols aligned in the dollar-strength direction — but that alignment was read
off the results, so it is not evidence. This module is the pre-registered follow-up:

* one orientation, fixed here: dollar up = USDX long, the other seven short;
* one pooled statistic per window: equal-risk oriented basket t-stat across the eight
  symbols on the common 5-min grid;
* family null over all 69 windows from *joint* row permutations of the aligned panel —
  the same time index applied to every column, so cross-symbol correlation is preserved
  (a pooled statistic's null depends on it).

Verdict rule fixed before running: the calibrated number is the family percentile of the
candidate window (03:00 +1h) among the 69 pooled windows over 200 permutation worlds.

Usage::

    python -m src.pooled_test [--worlds 200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, costs
from .registry import Registry
from .surrogates import stationary_bootstrap_indices
from .window_sweep import default_windows, segments_from_times

ORIENTATION = {
    "USDX": 1.0,
    "EURUSD": -1.0,
    "GBPUSD": -1.0,
    "XAUUSD": -1.0,
    "XAGUSD": -1.0,
    "XTIUSD": -1.0,
    "US500": -1.0,
    "NAS100": -1.0,
}
CANDIDATE = (3, 1)


def build_panel(symbols: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, float]]:
    """Common 5-min grid across symbols: (times, returns (T,K), last prices)."""
    closes = {}
    for sym in symbols:
        df = pd.read_parquet(config.PARQUET_DIR / f"{sym}_M1.parquet").set_index("ts")
        closes[sym] = df.resample("5min").last().dropna()["close"]
    wide = pd.concat(closes, axis=1).dropna()
    r = np.log(wide).diff().to_numpy()[1:]
    prices = {sym: float(wide[sym].iloc[-1]) for sym in wide.columns}
    return wide.index[1:], r, prices


def _pooled_t(P: np.ndarray, signs: np.ndarray) -> float:
    """Equal-risk oriented basket t-stat for a (D, K) per-day P&L matrix."""
    sd = P.std(axis=0, ddof=1)
    if np.any(sd == 0.0) or len(P) < 3:
        return float("nan")
    z = (P / sd) @ signs
    return float(z.mean() / (z.std(ddof=1) / np.sqrt(len(z))))


def _all_windows(S: np.ndarray, a, b, valid, signs, windows) -> np.ndarray:
    out = np.empty(len(windows))
    for w in range(len(windows)):
        m = valid[:, w]
        P = S[b[m, w], :] - S[a[m, w], :]
        out[w] = _pooled_t(P, signs)
    return out


def run_pooled(
    times: pd.DatetimeIndex,
    R: np.ndarray,
    signs: np.ndarray,
    windows: list[tuple[int, int]],
    n_worlds: int,
    seed: int = config.SEED,
    candidate: tuple[int, int] = CANDIDATE,
) -> dict:
    ns = times.to_numpy().astype("datetime64[ns]").astype("int64")
    a, b, valid = segments_from_times(ns, ns // (86_400 * 10**9), windows)
    K = R.shape[1]

    S0 = np.vstack([np.zeros((1, K)), np.cumsum(R, axis=0)])
    real = _all_windows(S0, a, b, valid, signs, windows)

    rng = np.random.default_rng(seed)
    null = np.empty((n_worlds, len(windows)))
    T = R.shape[0]
    for i in range(n_worlds):
        Rstar = R[stationary_bootstrap_indices(T, 1.0, rng)]
        Sstar = np.vstack([np.zeros((1, K)), np.cumsum(Rstar, axis=0)])
        null[i] = _all_windows(Sstar, a, b, valid, signs, windows)

    abs_null = np.abs(null)
    null_max = np.nanmax(abs_null, axis=1)
    real_max = float(np.abs(real).max())
    family_percentile = float((null_max < real_max).mean())

    ci = windows.index(candidate)
    candidate_t = float(real[ci])
    candidate_percentile = float((abs_null[:, ci] < abs(real[ci])).mean())
    window_own_percentiles = (abs_null < np.abs(real)[None, :]).mean(axis=0)

    # Per-symbol detail at the candidate window (equal notional per leg, for reporting)
    m = valid[:, ci]
    P = S0[b[m, ci], :] - S0[a[m, ci], :]
    days = np.unique(ns // (86_400 * 10**9))
    day_ids = days[m]
    cut = np.median(day_ids)
    symbols = None  # filled by caller via result["per_symbol"] alignment
    per_symbol = {
        "edge_bp": (P.mean(axis=0) * 1e4).tolist(),
        "t": [
            float(P[:, i].mean() / (P[:, i].std(ddof=1) / np.sqrt(len(P))))
            for i in range(K)
        ],
        "t_first_half": [
            float(
                P[day_ids <= cut, i].mean()
                / (P[day_ids <= cut, i].std(ddof=1) / np.sqrt((day_ids <= cut).sum()))
            )
            for i in range(K)
        ],
        "t_second_half": [
            float(
                P[day_ids > cut, i].mean()
                / (P[day_ids > cut, i].std(ddof=1) / np.sqrt((day_ids > cut).sum()))
            )
            for i in range(K)
        ],
        "n_days": [int(m.sum())] * K,
    }

    return {
        "config": {"worlds": n_worlds, "seed": seed, "windows": len(windows), "bars": int(T)},
        "real_t": real.tolist(),
        "null": {
            "window_t_mean": null.mean(axis=0).tolist(),
            "family_max_mean": float(null_max.mean()),
            "family_max_p95": float(np.percentile(null_max, 95)),
            "family_max_max": float(null_max.max()),
        },
        "family": {"real_max_abs_t": real_max, "percentile": family_percentile},
        "window_own_percentiles": window_own_percentiles.tolist(),
        "candidate": {
            "window": list(candidate),
            "t": candidate_t,
            "own_null_percentile": candidate_percentile,
        },
        "per_symbol": per_symbol,
        "windows": [list(w) for w in windows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "pooled_dollar.json")
    args = ap.parse_args()

    config.ensure_dirs()
    symbols = [s for s in config.SYMBOLS if s in ORIENTATION]
    windows = default_windows()

    # Register every window hypothesis before evaluation.
    reg = Registry(config.REGISTRY_PATH)
    for h, d in windows:
        reg.register(
            {
                "family": "dollar_pooled_v1",
                "entry_hour_utc": h,
                "hold_hours": d,
                "orientation": "dollar_strength",
            },
            generator="pooled_test",
            generator_version="1",
        )

    times, R, prices = build_panel(symbols)
    signs = np.array([ORIENTATION[s] for s in symbols])
    result = run_pooled(times, R, signs, windows, args.worlds)

    for j, (h, d) in enumerate(windows):
        hid = reg.register(
            {
                "family": "dollar_pooled_v1",
                "entry_hour_utc": h,
                "hold_hours": d,
                "orientation": "dollar_strength",
            },
            generator="pooled_test",
            generator_version="1",
        )
        reg.record_evaluation(
            hid,
            {
                "t_pooled": round(float(result["real_t"][j]), 3),
                "own_null_percentile": round(float(result["window_own_percentiles"][j]), 4),
            },
            {"null": {"mean_block": 1.0, "worlds": args.worlds, "joint": True}},
        )

    result["symbols"] = symbols
    result["candidate"]["costs"] = {
        s: costs.commission_round_trip_bp(s, prices[s]) for s in symbols
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    h, d = CANDIDATE
    print(f"panel: {result['config']['bars']:,} bars x {len(symbols)} symbols, "
          f"{args.worlds} joint-permutation worlds, {len(windows)} windows")
    print(f"\nfamily (all 69 windows, joint null): max |t_pooled| = {result['family']['real_max_abs_t']:.2f}"
          f" -> {100 * result['family']['percentile']:.1f}th percentile"
          f" (null mean {result['null']['family_max_mean']:.2f}, p95 {result['null']['family_max_p95']:.2f})")
    print(f"candidate {h:02d}:00 +{d}h: t_pooled = {result['candidate']['t']:.2f}"
          f"  (own-null percentile {100 * result['candidate']['own_null_percentile']:.1f}%)")

    print(f"\n{'symbol':>7} {'edge bp':>9} {'t':>7} {'t1':>7} {'t2':>7} {'comm rt bp':>11} {'BE spread':>10}")
    ps = result["per_symbol"]
    for i, sym in enumerate(symbols):
        comm = costs.commission_round_trip_bp(sym, prices[sym])
        comm_s = "n/a" if comm is None else f"{comm:.3f}"
        be = abs(ps["edge_bp"][i]) if comm is None else abs(ps["edge_bp"][i]) - comm
        print(f"{sym:>7} {ps['edge_bp'][i]:>9.4f} {ps['t'][i]:>7.2f} {ps['t_first_half'][i]:>7.2f}"
              f" {ps['t_second_half'][i]:>7.2f} {comm_s:>11} {be:>10.3f}")

    order = np.argsort(-np.abs(result["real_t"]))[:8]
    print("\ntop windows by |t_pooled| (family null reference above):")
    for j in order:
        hj, dj = windows[j]
        print(f"  {hj:02d}:00 +{dj}h  t={result['real_t'][j]:+7.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
