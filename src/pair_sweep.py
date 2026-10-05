"""Cross-asset dislocation reversion: fixed pair rules, pre-registered.

Six pairs with mechanism stories, fixed before running:

    US500/NAS100   sibling equity indices
    XAUUSD/XAGUSD  precious metals
    EURUSD/GBPUSD  European majors
    XAUUSD/USDX    gold vs the dollar index
    XTIUSD/USDX    oil vs the dollar index
    EURUSD/USDX    euro vs the dollar index

Rule: log-spread s_t = logA_t - beta_t * logB_t, beta_t from a rolling 2-day OLS of the
levels (no look-ahead: window ends at t). z_t = rolling 5-day z-score of s. Enter when
|z| >= z_entry (grid {2.0, 3.0}), trade the reversion direction, exit when |z| <= z_exit
(grid {0.0, 0.5}) or after max_hold = 1 day. One position per pair at a time.

Family: 6 pairs x 2 x 2 = 24 candidates. Statistic: |t| of the mean oriented trade P&L
(in bp of the A-leg). Null: joint row permutation of the pair's returns (mean_block=1) —
contemporaneous co-movement and marginals preserved, all serial structure (including any
reversion) destroyed. Family max calibrated over 200 worlds. Costs from src/costs.py,
B-leg commission scaled by |beta| (reported); spread adopted as 0 with break-even
reported.

Usage::

    python -m src.pair_sweep [--worlds 200]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, costs
from .registry import Registry
from .surrogates import stationary_bootstrap_indices

PAIRS = [
    ("US500", "NAS100"),
    ("XAUUSD", "XAGUSD"),
    ("EURUSD", "GBPUSD"),
    ("XAUUSD", "USDX"),
    ("XTIUSD", "USDX"),
    ("EURUSD", "USDX"),
]
Z_ENTRY = [2.0, 3.0]
Z_EXIT = [0.0, 0.5]
BETA_WINDOW = 576  # 2 days of 5-min bars
Z_WINDOW = 1440  # 5 days
MAX_HOLD = 288  # 1 day


@dataclass
class PairData:
    a: str
    b: str
    times: pd.DatetimeIndex
    la: np.ndarray  # log price A on the common grid
    lb: np.ndarray
    price_a: float
    price_b: float

    @property
    def name(self) -> str:
        return f"{self.a}/{self.b}"


def build_pair(a: str, b: str) -> PairData:
    closes = {}
    for sym in (a, b):
        df = pd.read_parquet(config.PARQUET_DIR / f"{sym}_M1.parquet").set_index("ts")
        closes[sym] = df.resample("5min").last().dropna()["close"]
    wide = pd.concat(closes, axis=1).dropna()
    return PairData(a, b, wide.index, np.log(wide[a].to_numpy()), np.log(wide[b].to_numpy()),
                    float(wide[a].iloc[-1]), float(wide[b].iloc[-1]))


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return out


def rolling_beta_z(la: np.ndarray, lb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """beta_t (levels OLS, trailing window ending at t) and z_t (spread z-score)."""
    mx = _rolling_mean(la, BETA_WINDOW)
    my = _rolling_mean(lb, BETA_WINDOW)
    mxy = _rolling_mean(la * lb, BETA_WINDOW)
    myy = _rolling_mean(lb * lb, BETA_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (mxy - mx * my) / np.maximum(myy - my * my, 0.0)
    # cumsum-based rolling stats cannot see leading NaNs (they propagate through the
    # whole cumulative sum), so the spread z-score is computed on the tail where beta
    # is defined, then padded back to full length.
    i0 = BETA_WINDOW - 1
    s = la[i0:] - beta[i0:] * lb[i0:]
    ms = _rolling_mean(s, Z_WINDOW)
    ms2 = _rolling_mean(s * s, Z_WINDOW)
    sd = np.sqrt(np.maximum(ms2 - ms * ms, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_tail = np.where(sd > 0, (s - ms) / sd, np.nan)
    z = np.full(len(la), np.nan)
    z[i0:] = z_tail
    return beta, z


def simulate_pair(la: np.ndarray, lb: np.ndarray, beta: np.ndarray, z: np.ndarray,
                  z_entry: float, z_exit: float) -> dict:
    """Sequential trade simulation. Returns entry/exit indices, oriented P&L, entry betas."""
    t = len(la)
    entries = np.flatnonzero(np.abs(z) >= z_entry)
    exits = np.flatnonzero(np.abs(z) <= z_exit)
    ent: list[int] = []
    ext: list[int] = []
    pnl: list[float] = []
    bet: list[float] = []
    pos = 0
    while pos < len(entries):
        e = int(entries[pos])
        j = np.searchsorted(exits, e, side="right")
        x = e + MAX_HOLD
        if j < len(exits):
            x = min(x, int(exits[j]))
        x = min(x, t - 1)
        if x > e and np.isfinite(beta[e]):
            ds = (la[x] - la[e]) - beta[e] * (lb[x] - lb[e])
            ent.append(e)
            ext.append(x)
            pnl.append(float(-np.sign(z[e]) * ds))
            bet.append(float(beta[e]))
        pos = int(np.searchsorted(entries, x, side="right"))
    return {
        "entry": np.asarray(ent, dtype=np.int64),
        "exit": np.asarray(ext, dtype=np.int64),
        "pnl": np.asarray(pnl, dtype=float),
        "beta": np.asarray(bet, dtype=float),
    }


def _tstat(pnl: np.ndarray) -> float:
    if len(pnl) < 5:
        return float("nan")
    sd = pnl.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(pnl.mean() / (sd / np.sqrt(len(pnl))))


def run_pair_sweep(pairs: list[PairData], n_worlds: int, seed: int = config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    combos = [(z0, zx) for z0 in Z_ENTRY for zx in Z_EXIT]

    real: dict[tuple, dict] = {}
    for p in pairs:
        beta, z = rolling_beta_z(p.la, p.lb)
        for z0, zx in combos:
            tr = simulate_pair(p.la, p.lb, beta, z, z0, zx)
            t = _tstat(tr["pnl"])
            cost_bp = None
            if costs.known_symbol(p.a) and costs.known_symbol(p.b):
                ca = costs.commission_round_trip_bp(p.a, p.price_a)
                cb = costs.commission_round_trip_bp(p.b, p.price_b)
                if ca is not None and cb is not None and len(tr["beta"]):
                    cost_bp = float(np.mean(ca + np.abs(tr["beta"]) * cb))
            edge_bp = float(tr["pnl"].mean() * 1e4) if len(tr["pnl"]) else float("nan")
            half = (
                [_tstat(tr["pnl"][tr["entry"] <= np.median(tr["entry"])]),
                 _tstat(tr["pnl"][tr["entry"] > np.median(tr["entry"])])]
                if len(tr["pnl"])
                else [float("nan"), float("nan")]
            )
            real[(p.name, z0, zx)] = {
                "pair": p.name,
                "z_entry": z0,
                "z_exit": zx,
                "n_trades": int(len(tr["pnl"])),
                "edge_bp": edge_bp,
                "t": t,
                "avg_beta": float(np.mean(np.abs(tr["beta"]))) if len(tr["beta"]) else float("nan"),
                "avg_hold_bars": float(np.mean(tr["exit"] - tr["entry"])) if len(tr["pnl"]) else float("nan"),
                "cost_bp": cost_bp,
                "break_even_spread_bp": None if (cost_bp is None or not np.isfinite(edge_bp))
                else abs(edge_bp) - cost_bp,
                "t_first_half": half[0],
                "t_second_half": half[1],
            }

    ra = {p.name: np.diff(p.la) for p in pairs}
    rb = {p.name: np.diff(p.lb) for p in pairs}
    null = np.empty((n_worlds, len(pairs) * len(combos)))
    for w in range(n_worlds):
        col = 0
        for p in pairs:
            idx = stationary_bootstrap_indices(len(ra[p.name]), 1.0, rng)
            la_s = np.concatenate([[0.0], np.cumsum(ra[p.name][idx])])
            lb_s = np.concatenate([[0.0], np.cumsum(rb[p.name][idx])])
            beta_s, z_s = rolling_beta_z(la_s, lb_s)
            for z0, zx in combos:
                tr = simulate_pair(la_s, lb_s, beta_s, z_s, z0, zx)
                null[w, col] = _tstat(tr["pnl"])
                col += 1
    abs_null = np.abs(null)
    null_max = abs_null.max(axis=1)

    keys = [(p.name, z0, zx) for p in pairs for (z0, zx) in combos]
    for j, key in enumerate(keys):
        real[key]["own_null_percentile"] = float((abs_null[:, j] < abs(real[key]["t"])).mean())
    rows = sorted(real.values(), key=lambda r: -abs(r["t"]) if np.isfinite(r["t"]) else 0.0)

    real_max = float(max(abs(r["t"]) for r in rows if np.isfinite(r["t"])))
    return {
        "config": {"worlds": n_worlds, "seed": seed, "pairs": [p.name for p in pairs],
                   "candidates": len(pairs) * len(combos), "beta_window": BETA_WINDOW,
                   "z_window": Z_WINDOW, "max_hold": MAX_HOLD},
        "null": {"family_max_mean": float(null_max.mean()),
                 "family_max_p95": float(np.percentile(null_max, 95)),
                 "family_max_max": float(null_max.max())},
        "family": {"real_max_abs_t": real_max, "percentile": float((null_max < real_max).mean())},
        "candidates": rows,
        "pair_stats": {
            p.name: {"n_bars": int(len(p.la)), "first": str(p.times[0]), "last": str(p.times[-1])}
            for p in pairs
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "pair_sweep.json")
    args = ap.parse_args()

    config.ensure_dirs()
    pairs = [build_pair(a, b) for a, b in PAIRS]

    reg = Registry(config.REGISTRY_PATH)
    for p in pairs:
        for z0 in Z_ENTRY:
            for zx in Z_EXIT:
                reg.register(
                    {"family": "pair_reversion_v1", "pair": [p.a, p.b], "z_entry": z0, "z_exit": zx},
                    generator="pair_sweep",
                    generator_version="1",
                )

    result = run_pair_sweep(pairs, args.worlds)
    for row in result["candidates"]:
        a, b = row["pair"].split("/")
        hid = reg.register(
            {"family": "pair_reversion_v1", "pair": [a, b], "z_entry": row["z_entry"], "z_exit": row["z_exit"]},
            generator="pair_sweep",
            generator_version="1",
        )
        reg.record_evaluation(
            hid,
            {
                "n_trades": row["n_trades"],
                "edge_bp": None if not np.isfinite(row["edge_bp"]) else round(row["edge_bp"], 5),
                "t": None if not np.isfinite(row["t"]) else round(row["t"], 3),
                "break_even_spread_bp": None if row["break_even_spread_bp"] is None else round(row["break_even_spread_bp"], 5),
                "own_null_percentile": round(row["own_null_percentile"], 4),
            },
            {"null": {"mean_block": 1.0, "worlds": args.worlds, "joint_pair": True}, "spread_bp": 0.0},
        )

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    c = result["config"]
    print(f"pairs {len(pairs)} x z_entry {Z_ENTRY} x z_exit {Z_EXIT} = {c['candidates']} candidates, "
          f"{c['worlds']} joint-permutation worlds")
    print(f"null family max |t|: mean {result['null']['family_max_mean']:.2f} | p95 {result['null']['family_max_p95']:.2f}"
          f" | max {result['null']['family_max_max']:.2f}")
    print(f"real family max |t| = {result['family']['real_max_abs_t']:.2f}"
          f" -> percentile {100 * result['family']['percentile']:.1f}%\n")

    hdr = (f"{'pair':>15} {'z0':>4} {'zx':>4} {'n':>5} {'edge bp':>8} {'t':>7} {'avg|b|':>7}"
           f" {'hold':>5} {'BE spr':>8} {'t1':>6} {'t2':>6} {'own%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for row in result["candidates"]:
        be = "n/a" if row["break_even_spread_bp"] is None else f"{row['break_even_spread_bp']:.3f}"
        print(f"{row['pair']:>15} {row['z_entry']:>4.1f} {row['z_exit']:>4.1f} {row['n_trades']:>5}"
              f" {row['edge_bp']:>8.3f} {row['t']:>7.2f} {row['avg_beta']:>7.2f} {row['avg_hold_bars']:>5.0f}"
              f" {be:>8} {row['t_first_half']:>6.2f} {row['t_second_half']:>6.2f} {row['own_null_percentile']:>6.3f}")

    print("\nper-pair panel:")
    for p in pairs:
        print(f"  {p.name:>15}  bars {len(p.la):,}  {p.times[0]:%Y-%m-%d} -> {p.times[-1]:%Y-%m-%d}"
              f"  last A {p.price_a:.4f}  last B {p.price_b:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
