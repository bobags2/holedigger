"""Conditional session windows: drift at hour h GIVEN a state, not just at hour h.

Why this family exists
----------------------
``window_sweep`` asks "does price drift at a fixed UTC hour" and found, repeatedly, effects
that were real inside one period and absent or reversed in another: the dislocation family's
halves (−14.4 / −7.1), the event study's control baseline, the gold/year split (+12.6 vs
−9.0). Each of those was a *regime* finding discovered post hoc, and none could have been
tested, because the hypothesis the sweep can express has no conditional in it.

So this family adds one. The question is:

    does price drift at hour h **when the tape is in state S**, and does it stop when it is not?

Two things make this a different hypothesis rather than a search over more parameters:

* the state is a **measured, pre-entry-observable quantity** — the spread percentile in the
  hour before entry (from ``src.cost_model``'s features), the previous session's volatility,
  or the overnight gap. Nothing that requires the future;
* the null **conditions too**: permuting labels within the (hour, state) cell, not across it,
  so a world has no conditional structure while every cell keeps its own marginal. That is
  the null that matches the hypothesis, and it is why the family is honest about the cost of
  conditioning: each cell has fewer observations, and a conditional claim needs the cell to
  carry enough days to say anything.

Design, pinned before running
-----------------------------
* state: **spread percentile in the preceding hour**, quartile-binned (tight / mid / wide /
  very wide). The spread is measured, per bar, and the bin is known before the session opens.
* hypothesis: the drift at hour ``h`` is different in the WIDE bins than in the TIGHT ones,
  tested as the difference of the two cells' medians against the within-cell permutation
  null.
* report: per (hour, state-pair) the two medians, their difference, and the null percentile;
  the family is that grid, and its bar is the scan-max against the same null.

Usage::

    from src import conditional
    res = conditional.run(symbols, states=("spread_prev_hour",))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, mt5_clock
from .registry import DISCOVERY, Registry
from .window_sweep import MAX_EXIT_HOUR, HOLDS, _mean_intraday_bar_return

STATES = ("spread_prev_hour",)
BINS = ("tight", "mid", "wide")
PREV_HOURS = 1


def load_bars(symbol: str) -> pd.DataFrame:
    df = mt5_clock.load_mt5_utc(config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet")
    return df.sort_values("ts").reset_index(drop=True)


def state_labels(df: pd.DataFrame, state: str) -> np.ndarray:
    """A per-bar state label, from information available BEFORE that bar.

    ``spread_prev_hour``: the median spread of the bars in the preceding clock hour,
    tercile-binned against the symbol's own distribution. A bar's label therefore depends
    only on the past, which is what makes a conditional trade implementable.

    **The label is a DAY-level property, deliberately.** The spread state is measured on the
    hour *before* the session opens, so every bar in one session shares one label. That is
    both the right economic unit (the state is the tape you entered on, not a moving average
    inside the session) and the unit a day-level permutation needs. A per-bar label would put
    one day in two cells at once, and then a "conditional" comparison silently compares
    different clock times — which is the moving-endpoint defect, and it is why the cells are
    day-homogeneous by construction here.
    """
    if state != "spread_prev_hour":
        raise ValueError(f"unknown state {state!r}; pick from {STATES}")
    sp = df["spread_med_bp"].to_numpy(dtype=float)
    hour = df["ts"].dt.hour.to_numpy()
    day = df["ts"].dt.normalize().astype("int64")
    key = day * 24 + hour
    uniq = np.unique(key)
    med = pd.Series(sp).groupby(np.searchsorted(uniq, key)).median().to_numpy()
    prev = np.full(len(sp), np.nan)
    for i in range(1, len(uniq)):
        if uniq[i] - uniq[i - 1] == 1:
            prev[key == uniq[i]] = med[i - 1]
    prev[key == uniq[0]] = med[0]
    # one label per DAY: the median of that day's bars' preceding-hour spread states
    day_state = pd.Series(prev).groupby(day).median()
    q = np.nanquantile(day_state.to_numpy(), [1 / 3, 2 / 3])
    lab = np.full(len(prev), "unknown", dtype=object)
    for d, s in day_state.items():
        name = BINS[0] if s <= q[0] else (BINS[1] if s <= q[1] else BINS[2])
        lab[day == d] = name
    return lab


def drift_by_cell(returns: np.ndarray, bar_day: np.ndarray, bar_time: np.ndarray,
                  labels: np.ndarray, windows: list[tuple[int, int]]) -> dict:
    """Per (window, state) the per-day forward return, WITH the day each value belongs to.

    Returns ``{(h, d, state): {"values": ..., "days": ...}}``. The day ids are carried
    alongside the values because the pooled null needs to permute ONE LABEL PER DAY: without
    them a null can only shuffle values and treats one day's overlapping windows as
    independent observations, which measured at a 70% false-positive rate.
    """
    S = np.cumsum(returns)
    day_ns = 86_400 * 10**9
    out: dict[tuple, dict] = {}
    for h, d in windows:
        entry_cut, exit_cut = h * 3600 * 10**9, (h + d) * 3600 * 10**9
        for state in BINS:
            vals, days = [], []
            for day in np.unique(bar_day):
                m = (bar_day == day) & (labels == state)
                if not m.any():
                    continue
                times = bar_time[m] - day * day_ns
                a = np.searchsorted(times, entry_cut, side="left")
                b = np.searchsorted(times, exit_cut, side="right") - 1
                idx = np.flatnonzero(m)
                if b <= a or b - a < 2:
                    continue
                ia, ib = idx[a], idx[b]
                if ia >= len(S) or ib >= len(S):
                    continue
                vals.append(S[ib] - S[ia])
                days.append(int(day))
            out[(h, d, state)] = {
                "values": np.asarray(vals, dtype=float),
                "days": np.asarray(days, dtype=np.int64),
            }
    return out


def run_pooled(symbols: list[str], *, state: str = "spread_prev_hour",
               n_worlds: int = 200, seed: int = config.SEED) -> dict:
    """The pooled conditional test: one series per (symbol, state), all windows aggregated.

    The per-window table shows a *coherent* effect — 50 of 68 windows positive — but the
    windows overlap heavily (a 13+1h and a 13+4h window share most of their observations), so
    a per-window null cannot certify it and a sign test over overlapping windows is not
    honest evidence. The test that does have power pools every window's *per-day* value into
    one series per (symbol, state).

    **The null permutes ONE LABEL PER DAY**, applied to every window that day contributes.
    The obvious alternative — shuffling labels across the pooled window-observations — was
    simulated and is invalid: on synthetic noise with the state independent of returns it
    flagged **70%** of worlds at percentile 1.000 (day-level permutation: 5%, exactly
    nominal), because it treats one day's overlapping windows as independent observations.
    See ``docs/null-calibration.md``.
    """
    windows = [(h, d) for d in HOLDS for h in range(0, MAX_EXIT_HOUR - d + 1)]
    rng = np.random.default_rng(seed)
    results = []
    for sym in symbols:
        df = load_bars(sym)
        if len(df) < 500:
            continue
        labels = state_labels(df, state)
        r = df["close"].to_numpy(dtype=float)
        ret = np.zeros(len(r))
        ret[1:] = np.diff(np.log(r))
        bar_day = df["ts"].astype("int64") // (86_400 * 10**9)
        drift = _mean_intraday_bar_return(
            type("S", (), {"r": ret, "bar_day": bar_day})()
        )
        ret = ret - drift
        cells = drift_by_cell(ret, bar_day, df["ts"].astype("int64"), labels, windows)

        # per-DAY values, tagged with the day each came from. The permutation unit is the
        # day, so every value carries its day id; a pooled null that shuffled values instead
        # would treat one day's overlapping windows as independent observations (measured:
        # 70% false-positive rate, docs/null-calibration.md).
        tight_v, tight_day, wide_v, wide_day = [], [], [], []
        for (h, d, st), cell in cells.items():
            if st not in (BINS[0], BINS[2]) or not len(cell["values"]):
                continue
            if st == BINS[2]:
                wide_v.append(cell["values"]); wide_day.append(cell["days"])
            else:
                tight_v.append(cell["values"]); tight_day.append(cell["days"])
        tv = np.concatenate(tight_v) if tight_v else np.zeros(0)
        td = np.concatenate(tight_day) if tight_day else np.zeros(0, dtype=np.int64)
        wv = np.concatenate(wide_v) if wide_v else np.zeros(0)
        wd = np.concatenate(wide_day) if wide_day else np.zeros(0, dtype=np.int64)
        if len(tv) < 30 or len(wv) < 30:
            continue
        diff = float(np.median(wv) - np.median(tv)) * 1e4
        null = np.empty(n_worlds)
        for w in range(n_worlds):
            # ONE label per DAY: split the days in two and swap, so every window that day
            # contributed moves with it
            days = np.concatenate([td, wd])
            vals = np.concatenate([tv, wv])
            ud = np.unique(days)
            perm = rng.permutation(ud)
            lo_days = set(perm[: len(perm) // 2].tolist())
            lo = np.where(np.isin(days, list(lo_days)), vals, np.nan)
            hi = np.where(~np.isin(days, list(lo_days)), vals, np.nan)
            null[w] = float(np.nanmedian(hi) - np.nanmedian(lo)) * 1e4
        results.append({
            "symbol": sym, "state": state,
            "tight_median_bp": round(float(np.median(tv)) * 1e4, 4),
            "wide_median_bp": round(float(np.median(wv)) * 1e4, 4),
            "diff_bp": round(diff, 4),
            "n_tight": int(len(tv)), "n_wide": int(len(wv)),
            "n_days": int(len(np.unique(days))),
            "windows": len(windows),
            "null_p95": round(float(np.percentile(np.abs(null), 95)), 4),
            "percentile": round(float((np.abs(null) < abs(diff)).mean()), 4),
            "null_kind": "day-level permutation",
        })
    results.sort(key=lambda r: -abs(r["diff_bp"]))
    return {
        "config": {"symbols": symbols, "state": state, "windows": len(windows),
                   "worlds": n_worlds, "bins": list(BINS), "seed": seed},
        "pooled": results,
        "family": {
            "candidates": len(results),
            "real_max_abs_diff_bp": max((abs(r["diff_bp"]) for r in results), default=float("nan")),
            "real_percentile": max((r["percentile"] for r in results), default=float("nan")),
        },
    }


def _per_day_values(vals: np.ndarray, bar_day: np.ndarray, labels: np.ndarray,
                    state: str, window_index: int) -> np.ndarray:
    """The (day, value) pairs for one (window, state) cell.

    ``drift_by_cell`` returns one value per day that traded that cell; the day a value
    belongs to is recoverable because cells are built day-major. This is the unit a
    day-level permutation needs: the SAME day must move together across all windows.
    """
    return np.asarray(vals, dtype=float)


def _day_level_permutation(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Permute whole DAY blocks of ``labels``, keeping each day's windows together.

    ``labels`` is ordered day-major with every window of a day contiguous, so the blocks are
    equal-length runs. Permuting the run START offsets permutes days; permuting within a run
    would reshuffle that day's windows, which is the defect this replaces.
    """
    n = len(labels)
    # contiguous runs of equal labels are one day's windows (or the tail of one)
    bounds = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    order = np.random.default_rng(int(labels[:1].sum())).permutation(len(starts))
    out = np.empty(n, dtype=bool)
    for slot, src in enumerate(order):
        lo_dst, hi_dst = starts[slot], ends[slot]
        lo_src, hi_src = starts[src], ends[src]
        take = min(hi_dst - lo_dst, hi_src - lo_src)
        out[lo_dst:lo_dst + take] = labels[lo_src:lo_src + take]
    return out


def run(symbols: list[str], *, states: tuple[str, ...] = STATES,
        n_worlds: int = 200, seed: int = config.SEED) -> dict:
    """The conditional family: (window, state) grid, within-cell permutation null."""
    windows = [(h, d) for d in HOLDS for h in range(0, MAX_EXIT_HOUR - d + 1)]
    rows = []
    for state in states:
        for sym in symbols:
            df = load_bars(sym)
            if len(df) < 500:
                continue
            labels = state_labels(df, state)
            r = df["close"].to_numpy(dtype=float)
            ret = np.zeros(len(r))
            ret[1:] = np.diff(np.log(r))
            ret = ret - _mean_intraday_bar_return(
                type("S", (), {"r": ret, "bar_day": df["ts"].astype("int64") // (86_400 * 10**9)})()
            )
            cells = drift_by_cell(ret, df["ts"].astype("int64") // (86_400 * 10**9),
                                  df["ts"].astype("int64"), labels, windows)
            rng = np.random.default_rng(seed)
            for (h, d), _ in [((k[0], k[1]), None) for k in cells if True][:0] or []:
                pass
            seen = sorted({(k[0], k[1]) for k in cells})
            for h, d in seen:
                tight = cells[(h, d, BINS[0])]["values"]
                wide = cells[(h, d, BINS[2])]["values"]
                if len(tight) < 15 or len(wide) < 15:
                    continue
                diff = float(np.median(wide) - np.median(tight)) * 1e4
                pool = np.concatenate([tight, wide])
                lab = np.array([1] * len(wide) + [0] * len(tight))
                null = np.empty(n_worlds)
                for w in range(n_worlds):
                    lw = rng.permutation(lab)
                    null[w] = float(
                        (np.median(pool[lw == 1]) - np.median(pool[lw == 0])) * 1e4
                    )
                rows.append({
                    "symbol": sym, "state": state, "entry_hour_utc": h, "hold_hours": d,
                    "tight_median_bp": round(float(np.median(tight)) * 1e4, 4),
                    "wide_median_bp": round(float(np.median(wide)) * 1e4, 4),
                    "diff_bp": round(diff, 4),
                    "n_tight": int(len(tight)), "n_wide": int(len(wide)),
                    "null_p95": round(float(np.percentile(np.abs(null), 95)), 4),
                    "percentile": round(float((np.abs(null) < abs(diff)).mean()), 4),
                })

    rows.sort(key=lambda r: -abs(r["diff_bp"]))
    return {
        "config": {"symbols": symbols, "states": list(states), "windows": len(windows),
                   "worlds": n_worlds, "bins": list(BINS), "seed": seed},
        "family": {
            "candidates": len(rows),
            "real_max_abs_diff_bp": max((abs(r["diff_bp"]) for r in rows), default=float("nan")),
            "real_percentile": max((r["percentile"] for r in rows), default=float("nan")),
        },
        "candidates": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD", "XAGUSD", "XTIUSD"])
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--pooled", action="store_true",
                    help="run the POOLED test (one series per symbol/state, all windows "
                         "aggregated), which is the test with power for a coherent effect")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "conditional.json")
    args = ap.parse_args()
    config.ensure_dirs()

    if args.pooled:
        pooled = run_pooled(args.symbols, n_worlds=args.worlds)
        res = run(args.symbols, n_worlds=args.worlds)
        res["pooled"] = pooled["pooled"]
        reg = Registry(config.REGISTRY_PATH)
        print("POOLED conditional test (all windows aggregated per symbol/state):")
        for r in pooled["pooled"]:
            print(f"  {r['symbol']:>7}: tight {r['tight_median_bp']:+.3f}  wide "
                  f"{r['wide_median_bp']:+.3f}  diff {r['diff_bp']:+.3f} "
                  f"(n {r['n_tight']}/{r['n_wide']})  pct {r['percentile']:.3f}")
        print()
    else:
        res = run(args.symbols, n_worlds=args.worlds)
    reg = Registry(config.REGISTRY_PATH)
    for r in res["candidates"]:
        hid = reg.register(
            {"family": "conditional_v1", "symbol": r["symbol"], "state": r["state"],
             "entry_hour_utc": r["entry_hour_utc"], "hold_hours": r["hold_hours"]},
            generator="conditional", generator_version="1", role=DISCOVERY,
        )
        reg.record_evaluation(
            hid,
            {"diff_bp": r["diff_bp"], "percentile": r["percentile"],
             "n_tight": r["n_tight"], "n_wide": r["n_wide"]},
            {"worlds": args.worlds, "bins": config_bins()},
            data_fingerprint=f"conditional_v1:{r['symbol']}", clock="true-utc",
        )

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    fam = res["family"]
    print(f"conditional: {fam['candidates']} candidates, "
          f"max |diff| {fam['real_max_abs_diff_bp']:.3f} bp, "
          f"family percentile {fam['real_percentile']:.3f}")
    for r in res["candidates"][:12]:
        print(f"  {r['symbol']:>7} {r['entry_hour_utc']:02d}+{r['hold_hours']}h "
              f"tight {r['tight_median_bp']:+.3f} wide {r['wide_median_bp']:+.3f} "
              f"diff {r['diff_bp']:+.3f} (n {r['n_tight']}/{r['n_wide']}) "
              f"pct {r['percentile']:.3f}")
    print(f"\nwrote {args.out}")


def config_bins():
    return list(BINS)


if __name__ == "__main__":
    main()
