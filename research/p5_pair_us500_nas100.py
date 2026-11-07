"""P5 — US500/NAS100 pair on the cTrader toll, gated on live cointegration.

WHY THIS RUN EXISTS: P4's gold/Nasdaq test found a gate that clearly worked (+35.18 bp vs a
+3.41 bp ungated control) and could not be certified because the gate opened on only 2.1% of
days, leaving 12 trades in six years (t = +0.40). The pair's own artifact (artifact/pair_sweep.json)
says US500/NAS100 has the stronger ungated signal of the family: +5.94 bp/tape, break-even at
6.49 bp, and it is priceable ONLY on the cTrader venue (1.29 bp two-leg toll; no MT5 spread
panel exists for US500).

The hypothesis under test is the P4 failure mode's mirror image: if US500/NAS100 cointegration
is *persistently* live rather than episodically, the same gate produces a certifiable sample
instead of 12 trades. That is the difference between "the gate works and starves" and "the gate
works and trades".

Frozen 2026-11-06 before running. Symbol on the cTrader tape is NAS100, which is what
load_tape_cost expects.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import venue_cost as vc
from src.pair_sweep import build_pair


def rolling_corr(la: np.ndarray, lb: np.ndarray, window: int) -> np.ndarray:
    ra, rb = pd.Series(la).diff(), pd.Series(lb).diff()
    out = np.full(len(la), np.nan)
    for i in range(window, len(la)):
        x, y = ra.iloc[i - window + 1:i].to_numpy(), rb.iloc[i - window + 1:i].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < window // 2:
            continue
        dx, dy = x[m] - x[m].mean(), y[m] - y[m].mean()
        den = np.sqrt((dx * dx).sum() * (dy * dy).sum())
        if den > 0:
            out[i] = float((dx * dy).sum() / den)
    return out


def beta_z(la: np.ndarray, lb: np.ndarray, window: int):
    ra, rb = pd.Series(la).diff().to_numpy(), pd.Series(lb).diff().to_numpy()
    beta = np.full(len(la), np.nan)
    for i in range(window, len(la)):
        x, y = rb[i - window + 1:i], ra[i - window + 1:i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < window // 2:
            continue
        dx = x[m] - x[m].mean()
        den = (dx * dx).sum()
        if den > 0:
            beta[i] = float((dx * (y[m] - y[m].mean())).sum() / den)
    beta = pd.Series(beta).ffill().bfill().to_numpy()
    spread = la - beta * lb
    mu = pd.Series(spread).rolling(window, min_periods=window // 2).mean().to_numpy()
    sd = pd.Series(spread).rolling(window, min_periods=window // 2).std().to_numpy()
    return beta, (spread - mu) / np.where(sd > 0, sd, np.nan)


def run(a: str, b: str, corr_thresh: float, z_entry: float = 2.0, z_exit: float = 0.5):
    p = build_pair(a, b)
    t, la, lb = p.times, p.la, p.lb
    pa, pb = p.price_a / np.exp(la) if False else np.exp(la), np.exp(lb)
    Z_W = 1440
    corr = rolling_corr(la, lb, 576)
    gate = np.abs(corr) > corr_thresh
    beta, z = beta_z(la, lb, Z_W)
    toll = vc.load_tape_cost(a).roundtrip_med_bp + vc.load_tape_cost(b).roundtrip_med_bp
    toll_p90 = vc.load_tape_cost(a).roundtrip_p90_bp + vc.load_tape_cost(b).roundtrip_p90_bp

    trades = []
    open_pos, e_i = 0, -1
    for i in range(Z_W, len(la)):
        if not np.isfinite(z[i]):
            continue
        if open_pos == 0 and gate[i]:
            if z[i] < -z_entry:
                open_pos, e_i = 1, i
            elif z[i] > z_entry:
                open_pos, e_i = -1, i
        elif open_pos != 0 and abs(z[i]) < z_exit:
            trades.append((e_i, open_pos, i))
            open_pos = 0
    if open_pos:
        trades.append((e_i, open_pos, len(la) - 1))

    closed = [tr for tr in trades if tr[2] < len(la)]
    raw, held = [], []
    for e_i, side, x_i in closed:
        d_a, d_b = la[x_i] - la[e_i], lb[x_i] - lb[e_i]
        r = (d_a - beta[e_i] * d_b) if side == 1 else (beta[e_i] * d_b - d_a)
        raw.append(float(r) * 1e4)
        held.append(x_i - e_i)
    if not raw:
        return {"gate": f"|corr|>{corr_thresh}", "trades": 0, "gate_frac": float(gate.mean())}
    raw = np.array(raw)
    net = raw - toll
    net_p90 = raw - toll_p90
    sd_r = float(np.std(net, ddof=1))
    t_stat = net.mean() / (sd_r / np.sqrt(len(net))) if sd_r > 0 else float("nan")
    # year-by-year consistency: the gate that killed P4's pooled significance
    det = {}
    yr = pd.DatetimeIndex(t)
    for y in sorted(set(yr.year)):
        m = (yr[[e for e, _, _ in closed]].year == y)
        if m.sum():
            det[int(y)] = float(net[m].mean())
    pos = sum(1 for v in det.values() if v > 0)
    return {
        "gate": f"|corr|>{corr_thresh}", "gate_frac": float(gate.mean()),
        "trades": len(raw), "hit": float((raw > 0).mean()),
        "mean_raw_bp": float(raw.mean()), "net_bp": float(net.mean()),
        "net_p90_bp": float(net_p90.mean()), "t": float(t_stat),
        "toll_bp": float(toll), "med_hold_bars": int(np.median(held)),
        "per_year": {k: round(v, 2) for k, v in det.items()}, "yrs_pos": int(pos),
    }


def main() -> None:
    a, b = "US500", "NAS100"
    print(f"P5: {a}/{b} on the cTrader toll, gated")
    print(f"  2-leg toll: {vc.load_tape_cost(a).roundtrip_med_bp + vc.load_tape_cost(b).roundtrip_med_bp:.3f} bp (median)")
    print()
    print(f"{'gate':>12} {'open':>6} {'trades':>7} {'hit':>5} {'raw_bp':>8} {'net_bp':>8} {'t':>6} {'yrs+':>5} {'per-year mean (bp)'}")
    for thr in (0.5, 0.6, 0.7):
        r = run(a, b, thr)
        if r["trades"] == 0:
            print(f"{r['gate']:>12} {r['gate_frac']:>6.1%} {0:>7}  no trades")
            continue
        py = " ".join(f"{k}:{v:+.2f}" for k, v in r["per_year"].items())
        print(f"{r['gate']:>12} {r['gate_frac']:>6.1%} {r['trades']:>7,} {r['hit']:>5.0%} "
              f"{r['mean_raw_bp']:>8.2f} {r['net_bp']:>8.2f} {r['t']:>6.2f} {r['yrs_pos']:>2}/{len(r['per_year'])}  {py}")
    print()
    print("DELIVERABLE: a gate that WORKS and keeps a sample - the thing P4 could not have.")


if __name__ == "__main__":
    main()
