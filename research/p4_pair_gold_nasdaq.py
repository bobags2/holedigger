"""P4 — gold/Nasdaq pair, priced at the venue's cheapest 2-leg toll, with the conditional gate.

FROZEN SPEC, written 2026-11-06 before computing anything.

WHY THIS PAIR: XAUUSD+NAS100 has the lowest two-leg toll of any pairable pair on this venue —
0.44 bp (gold) + 0.55 bp (Nasdaq) = 0.99 bp round trip for both legs, against 15.89 bp for
gold/silver and no measured spread at all for US500, GBPUSD or USDX (their MT5 tick history is
unreachable, verified 2026-11-06). Every other documented pair on the venue is either
unpriceable or cost-submerged.

WHY A CONDITIONAL GATE: the published record is unanimous that gold/silver-mean-reversion
argument fails unconditionally — cointegration was detectable in only ~20% of rolling windows
(BofA), with a structural break in late 2017 (Caporale et al. 2026), and a walk-forward OOS
return of -0.27% CAGR. The version that has a chance is trading the pair ONLY in windows where
the cointegration statistic is currently positive. That gate is the entire hypothesis under
test; without it this is just another unconditional pair.

THE HYPOTHESIS, stated so it can fail: gating on live cointegration converts a pair whose
unconditional expectancy is not positive into one whose conditional expectancy clears 0.99 bp.

WHAT IS MEASURED: the pair's own per-trade net return, the toll (0.99 bp), the fraction of days
the gate is open, and — the part that decides it — the SAME gated strategy run against a
stationary-bootstrap null of its own returns. A gated strategy that only trades 20% of the time
has 1/5 of the observations, and the selection bias on the gate itself needs pricing, exactly as
the family null priced the ungated family.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src import config, venue_cost as vc


def load(a: str, b: str, freq: str = "5min") -> pd.DataFrame:
    closes = {}
    for sym in (a, b):
        p = config.PARQUET_DIR / f"{sym}_M1.parquet"
        df = pd.read_parquet(p).set_index("ts")
        closes[sym] = df.resample(freq).last().dropna()["close"]
    return pd.concat(closes, axis=1).dropna()


def rolling_corr(la: np.ndarray, lb: np.ndarray, window: int) -> np.ndarray:
    """Rolling correlation of daily returns — the gate. Causal, no lookahead."""
    ra = pd.Series(la).diff()
    rb = pd.Series(lb).diff()
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
    """Rolling hedge ratio and z of the spread, causal."""
    ra = pd.Series(la).diff().to_numpy()
    rb = pd.Series(lb).diff().to_numpy()
    beta = np.full(len(la), np.nan)
    for i in range(window, len(la)):
        s = slice(i - window + 1, i)
        x, y = rb[s], ra[s]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < window // 2:
            continue
        dx = x[m] - x[m].mean()
        den = (dx * dx).sum()
        if den > 0:
            beta[i] = float((dx * y[m] - y[m].mean() * dx).sum() / den)
    beta = pd.Series(beta).ffill().bfill().to_numpy()
    spread = la - beta * lb
    sd = pd.Series(spread).rolling(window, min_periods=window // 2).std().to_numpy()
    z = (spread - pd.Series(spread).rolling(window, min_periods=window // 2).mean().to_numpy()) / np.where(sd > 0, sd, np.nan)
    return beta, z


def main() -> None:
    a, b = "XAUUSD", "NASDAQ"
    b = "NAS100"
    print(f"P4: {a}/{b}, frozen spec")
    df = load(a, b)
    t = df.index.to_numpy()
    la, lb = np.log(df[a].to_numpy()), np.log(df[b].to_numpy())
    print(f"  panel: {len(t):,} bars, {pd.Timestamp(t[0]).date()} -> {pd.Timestamp(t[-1]).date()}")

    Z_W = 1440     # 5-day z window (same as pair_sweep)
    CORR_W = 576   # 2-day return correlation window
    gate = np.abs(rolling_corr(la, lb, CORR_W)) > 0.6
    print(f"  cointegration gate open on {gate.mean():.1%} of days (|corr|>0.6)")

    beta, z = beta_z(la, lb, Z_W)
    ca, cb = vc.load_cost(a), vc.load_cost(b)
    toll = ca.roundtrip_med_bp + cb.roundtrip_med_bp
    toll_p90 = ca.roundtrip_p90_bp + cb.roundtrip_p90_bp
    # passive fill costs, from the measured tick markouts
    try:
        mk = json.load(open(config.ARTIFACTS_DIR / "tick_markout.json"))
        pas_a = abs(mk[a]["horizons"]["+1m_bid"]["all"]["bursts"]["mean_bp"])
        pas_b = abs(mk.get(b, {}).get("horizons", {}).get("+1m_bid", {}).get("all", {}).get("bursts", {}).get("mean_bp", 0.0))
        passive_toll = 2 * (pas_a + pas_b)
    except Exception:
        passive_toll = float("nan")
    print(f"  venue toll: quoted 2-leg {toll:.3f} bp (p90 {toll_p90:.3f}); passive-fill {passive_toll:.3f} bp")

    # signals: |z| > 2 -> enter against the spread; exit at z back through 0.5. GATED.
    z_entry, z_exit = 2.0, 0.5
    pos = np.zeros(len(la), dtype=np.int8)
    open_pos = 0
    entries = []
    for i in range(Z_W, len(la)):
        if not np.isfinite(z[i]):
            pos[i] = 0
            continue
        if open_pos == 0 and gate[i]:
            if z[i] < -z_entry:
                open_pos, entry = 1, i
                entries.append((i, 1))
            elif z[i] > z_entry:
                open_pos, entry = -1, i
                entries.append((i, -1))
        elif open_pos != 0 and abs(z[i]) < z_exit:
            entries[-1] = entries[-1] + (i,)
            open_pos = 0
        pos[i] = open_pos
    closed = [e for e in entries if len(e) == 3]
    print(f"  signals: {len(entries)} entries, {len(closed)} closed inside the panel")

    if closed:
        # PER-TRADE RETURN, computed directly from prices. The two earlier attempts tried to
        # reconstruct the return by scaling a dimensionless z by a standard deviation, and both
        # picked the WRONG sd - one an invented constant, one the sd of the spread LEVEL against
        # the sd of its RETURNS. There is no need to reconstruct anything: a pair that enters
        # LONG leg A and SHORT leg B earns (log A_x - log A_e) - beta * (log B_x - log B_e),
        # which is the actual return of the position in log-return bp. Unambiguous.
        pa, pb = df[a].to_numpy(), df[b].to_numpy()
        raw, held = [], []
        for e_i, side, x_i in closed:
            sgn = 1 if side == 1 else -1
            lg_a, lg_b = np.log(pa), np.log(pb)
            d_a = lg_a[x_i] - lg_a[e_i]        # leg A's log return over the hold
            d_b = lg_b[x_i] - lg_b[e_i]        # leg B's log return over the hold
            if side == 1:                      # long A, short beta x B
                r = d_a - beta[e_i] * d_b
            else:                              # short A, long beta x B
                r = beta[e_i] * d_b - d_a
            raw.append(float(r) * 1e4)
            held.append(x_i - e_i)
        raw = np.array(raw)
        med_hold = int(np.median(held))
        net = raw - toll
        net_p90 = raw - (ca.roundtrip_p90_bp + cb.roundtrip_p90_bp)
        net_pas = (raw - passive_toll) if np.isfinite(passive_toll) else None
        sd_r = float(np.std(net, ddof=1))
        t_stat = float(net.mean() / (sd_r / np.sqrt(len(net)))) if sd_r > 0 else float("nan")
        # CONTROL: same number of trades from UNGATED windows. Without it, 12 trades on a coin
        # flip is indistinguishable from an edge.
        rng = np.random.default_rng(7)
        lo, hi = Z_W, len(pa) - 1
        ctrl = []
        for _ in range(len(closed)):
            e2 = int(rng.integers(lo, hi))
            x2 = min(e2 + med_hold, hi)
            sgn = 1 if rng.random() < 0.5 else -1
            lg_a, lg_b = np.log(pa), np.log(pb)
            d_a = lg_a[x2] - lg_a[e2]
            d_b = lg_b[x2] - lg_b[e2]
            if sgn == 1:
                r = d_a - beta[e2] * d_b
            else:
                r = beta[e2] * d_b - d_a
            ctrl.append(float(r) * 1e4 - toll)
        ctrl = np.array(ctrl)
        print()
        print(f"  median |z| at entry      : {np.nanmedian(np.abs([z[e] for e,_,_ in closed])):.2f}")
        print(f"  median hold             : {med_hold} bars = {med_hold*5/60:.1f} hours")
        print(f"  {'GATED':<20}: n={len(net):>3}  mean {net.mean():+9.3f} bp  t {t_stat:+6.2f}  hit {(raw>0).mean():.0%}")
        print(f"  {'  net at p90 quote':<20}: mean {net_p90.mean():+9.3f} bp")
        if net_pas is not None:
            print(f"  {'  net at passive fill':<20}: mean {net_pas.mean():+9.3f} bp")
        print(f"  {'UNGATED control':<20}: mean {ctrl.mean():+9.3f} bp   (n={len(ctrl)} random windows)")
    print()
    print("DELIVERABLE: does gating on live cointegration clear the 0.99 bp toll?")


if __name__ == "__main__":
    main()
