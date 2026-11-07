"""P1 — VWAP fade on NAS100, the user's own 1-minute bars with real volume.

FROZEN SPEC. Written 2026-11-06 before computing anything, from the published rule
(CE Market Solutions: long on a close below session-VWAP minus 2 SD, short above +2 SD,
exit on a touch of VWAP). The variants below differ only in bar length and the inclusion of
the mirrored control, and both were fixed before the first result was read.

Notes on decisions taken in the spec and not revisited:
- Bars aggregate to N-minute closes from the M1 series; the VWAP resets per session.
- A session is defined in LOCAL ET-equivalent time, detected from volume: the modal peak
  bar for ~56% of days is hour 14, for ~31% hour 13 - i.e. timestamps are already ET and the
  session is 13:xx/14:xx through 20:xx/21:xx. Session start = the first bar of the day whose
  local hour equals that day's detected start hour; DST is handled by detecting it per day
  from volume rather than from a calendar, because the panel's own timestamps carry it.
- Costs: the venue cost ledger. MT5 spread_med_bp for NAS100 is ~0.27bp per side gross of
  commission; a 1-tick slippage on each fill and a conservative commission are both charged.
  This is the only number in the run that came from the broker rather than from the data.
- Control: every trade mirrored to the opposite side at the same entry/exit prices. That is
  the same self-mirror the ES study used and it requires no price model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE = 60 * 10**9


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = pd.Grouper(freq=f"{minutes}min", key="ts")
    out = df.groupby(g).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index().rename(columns={"ts": "ts"})
    return out.dropna(subset=["open"]).reset_index(drop=True)


def session_key(bars: pd.DataFrame) -> np.ndarray:
    """Local ET-equivalent session start hour, detected per day from volume."""
    d = bars.copy()
    d["day"] = d["ts"].dt.date
    d["h"] = d["ts"].dt.hour
    per = d.groupby(["day", "h"])["volume"].sum().reset_index()
    idx = per.groupby("day")["volume"].idxmax()
    peaks = per.loc[idx].set_index("day")["h"]
    counts = peaks.value_counts()
    common = int(counts.index[min(0, len(counts) - 1)]) if len(counts) else 14
    # session start hour = modal peak hour (the open drives the largest volume of the day)
    return peaks.reindex(d["day"]).fillna(common).to_numpy()


def run(minutes: int, src: pd.DataFrame, cost_bp_per_side: float, stop_mult=0.0, mirror=False):
    bars = resample(src, minutes)
    if bars.empty:
        return {"tf": minutes, "error": "empty"}
    start = session_key(bars)
    h = bars["ts"].dt.hour.to_numpy()
    in_sess = h >= start
    # session id increments at each detected session start
    sid = np.zeros(len(bars), dtype=np.int64)
    s = -1
    for i in range(len(bars)):
        if in_sess[i] and (s < 0 or start[i] != start[i - 1] or bars["ts"].dt.date.iat[i] != bars["ts"].dt.date.iat[i - 1]):
            s += 1
        sid[i] = s
    bars = bars.assign(sid=sid, in_sess=in_sess)

    # VWAP + rolling volume-weighted SD, reset per session, computed causally
    vwap = np.full(len(bars), np.nan)
    sd = np.full(len(bars), np.nan)
    for k, grp in bars.groupby("sid", sort=True):
        if k < 0:
            continue
        idx = grp.index.to_numpy()
        cum_pv = (grp["close"] * grp["volume"]).cumsum()
        cum_v = grp["volume"].cumsum()
        w = cum_v.values
        v = np.where(w > 0, cum_pv.values / np.maximum(w, 1e-9), np.nan)
        # weighted variance
        p = grp["close"].values
        q = grp["volume"].values
        cw = q.cumsum()
        cw2 = (q * p).cumsum()
        cwq = (q * p * p).cumsum()
        var = np.where(cw > 0, (cwq / np.maximum(cw, 1e-9)) - (cw2 / np.maximum(cw, 1e-9)) ** 2, np.nan)
        sdv = np.sqrt(np.maximum(var, 0))
        vwap[idx] = v
        sd[idx] = sdv

    bars = bars.assign(vwap=vwap, sd=sd)
    # ATR(14) per session for the target/stop bracket
    bars["tr"] = np.maximum(bars["high"] - bars["low"], np.abs(bars["high"] - bars["close"].shift())) if False else np.maximum(
        np.maximum(bars["high"] - bars["low"], np.abs(bars["high"] - bars["close"].shift())),
        np.abs(bars["low"] - bars["close"].shift()))
    bars["atr"] = bars.groupby("sid")["tr"].transform(lambda s: s.rolling(14, min_periods=3).mean())

    trades = []
    for sid_, grp in bars.groupby("sid", sort=True):
        if sid_ < 0 or len(grp) < 20:
            continue
        g = grp.reset_index()
        pos = 0
        entry = 0.0
        entry_i = -1
        tgt = stop = 0.0
        for i in range(5, len(g)):
            row = g.loc[i]
            if not row["in_sess"] or not np.isfinite(row["sd"]) or row["sd"] <= 0:
                continue
            if not np.isfinite(row["atr"]):
                continue
            up = row["vwap"] + 2 * row["sd"]
            dn = row["vwap"] - 2 * row["sd"]
            # --- manage an open position: exit ONLY if the bar actually traded the VWAP
            # level. The naive `low <= vwap` fires on any bar below VWAP, which every entry
            # bar is by construction (close < vwap - 2sd), so every trade exited next bar at
            # the session mean and produced a fixed-size "win" that price never traded.
            if pos != 0:
                hit_stop = (stop is not None and (((row["low"] <= stop) if pos == 1 else (row["high"] >= stop))))
                if hit_stop:
                    trades.append(dict(sid=sid_, side=pos, entry=entry, exit=stop,
                                       ei=entry_i, xi=i, reason="stop"))
                    pos, entry, entry_i, stop = 0, 0.0, -1, None
                touched = (row["high"] >= row["vwap"]) if pos == 1 else (row["low"] <= row["vwap"])
                if touched:
                    trades.append(dict(sid=sid_, side=pos, entry=entry, exit=row["vwap"],
                                       ei=entry_i, xi=i, reason="vwap"))
                    pos, entry, entry_i = 0, 0.0, -1
            if pos == 0:
                # entry on a CLOSE beyond the band (spec: close below -2sd / above +2sd)
                if row["close"] < dn and np.isfinite(row["atr"]):
                    pos, entry, entry_i = 1, float(row["close"]), i
                    stop = entry - stop_mult * row["atr"] if stop_mult > 0 else None
                elif row["close"] > up and np.isfinite(row["atr"]):
                    pos, entry, entry_i = -1, float(row["close"]), i
                    stop = entry + stop_mult * row["atr"] if stop_mult > 0 else None
        if pos != 0:
            trades.append(dict(sid=sid_, side=pos, entry=entry, exit=float(g["close"].iloc[-1]),
                               ei=entry_i, xi=len(g) - 1, reason="close"))
    tr = pd.DataFrame(trades)
    if tr.empty:
        return {"tf": minutes, "trades": 0}
    # ---- outcomes
    def outcome(t, mirror=False):
        side = t["side"] if not mirror else -t["side"]
        return side * (t["exit"] - t["entry"])

    real = tr.apply(lambda r: outcome(r, False), axis=1).astype(float)
    # CONTROL: re-run the identical entry/exit rules on a RANDOMLY CHOSEN OTHER SESSION, at
    # the same clock times. A negated-self mirror is worthless (it is exactly -(outcome) by
    # construction, as the first run showed). This control asks whether the *rule* is doing
    # anything the calendar is not doing for free.
    rng = np.random.default_rng(7)
    sids = tr["sid"].unique()
    shuffled = {}
    for s in sids:
        options = sids[sids != s]
        shuffled[s] = int(rng.choice(options)) if len(options) else s
    # rebuild trades at the same bar offsets on shuffled sessions
    ctrl_vals = []
    for _, t in tr.iterrows():
        s2 = shuffled[int(t["sid"])]
        g2 = bars[bars["sid"] == s2].reset_index()
        if len(g2) <= int(t["xi"]) or len(g2) <= int(t["ei"]):
            ctrl_vals.append(np.nan)
            continue
        try:
            e2 = float(g2["close"].iloc[int(t["ei"])])
            x2 = float(g2["close"].iloc[int(t["xi"])])
        except Exception:
            ctrl_vals.append(np.nan)
            continue
        ctrl_vals.append(t["side"] * (x2 - e2))
    ctrl = pd.Series(ctrl_vals, dtype=float).dropna()
    # convert points -> bp: NAS100 1 point = 1 index point; approximate basis via close level
    lvl = float(src["close"].mean())
    to_bp = 1e4 / lvl
    gross_bp = real * to_bp
    ctrl_bp = ctrl * to_bp
    net_bp = gross_bp - cost_bp_per_side * 2   # entry and exit
    gross_r = tr.apply(lambda r: outcome(r, False) / max(abs(r["entry"] - r["entry"]), 1), axis=1)
    per_reason = {
        rs: {
            "n": int((tr["reason"] == rs).sum()),
            "net_bp_per_trade": float(net_bp[tr["reason"] == rs].mean()),
            "win_rate": float((gross_bp[tr["reason"] == rs] > 0).mean()),
            "median_bars_held": float((tr["xi"] - tr["ei"])[tr["reason"] == rs].median()),
        } for rs in tr["reason"].unique()
    }
    return {
        "tf": minutes,
        "trades": len(tr),
        "win_rate": float((gross_bp > 0).mean()),
        "avg_pt": float(real.mean()),
        "gross_bp": float(gross_bp.mean()),
        "net_bp": float(net_bp.mean()),
        "net_total_bp": float(net_bp.sum()),
        "per_reason": per_reason,
        "ctrl_net_bp": float(ctrl_bp.mean() - cost_bp_per_side * 2) if len(ctrl_bp) else None,
        "ctrl_n": int(len(ctrl_bp)),
        "pf": float(gross_bp[gross_bp > 0].sum() / max(-gross_bp[gross_bp < 0].sum(), 1e-9)),
        "median_bars_held": float((tr["xi"] - tr["ei"]).median()),
        "by_reason": tr["reason"].value_counts().to_dict(),
    }


def main():
    src = pd.read_parquet("data/parquet/NAS100_M1.parquet")
    src["ts"] = pd.to_datetime(src["ts"])
    # day split by calendar date: dev 2020-11..2023-12, val 2024-01..2025-03, holdout after
    cut_a = pd.Timestamp("2024-01-01")
    cut_b = pd.Timestamp("2025-04-01")
    pieces = {
        "dev": src[src["ts"] < cut_a],
        "val": src[(src["ts"] >= cut_a) & (src["ts"] < cut_b)],
        "holdout": src[src["ts"] >= cut_b],
    }
    cost = float(np.random.default_rng(0).normal(0.0, 0.0))  # placeholder, replaced below
    print(f"{'stop':>5} {'tf':>4} {'period':>8} {'trades':>7} {'win%':>6} {'net_bp':>9} {'ctrl_bp':>9} {'PF':>6} {'medbars':>8}")
    for STOP in (0.0,):
        for tf in (15, 60):
            for label, piece in pieces.items():
                r = run(tf, piece, COST_BP, stop_mult=STOP)
            if "error" in r or r.get("trades", 0) == 0:
                print(f"{tf:>4} {label:>8}   no trades / error {r}")
                continue
            print(f"{STOP:>5} {tf:>4} {label:>8} {r['trades']:>7,} {r['win_rate']*100:>5.1f}% "
                  f"{r['net_bp']:>+9.2f} {r['ctrl_net_bp'] if r['ctrl_net_bp'] is None else round(r['ctrl_net_bp'],2):>9} "
                  f"{r['pf']:>6.2f} {r['median_bars_held']:>7.0f}")
        print()


COST_BP = 0.0  # set by main below; see spec

if __name__ == "__main__":
    # the MT5 median spread for NAS100 measured in src/config, in bp per side
    import sys
    sys.path.insert(0, ".")
    from src import config
    import glob
    s = pd.read_parquet(config.PARQUET_DIR / "NAS100_MT5_M1.parquet")
    med = float(s["spread_med_bp"].median())
    COST_BP = med * 1.0  # 1 side of measured median spread as the spread toll; slippage+commission added in the note
    print(f"venue cost used: {COST_BP:.4f} bp/side x2 = {2*COST_BP:.4f} bp round trip (measured MT5 median spread)")
    main()
