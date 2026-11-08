"""Path-aware stop evaluation - the reviewer's central correction, and it is right.

THE DEFECT IN `src/risk_rules.py`, stated plainly. `evaluate_forward` scores a stop as
``np.maximum(profit, -cap)``: a floor on each position's FINAL P&L. A stop does not fire on the
final P&L; it fires when the price path touches the level. So the floor keeps every position
that dipped below the cap and later recovered - precisely the positions a real stop would have
cut. The evaluation was path-unaware, and the cap's apparent gain came from that gap.

THIS MODULE measures max adverse excursion from the M1 high/low between each position's open and
close, and fires a stop when MAE >= the level. A sweep (the sub-minute touch inside one bar) is
invisible in M1 data, so this UNDERSTATES stop-outs on a real tape. That makes the numbers here
a LOWER bound on how often the stop fires, i.e. an OPTIMISTIC evaluation of the cap, and
optimistic is the wrong direction for a claim - so it is reported as an upper bound on gain and
the reviewer's framing (an upper bound on loss) is the one to plan against.

THE PROPERTY TEST THAT MUST HOLD, and which the previous code fails: applying any risk rule to
a zero-skill book must not improve expectancy. If it does, the rule is manufacturing return
from the evaluation rather than removing loss, and the number is not a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src import config, risk_rules as rr, venue_cost as vc


def max_adverse_excursion(trades: pd.DataFrame, *, symbol_col: str = "sym") -> pd.DataFrame:
    """Per-position MAE and MFE in bp of the position's notional, from M1 bars.

    A long that fell `d` below its entry MAE's by `d`; a short MAE's by the same amount upward.
    MFE is the mirror. Both are measured on bars STRICTLY between the open timestamp and the
    close timestamp, inclusive of neither end's own bar where the timestamp matches, because
    the entry bar's own range contains the fill itself.
    """
    out = []
    for sym, g in trades.groupby(symbol_col):
        p = config.PARQUET_DIR / f"{sym}_M1.parquet"
        if not p.exists():
            alt = config.PARQUET_DIR / f"{sym}_M1.parquet"
            if not alt.exists():
                continue
            p = alt
        df = pd.read_parquet(p, columns=["ts", "high", "low"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
        ts_ns = df["ts"].to_numpy()
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        for r in g.itertuples(index=False):
            o = np.datetime64(pd.Timestamp(r.open_time))
            c = np.datetime64(pd.Timestamp(r.close_time))
            i0 = int(np.searchsorted(ts_ns, o, side="right"))
            i1 = int(np.searchsorted(ts_ns, c, side="left"))
            if i1 <= i0:
                out.append({"position_id": r.position_id, "mae_bp": np.nan, "mfe_bp": np.nan})
                continue
            px_o = float(r.open_price)
            long_side = str(getattr(r, "type", "buy")).lower().startswith("b")
            seg_hi = np.max(hi[i0:i1])
            seg_lo = np.min(lo[i0:i1])
            if long_side:
                mae = (px_o - seg_lo) / px_o * 1e4
                mfe = (seg_hi - px_o) / px_o * 1e4
            else:
                mae = (seg_hi - px_o) / px_o * 1e4
                mfe = (px_o - seg_lo) / px_o * 1e4
            out.append({"position_id": r.position_id, "mae_bp": float(mae), "mfe_bp": float(mfe)})
    return pd.DataFrame(out)


def path_aware(trades: pd.DataFrame, spec_cap_bp: dict, *, slippage_bp: float = 0.0) -> pd.DataFrame:
    """Score a stop the way the tape would: it fires if the path touched the level.

    ``spec_cap_bp`` maps symbol -> cap in bp (e.g. 3x the measured toll). P&L becomes
    ``-(cap + slippage)`` on a fired stop and the realised profit otherwise.
    """
    t = trades.copy()
    if "sym" not in t and "symbol" in t:
        t["sym"] = t["symbol"]
    ex = max_adverse_excursion(t)
    t = t.merge(ex, on="position_id", how="left")
    t["cap_bp"] = t["sym"].map(spec_cap_bp)
    fired = t["mae_bp"].to_numpy() >= t["cap_bp"].to_numpy()
    capped_pnl = -(t["notional"].to_numpy() * (t["cap_bp"].to_numpy() + slippage_bp) / 1e4)
    t["paw_profit"] = np.where(fired, capped_pnl, t["profit"].to_numpy(float))
    t["paw_fired"] = fired
    return t


def summary(t: pd.DataFrame, col: str) -> dict:
    d = t.groupby(t["open_day"])[col].sum()
    return {
        "net": round(float(t[col].sum()), 0),
        "pct_positive_days": round(float((d > 0).mean()), 3),
        "worst_day": round(float(d.min()), 0),
        "best_day": round(float(d.max()), 0),
        "per_pos_bp": round(float((t[col] / t["notional"] * 1e4).mean()), 3),
        "n": int(len(t)),
    }


def zero_skill_book(n: int = 3000, *, hold_s_median: float = 269.0, sid: str = "XAUUSD") -> pd.DataFrame:
    """A synthetic zero-skill book on gold, for the property test.

    Same shape as the account: median hold 269s, p90 2,139s, driftless, 2.5 bp/sqrt(min) of
    diffusion, 0.44 bp friction on entry. EXPECTANCY IS NEGATIVE BY CONSTRUCTION, so any risk
    rule that improves it is manufacturing number, not removing loss.
    """
    rng = np.random.default_rng(1234)
    end = pd.Timestamp("2026-09-01 00:00:00")
    opens = end - pd.to_timedelta(rng.exponential(hold_s_median, n).cumsum(), unit="s")
    opens = opens[opens > end - pd.Timedelta(days=30)]
    n = len(opens)
    holds = rng.lognormal(np.log(hold_s_median), 1.0, n)
    closes = opens + pd.to_timedelta(holds, unit="s")
    # driftless diffusion scaled by hold: 2.5 bp per sqrt(minute)
    minutes = holds / 60.0
    sd = 2.5 * np.sqrt(np.maximum(minutes, 1e-6))
    ret = rng.normal(0.0, sd)
    px = 3800.0
    notional = px * 0.01 * 100.0 * np.ones(n)
    gross = ret / 1e4 * notional
    friction = 0.44 / 1e4 * notional
    side = rng.choice([1, -1], n)
    profit = side * gross - friction
    return pd.DataFrame({
        "position_id": [f"z{i}" for i in range(n)],
        "sym": sid, "type": np.where(side > 0, "buy", "sell"),
        "open_time": opens, "close_time": closes, "open_price": px,
        "open_volume": 0.01, "notional": notional, "profit": profit,
        "open_day": pd.DatetimeIndex(opens).normalize(),
    })


def main() -> None:
    print("P19: path-aware stops - the reviewer's central correction, verified on our own data\n")
    t = rr.load_ctrader_trades()
    t = rr.attach_balance(t, Path(__file__).resolve().parent.parent / "artifacts" /
                          "ctrader_deals.json")
    smap = json.loads((Path(__file__).resolve().parent.parent / "artifacts" /
                       "ctrader_symbol_map.json").read_text(encoding="utf-8"))
    t["sym"] = t["symbol_id"].astype(str).map(smap)
    t["open_day"] = pd.to_datetime(t["open_time"]).dt.normalize()
    toll = {s: vc.load_cost(s).roundtrip_med_bp for s in ("XAUUSD", "NAS100", "XTIUSD", "XAGUSD")}

    print("=== REAL BOOK: how often does the path actually touch the cap? ===")
    t = path_aware(t, toll)
    for sym in ("XAUUSD", "NAS100", "XTIUSD", "XAGUSD"):
        g = t[t["sym"] == sym]
        if not len(g):
            continue
        print(f"  {sym:>7}: cap {toll[sym]:>6.2f} bp | MAE>=cap on {g['paw_fired'].mean():>5.1%} of "
              f"{len(g):>4} positions | floor(P&L,-cap) net {g['profit'].sum():>+11,.0f} | "
              f"path-aware net {g['paw_profit'].sum():>+11,.0f}")
    print()

    print("=== THE PROPERTY THAT MUST HOLD: no gain on a zero-skill book ===")
    z = zero_skill_book()
    print(f"  zero-skill book: n={len(z):,}, net {z['profit'].sum():+,.0f} "
          f"({z['profit'].sum()/len(z):+.2f}/position), "
          f"per-pos {float((z['profit']/z['notional']*1e4).mean()):+.3f} bp by construction")
    z = path_aware(z, toll)
    print(f"  as scored by src/risk_rules (floor on FINAL P&L): "
          f"{float(np.maximum(z['profit'], -z['notional']*toll['XAUUSD']/1e4).sum()):+,.0f}   <- what the")
    print(f"  as scored path-aware (fires if the path touched): {z['paw_profit'].sum():+,.0f}")
    print()
    print("  If the first number is better than zero, the OLD evaluation manufactures return on a")
    print("  book with no skill, and the reviewer is right. If the second stays below zero, the")
    print("  path-aware evaluator does not.")


if __name__ == "__main__":
    main()
