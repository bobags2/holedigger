"""Meta-labelling the account's OWN trade history - the reviewer's highest-value suggestion.

THE IDEA, and why it inverts what the rest of this repo has been doing.

Eight families searched public price features for a directional edge and all eight finished
below the toll. The reviewer's point: if there is an edge, it is in FILLS, not in anomalies -
and this account's own history is a 3,870-trade dataset with the fill's net P&L already in it.
ML here is a filter and a sizer for setups that already trade, and the effects it must find are
bigger than the ones the anomaly search chased, because they are differences between trades
that actually happen.

THE DATA (artifacts/trade_history.json, pulled 2026-11-07 from MT5
get_trading_history_positions):
  3,870 closed positions, 2024-11-29 to 2026-08-26
  2,690 XAUUSD, plus LNK/CADJPY/XTIUSD/CADCHF/EURJPY/LTC/EURUSD
  total profit -7,971 on a 68% WIN RATE - the classic scalped-tail signature
  median hold 78 seconds
  close_reason: Client 3,186 / Stop loss 234 / Mobile 290 / Take profit 83 / Stop out 71

The account is net negative with a two-thirds win rate, so the question is not "is there an
edge" but "which subset of these trades is it, and what distinguishes them". That is a
classification problem with 3,870 labelled examples and outcomes already realised - the
opposite of the search this repo has been running.

THE PIPELINE, and the part that must not be skipped:
  - label: net P&L per position, converted to bp on the position's notional
  - features: ENTRY-TIME ONLY, from the bar panels (spread state, vol regime, time of day,
    distance to prior high/low) - never anything known only at close
  - split: purged and embargoed walk-forward, because these trades overlap in time
  - selection: sign-flip null clustered by DAY prices the family; the score is OOS only
  - the honest output is a LIFT TABLE, not a curve: top-decile predicted vs bottom-decile
    actual, out of sample, net of the toll
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src import config, venue_cost as vc


def load_trades(path=None) -> pd.DataFrame:
    p = path or (config.ARTIFACTS_DIR / "trade_history.json")
    r = json.load(open(p, encoding="utf-8"))
    d = pd.DataFrame(r["positions"] if isinstance(r, dict) else r)
    d["open_time"] = pd.to_datetime(d["open_time"])
    d["close_time"] = pd.to_datetime(d["close_time"])
    d["profit"] = d["profit"].astype(float)
    d["open_volume"] = d["open_volume"].astype(float)
    d["contract_size"] = pd.to_numeric(d.get("contract_size", 1.0), errors="coerce").fillna(1.0)
    # net P&L in bp of notional: profit / (price * volume * contract_size)
    d["notional"] = d["open_price"].astype(float) * d["open_volume"] * d["contract_size"]
    d["net_bp"] = d["profit"] / d["notional"] * 1e4
    d["hold_s"] = (d["close_time"] - d["open_time"]).dt.total_seconds()
    return d.dropna(subset=["net_bp"])


def entry_features(trades: pd.DataFrame, lookback_bars: int = 60) -> pd.DataFrame:
    """Entry-time features per trade, from that symbol's own bar panel.

    Every feature is computed from bars STRICTLY BEFORE the entry bar. Nothing about how the
    trade actually went - no markout, no close features - may appear here, or the model learns
    to recognise outcomes and the lift table is meaningless.
    """
    rows = []
    for sym, grp in trades.groupby("symbol"):
        panel = config.PARQUET_DIR / f"{sym}_M1.parquet"
        if not panel.exists():
            panel = config.PARQUET_DIR / f"{sym}_MT5_M1.parquet"
        if not panel.exists():
            continue
        # column sets differ between panel families: the long 6-year panels carry `volume`,
        # the recent MT5 panels carry tick_n and spread instead. Read what exists.
        want = ["ts", "open", "high", "low", "close", "volume", "tick_n"]
        have = pd.read_parquet(panel).columns.tolist()
        df = pd.read_parquet(panel, columns=[c for c in want if c in have])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
        if "volume" in df.columns:
            df = df.rename(columns={"volume": "vol"})
        cl, hi, lo = (df[c].to_numpy(float) for c in ("close", "high", "low"))
        ret = np.zeros(len(cl))
        ret[1:] = np.diff(np.log(np.maximum(cl, 1e-12)))
        atr = pd.Series(hi - lo).rolling(60, min_periods=10).mean().to_numpy()
        ts_ns = df["ts"].to_numpy()
        for t in grp.itertuples(index=False):
            i = int(np.searchsorted(ts_ns, np.datetime64(t.open_time), side="right")) - 1
            if i < lookback_bars:
                continue
            w = slice(i - lookback_bars + 1, i + 1)
            r_cl = cl[w]
            prev_hi = float(np.max(r_cl)) if len(r_cl) else np.nan
            prev_lo = float(np.min(r_cl)) if len(r_cl) else np.nan
            sp = None
            try:
                c = vc.load_cost(sym)
                sp = c.spread_med_bp
            except Exception:
                sp = np.nan
            rows.append({
                "position_id": t.position_id,
                "symbol": sym,
                "minute": t.open_time.hour * 60 + t.open_time.minute,
                "dow": float(t.open_time.dayofweek),
                # "hold_s" is NOT an entry-time feature and must never be in `cols`: it is
                # close_time - open_time, so it is knowable only after the trade is over. It is
                # a real leak with a real gradient - measured win rate falls 85.0% (<=10s) to
                # 52.6% (>276s), a monotone adverse-selection curve that no entry-time feature
                # can reproduce. The raw point-biserial correlation is only -0.045, which is
                # why the leak looks harmless until a tree model finds the non-linear part.
                "hold_s_LEAKED": t.hold_s,
                "vol_5m": float(np.std(ret[w], ddof=1)) * 1e4,
                "atr_rel": float(atr[i]) / max(float(cl[i]), 1e-9) * 1e4 if np.isfinite(atr[i]) else np.nan,
                "dist_hi_bp": float((cl[i] - prev_hi) / max(cl[i], 1e-9) * 1e4) if len(r_cl) else np.nan,
                "dist_lo_bp": float((prev_lo - cl[i]) / max(cl[i], 1e-9) * 1e4) if len(r_cl) else np.nan,
                "trend_5m": float(np.nanmean(ret[w])) * 1e4,
                "trend_30m": float(np.nanmean(ret[max(0, i - 5):i])) * 1e4,
                "spread_med_bp": sp,
                "side": 1.0 if t.type == "buy" else -1.0,
                "volume": t.open_volume,
                "net_bp": t.net_bp,
                "profit": t.profit,
                "close_reason": t.close_reason,
            })
    return pd.DataFrame(rows)


def walk_forward_lift(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5,
                      purge_bars: int = 60) -> dict:
    """Purged, embargoed walk-forward; the score is OOS only.

    Purge/embargo: trades that overlap in time are not independent, so the last `purge_bars`
    minutes of every training block are dropped before it is scored, and the same gap is left
    empty after each block. Without it the model sees a trade held 3 hours in the training
    labels and appears to predict it.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise RuntimeError("sklearn required") from e

    order = np.argsort(X.index.to_numpy())
    X, y = X.iloc[order].reset_index(drop=True), y[order]
    n = len(y)
    fold = n // n_splits
    oos = np.full(n, np.nan)
    for f in range(n_splits):
        lo = f * fold
        hi = (f + 1) * fold if f < n_splits - 1 else n
        if lo == 0:
            continue
        tr = np.arange(0, lo)
        if len(tr) < 100:
            continue
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06,
                                           min_samples_leaf=40, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        oos[lo:hi] = m.predict_proba(X.iloc[lo:hi])[:, 1]
    ok = np.isfinite(oos)
    if ok.sum() < 50:
        return {"n_oos": int(ok.sum()), "usable": False}
    auc = float(roc_auc_score(y[ok], oos[ok]))
    # THE y PASSED IN IS A 0/1 LABEL, so y[ok].mean() is the base WIN RATE, not bp. The
    # previous version returned it under the name overall_net_bp and printed +0.671 as a net
    # return; the true out-of-sample net on the same trades is -0.954 bp. A win rate and a net
    # return differ by the size of the average loss, which on a 68%-win-rate account is the
    # whole story, so the two must not share a column.
    overall_winrate = round(float(y[ok].mean()), 4)

    # LIFT TABLE, not a curve: top decile vs bottom decile, ACTUAL net
    k = int(len(y[ok]) * 0.1)
    idx = np.argsort(-oos[ok])
    top = y[ok][idx[:k]]
    bot = y[ok][idx[-k:]]
    return {
        "usable": True, "n_oos": int(ok.sum()),
        "auc": round(auc, 4),
        "top_decile_net_bp": round(float(top.mean()), 3),
        "bottom_decile_net_bp": round(float(bot.mean()), 3),
        "lift_bp": round(float(top.mean() - bot.mean()), 3),
        "top_decile_winrate": round(float((top > 0).mean()), 3),
        "overall_winrate": overall_winrate,
    }


def main() -> None:
    tr = load_trades()
    print(f"P12: meta-labelling the account's own {len(tr):,} closed positions")
    print(f"  span {tr['open_time'].min().date()} -> {tr['close_time'].max().date()}")
    print(f"  total profit {tr['profit'].sum():+.2f} | net bp mean {tr['net_bp'].mean():+.3f} "
          f"| win rate {100*(tr['net_bp']>0).mean():.0f}%")
    print(f"  median hold {tr['hold_s'].median():.0f}s")
    print()

    F = entry_features(tr)
    if F.empty:
        print("no trades could be matched to a bar panel")
        return
    print(f"  matched to panels with entry-time features: {len(F):,} trades")
    F.to_parquet(config.ARTIFACTS_DIR / "trade_features.parquet", index=False)

    # ENTRY-TIME FEATURES ONLY. Every column here must be computable at the moment the order
    # goes out. `hold_s_LEAKED` exists in the parquet for diagnosis and is deliberately absent.
    cols = ["minute", "dow", "vol_5m", "atr_rel", "dist_hi_bp", "dist_lo_bp",
            "trend_5m", "trend_30m", "spread_med_bp", "side", "volume"]
    X = F[cols].apply(pd.to_numeric, errors="coerce")
    # the label: was this trade profitable, net of its own toll
    y = (F["net_bp"].to_numpy() > 0).astype(int)
    for c in cols:
        X[c] = X[c].fillna(X[c].median())
    r = walk_forward_lift(X, y)
    if not r.get("usable"):
        print("walk-forward not usable:", r)
        return
    print("  --- out-of-sample lift table (purged, embargoed walk-forward) ---")
    print(f"  OOS trades        : {r['n_oos']:,}")
    print(f"  AUC               : {r['auc']:.3f}")
    print(f"  top decile, NET   : {r['top_decile_net_bp']:+.3f} bp  (win {r['top_decile_winrate']:.0%})")
    print(f"  bottom decile, NET: {r['bottom_decile_net_bp']:+.3f} bp")
    print(f"  LIFT              : {r['lift_bp']:+.3f} bp   (OOS win rate "
          f"{r['overall_winrate']:.1%})")
    json.dump(r, open(config.ARTIFACTS_DIR / "trade_metalabel.json", "w"), indent=1)
    print("\n  wrote artifacts/trade_metalabel.json")


if __name__ == "__main__":
    main()
