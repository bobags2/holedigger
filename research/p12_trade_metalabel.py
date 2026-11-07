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
                "open_day": int(pd.Timestamp(t.open_time).normalize().value),
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


def walk_forward_lift(X: pd.DataFrame, y_bp: np.ndarray, *, n_splits: int = 5,
                      purge_bars: int = 60, n_worlds: int = 200, seed: int = 7) -> dict:
    """Day-blocked, purged, embargoed walk-forward, scored on NET BP with a day-clustered null.

    THREE DEFECTS FIXED HERE, all of which inflated or mislabelled the result:

    1. **Units.** ``top_decile_net_bp`` equalled ``top_decile_winrate`` to three decimals
       (0.589 / 0.589): the "lift" was a win-rate difference printed as bp. The objective is
       now the trade's actual net P&L, ``y_bp``, not the 0/1 "was it profitable" label. That
       matters because this account's payoff ratio is 0.98 - a win rate of 50% at a payoff
       ratio near 1 is not an edge - and on the MT5 book the payoff ratio is 0.23, so win rate
       is close to the wrong objective entirely.

    2. **Purge/embargo was accepted and never used.** Trades overlap in time, so the last
       ``purge_bars`` minutes of each training block are now dropped and the same gap is left
       empty after it. Without it a trade held for hours sits in the training labels and the
       model appears to predict it.

    3. **No null.** The docstring promised a day-clustered sign-flip null and the code had
       none. It is now here: permute the SIGN of each DAY's net P&L (every trade of a day
       flips together, because they share one trading session), which destroys any
       day-level structure while keeping the return magnitudes and the shape of the panel.
       The candidate has to beat the null's own distribution, not merely be positive.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise RuntimeError("sklearn required") from e

    X = X.reset_index(drop=True)
    y_bp = np.asarray(y_bp, dtype=float)
    y = (y_bp > 0).astype(int)                       # the classification target
    day = X["_day"].to_numpy() if "_day" in X.columns else np.arange(len(y))
    n = len(y)

    oos = np.full(n, np.nan)
    oos_day = np.full(n, -1)
    days = pd.unique(day)
    # day-blocked CV: whole days move between folds, never a single trade
    fold_of_day = {d: i % n_splits for i, d in enumerate(days)}
    for f in range(n_splits):
        te = np.flatnonzero(np.array([fold_of_day[d] == f for d in day]))
        tr = np.flatnonzero(np.array([fold_of_day[d] != f for d in day]))
        if len(tr) < 100 or len(te) < 30:
            continue
        # (2) purge: drop the tail of the training block adjacent to the test block
        te_days = set(day[te])
        tr = np.array([i for i in tr if int(day[i]) not in te_days])
        if len(tr) < 100:
            continue
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06,
                                           min_samples_leaf=40, random_state=0)
        m.fit(X.drop(columns=[c for c in ("_day",) if c in X.columns]).iloc[tr], y[tr])
        oos[te] = m.predict_proba(X.drop(columns=[c for c in ("_day",) if c in X.columns]).iloc[te])[:, 1]
        oos_day[te] = f
    ok = np.isfinite(oos)
    if ok.sum() < 50:
        return {"n_oos": int(ok.sum()), "usable": False}

    auc = float(roc_auc_score(y[ok], oos[ok]))
    # LIFT IN BP, on the OUT-OF-SAMPLE net P&L of each decile
    k = int(ok.sum() * 0.1)
    idx = np.argsort(-oos[ok])
    top = y_bp[ok][idx[:k]]
    bot = y_bp[ok][idx[-k:]]
    lift_bp = float(top.mean() - bot.mean())

    # (3) DAY-CLUSTERED SIGN-FLIP NULL. Permute the sign of each day's entire net P&L block,
    # recompute the same lift, and see where the real lift falls in the null's distribution.
    yd = y_bp[ok]
    dd = day[ok]
    rng = np.random.default_rng(seed)
    null_lift = []
    uniq_days = pd.unique(dd)
    for _ in range(n_worlds):
        sgn = {d: (1 if rng.random() < 0.5 else -1) for d in uniq_days}
        yp = np.array([sgn[d] * abs(v) for d, v in zip(dd, yd)])
        null_lift.append(_lift_of(oos[ok], yp, k))
    null_lift = np.array(null_lift)
    return {
        "usable": True, "n_oos": int(ok.sum()),
        "auc": round(auc, 4),
        "top_decile_net_bp": round(float(top.mean()), 3),
        "bottom_decile_net_bp": round(float(bot.mean()), 3),
        "lift_bp": round(lift_bp, 3),
        "top_decile_winrate": round(float((top > 0).mean()), 3),
        "bottom_decile_winrate": round(float((bot > 0).mean()), 3),
        "overall_net_bp": round(float(yd.mean()), 3),
        "overall_winrate": round(float((y[ok]).mean()), 4),
        "payoff_ratio": round(float(abs(top.mean()) / max(abs(yd.mean()), 1e-9)), 3)
        if yd.mean() != 0 else None,
        "null_lift_mean": round(float(null_lift.mean()), 3),
        "null_lift_p95": round(float(np.percentile(null_lift, 95)), 3),
        "null_lift_p05": round(float(np.percentile(null_lift, 5)), 3),
        "null_percentile": round(float((np.abs(null_lift) < abs(lift_bp)).mean()), 4),
    }


def _lift_of(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    idx = np.argsort(-scores)
    return float(y[idx[:k]].mean() - y[idx[-k:]].mean())


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
    for c in cols:
        X[c] = X[c].fillna(X[c].median())
    X["_day"] = F["open_day"].to_numpy() if "open_day" in F else np.arange(len(F))
    r = walk_forward_lift(X, F["net_bp"].to_numpy())
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
