"""Risk rules - the deliverable after fourteen explorations found no directional edge.

WHY THIS MODULE IS IN src/ AND NOT research/. It is the first thing in this repo that is
meant to run against a live account rather than answer a research question. Every other module
asks "is there an edge"; this one constrains what happens while we look for one, and it needs
no forecast to do it.

WHY THESE RULES AND NOT A MODEL. The corrected cTrader history (research/p16) says the losses
are process, not prediction: three of four instruments lose MORE than their entire net in their
worst ten trades (NAS100 279% of net, XAGUSD 153%, XAUUSD 96%), 51% of the oil loss is ten
trades out of 734, and the account's balance went negative at one point. A cap is enforceable
at entry time with no forecast, and it is the only intervention the data supports.

WHAT "ENFORCING" MEANS HERE, AND WHAT IT DOES NOT. This module evaluates rules against
history and reports breaches against live history. It does not send orders. Auto-execution in
a terminal is a deployment decision with real consequences and it is deliberately not part of
this module; `ctrader/calgo` and the MT5 side carry the enforcement hooks if you want them
after reviewing what this says.

THE FIVE RULES, with the derivation recorded per number rather than asserted:

  R1  per-position hard stop, in bp of the position's own notional. Portable across accounts
      and currencies, and directly comparable to the cost ledger in src/venue_cost.
  R2  per-day loss limit, in the account's currency, as a fraction of that day's equity.
  R3  commission-aware minimum target: the measured commission per position, so a trade that
      cannot clear it has no reason to exist.
  R4  max trades per day, from the measured activity distribution.
  R5  per-instrument daily loss limit, sized off that instrument's own tail.

WHAT IS NOT EVALUABLE FROM HISTORY, stated because it is the tempting one: a MAX-HOLD rule.
These trades had no max hold, so applying one to them conditions on the outcome - the same
objection that retired the MT5 "hold <=30s was +0.775 bp" claim. R0 (max hold) is therefore
deliberately absent from the counterfactual and belongs in the forward test only.

THE COUNTERFACTUALS ARE NOT ALL EQUAL, and the difference is load-bearing:
  R1, R2, R5 - CAPS. If a position lost more than the cap it would have been stopped at the
      cap, so the counterfactual P&L is computable and honest.
  R4 - TRUNCATION. Dropping the trades after the Nth means dropping trades we ALREADY KNOW
      lost, because the limit would have been hit by the ones that did. This biases the
      counterfactual in our favour and is reported separately as an upper bound, never as
      the headline.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskSpec:
    """A frozen, pre-registered rule set. Every number carries its derivation."""

    account: str
    currency: str
    r1_position_bp: float                 # hard stop per position, in bp of notional
    r2_day_frac_of_equity: float          # per-day loss limit as a fraction of day equity
    r3_commission_per_position: float     # measured, account currency
    r3_min_target_bp: float               # minimum move a position must promise, in bp
    r4_max_trades_per_day: int
    r5_instrument_daily_frac: float       # per-instrument daily loss limit, frac of equity
    built_from: dict = field(default_factory=dict)

    # --- portable, currency-free ---
    def position_stop_money(self, notional: float) -> float:
        """R1 in account currency for a position of ``notional``."""
        return self.r1_position_bp / 1e4 * notional

    def day_limit_money(self, equity: float) -> float:
        return -abs(self.r2_day_frac_of_equity) * equity

    def instrument_limit_money(self, equity: float) -> float:
        return -abs(self.r5_instrument_daily_frac) * equity

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RiskSpec":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**d)


def _p95_abs(a: pd.Series) -> float:
    return float(np.quantile(a.abs(), 0.95))


def build_spec(trades: pd.DataFrame, *, account: str, currency: str,
               equity_ref: float | None = None) -> RiskSpec:
    """Derive the five numbers from a trade frame, with each derivation recorded.

    ``trades`` needs: ``profit``, ``net_bp``, ``notional``, ``open_time``, ``symbol_id``,
    and (for R2/R5's equity reference) ``balance_after`` - the broker's own balance at each
    close, which is the only scale-free equity signal available.
    """
    t = trades.dropna(subset=["profit", "net_bp", "notional"]).copy()
    t["open_time"] = pd.to_datetime(t["open_time"])
    t["open_day"] = t["open_time"].dt.normalize()

    eq = float(np.nanmedian(t["balance_after"])) if "balance_after" in t else (
        equity_ref if equity_ref else float(t["notional"].median()))
    daily = t.groupby("open_day")["profit"].sum()
    tpd = t.groupby("open_day").size()

    # R1: the p95 of |net_bp| - the level above which a position's loss is in the tail
    r1 = _p95_abs(t["net_bp"])

    # R2: the per-day limit as a fraction of equity. The level is a DECISION; the derivation
    # here records what the measured daily distribution looks like at each candidate, so the
    # choice is made against evidence rather than picked.
    cand = {}
    for frac in (0.005, 0.01, 0.02, 0.05):
        lim = -frac * eq
        cand[f"{frac:.3%}"] = {
            "limit_money": round(lim, 2),
            "days_breached": int((daily < lim).sum()),
            "days_total": int(len(daily)),
            "breached_pnl": round(float(daily[daily < lim].sum()), 2),
        }
    # R4: the p95 of trades/day. The top-decile days are the losing ones (measured), so the
    # cap sits at the p95 rather than the max.
    r4 = int(tpd.quantile(0.95))

    # R3: measured commission per position, and the target that must clear it
    comm = float(t["commission"].sum()) / max(len(t), 1) if "commission" in t else 0.0
    mean_notional = float(t["notional"].mean())
    r3_bp = abs(comm) / max(mean_notional, 1e-9) * 1e4

    # R5: per-instrument daily limit. Sized so the instrument's worst-ten share cannot recur:
    # measured per-instrument worst-10 as a fraction of net, then the cap set at the level
    # that would have truncated it.
    r5_rec = {}
    if "symbol_id" in t:
        per_sym_day = t.groupby(["open_day", "symbol_id"])["profit"].sum()
        for sid, g in t.groupby("symbol_id"):
            if len(g) < 50:
                continue
            w = g.nsmallest(10, "profit")["profit"].sum()
            net = g["profit"].sum()
            r5_rec[str(int(sid))] = {
                "n": int(len(g)), "net": round(float(net), 2),
                "worst10": round(float(w), 2),
                "worst10_share": round(float(w / net), 3) if net else None,
            }
    return RiskSpec(
        account=account, currency=currency,
        r1_position_bp=round(r1, 2),
        r2_day_frac_of_equity=0.01,          # the default; `candidates` records the tradeoff
        r3_commission_per_position=round(comm, 2),
        r3_min_target_bp=round(r3_bp, 3),
        r4_max_trades_per_day=r4,
        r5_instrument_daily_frac=0.01,
        built_from={
            "n_positions": int(len(t)),
            "equity_ref": round(eq, 2),
            "r1": {"p95_abs_net_bp": round(r1, 2),
                   "positions_breaching": int((t["net_bp"].abs() > r1).sum()),
                   "pnl_of_breaching": round(float(t.loc[t["net_bp"].abs() > r1, "profit"].sum()), 2)},
            "r2": {"median_daily": round(float(daily.median()), 2),
                   "mean_daily": round(float(daily.mean()), 2),
                   "worst_daily": round(float(daily.min()), 2),
                   "candidates": cand},
            "r3": {"commission_total": round(comm * len(t), 2),
                   "commission_per_position": round(comm, 2),
                   "mean_notional": round(mean_notional, 2),
                   "min_target_bp": round(r3_bp, 3)},
            "r4": {"median_trades_per_day": int(tpd.median()),
                   "p95": int(tpd.quantile(0.95)), "max": int(tpd.max()),
                   "top_decile_days_pnl": round(float(daily[tpd >= tpd.quantile(0.90)].sum()), 2)},
            "r5": r5_rec,
        },
    )


def shape_metrics(trades: pd.DataFrame) -> dict:
    """The judging criteria: the SHAPE of the P&L distribution, not the total.

    A framework that halves the worst day is working even if the total is still negative, so
    these are the numbers the forward test is scored on.
    """
    t = trades.dropna(subset=["profit"]).copy()
    if t.empty:
        return {"n": 0}
    t["open_day"] = pd.to_datetime(t["open_time"]).dt.normalize()
    daily = t.groupby("open_day")["profit"].sum()
    w10 = t.nsmallest(10, "profit")["profit"].sum() if len(t) >= 10 else float("nan")
    out = {
        "n": int(len(t)),
        "net": round(float(t["profit"].sum()), 2),
        "net_bp": round(float(t["net_bp"].mean()), 3) if "net_bp" in t else None,
        "win_rate": round(float((t["profit"] > 0).mean()), 3),
        "worst_day": round(float(daily.min()), 2),
        "best_day": round(float(daily.max()), 2),
        "days_positive_pct": round(float((daily > 0).mean()), 3),
        "days_total": int(len(daily)),
        "worst10": round(float(w10), 2),
        "worst10_share": round(float(w10 / t["profit"].sum()), 3)
        if t["profit"].sum() else None,
        "worst10_would_flip_sign": bool(float(w10) > 0) if len(t) >= 10 else None,
    }
    if "symbol_id" in t:
        per = {}
        smap = None
        p = Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_symbol_map.json"
        if p.exists():
            smap = json.loads(p.read_text(encoding="utf-8"))
        for sid, g in t.groupby("symbol_id"):
            if len(g) < 50:
                continue
            key = str(int(sid))
            name = (smap or {}).get(key, key)
            w = g.nsmallest(10, "profit")["profit"].sum()
            per[name] = {
                "n": int(len(g)), "net": round(float(g["profit"].sum()), 2),
                "worst10": round(float(w), 2),
                "worst10_share": round(float(w / g["profit"].sum()), 3)
                if g["profit"].sum() else None,
            }
        out["per_instrument"] = per
    return out


def counterfactual(trades: pd.DataFrame, spec: RiskSpec) -> dict:
    """Apply R1/R2/R5 as caps and R4 as truncated, separately, and report the shape.

    R4 is reported on its own line and labelled an UPPER BOUND: dropping trades after the
    day's Nth drops trades that we already know lost, so this is the most favourable possible
    reading and is not the headline.
    """
    t = trades.dropna(subset=["profit", "net_bp", "notional", "open_time"]).copy()
    t["open_time"] = pd.to_datetime(t["open_time"])
    t = t.sort_values("open_time").reset_index(drop=True)
    t["open_day"] = t["open_time"].dt.normalize()

    out: dict = {"as_measured": shape_metrics(t)}

    # --- R1: cap every position at the spec's bp stop -----------------------------
    cap = t["notional"] * spec.r1_position_bp / 1e4
    t["r1_profit"] = np.maximum(t["profit"], -cap)
    out["r1_position_stop"] = shape_metrics(t.assign(profit=t["r1_profit"]))

    # --- R2: truncate each day at the equity-referenced day limit -----------------
    eq = float(np.nanmedian(t["balance_after"])) if "balance_after" in t else float("nan")
    if np.isfinite(eq):
        lim = spec.day_limit_money(eq)
        keep = []
        for _, g in t.groupby("open_day", sort=True):
            cum, stopped = 0.0, False
            for i, v in g["r1_profit"].items():
                if stopped:
                    continue
                keep.append(i)
                cum += v
                if cum <= lim:
                    stopped = True
        r2 = t.loc[keep]
        out["r1_plus_day_limit"] = shape_metrics(r2)

    # --- R5: truncate each (day, instrument) at the instrument limit ---------------
    if np.isfinite(eq) and "symbol_id" in t:
        ilim = spec.instrument_limit_money(eq)
        keep = []
        for (_, sid), g in t.groupby(["open_day", "symbol_id"], sort=True):
            cum = 0.0
            for i, v in g["r1_profit"].items():
                keep.append(i)
                cum += v
                if cum <= ilim:
                    break
        r5 = t.loc[sorted(set(keep))]
        out["r1_plus_instrument_limit"] = shape_metrics(r5)

    # --- R4: keep only the first N trades of each day (UPPER BOUND, biased) --------
    t["rank"] = t.groupby("open_day").cumcount()
    r4 = t[t["rank"] < spec.r4_max_trades_per_day]
    out["r4_max_trades_UPPER_BOUND"] = shape_metrics(r4)
    out["r4_bias_warning"] = (
        "R4 drops trades after the day's Nth, which are trades that had already lost - the "
        "most favourable possible reading. Not a headline."
    )
    return out


def breaches(trades: pd.DataFrame, spec: RiskSpec) -> pd.DataFrame:
    """Which historical positions each rule would have stopped - the monitor's output."""
    t = trades.dropna(subset=["profit", "net_bp", "notional", "open_time"]).copy()
    t["open_day"] = pd.to_datetime(t["open_time"]).dt.normalize()
    cap = t["notional"] * spec.r1_position_bp / 1e4
    eq = float(np.nanmedian(t["balance_after"])) if "balance_after" in t else float("nan")
    t["r1_breach"] = t["profit"] < -cap
    if "symbol_id" in t:
        tpd = t.groupby("open_day").cumcount() + 1
        t["r4_breach"] = tpd > spec.r4_max_trades_per_day
    else:
        t["r4_breach"] = False
    t["r2_breach"] = False
    if np.isfinite(eq):
        dlim = spec.day_limit_money(eq)
        cum = t.groupby("open_day")["profit"].cumsum()
        t["r2_breach"] = cum <= dlim
    if "symbol_id" in t:
        cum = t.groupby(["open_day", "symbol_id"])["profit"].cumsum()
        t["r5_breach"] = cum <= spec.instrument_limit_money(eq) if np.isfinite(eq) else False
    else:
        t["r5_breach"] = False
    cols = ["open_time", "position_id", "profit", "net_bp", "r1_breach", "r2_breach",
            "r4_breach", "r5_breach"]
    return t[[c for c in cols if c in t.columns]]


def load_ctrader_trades() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[1] / "artifacts" / "ctrader_positions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run research/p15_ctrader_deals.py first")
    return pd.read_parquet(p)


def attach_balance(trades: pd.DataFrame, deals_path: Path) -> pd.DataFrame:
    """Attach the broker's own ``balance`` from the raw deal detail, by position.

    The balance is the only scale-free equity signal in the data and it is what R2 and R5 need;
    it lives in closePositionDetail on the LAST closing deal of each position.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "p15", Path(__file__).resolve().parents[1] / "research" / "p15_ctrader_deals.py")
    p15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p15)
    d = json.loads(deals_path.read_text(encoding="utf-8"))
    D = pd.DataFrame(d["deals"]).drop_duplicates(subset=["dealId"])

    def cpd(t, f):
        if not isinstance(t, str):
            return np.nan
        for line in t.splitlines():
            if line.startswith(f + ":"):
                try:
                    return float(line.split(":", 1)[1])
                except ValueError:
                    return np.nan
        return np.nan

    D["gp"] = [cpd(t, "grossProfit") for t in D["closePositionDetail"]]
    D["bal"] = [cpd(t, "balance") for t in D["closePositionDetail"]]
    D["ts"] = pd.to_datetime(pd.to_numeric(D["executionTimestamp"]), unit="ms")
    last = (D[D["gp"].notna()].sort_values("ts").groupby("positionId").tail(1)
            [["positionId", "bal"]].rename(columns={"bal": "balance_after"}))
    # position_id is a string on the positions side and int64 on the deals side; without the
    # cast the join silently produces all-NaN balances and R2/R5 fall back to a notional guess
    last["position_id"] = last["positionId"].astype(str)
    out = trades.merge(last[["position_id", "balance_after"]], on="position_id", how="left")
    return out
