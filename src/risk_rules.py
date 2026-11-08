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
  R1, R2, R5 - CAPS, AND THEIR EVALUATION IS PATH-UNAWARE, WHICH IS THE DEFECT THIS MODULE
      HAD. Flooring the final P&L at the cap keeps every position that dipped below the cap and
      later recovered - exactly the positions a real stop would have cut. Verified on a
      zero-skill synthetic book with negative expectancy by construction: the floor scored
      +2,120 where the path-aware evaluation scored -798 (research/p19). The path-aware
      evaluator is research/p19_path_aware_stops.py; nothing here should be used as a result
      until it is scored that way.
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

#: cTrader money fields are in CENTS on this account. Verified live against ProtoOATraderReq on
#: account 47477688: ``balance: 54`` with ``moneyDigits: 2``, so the live balance is **$0.54**
#: and the history's 112,508 is **$1,125.08**. ``leverageInCents: 50000`` is **500x levered**.
#: Every USD figure this repo quoted before this line was 100x too large; net -602,934 is
#: **-$6,029.34**, mean notional 4,043,835 is **$40,438**, median balance 6,272 is **$62.72**.
MONEY_DIVISOR = 100.0


def load_money(cents: float) -> float:
    """Convert a cTrader money field from cents to the deposit currency."""
    return float(cents) / MONEY_DIVISOR


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


# ---------------------------------------------------------------------------
# Forward test - frozen spec, judged on shape, with a day-clustered null
# ---------------------------------------------------------------------------

def forward_test(symbol_to_toll_bp: dict[str, float], *, cap_multiple: float = 3.0,
                 equity_ref: float = 0.0, out_dir=None) -> Path:
    """Write the frozen regulation spec the live monitor runs against.

    ``symbol_to_toll_bp`` is per instrument at per-hour granularity when available, so the cap
    is tighter in the hours the ledger says are cheap. It carries NO fitted numbers: every value
    is either a measured venue toll or the frozen multiplier.
    """
    spec = {
        "version": 1,
        "cap_multiple": cap_multiple,
        "cap_bp": {s: round(cap_multiple * tol, 4) for s, tol in symbol_to_toll_bp.items()},
        "equity_ref": equity_ref,
        # the judging criteria, fixed before the run
        "judge_on": ["days_positive_pct", "worst_day", "worst10_share", "trades_per_day_kept"],
        "target": {"days_positive_pct_min": 0.50,
                   "worst10_share_max": 0.50,
                   "book_untouched_min": 0.80},
        "null": "day-clustered sign-flip, per day, 200 worlds",
    }
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "artifacts"
    if not out_dir.exists():
        raise FileNotFoundError(
            f"spec output directory {out_dir} does not exist - create it or point out_dir at a "
            "directory that does. This function does not silently mkdir: a caller that hands it "
            "an unchecked path is exactly how a spec ends up half-written or missing.")
    p = out_dir / "forward_spec.json"
    p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return p


def evaluate_forward(trades: pd.DataFrame, spec: dict, *, by_hour: bool = True,
                     toll_by_hour: dict | None = None) -> dict:
    """Score a book against the frozen forward spec.

    *** THIS EVALUATION IS PATH-UNAWARE AND ITS HEADLINE NUMBERS ARE NOT RESULTS. ***

    It floors each position's final P&L at the cap. A stop fires when the price path TOUCHES
    the level, so this keeps every position that dipped below the cap and recovered - the ones
    a real stop would have cut. The zero-skill property test proves it: on a synthetic book
    built to have negative expectancy by construction, this function returns a POSITIVE net.

    Use ``research/p19_path_aware_stops.path_aware`` for any claim that will be acted on. This
    remains only as the place the frozen spec is defined, and for the property test to fail.
    """
    """Score a book against the frozen forward spec, and null it by day.

    The null keeps the day-clustered structure honest: permuting the sign of whole days is the
    same construction the rest of this repo uses. It stops an unclustered null from calling a
    coin-flip day count significant (on 84 days the unclustered null sits at ~50%). It CANNOT
    catch an evaluation artifact - the path-unaware floor passed this null - so a null result
    here is necessary, never sufficient.
    """
    t = trades.dropna(subset=["profit", "notional", "open_time"]).copy()
    t["open_time"] = pd.to_datetime(t["open_time"])
    t["open_day"] = t["open_time"].dt.normalize()
    if "sym" not in t and "symbol" in t:
        t["sym"] = t["symbol"]
    cap_bp = spec["cap_bp"]

    # THE CAP, ONE WAY. A position is FLOORED at -(cap x notional): the loss cannot exceed
    # the cap, the gain is untouched. The first version of this used np.minimum, which capped
    # the WINNERS and turned a +1.21M book into -3.73M - and it passed a smoke run because the
    # number it printed was plausible-looking. np.maximum(profit, -cap) is the only correct
    # form.
    def cap_bp_of(row) -> float:
        sym = str(row.get("sym") or "")
        bp = cap_bp.get(sym)
        if bp is None:
            return np.inf
        if toll_by_hour and (sym, row["open_time"].hour) in toll_by_hour:
            bp = toll_by_hour[(sym, row["open_time"].hour)]
        return bp

    # symbols without a measured toll in the spec are NOT capped - they keep their own P&L.
    # Mapping them to NaN silently is how the first version produced a +nan net and a 98%
    # win rate: shape_metrics then averaged over NaN-poisoned profits.
    profit = t["profit"].to_numpy(dtype=float)
    notional = t["notional"].to_numpy(dtype=float)
    with np.errstate(over="ignore"):
        bpv = np.array([cap_bp_of(r) for _, r in t.iterrows()], dtype=float)
    has_cap = np.isfinite(bpv) & (bpv > 0) & np.isfinite(notional)
    cap_money = np.where(has_cap, bpv / 1e4 * notional, np.inf)
    capped = np.maximum(profit, -cap_money)          # floor the loss, never the gain
    t["capped_profit"] = np.where(has_cap, capped, profit)
    t["r1_breach"] = has_cap & (profit < -cap_money)

    # "kept" here means the trade still happened with its original P&L, NOT that it traded at
    # all - every position in this book was already executed. A cap truncates a loss, it does
    # not prevent an entry, so this measures how much of the book the rules left untouched and
    # renamed from the misleading "trades_per_day_kept" it was first called.
    kept_frac = float((t["profit"] == t["capped_profit"]).mean())
    traded_frac = 1.0   # all positions executed; the cap is a stop, not an entry filter

    res = shape_metrics(t.assign(profit=t["capped_profit"]))
    res["book_untouched_frac"] = round(kept_frac, 3)
    res["trades_executed_frac"] = round(traded_frac, 3)
    # positions_capped is the count whose P&L the cap actually changed, not the count that
    # carried a cap in the spec - the first version printed 3,356 because it counted has_cap
    res["positions_capped"] = int((t["profit"].to_numpy() != t["capped_profit"].to_numpy()).sum())

    # day-clustered sign-flip null on the capped book's daily totals
    daily = t.assign(profit=t["capped_profit"]).groupby("open_day")["capped_profit"].sum()
    rng = np.random.default_rng(20261107)
    obs = float((daily > 0).mean())
    dist = []
    for _ in range(200):
        sgn = pd.Series(rng.choice([-1.0, 1.0], len(daily)), index=daily.index)
        dist.append(float(((daily * sgn) > 0).mean()))
    dist = np.array(dist)
    res["days_positive"] = round(obs, 4)
    res["null_days_positive_mean"] = round(float(dist.mean()), 4)
    res["null_days_positive_p95"] = round(float(np.percentile(dist, 95)), 4)
    res["days_positive_null_percentile"] = round(float((dist < obs).mean()), 4)

    tg = spec.get("target", {})
    res["meets_target"] = {
        "days_positive_pct": res.get("days_positive_pct", obs) >= tg.get("days_positive_pct_min", 0.5),
        "worst10_share": (res.get("worst10_share") or 0) <= tg.get("worst10_share_max", 0.5),
        "book_untouched": kept_frac >= tg.get("book_untouched_min", 0.8),
    }
    return res


# ---------------------------------------------------------------------------
# Sizing - the finding the framework missed, and the one that matters most
# ---------------------------------------------------------------------------

def leverage_profile(trades: pd.DataFrame) -> dict:
    """Size versus equity, measured. This is what the cap framework never looked at.

    A stop in bp of notional says nothing about size relative to the account, and size is the
    actual mechanism of ruin. On the corrected cTrader book: mean notional 4,043,835 against a
    median balance of 6,272 (645x), a median per-position loss of 7.0% of balance, a p95 of
    67.2%, and R1's 37.9 bp stop sizing a single position at 244% of the account.
    """
    t = trades.dropna(subset=["profit", "notional", "balance_after"]).copy()
    if t.empty:
        return {"n": 0}
    bal = t["balance_after"].to_numpy(float)
    notional = t["notional"].to_numpy(float)
    loss_pct = np.where(bal > 0, -t["profit"].to_numpy(float) / np.where(bal > 0, bal, np.nan) * 100.0,
                        np.nan)
    l = loss_pct[t["profit"].to_numpy() < 0]
    l = l[np.isfinite(l)]
    return {
        "n": int(len(t)),
        "median_balance": round(float(np.nanmedian(bal)), 2),
        "mean_notional": round(float(notional.mean()), 2),
        "median_notional": round(float(np.nanmedian(notional)), 2),
        "leverage_mean_notional_over_median_balance": round(
            float(notional.mean() / max(np.nanmedian(bal), 1e-9)), 1),
        "median_loss_pct_of_balance": round(float(np.median(l)), 2) if len(l) else None,
        "p95_loss_pct_of_balance": round(float(np.quantile(l, 0.95)), 2) if len(l) else None,
        "max_loss_pct_of_balance": round(float(np.nanmax(l)), 1) if len(l) else None,
    }


def size_for(equity: float, *, risk_frac: float, stop_bp: float,
             max_leverage: float = 10.0, min_notional: float = 0.0) -> float:
    """Position notional that risks ``risk_frac`` of equity over a ``stop_bp`` stop.

    risk_frac 0.005 with a 10 bp stop on 1,125 of equity gives notional
    1125 * 0.005 / (10/1e4) = 5,625 - against the 40k the account actually traded.

    ``min_notional`` is the VENUE's minimum tradable notional (gold's 0.01 lot is about \$380).
    If the sizing rule lands below it, the correct answer is 0 - DO NOT TRADE - because the
    venue floor would force a size many times the intended risk. The live monitor on a \$0.54
    account computed \$2.70 of notional and would have been unable to trade it at all; returning
    0 with a reason is the honest output, and it is what keeps a small account from being forced
    into an oversized position by a rounding floor.
    """
    if stop_bp <= 0:
        raise ValueError("stop_bp must be positive")
    risk_money = equity * risk_frac
    notional = risk_money / (stop_bp / 1e4)
    notional = min(notional, equity * max_leverage)
    if notional < min_notional:
        return 0.0
    return notional


def sizing_recommendation(equity: float, *, stop_bp: float,
                           risk_frac: float = 0.005,
                           max_leverage: float = 10.0,
                           min_notional: float = 0.0) -> dict:
    """With ``tradable``: False when the rule lands below the venue's minimum notional.

    The monitor returns that instead of a number, because the alternative - rounding up to the
    venue minimum - silently trades a size the risk rule never sanctioned.
    """
    n = size_for(equity, risk_frac=risk_frac, stop_bp=stop_bp,
                 max_leverage=max_leverage, min_notional=min_notional)
    return {
        "equity": round(equity, 2),
        "stop_bp": round(stop_bp, 3),
        "risk_frac": risk_frac,
        "max_leverage": max_leverage,
        "min_notional": min_notional,
        "notional": round(n, 2),
        "tradable": n > 0,
        "reason": "" if n > 0 else (
            f"equity {equity:.2f} at {risk_frac:.3%} risk over a {stop_bp:.1f}bp stop gives "
            f"notional below the venue minimum {min_notional:.2f} - do not trade"),
        "risk_money_per_trade": round(equity * risk_frac, 2),
        "worst_day_at_this_size": round(equity * risk_frac * 5, 2),
    }


def kill_switch(equity_history: list[float], *, peak_dd_frac: float = 0.25) -> dict:
    """Account-level drawdown stop on REALISED + FLOATING equity, in event order.

    The rules that ran on final-P&L accumulations cannot see this, which is why the account
    reached -13 against a 1% day limit: the limit was checked on the wrong series.
    """
    e = np.asarray(equity_history, dtype=float)
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    peak = np.maximum.accumulate(e)
    dd = 1.0 - e / np.where(peak > 0, peak, np.nan)
    i = int(np.nanargmax(dd)) if np.isfinite(dd).any() else 0
    return {
        "n": len(e),
        "start": round(float(e[0]), 2),
        "trough": round(float(e[i]), 2),
        "trough_index": i,
        "max_drawdown_pct": round(float(np.nanmax(dd) * 100), 1),
        "drawdown_stop_triggered": bool(np.nanmax(dd) >= peak_dd_frac),
        "drawdown_stop_pct": peak_dd_frac * 100,
    }
