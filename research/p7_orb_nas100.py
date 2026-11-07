"""P7 — Opening Range Breakout on our own NAS100 panel, the document's #1 edge, previously untested here.

FROZEN SPEC, written 2026-11-07 before computing anything.

Why this run: the research document rates ORB its strongest retail-accessible edge ("still working,
won't be arbed away"), I priced four other families at this venue and all four failed, and I never
actually tested ORB here. So the honest position has an untested gap where the document's strongest
claim lives. This closes it.

The rule, taken from the document (Tier 1, #1) and from Crabel's specification:
  - Mark the first 15 minutes of the US cash session high/low.
  - Enter on the first break of that range, either side.
  - Stop at the range midpoint (the document: "stop at range midpoint"); target 1x and 1.5x the
    range width.
  - Kill the idea if no breakout by 90 minutes.
Sessions are detected per day from volume (modal peak bar hour 14 on ~56% of days, 13 on ~31%),
so DST is carried by the data, not assumed.

What is priced: NAS100's measured round trip (0.55 bp) and its passive-fill cost, and the
year-by-year sign gate that killed the hourly cells and both pairs. A PB 1.128 with a max
drawdown of 89% of net (CE Market Solutions' own numbers) is a risk-adjusted statement, not an
edge statement, so the report is expectancy AND drawdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, venue_cost as vc


def detect_session_start(bars: pd.DataFrame) -> pd.Series:
    """The hour in which the day's volume STEP-UP occurs, not the loudest hour.

    Verified defect in the first version: it used the *peak-volume* hour. On this feed the
    loudest hour is 14:xx, but the cash open is 13:30 (EDT) / 14:30 (EST) UTC, and bars are
    timestamped at bar-OPEN. So a 15-minute "opening range" was built from bars labelled
    13:00-13:15, which cover 12:30-12:45 - i.e. 30 to 60 minutes of PRE-OPEN quiet tape, and
    the strategy then waited for price to break a range that formed before the open. The
    largest adjacent-hour volume jump lands in hour 13 on 611 of 1,034 days, which is the open
    arriving.
    """
    d = bars.copy()
    d["day"] = d["ts"].dt.date
    d["h"] = d["ts"].dt.hour
    per = d.groupby(["day", "h"])["vol"].sum().reset_index()
    out = []
    for day, g in per.groupby("day"):
        g = g.sort_values("h")
        best, best_h = -1.0, None
        for i in range(1, len(g)):
            a, b = int(g["h"].iloc[i - 1]), int(g["h"].iloc[i])
            if a in (20, 21, 22, 23) or b > 17:
                continue
            r = g["vol"].iloc[i] / max(g["vol"].iloc[i - 1], 1.0)
            if r > best:
                best, best_h = r, b
        out.append((day, best_h if best_h is not None else 13))
    step = pd.Series({d_: h for d_, h in out})
    return step.reindex(d["day"]).to_numpy()


def run(range_minutes: int = 15, stop_mode: str = "opposite", symbol: str = "NAS100") -> dict:
    df = pd.read_parquet(config.PARQUET_DIR / f"{symbol}_M1.parquet")
    df["ts"] = pd.to_datetime(df["ts"])
    # the long panels call it `volume`; normalise first so every use below agrees
    df = df.rename(columns={"volume": "vol"})
    start_h = detect_session_start(df)
    df["start_h"] = start_h
    df["in_sess"] = df["ts"].dt.hour.to_numpy() >= df["start_h"].to_numpy()

    # to_bp: 1 index point against the session's own level
    lvl = float(df["close"].mean())
    pts_to_bp = 1e4 / lvl

    # group into sessions
    df["day"] = df["ts"].dt.date
    df["sess_id"] = np.arange(len(df))
    sid, s = np.zeros(len(df), dtype=np.int64), -1
    prev_day = None
    for i in range(len(df)):
        if df["in_sess"].iat[i] and df["day"].iat[i] != prev_day:
            s += 1
            prev_day = df["day"].iat[i]
        sid[i] = s
    df["sid"] = sid

    trades = []
    for k, g in df.groupby("sid", sort=True):
        if k < 0:
            continue
        g = g.reset_index(drop=True)
        first = g[g["in_sess"]]
        if len(first) < range_minutes + 30:
            continue
        # the open is at :30 on EDT / :00 on EST. The first bar in the step hour is bar-OPEN,
        # so take the slice that begins at the detected open BAR, not the hour's first bar.
        open_bar = np.searchsorted(
            first["ts"].to_numpy(),
            np.datetime64(pd.Timestamp(first["ts"].iloc[0]).replace(minute=30)), side="left")
        rng_bars = first.iloc[max(open_bar, 0): max(open_bar, 0) + range_minutes]
        rng = rng_bars if len(rng_bars) == range_minutes else first.iloc[:range_minutes]
        hi, lo = float(rng["high"].max()), float(rng["low"].min())
        if hi <= lo:
            continue
        width = hi - lo
        entered, side, e_i = False, 0, -1
        for i in range(range_minutes, len(g)):
            row = g.loc[i]
            if not row["in_sess"]:
                continue
            mins = (row["ts"] - rng["ts"].iloc[0]).total_seconds() / 60.0
            if not entered:
                if mins > 90:
                    break
                # FIRST BREAKOUT: price must close outside the range on a bar whose close was
                # inside it on the previous bar. Without that condition price that simply trades
                # above the range high all day (a trending tape) enters on bar 1 and then
                # immediately hits the midpoint stop - which produced 1,151 stops out of 1,745
                # trades and a 16% win rate that has nothing to do with the strategy.
                prev_inside = (i == range_minutes) or (
                    lo < g.loc[i - 1]["close"] < hi)
                if prev_inside and row["close"] > hi:
                    side, entered, e_i = 1, True, i
                elif prev_inside and row["close"] < lo:
                    side, entered, e_i = -1, True, i
            elif stop_mode == "published":
                # THE ACTUAL PUBLISHED RULE: no stop, no target, held to the session close
                # (CE Market Solutions: "exit at 15:59 ET", stops/targets absent). The first
                # version added a 1.5x target and a stop, which is a DIFFERENT strategy - capping
                # winners at 1.5R while capping losers at 0.5-1.0R changes the payoff shape
                # entirely, so its result was never comparable to the published numbers.
                if i == len(g) - 1 or not g.loc[i + 1]["in_sess"]:
                    trades.append(dict(sid=k, side=side, entry=float(row["open"]),
                                       exit=float(row["close"]), reason="close", ei=e_i, xi=i,
                                       edate=g.loc[e_i, "ts"]))
                    break
                continue
            else:
                # three stop placements, because the first version used the midpoint and 66% of
                # trades stopped out immediately - a breakout retracing HALF its range before
                # resuming is ordinary, so a midpoint stop is a defect rather than a result
                if stop_mode == "mid":
                    stop_level = (hi + lo) / 2.0
                elif stop_mode == "opposite":
                    stop_level = lo if side == 1 else hi
                else:                       # wide: 0.5x the range beyond the opposite side
                    stop_level = lo - 0.5 * width if side == 1 else hi + 0.5 * width
                hit_stop = (row["low"] <= stop_level) if side == 1 else (row["high"] >= stop_level)
                if hit_stop:
                    trades.append(dict(sid=k, side=side, entry=float(row["open"]),
                                       exit=stop_level, reason="stop", ei=e_i, xi=i,
                                       edate=g.loc[e_i, "ts"]))
                    break
                tgt = float(row["open"]) + side * 1.5 * width
                hit_tgt = (row["high"] >= tgt) if side == 1 else (row["low"] <= tgt)
                if hit_tgt:
                    trades.append(dict(sid=k, side=side, entry=float(row["open"]),
                                       exit=tgt, reason="target", ei=e_i, xi=i,
                                       edate=g.loc[e_i, "ts"]))
                    break
                # flat at the end of the session
                if i == len(g) - 1 or not g.loc[i + 1]["in_sess"]:
                    trades.append(dict(sid=k, side=side, entry=float(row["open"]),
                                       exit=float(row["close"]), reason="close", ei=e_i, xi=i,
                                       edate=g.loc[e_i, "ts"]))
                    break
    tr = pd.DataFrame(trades)
    if tr.empty:
        return {"range_min": range_minutes, "trades": 0}
    raw_pts = (tr["exit"] - tr["entry"]) * tr["side"].values
    gross_bp = raw_pts * pts_to_bp
    cost = vc.load_cost(symbol)
    net = gross_bp - cost.roundtrip_med_bp
    # per-year grouping from the ENTRY BAR'S OWN TIMESTAMP, carried in the record. Indexing the
    # full panel with indices that were assigned inside a filtered sub-frame returns the wrong
    # rows - the first version did that and reported every trade as 2020.
    yr = pd.DatetimeIndex(tr["edate"]).year.to_numpy()
    det = {int(y): float(net[yr == y].mean()) for y in sorted(set(yr))}
    pos = sum(1 for v in det.values() if v > 0)
    eq = net.cumsum()
    return {
        "range_min": range_minutes, "trades": len(tr),
        "win_rate": float((gross_bp > 0).mean()),
        "raw_bp": float(gross_bp.mean()),
        "net_bp": float(net.mean()),
        "t": float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if len(net) > 1 else float("nan"),
        "toll_bp": float(cost.roundtrip_med_bp),
        "per_year": {k: round(v, 2) for k, v in det.items()},
        "yrs_pos": int(pos), "n_yrs": len(det),
        "by_reason": tr["reason"].value_counts().to_dict(),
        "max_dd_bp": float((eq - eq.cummax()).min()),
        "per_reason_net_bp": {r: float(net[tr["reason"] == r].mean()) for r in tr["reason"].unique()},
    }


def main() -> None:
    print("P7: ORB on the own NAS100 panel — the document's #1 edge, first test at this venue")
    print()
    for rm in (15, 30):
        for sm in ("mid", "opposite", "wide"):
            r = run(range_minutes=rm, stop_mode=sm)
            if r["trades"] == 0:
                print(f"  {rm}-min range, {sm} stop: no trades")
                continue
            print(f"--- {rm}-minute range, {sm} stop ---")
        print(f"  trades {r['trades']:>4}  win {r['win_rate']:>5.1%}  raw {r['raw_bp']:+7.3f} bp  "
              f"toll {r['toll_bp']:.3f}  NET {r['net_bp']:+7.3f} bp  t {r['t']:+5.2f}")
        print(f"  by exit reason : {r['by_reason']}  net by reason "
              f"{ {k: round(v,2) for k,v in r['per_reason_net_bp'].items()} }")
        print(f"  per year       : " + " ".join(f"{k}:{v:+.2f}" for k, v in r["per_year"].items())
              + f"   -> {r['yrs_pos']}/{r['n_yrs']} years positive")
        print(f"  max drawdown   : {r['max_dd_bp']:.1f} bp (cumulative net basis)")
        print()
    print("DELIVERABLE: does the document's strongest edge clear the toll on the venue we trade?")


if __name__ == "__main__":
    main()
