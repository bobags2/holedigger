"""cTrader deal history - the account that was never analysed.

WHY THIS EXISTS. The P12 meta-labelling ran on 3,870 closed positions pulled from the LOCAL
MT5 bridge (verified: login 281766, "Jadon Yeo 001", FusionMarkets-Live, real, CAD, balance
0.0 - so those positions are historical and the account is now empty). The cTrader bridge was
never queried for history at all, and cTrader is the more-used account. The meta-labeling
result - AUC 0.512 after the hold_s leak was removed - is therefore a statement about ONE
account's fills, and the more traded one has never been looked at.

The compiled protobufs already carry PROTO_OA_DEAL_LIST_REQ/RES and
PROTO_OA_DEAL_LIST_BY_POSITION_ID_REQ/RES; this module wires them up rather than adding a
convention to the bridge.

WHAT COMES BACK and what to expect. cTrader's Deal list gives (ctidTraderAccountId,
positionId, volume, filledVolume, symbolId, createTimestamp, executionTimestamp,
executionPrice, baseToUsdConversionRate, closePositionDetail{...}). Two entries per round
trip - the opening deal and the closing deal - so the position-level P&L is the sum of the two
sides, then converted to bp of notional. That is a different shape from MT5's one-row-per-
closed-position, so the loader normalises to the same schema as
research/p12_trade_metalabel.load_trades so everything downstream is reusable.

The broker pricing in bp is `executionPrice` adjusted by `baseToUsdConversionRate`, which makes
cross-currency P&L comparable; the volume is in the symbol's contract units.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))               # holedigger (for artifacts/ + src/)
sys.path.insert(0, str(_ROOT.parent / "ctrader"))   # the cTrader bridge repo

import json

import numpy as np
import pandas as pd

from ctbridge import client as cb  # exposes OpenApiClient
from ctbridge.messages import OpenApiMessages_pb2 as oa


def _pb_fields(msg) -> dict:
    return {f.name: getattr(msg, f.name) for f in msg.DESCRIPTOR.fields}


async def pull_deals(days: int = 730, out: Path | None = None) -> list[dict]:
    """Authenticate against the live cTrader bridge and page the deal list."""
    root = _ROOT.parent / "ctrader"
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    t = cfg["ctrader"]

    # the constructor is keyword-only and carries the tokens explicitly
    cl = cb.OpenApiClient(
        client_id=t["client_id"], client_secret=t["client_secret"],
        access_token=t["access_token"], account_id=t.get("account_id"), live=bool(t.get("live", True)),
    )
    await cl.start()      # start() authenticates and resolves account_id itself
    try:
        acct = cl.account_id or t.get("account_id")
        if acct is None:
            raise SystemExit("no account id resolved from the cTrader auth response")
        print(f"authenticated as cTrader account {acct}")

        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        to_ms = int(now.timestamp() * 1000)
        from_ms = int((now - timedelta(days=days)).timestamp() * 1000)

        deals: list[dict] = []
        # PAGE, and sub-page when a window TRUNCATES. Verified on the first run: every 30-day
        # window from 2026-05-31 on returned exactly maxRows, so the last five months were cut
        # off at 1,000 deals and the tail was missing without any error. A window that returns
        # exactly maxRows is not "complete", it is "there may be more".
        async def _page(lo_ms: int, hi_ms: int, depth: int = 0) -> None:
            req = oa.ProtoOADealListReq()
            req.ctidTraderAccountId = int(acct)
            req.fromTimestamp = int(lo_ms)
            req.toTimestamp = int(hi_ms)
            req.maxRows = 1000
            res = await cl.send(req)
            got = list(getattr(res, "deal", []))
            for d in got:
                deals.append(_pb_fields(d))
            # split in half while the window is saturating and still spans more than an hour
            if len(got) >= 1000 and (hi_ms - lo_ms) > 3600_000:
                mid = (lo_ms + hi_ms) // 2
                await _page(lo_ms, mid, depth + 1)
                await _page(mid, hi_ms, depth + 1)

        step = 30 * 86400 * 1000
        lo = from_ms
        npwin = 0
        while lo < to_ms:
            hi = min(lo + step, to_ms)
            before = len(deals)
            await _page(lo, hi)
            npwin += 1
            print(f"  {datetime.utcfromtimestamp(lo/1000).date()} .. "
                  f"{datetime.utcfromtimestamp(hi/1000).date()}: {len(deals)-before:,} deals")
            lo = hi

        if out:
            out.write_text(json.dumps({"account_id": acct, "deals": deals}, default=str), encoding="utf-8")
        return deals
    finally:
        await cl.close()


def _cpd(d: "pd.DataFrame", field: str) -> "np.ndarray":
    """Pull ``field`` out of cTrader's newline-embedded ``closePositionDetail`` strings.

    The field is a single string like ``"entryPrice: 29758.22\ngrossProfit: -740\nswap: 0\n..."``,
    so it needs line-splitting, not a column read.
    """
    out = np.full(len(d), np.nan)
    col = d.get("closePositionDetail")
    if col is None:
        return out
    for i, t in enumerate(col):
        if not isinstance(t, str):
            continue
        for line in t.splitlines():
            if line.startswith(field + ":"):
                try:
                    out[i] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    out[i] = np.nan
                break
    return out


def to_positions(deals: list[dict]) -> pd.DataFrame:
    """Positions from the deal list, in the same shape the MT5 loader produces.

    REBUILT AFTER READING THE ACTUAL SHAPE, because the first version returned 0 rows from
    13,334 real deals. Three things the column view hides:

    1. ``dealStatus`` is the integer ``2`` for every row - not the word FILLED - so a
       "contains FILLED" filter matches nothing and silently empties the frame.
    2. Positions have 4 fills, not 2, for 2,391 of them (partial and scale-out exits), so an
       "open + close pair" assumption drops about 71% of the history.
    3. ``closePositionDetail`` carries the broker's own ``grossProfit``, which is correct under
       partial fills, so reconstructing P&L from prices is both unnecessary and wrong.
    """
    if not deals:
        return pd.DataFrame()
    d = pd.DataFrame(deals)
    d = d[pd.to_numeric(d.get("dealStatus"), errors="coerce") == 2].copy()
    for c in ("positionId", "symbolId", "tradeSide", "filledVolume", "volume", "executionPrice",
              "executionTimestamp"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ts"] = pd.to_datetime(d["executionTimestamp"], unit="ms")
    d["grossProfit"] = _cpd(d, "grossProfit")
    d = d.dropna(subset=["positionId", "ts", "executionPrice"]).sort_values("ts")

    rows = []
    for pid, g in d.groupby("positionId", sort=True):
        if not len(g):
            continue
        opening = g.iloc[0]
        closing = g.iloc[-1]
        side = 1.0 if int(opening.get("tradeSide", 1) or 1) == 1 else -1.0
        vol = float(opening.get("filledVolume", 0) or 0)
        px_o = float(opening.get("executionPrice", np.nan))
        px_c = float(closing.get("executionPrice", np.nan))
        notional = vol * max(px_o, 1e-9)
        gp = g["grossProfit"].dropna()
        pnl = float(gp.iloc[0]) if len(gp) else side * (px_c - px_o) * vol
        rows.append({
            "position_id": str(pid),
            "symbol_id": int(opening.get("symbolId", -1)) if pd.notna(opening.get("symbolId")) else -1,
            "type": "buy" if side > 0 else "sell",
            "open_time": opening["ts"],
            "close_time": closing["ts"],
            "open_price": px_o,
            "close_price": px_c,
            "open_volume": vol,
            "n_fills": int(len(g)),
            "notional": notional,
            "profit": pnl,
            "net_bp": pnl / max(notional, 1e-9) * 1e4,
            "hold_s": (closing["ts"] - opening["ts"]).total_seconds(),
            "closed": bool(len(gp)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "artifacts" / "ctrader_deals.json"
    deals = asyncio.run(pull_deals(days=730, out=out))
    print(f"\n{len(deals):,} deals -> {out}")
    p = to_positions(deals)
    if not p.empty:
        print(f"reconstructed {len(p):,} closed positions "
              f"({p['open_time'].min().date()} -> {p['close_time'].max().date()})")
        out2 = out.with_name("ctrader_positions.parquet")
        p.to_parquet(out2, index=False)
        print(f"  wrote {out2}")
        print(f"  net_bp mean {p['net_bp'].mean():+.3f}  win {100*(p['net_bp'] > 0).mean():.0f}%")


if __name__ == "__main__":
    main()
