"""MT5 MCP -> holedigger parquet ingest (resumable, budget-bounded).

Pulls tick history (true bid/ask) per symbol in bounded chunks and derives M1 bars from
the *mid* price, carrying exact per-bar spread statistics. That is the dataset the
existing sweeps consume: the same window/segment machinery, but with spread measured
per bar instead of taped.

Resume: progress lives in ``data/mt5_state_<SYM>.json``; every chunk rewrites the parquet
and the state, so an interrupted run just continues on the next invocation. ``--budget``
stops the run cleanly before the host's timeout so a partial pass never loses work.

Timestamps are stored exactly as the terminal returns them ("terminal local"); the
terminal clock is recorded in the manifest for later UTC conversion.

Usage::

    python -m src.mt5_ingest --symbols EURUSD XAUUSD --days 350 --budget 250
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .mt5 import Mt5

CHUNK_HOURS = 12
TICK_LIMIT = 100000
CHUNK_DAYS = 14

# Terminal-local is UTC+3, measured by the EURUSD cross-source shift scan against the
# cTrader (UTC-labelled) parquet: return correlation peaks at a -3h shift. UTC = local - 3.
MT5_TZ_OFFSET_HOURS = 3.0


def _p90(s) -> float:
    """Percentile helper that survives the empty series pandas passes to lambdas."""
    s = pd.Series(s).dropna()
    return float(np.percentile(s, 90)) if len(s) else float("nan")


def _pull_range(mt5: Mt5, symbol: str, f: str, t: str) -> pd.DataFrame:
    """One tick request; halves the window and recurses if the limit truncates it."""
    payload = mt5.ticks_history(symbol, datetime_from=f, datetime_to=t, limit=TICK_LIMIT)
    rows = payload.get("history", []) if isinstance(payload, dict) else payload
    start, end = datetime.fromisoformat(f), datetime.fromisoformat(t)
    if len(rows) >= TICK_LIMIT and (end - start) > timedelta(minutes=10):
        mid = start + (end - start) / 2
        return pd.concat(
            [_pull_range(mt5, symbol, f, mid.strftime("%Y-%m-%dT%H:%M:%S")),
             _pull_range(mt5, symbol, mid.strftime("%Y-%m-%dT%H:%M:%S"), t)],
            ignore_index=True,
        )
    if len(rows) >= TICK_LIMIT:
        print(f"  [warn] {symbol} {f}->{t}: still at limit {TICK_LIMIT} at minimum chunk")
    if not rows:
        return pd.DataFrame(columns=["ts", "bid", "ask"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["time_ms"])
    return df[["ts", "bid", "ask"]]


def pull_ticks(mt5: Mt5, symbol: str, day_from: datetime, day_to: datetime) -> pd.DataFrame:
    """Tick history for [day_from, day_to), split so no response hits the row limit."""
    frames = []
    cursor = day_from
    while cursor < day_to:
        chunk_end = min(cursor + timedelta(hours=CHUNK_HOURS), day_to)
        f, t = cursor.strftime("%Y-%m-%dT%H:%M:%S"), chunk_end.strftime("%Y-%m-%dT%H:%M:%S")
        frames.append(_pull_range(mt5, symbol, f, t))
        cursor = chunk_end
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["ts", "bid", "ask"])
    return pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)


def mid_minute_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """M1 bars from the mid price, with per-bar spread statistics (in bp)."""
    empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "tick_n",
                                  "spread_med_bp", "spread_mean_bp", "spread_p90_bp"])
    if not len(ticks):
        return empty
    mid = (ticks["bid"].to_numpy(dtype=float) + ticks["ask"].to_numpy(dtype=float)) / 2.0
    spread = ticks["ask"].to_numpy(dtype=float) - ticks["bid"].to_numpy(dtype=float)
    d = pd.DataFrame({"ts": ticks["ts"].to_numpy(), "mid": mid, "sp": spread})
    g = d.groupby(pd.Grouper(key="ts", freq="1min"))
    bars = g.agg(open=("mid", "first"), high=("mid", "max"), low=("mid", "min"), close=("mid", "last"),
                 tick_n=("mid", "size"), spread_med_bp=("sp", "median"), spread_mean_bp=("sp", "mean"),
                 spread_p90_bp=("sp", _p90))
    bars = bars.reset_index()
    # Grouper emits a row per calendar minute: drop empty ones (no ticks in the minute)
    # or they become NaN prices that poison every cumulative sum downstream.
    bars = bars[bars["tick_n"] > 0].reset_index(drop=True)
    # spreads arrive in price units; normalise to bp of the bar's mid close
    px = bars["close"].to_numpy(dtype=float)
    for col in ("spread_med_bp", "spread_mean_bp", "spread_p90_bp"):
        bars[col] = bars[col].to_numpy(dtype=float) / px * 1e4
    return bars


# --------------------------------------------------------------------- resume state
def _state_path(symbol: str) -> Path:
    return config.DATA_DIR / f"mt5_state_{symbol}.json"


def _parquet_path(symbol: str) -> Path:
    return config.PARQUET_DIR / f"{symbol}_MT5_M1.parquet"


def _load_state(symbol: str, default_from: datetime, until: datetime) -> dict:
    p = _state_path(symbol)
    if p.exists():
        state = json.loads(p.read_text(encoding="utf-8"))
        # an extended window invalidates earlier coverage: drop it and start over
        if state.get("start") and default_from.isoformat() < state["start"]:
            p.unlink(missing_ok=True)
            _parquet_path(symbol).unlink(missing_ok=True)
        else:
            return state
    return {"symbol": symbol, "start": default_from.isoformat(),
            "next_from": default_from.isoformat(), "until": until.isoformat(),
            "done": False, "ticks": 0, "chunks": 0}


def _save_state(symbol: str, state: dict) -> None:
    _state_path(symbol).write_text(json.dumps(state, indent=2), encoding="utf-8")


def ingest_chunk(mt5: Mt5, symbol: str, start: datetime, until: datetime,
                 chunk_days: int = CHUNK_DAYS) -> dict:
    """Pull the next chunk for one symbol, append to its parquet, advance the state."""
    state = _load_state(symbol, start, until)
    next_from = datetime.fromisoformat(state["next_from"])
    until = datetime.fromisoformat(state["until"])
    if state.get("done") or next_from >= until:
        state["done"] = True
        _save_state(symbol, state)
        return state

    chunk_end = min(next_from + timedelta(days=chunk_days), until)
    ticks = pull_ticks(mt5, symbol, next_from, chunk_end)
    bars = mid_minute_bars(ticks)
    # MT5 stamps terminal-local (measured UTC+3); store UTC so every source and every
    # window grid uses the same clock, and equity sessions land inside the 00-20h grid.
    bars["ts"] = bars["ts"] - pd.Timedelta(hours=MT5_TZ_OFFSET_HOURS)
    parquet = _parquet_path(symbol)
    state.update({
        "next_from": chunk_end.isoformat(),
        "done": chunk_end >= until,
        "ticks": int(state["ticks"] + len(ticks)),
        "chunks": int(state["chunks"] + 1),
    })
    if not len(bars):
        # no ticks in the window (holiday gap, thin listing): advance the state only
        _save_state(symbol, state)
        return state
    if parquet.exists():
        have = pd.read_parquet(parquet)
        have = have[have["tick_n"] > 0]  # legacy runs wrote empty-minute rows
        if len(have):
            bars = pd.concat([have, bars], ignore_index=True).sort_values("ts").reset_index(drop=True)
    bars.to_parquet(parquet, index=False)
    state["bars"] = int(len(bars))

    state.update({
        "next_from": chunk_end.isoformat(),
        "done": chunk_end >= until,
        "ticks": int(state["ticks"] + len(ticks)),
        "chunks": int(state["chunks"] + 1),
        "bars": int(len(bars)),
    })
    _save_state(symbol, state)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None, help="explicit symbol list (or use --sector)")
    ap.add_argument("--days", type=int, default=350, help="days of history to fetch")
    ap.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (overrides --days)")
    ap.add_argument("--budget", type=float, default=250.0, help="wall-clock seconds before stopping cleanly")
    ap.add_argument("--sector", default=None, help="resolve symbols by MT5 sector (e.g. Technology) instead of --symbols")
    ap.add_argument("--shard", default=None, help="take a strided slice k/n of the resolved list (e.g. 0/3), so parallel jobs split the universe")
    args = ap.parse_args()

    config.ensure_dirs()
    mt5 = Mt5()
    info = mt5.initialize()
    try:
        clock = mt5.call("get_time_information")
    except Exception:
        clock = None

    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    start = datetime.strptime(args.date_from, "%Y-%m-%d") if args.date_from else end - timedelta(days=args.days)
    started = time.time()

    if args.sector:
        symbols = [s["symbol"] for s in mt5.symbols()["symbols"] if s.get("sector") == args.sector]
    else:
        symbols = args.symbols or []
    if not symbols:
        raise SystemExit("nothing to pull: pass --symbols or --sector")
    if args.shard:
        k, n = (int(p) for p in args.shard.split("/"))
        symbols = symbols[k::n]
    for sym in symbols:
        while True:
            state = ingest_chunk(mt5, sym, start, end)
            if state["done"]:
                break
            if time.time() - started > args.budget:
                print(f"[budget] {sym} incomplete: {state.get('bars', 0):,} bars so far, "
                      f"{state['ticks']:,} ticks; re-run to continue")
                break
        el = round(time.time() - started, 1)
        st = _load_state(sym, start, end)
        print(f"{sym:>7}: bars={st.get('bars', 0):,} ticks={st.get('ticks', 0):,} "
              f"done={st.get('done')} elapsed={el}s")

    out = config.DATA_DIR / "mt5_manifest.json"
    merged = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text(encoding="utf-8")).get("symbols", {})
        except (json.JSONDecodeError, OSError):
            merged = {}
    # shards overwrite each other's manifest: merge so the record stays complete
    merged.update({sym: _load_state(sym, start, end) for sym in symbols})
    payload = {
        "source": "mt5_mcp",
        "server": info.get("serverInfo"),
        "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_terminal_local": [start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")],
        "timestamps_stored_as": "UTC (terminal local + %+.0fh)" % -MT5_TZ_OFFSET_HOURS,
        "tz_offset_hours": MT5_TZ_OFFSET_HOURS,
        "terminal_clock": clock,
        "run_seconds": round(time.time() - started, 1),
        "symbols": merged,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
