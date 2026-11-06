"""Pull and RETAIN raw ticks, so tick-level markout and adverse selection become answerable.

Why this module exists
----------------------
The MT5 ingest deliberately summarises ticks into per-minute bars (``tick_n`` plus spread
quantiles) and discards the tape. That is the right trade for the sweep families, which need
years of bars, not millions of ticks. It also means one question cannot be answered: **does a
passive fill earn the spread, or get run over by informed flow?** That question is about the
order of prints within a bar — who crossed whom, in which direction — and no summary retains
it.

So this module pulls the raw tick history and keeps it: bid/ask/mid, volume, and flags, at
true UTC. One symbol at a time, resumable, budget-bounded, exactly like ``src.mt5_ingest`` —
a long pull must never lose work to a host timeout.

What it writes
--------------
``data/parquet/{SYMBOL}_MT5_ticks.parquet`` with columns ``ts`` (true UTC), ``bid``, ``ask``,
``mid``, ``volume``, ``flags``, plus a ``clock`` metadata tag recorded through
``mt5_clock.write_clock_tag_from_frame`` so the file declares its own clock. Rows are appended
in the same resumable way as the bar ingest; an interrupted run continues on the next
invocation.

Usage::

    python -m src.tick_ingest --symbols XAGUSD --days 14 --budget 240
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, mt5_clock
from .mt5 import Mt5

CHUNK_HOURS = 12
TICK_LIMIT = 100_000


def _ticks_path(symbol: str) -> Path:
    return config.PARQUET_DIR / f"{symbol}_MT5_ticks.parquet"


def _state_path(symbol: str) -> Path:
    return config.DATA_DIR / f"mt5_ticks_state_{symbol}.json"


def _frame_from_raw(rows) -> pd.DataFrame:
    """Normalise whatever the MCP tool returned into the canonical tick frame.

    The MCP payload's field names are not assumed: whatever the terminal returns is mapped by
    position-insensitive key matching for the handful of fields this study needs, and the
    frame is returned even when some are absent (they become NaN and are dropped downstream).
    """
    if not rows:
        return pd.DataFrame(columns=["ts", "bid", "ask", "mid", "volume", "flags"])
    df = pd.DataFrame(rows)
    lower = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return df[lower[n]]
        return pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "ts": pd.to_datetime(pick("time", "timestamp", "ts"), errors="coerce", utc=True).dt.tz_localize(None),
        "bid": pd.to_numeric(pick("bid"), errors="coerce"),
        "ask": pd.to_numeric(pick("ask"), errors="coerce"),
        "last": pd.to_numeric(pick("last", "price"), errors="coerce"),
        "volume": pd.to_numeric(pick("volume", "volume_real", "size"), errors="coerce"),
        "flags": pick("flags", "flag"),
    })
    # mid: the terminal's own mid when it reports one, else the bid/ask midpoint
    mid = pd.to_numeric(pick("mid"), errors="coerce")
    calc = (out["bid"] + out["ask"]) / 2.0
    out["mid"] = mid.where(mid.notna(), calc)
    out = out.dropna(subset=["ts"])
    if len(out):
        out = out[(out["bid"] > 0) & (out["ask"] > 0)]
    return out.reset_index(drop=True)


def pull_ticks(mt5: Mt5, symbol: str, start: datetime, until: datetime) -> pd.DataFrame:
    """Raw ticks for one window, in bounded chunks, concatenated in time order."""
    frames = []
    t = start
    while t < until:
        chunk_end = min(t + timedelta(hours=CHUNK_HOURS), until)
        try:
            rows = mt5.ticks_history(
                symbol,
                datetime_from=t.isoformat(),
                datetime_to=chunk_end.isoformat(),
                limit=TICK_LIMIT,
            )
        except Exception as exc:  # a dead chunk must not kill the run
            print(f"  {t:%Y-%m-%d %H:%M} -> {chunk_end:%H:%M}: {type(exc).__name__} {exc}")
            rows = []
        if isinstance(rows, dict):
            rows = rows.get("ticks") or rows.get("data") or rows.get("result") or []
        f = _frame_from_raw(rows)
        if len(f):
            frames.append(f)
        t = chunk_end
    if not frames:
        return pd.DataFrame(columns=["ts", "bid", "ask", "mid", "volume", "flags"])
    return pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)


def ingest_chunk(mt5: Mt5, symbol: str, start: datetime, until: datetime) -> dict:
    """Pull the next window's ticks, append to the parquet, advance the state.

    Same contract as ``src.mt5_ingest.ingest_chunk``: the clock is recorded in the parquet's
    own metadata (never only in the sidecar state, which had a crash window where the file and
    the tag could disagree), and an interrupted run continues on the next invocation.
    """
    state_path = _state_path(symbol)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"symbol": symbol, "start": start.isoformat(),
                 "next_from": start.isoformat(), "until": until.isoformat(),
                 "done": False, "ticks": 0}
    next_from = datetime.fromisoformat(state["next_from"])
    until = datetime.fromisoformat(state["until"])
    if state.get("done") or next_from >= until:
        state["done"] = True
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    # one working window per invocation, so a budget-bound run stops cleanly mid-history
    window_end = min(next_from + timedelta(days=7), until)
    ticks = pull_ticks(mt5, symbol, next_from, window_end)
    ticks["ts"] = mt5_clock.broker_to_utc(ticks["ts"])

    path = _ticks_path(symbol)
    if path.exists() and len(ticks):
        have = pd.read_parquet(path)
        ticks = pd.concat([have, ticks], ignore_index=True).sort_values("ts").reset_index(drop=True)
    if len(ticks):
        mt5_clock.write_clock_tag_from_frame(ticks, path)
    state.update({
        "next_from": window_end.isoformat(),
        "done": window_end >= until,
        "ticks": int(state["ticks"] + len(ticks)),
    })
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=14, help="history depth per symbol")
    ap.add_argument("--budget", type=float, default=240.0, help="wall-clock seconds before stopping cleanly")
    args = ap.parse_args()
    config.ensure_dirs()

    mt5 = Mt5()
    until = datetime.utcnow()
    start = until - timedelta(days=args.days)
    t0 = time.time()
    for sym in args.symbols:
        while time.time() - t0 < args.budget:
            state = ingest_chunk(mt5, sym, start, until)
            if state.get("done"):
                p = _ticks_path(sym)
                n = len(pd.read_parquet(p)) if p.exists() else 0
                print(f"{sym}: done, {n} ticks retained in {p.name}")
                break
        else:
            print(f"{sym}: budget reached; run again to continue")


if __name__ == "__main__":
    main()
