"""MT5 economic calendar -> event table (parquet).

Two-stage pull (per the MCP schema):
  1. ``economic_calendar_list_values`` gives dated *values* (time, actual/forecast),
     interpreted in trade-server time = UTC+3 (same offset measured elsewhere).
  2. ``economic_calendar_get_event_by_id`` gives the event *description* (name,
     importance) — the values payload alone carries no usable impact ranking
     (``impact_type`` is mostly "not set").

Usage::

    python -m src.events --from 2026-01-01 --to 2026-11-01
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config
from .mt5 import Mt5

SERVER_UTC_OFFSET_HOURS = 3.0  # trade-server time = UTC+3
CHUNK_DAYS = 3  # observed density: ~50 events/day, limit 1000/call


def pull_values(m: Mt5, start: datetime, end: datetime, limit: int = 1000) -> list[dict]:
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        f, t = cursor.strftime("%Y-%m-%dT%H:%M:%S"), chunk_end.strftime("%Y-%m-%dT%H:%M:%S")
        payload = m.call(
            "economic_calendar_list_values", {"datetime_from": f, "datetime_to": t, "limit": limit}
        )
        rows.extend(payload.get("values", []) if isinstance(payload, dict) else payload)
        cursor = chunk_end
    return rows


def pull_descriptions(m: Mt5, event_ids: list[str], cache_path: Path) -> dict[str, dict]:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [e for e in set(event_ids) if e not in cache]
    for e in todo:
        try:
            cache[e] = m.call("economic_calendar_get_event_by_id", {"event_id": e})
        except Exception:
            cache[e] = {}
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def _event_meta(raw: dict, country_ccy: dict[str, str] | None = None) -> dict:
    """Description payload = {"ok":..., "event": {...}}."""
    e = raw.get("event", raw) if isinstance(raw, dict) else {}
    ccy = e.get("currency", "")
    if not ccy and country_ccy is not None:
        ccy = country_ccy.get(str(e.get("country_id", "")), "")
    return {
        "name": e.get("name", ""),
        "importance": e.get("importance", ""),
        "country_id": str(e.get("country_id", "")),
        "currency": ccy,
    }


def pull_country_currency(m: Mt5, cache_path: Path) -> dict[str, str]:
    """Map MT5 country_id -> ISO currency (description payload has no currency code)."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache:
        return cache
    payload = m.call("economic_calendar_list_countries", {})
    rows = payload.get("countries", payload) if isinstance(payload, dict) else payload
    for c in rows:
        cache[str(c["id"])] = c.get("currency", "")
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def pull_country_currency_cache(cache_path: Path) -> dict[str, str]:
    """Cached map only (build_table runs outside a pull session)."""
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def build_table(rows: list[dict], descriptions: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ts", "event_id", "name", "importance", "actual", "forecast", "prev"])
    df["event_id"] = df["event_id"].astype(str)
    country_ccy = pull_country_currency_cache(config.DATA_DIR / "country_currency.json")
    meta = pd.DataFrame(
        [
            {"event_id": e, **_event_meta(d, country_ccy)}
            for e, d in descriptions.items()
        ]
    )
    df = df.merge(meta, on="event_id", how="left")
    # the merge can coerce ts to object; force datetime64 before anything stores it
    df["ts"] = (pd.to_datetime(df["time"]) - pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)).astype("datetime64[ns]")
    df = df.rename(
        columns={
            "actual_value": "actual",
            "forecast_value": "forecast",
            "prev_value": "prev",
        }
    )
    keep = ["ts", "event_id", "name", "importance", "currency", "actual", "forecast", "prev"]
    return df[[c for c in keep if c in df.columns]].sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD (server time)")
    ap.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD (server time)")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "events.parquet")
    args = ap.parse_args()

    config.ensure_dirs()
    m = Mt5()
    m.initialize()
    start = datetime.strptime(args.date_from, "%Y-%m-%d")
    end = datetime.strptime(args.date_to, "%Y-%m-%d")

    rows = pull_values(m, start, end)
    print(f"pulled {len(rows)} dated values from {args.date_from} to {args.date_to}")
    if not rows:
        raise SystemExit("no calendar rows returned")
    descriptions = pull_descriptions(m, [str(r["event_id"]) for r in rows],
                                    config.DATA_DIR / "event_descriptions.json")
    pull_country_currency(m, config.DATA_DIR / "country_currency.json")
    df = build_table(rows, descriptions)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} events, {df['event_id'].nunique()} series, {df['ts'].min()} -> {df['ts'].max()}")
    print(f"importance mix: {df['importance'].value_counts().to_dict()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
