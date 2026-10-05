"""CSV -> parquet + manifest. The CSV stays the immutable source of truth.

Writes per symbol:
  ``{symbol}_M1_raw.parquet``  — everything, sorted, deduplicated
  ``{symbol}_M1.parquet``      — curated subset from ``uniform_start`` (honest M1 only)

and ``data/manifest.json`` with source hashes and row/span parity after a read-back.

Usage::

    python -m src.ingest [--symbols EURUSD ...] [--history-dir PATH] [--tag TAG]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .audit import load_history_csv


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_history(path: Path) -> pd.DataFrame:
    df, ts = load_history_csv(path)
    df = df.rename(columns=str.lower).assign(ts=ts.to_numpy())
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df


def curated_start_for(symbol: str, audit_report_path: Path) -> str:
    if audit_report_path.exists():
        with audit_report_path.open("r", encoding="utf-8") as f:
            for r in json.load(f):
                if r["symbol"] != symbol:
                    continue
                if r.get("uniform_start_ts"):
                    return r["uniform_start_ts"]
                if r.get("uniform_start_month"):
                    return f"{r['uniform_start_month']}-01"
    return config.CURATED_START


def parity_check(df: pd.DataFrame, path: Path) -> dict:
    back = pd.read_parquet(path)
    same_rows = len(back) == len(df)
    same_span = bool(back["ts"].iloc[0] == df["ts"].iloc[0] and back["ts"].iloc[-1] == df["ts"].iloc[-1])
    close_close = bool(np.isclose(back["close"].sum(), df["close"].sum(), rtol=1e-9))
    nans = int(back.isna().sum().sum())
    return {"rows": same_rows, "span": same_span, "close_sum": close_close, "nans": nans}


def ingest_symbol(symbol: str, history_dir: Path, parquet_dir: Path, audit_report_path: Path) -> dict:
    src = history_dir / f"{symbol}_M1.csv"
    df = read_history(src)
    start = curated_start_for(symbol, audit_report_path)
    df_cur = df[df["ts"] >= pd.Timestamp(start)].reset_index(drop=True)

    raw_path = parquet_dir / f"{symbol}_M1_raw.parquet"
    cur_path = parquet_dir / f"{symbol}_M1.parquet"
    df.to_parquet(raw_path, index=False)
    df_cur.to_parquet(cur_path, index=False)

    entry = {
        "symbol": symbol,
        "csv": str(src),
        "csv_sha256": sha256_file(src),
        "csv_bytes": src.stat().st_size,
        "rows": len(df),
        "rows_curated": len(df_cur),
        "first": str(df["ts"].iloc[0]),
        "last": str(df["ts"].iloc[-1]),
        "curated_start": start,
        "first_curated": str(df_cur["ts"].iloc[0]),
        "parquet_raw": str(raw_path),
        "parquet_curated": str(cur_path),
        "parity_raw": parity_check(df, raw_path),
        "parity_curated": parity_check(df_cur, cur_path),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ok = all(
        [
            entry["parity_raw"]["rows"],
            entry["parity_raw"]["span"],
            entry["parity_raw"]["close_sum"],
            entry["parity_curated"]["rows"],
            entry["parity_curated"]["span"],
            entry["parity_curated"]["close_sum"],
            entry["parity_raw"]["nans"] == 0,
            entry["parity_curated"]["nans"] == 0,
        ]
    )
    entry["parity_ok"] = ok
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    ap.add_argument("--history-dir", type=Path, default=config.HISTORY_DIR)
    ap.add_argument("--tag", default=None, help="suffix for manifest name (default: manifest.json)")
    args = ap.parse_args()

    config.ensure_dirs()
    audit_report_path = config.ARTIFACTS_DIR / "audit_report.json"
    entries = []
    for symbol in args.symbols:
        print(f"[ingest] {symbol}", flush=True)
        entry = ingest_symbol(symbol, args.history_dir, config.PARQUET_DIR, audit_report_path)
        if not entry["parity_ok"]:
            raise SystemExit(f"parity FAILED for {symbol}: {entry['parity_raw']} {entry['parity_curated']}")
        entries.append(entry)

    name = f"manifest{('_' + args.tag) if args.tag else ''}.json"
    out = config.DATA_DIR / name
    with out.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)

    table = pd.DataFrame(
        [
            {
                "symbol": e["symbol"],
                "rows": e["rows"],
                "curated": e["rows_curated"],
                "start": e["curated_start"],
                "first": e["first"][:10],
                "last": e["last"][:10],
                "parity": e["parity_ok"],
            }
            for e in entries
        ]
    )
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
