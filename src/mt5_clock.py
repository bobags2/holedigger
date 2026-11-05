"""The MT5 trade-server clock, per date — not a constant.

Fusion's MT5 server follows US DST: **UTC+3 through US summer time, UTC+2 through US
standard time**. Measured three independent ways on the 350-day FX/metals panel
(2025-11-19 .. 2026-11-04, `data/parquet/*_MT5_M1.parquet`):

* cross-source: the cTrader history (true-UTC labels) aligns with the MT5 panel at a
  0h shift through US summer and a +1h shift through US standard time, switching
  between 2026-03-06 and 2026-03-09. At the switch-corrected shift the 5-min closes
  match 0.91-0.96 with return correlation ~1.00; at a constant shift the winter months
  match 0.02-0.03 with correlation ~0.00;
* session opens: the MT5 panel's EURUSD weekly open sits at 21:02 in every month, while
  cTrader's moves 22:02 -> 21:02 exactly on 2026-03-08 and back on 2025-11-02 — the two
  US DST Sundays;
* the terminal's own snapshot in `data/mt5_manifest.json`: trade server
  2026-11-05T02:25:52 against pulled_utc 2026-11-05T00:25:53Z => +2h after US DST ended.

Two clocks therefore exist around this source:

* **server stamps** — what the terminal returns (``chart_history`` ticks). Convert with
  :func:`broker_to_utc`; nothing on disk holds these any more.
* **parquet stamps** — what `data/parquet/*_MT5_M1.parquet` holds: written by
  `src/mt5_ingest.py`, which subtracted a *fixed* 3h (``LEGACY_INGEST_OFFSET_HOURS``).
  Those stamps are exact through US summer time and 1h early through US standard time,
  so every reader of an MT5 parquet must pass its ``ts`` column through
  :func:`stored_to_utc`. That is the fix for the 2025-11-19..2026-11-04 panels; a future
  re-pull writes true UTC (see `src/mt5_ingest.py`), after which the repair is a no-op by
  construction only for symbols whose ingest state says ``clock == "utc"``.

Usage::

    from src import mt5_clock
    df["ts"] = mt5_clock.stored_to_utc(df["ts"])   # parquet -> true UTC
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Trade-server offsets, in hours, measured (see module docstring).
SERVER_OFFSET_DST_HOURS = 3.0
SERVER_OFFSET_STANDARD_HOURS = 2.0

# The constant the ingest used for the existing panels. Kept as the definition of what
# `stored_to_utc` repairs, so the two modules cannot drift apart.
LEGACY_INGEST_OFFSET_HOURS = 3.0

# US DST runs 02:00 local on the second Sunday of March to 02:00 local on the first
# Sunday of November; local means US Eastern, so those instants are 07:00Z and 06:00Z.
DST_START_UTC_HOUR = 7
DST_END_UTC_HOUR = 6


def _second_sunday_of_march(year: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=3, day=1)
    return first + pd.Timedelta(days=int((6 - first.dayofweek) % 7) + 7, hours=DST_START_UTC_HOUR)


def _first_sunday_of_november(year: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=11, day=1)
    return first + pd.Timedelta(days=int((6 - first.dayofweek) % 7), hours=DST_END_UTC_HOUR)


def utc_offset_hours(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Trade-server offset (hours) in force at each UTC instant.

    ``3.0`` inside the US DST window, ``2.0`` outside it. A tz-aware index is accepted
    (instants are compared in UTC).
    """
    idx = pd.DatetimeIndex(dates)
    if len(idx) == 0:
        return np.empty(0, dtype="float64")
    years, year_pos = np.unique(idx.year.to_numpy(), return_inverse=True)
    starts = np.array([_second_sunday_of_march(int(y)).value for y in years], dtype="int64")
    ends = np.array([_first_sunday_of_november(int(y)).value for y in years], dtype="int64")
    v = idx.asi8
    in_dst = (v >= starts[year_pos]) & (v < ends[year_pos])
    return np.where(in_dst, SERVER_OFFSET_DST_HOURS, SERVER_OFFSET_STANDARD_HOURS)


def _offset_of_stamps(ts: pd.DatetimeIndex, coarse_utc_hint_hours: float) -> np.ndarray:
    """Offset in force at each stamp, from a coarse UTC estimate of the instant.

    ``coarse_utc_hint_hours`` is the largest possible server offset for that stamp
    representation: 3h for a raw server stamp, 0h for a parquet stamp (which already had
    3h removed). Both hints land on the same instant for a corresponding pair of stamps,
    so the two conversions agree exactly.

    The estimate is 0-1h off the true instant, which can only misclassify a stamp whose
    true UTC falls in the hour before a changeover: Sunday 06:00-08:00 UTC around the two
    DST Sundays. No bar exists there — every instrument in the panel is closed until
    ~21:00 UTC on Sunday (checked: 0 of 39,695 Sunday bars in all 115 MT5 parquets
    before 20:00).
    """
    return utc_offset_hours(ts - pd.Timedelta(hours=coarse_utc_hint_hours))


def _as_input_kind(values: pd.DatetimeIndex | pd.Series, out: pd.DatetimeIndex):
    return pd.Series(out, index=values.index) if isinstance(values, pd.Series) else out


def broker_to_utc(ts: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex | pd.Series:
    """Terminal (trade-server) stamps -> true UTC: subtract the per-date offset."""
    idx = pd.DatetimeIndex(ts)
    off = _offset_of_stamps(idx, LEGACY_INGEST_OFFSET_HOURS)
    return _as_input_kind(ts, idx - pd.to_timedelta(off, unit="h"))


def stored_to_utc(ts: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex | pd.Series:
    """MT5 *parquet* stamps -> true UTC.

    The panels on disk carry ``server - 3h`` from the old constant-offset ingest, so the
    correction is +1h through US standard time and 0h through US summer time. Every
    reader of ``data/parquet/*_MT5_M1.parquet`` must apply this before using the stamps.
    """
    idx = pd.DatetimeIndex(ts)
    off = _offset_of_stamps(idx, 0.0)
    return _as_input_kind(ts, idx + pd.to_timedelta(LEGACY_INGEST_OFFSET_HOURS - off, unit="h"))
