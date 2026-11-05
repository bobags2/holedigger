"""Tests for the per-date MT5 trade-server clock (src/mt5_clock.py).

Pure and deterministic: no data files, no network. Covers the DST rule itself
(second Sunday of March / first Sunday of November, before and after the 02:00 local
transition) and the two conversions the pipeline uses.

Run:  python -m tests.test_mt5_clock
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import mt5_clock as clock


def test_deep_winter_and_deep_summer() -> None:
    # US standard time: server = UTC+2, parquet stamps are 1h early.
    winter = pd.DatetimeIndex(["2026-01-15T12:00:00"])
    assert clock.utc_offset_hours(winter).tolist() == [2.0]
    assert clock.broker_to_utc(winter)[0] == pd.Timestamp("2026-01-15T10:00:00")
    assert clock.stored_to_utc(winter)[0] == pd.Timestamp("2026-01-15T13:00:00")

    # US summer time: server = UTC+3, parquet stamps are already true UTC.
    summer = pd.DatetimeIndex(["2026-07-15T12:00:00"])
    assert clock.utc_offset_hours(summer).tolist() == [3.0]
    assert clock.broker_to_utc(summer)[0] == pd.Timestamp("2026-07-15T09:00:00")
    assert clock.stored_to_utc(summer)[0] == pd.Timestamp("2026-07-15T12:00:00")


def test_second_sunday_of_march_transition() -> None:
    # 2026-03-08 is the second Sunday of March; 02:00 local = 07:00 UTC. The DST window
    # is half-open, [start, end), so the boundary instant itself is already summer time.
    assert clock.utc_offset_hours(pd.DatetimeIndex(["2026-03-08T06:59:59"])).tolist() == [2.0]
    assert clock.utc_offset_hours(
        pd.DatetimeIndex(["2026-03-08T07:00:00", "2026-03-08T07:00:01"])
    ).tolist() == [3.0, 3.0]


def test_first_sunday_of_november_transition() -> None:
    # 2026-11-01 is the first Sunday of November; 02:00 local = 06:00 UTC.
    assert clock.utc_offset_hours(pd.DatetimeIndex(["2026-11-01T05:59:59"])).tolist() == [3.0]
    assert clock.utc_offset_hours(
        pd.DatetimeIndex(["2026-11-01T06:00:00", "2026-11-01T06:00:01"])
    ).tolist() == [2.0, 2.0]


def test_rule_holds_in_other_years() -> None:
    # 2nd Sunday of March / 1st Sunday of November, several years.
    cases = {
        2025: ("2025-03-09", "2025-11-02"),
        2026: ("2026-03-08", "2026-11-01"),
        2027: ("2027-03-14", "2027-11-07"),
        2028: ("2028-03-12", "2028-11-05"),
    }
    for _, (start, end) in cases.items():
        s = pd.Timestamp(start) + pd.Timedelta(hours=clock.DST_START_UTC_HOUR)
        e = pd.Timestamp(end) + pd.Timedelta(hours=clock.DST_END_UTC_HOUR)
        assert s.day_name() == "Sunday" and e.day_name() == "Sunday", (start, end)
        probes = pd.DatetimeIndex([s - pd.Timedelta(hours=1), s, e - pd.Timedelta(hours=1), e])
        assert clock.utc_offset_hours(probes).tolist() == [2.0, 3.0, 3.0, 2.0], start


def test_measured_session_opens_get_the_measured_correction() -> None:
    """The two fingerprints that measured the bug, as regression values.

    The MT5 panel's stamps are the parquet clock: EURUSD's weekly open reads 21:02 in
    every month, while cTrader's true-UTC open is 22:02 in US standard time. After the
    repair the MT5 open must land on 22:02 for the winter Sunday and stay 21:02 for the
    summer one.
    """
    stamps = pd.DatetimeIndex(["2025-11-30T21:02:00", "2026-04-12T21:02:00",
                               "2026-10-30T20:59:00", "2026-11-01T21:02:00"])
    fixed = pd.DatetimeIndex(clock.stored_to_utc(stamps))
    assert fixed.tolist() == [
        pd.Timestamp("2025-11-30T22:02:00"),   # winter Sunday open -> 22:02 true UTC
        pd.Timestamp("2026-04-12T21:02:00"),   # summer Sunday open -> 21:02, unchanged
        pd.Timestamp("2026-10-30T20:59:00"),   # last summer bar stays put
        pd.Timestamp("2026-11-01T22:02:00"),   # post-changeover Sunday open moves +1h
    ]


def test_input_kind_is_preserved_and_empty_is_safe() -> None:
    s = pd.Series(pd.to_datetime(["2026-01-15T12:00:00", "2026-07-15T12:00:00"]), index=[7, 9])
    out = clock.stored_to_utc(s)
    assert isinstance(out, pd.Series) and out.index.tolist() == [7, 9]
    assert out.tolist() == [pd.Timestamp("2026-01-15T13:00:00"), pd.Timestamp("2026-07-15T12:00:00")]

    idx = clock.broker_to_utc(pd.DatetimeIndex([]))
    assert isinstance(idx, pd.DatetimeIndex) and len(idx) == 0
    assert clock.utc_offset_hours(pd.DatetimeIndex([])).shape == (0,)


def test_parquet_and_server_clock_agree() -> None:
    """A parquet stamp is the server stamp minus 3h; the two conversions must agree."""
    stored = pd.DatetimeIndex(pd.date_range("2025-11-20", "2026-11-04", freq="6h"))
    server = stored + pd.Timedelta(hours=clock.LEGACY_INGEST_OFFSET_HOURS)
    assert clock.stored_to_utc(stored).equals(clock.broker_to_utc(server))
    # and the correction is exactly 1h, and only, through US standard time
    delta = np.asarray(clock.stored_to_utc(stored) - stored) / np.timedelta64(1, "h")
    standard = clock.utc_offset_hours(stored) == clock.SERVER_OFFSET_STANDARD_HOURS
    assert set(np.unique(np.round(delta, 6))) == {0.0, 1.0}
    assert np.array_equal(delta == 1.0, standard)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
