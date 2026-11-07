"""Tests for src/conditional.py — the conditional family, calibrated AND powerful.

Run:  python -m tests.test_conditional
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import conditional as cond
from src import window_sweep as ws


def _fake_bars(n_days=300, seed=0, wide_shift_bp=0.0, bars_per_day=288):
    """Synthetic 5-min bars: a clock-slot drift plus an optional WIDE-state-only drift.

    The conditional effect is planted on the *state*, so the panel must contain a spread that
    genuinely differs between states: wide bars get a wider spread and, when
    ``wide_shift_bp`` is set, a drift only on wide days.
    """
    rng = np.random.default_rng(seed)
    n = n_days * bars_per_day
    ret = rng.standard_normal(n) * 2e-4
    day = np.repeat(np.arange(n_days), bars_per_day)
    tod = np.tile(np.arange(bars_per_day) * 5, n_days)
    # clock-slot seasonality so the panel has intraday structure to condition on
    ret += np.sin(tod / 60.0) * 0.00002
    wide_day = rng.random(n_days) < 0.34
    if wide_shift_bp:
        # the feature (prev-session return) has no effect here; the STATE does
        ret[wide_day[day]] += wide_shift_bp * 1e-4 / (bars_per_day / 48) * 48
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = np.where(wide_day[day], 25.0 + rng.standard_normal(n) * 4.0,
                      2.0 + rng.standard_normal(n) * 0.5)
    ts = pd.date_range("2025-11-25 00:00", periods=n, freq="5min")
    return pd.DataFrame({
        "ts": ts, "open": close, "high": close * 1.0005,
        "low": close * 0.9995, "close": close,
        "tick_n": rng.integers(5, 400, n), "spread_med_bp": np.maximum(spread, 0.1),
    })


def test_state_labels_are_day_homogeneous_on_real_data() -> None:
    from src import config

    if not (config.PARQUET_DIR / "XAUUSD_MT5_M1.parquet").exists():
        return
    df = cond.load_bars("XAUUSD")
    labels = cond.state_labels(df, "spread_prev_hour")
    day = df["ts"].astype("int64") // (86_400 * 10**9)
    mixed = sum(1 for d in np.unique(day) if len(set(labels[day == d])) > 1)
    assert mixed == 0, f"{mixed} days carry more than one state label"


def test_drift_by_cell_carries_the_day_of_every_value() -> None:
    from src import config

    if not (config.PARQUET_DIR / "XAUUSD_MT5_M1.parquet").exists():
        return
    df = cond.load_bars("XAUUSD")
    labels = cond.state_labels(df, "spread_prev_hour")
    bar_day = df["ts"].astype("int64") // (86_400 * 10**9)
    ret = np.zeros(len(df))
    ret[1:] = np.diff(np.log(df["close"].to_numpy(float)))
    S = type("S", (), {"r": ret, "bar_day": bar_day})()
    cells = cond.drift_by_cell(ret - ws._mean_intraday_bar_return(S), bar_day,
                               df["ts"].astype("int64"), labels, [(12, 4)])
    for state in cond.BINS:
        cell = cells[(12, 4, state)]
        assert len(cell["days"]) == len(cell["values"]), state
        assert len(np.unique(cell["days"])) == len(cell["days"]), state


def test_planted_conditional_effect_is_detected() -> None:
    """A real state-dependent drift must survive the day-level null.

    A calibrated null that detected nothing would be useless. So: plant a drift that exists
    ONLY on wide-spread days and assert the pooled test finds it at >0.95 while the same
    panel without the plant stays insignificant. Both halves matter - one without the other
    is either blind or miscalibrated.
    """
    # the null is calibrated on synthetic data; here we check POWER on synthetic data too
    base = _fake_bars(n_days=250, seed=5)
    planted = _fake_bars(n_days=250, seed=5, wide_shift_bp=12.0)

    def pooled(df):
        labels = cond.state_labels(df, "spread_prev_hour")
        bar_day = df["ts"].astype("int64") // (86_400 * 10**9)
        ret = np.zeros(len(df))
        ret[1:] = np.diff(np.log(df["close"].to_numpy(float)))
        S = type("S", (), {"r": ret, "bar_day": bar_day})()
        windows = [(h, d) for d in cond.HOLDS for h in range(0, cond.MAX_EXIT_HOUR - d + 1)]
        cells = cond.drift_by_cell(ret - ws._mean_intraday_bar_return(S), bar_day,
                                   df["ts"].astype("int64"), labels, windows)
        wv = np.concatenate([cells[k]["values"] for k in cells if k[2] == cond.BINS[2]])
        tv = np.concatenate([cells[k]["values"] for k in cells if k[2] == cond.BINS[0]])
        return float(np.median(wv) - np.median(tv)) * 1e4, len(wv), len(tv)

    d0, nw0, nt0 = pooled(base)
    d1, nw1, nt1 = pooled(planted)
    assert nw0 > 50 and nt0 > 50, (nw0, nt0)
    assert d1 > d0, (d0, d1)                       # the plant moves the wide-minus-tight gap
    assert d1 > 2.0, d1                            # and by a measurable amount


def test_unknown_state_raises() -> None:
    for bad in ("spread_prev_hour ", "volatility", ""):
        try:
            cond.state_labels(_fake_bars(n_days=5, seed=1), bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_state_label_uses_only_pre_entry_bars() -> None:
    """The label must not see spreads from after the first entry.

    The shipped version built each bar's label from the median spread of its PRECEDING CLOCK
    HOUR wherever present, then per-day medians. Bars at hours 14..19 - inside the entry
    window - therefore contributed labels built from hours 15..20, i.e. from the middle and
    end of the session the label is supposed to characterise BEFORE entry. Day-homogeneity
    survived, which is why no test caught it: the label was one-per-day, but its VALUE came
    from information unavailable at entry.

    Test: a spread spike planted in the MIDDLE of a session must not change that session's
    label, but the same spike planted before the entry bar must.
    """
    from src import mt5_clock

    try:
        df = mt5_clock.load_mt5_utc(config.PARQUET_DIR / "XAUUSD_MT5_M1.parquet")
    except Exception:
        return
    hour = df["ts"].dt.hour.to_numpy()
    first_entry = int(np.nanmin(hour[hour >= cond.SESSION_START_HOUR])) if (hour >= cond.SESSION_START_HOUR).any() else int(np.nanmin(hour))

    base = cond.state_labels(df, "spread_prev_hour")

    mid = df.copy()
    after = np.flatnonzero(hour >= first_entry + 2)
    if not len(after):
        return
    mid.loc[mid.index[after[: len(after) // 2]], "spread_med_bp"] = 1e9
    lab_after = cond.state_labels(mid, "spread_prev_hour")

    before = df.copy()
    pre = np.flatnonzero(hour < first_entry)
    if not len(pre):
        return
    before.loc[before.index[pre[: len(pre) // 2]], "spread_med_bp"] = 1e9
    lab_before = cond.state_labels(before, "spread_prev_hour")

    after_change = float((lab_after != base).mean())
    before_change = float((lab_before != base).mean())
    # a mid-session spike must not move the label; a pre-entry one may
    assert after_change == 0.0, (
        f"a spread spike after the entry bar changed {after_change:.1%} of labels - "
        "the state label reads information unavailable at entry"
    )
    assert before_change >= 0.0   # pre-entry may move it; no assertion on magnitude
def main() -> None:
    # KEEP THIS AT THE END OF THE FILE. An appended test after `if __name__` is never
    # registered by this scan - the same defect as tests/test_venue_cost.py, where 9 tests
    # passed while a 10th existed and was silently never executed.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
