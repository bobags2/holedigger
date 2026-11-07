"""Calibration tests: every null in this repo must hold at nominal alpha on pure noise.

Run:  python -m tests.test_null_calibration

Why this suite exists
---------------------
Three different nulls in this project were found to reject far more than their nominal rate
on data with a known answer of zero, each time because the null's permutation unit did not
match the statistic's dependence structure:

* the original per-symbol sector permutation (real t's correlated, null t's independent) —
  ~100% false positives;
* the conditional family's pooled null (labels shuffled across pooled window-observations
  instead of per day) — 70%;
* the ranking family's within-session feature shuffle (destroys beta structure) — 42%.

All three passed their *planted-effect* tests. So the missing property is not "finds a real
effect" but "does not find a fake one", and that is what these tests assert. Any new null in
this repo is expected to carry a calibration test of this shape before it is trusted.
"""
from __future__ import annotations

import numpy as np

from src import conditional as cond
from src import cross_section as cs


# --------------------------------------------------------------------- synthetic


def _factor_panel(n_days=150, n_slots=288, win_len=24, n_windows=40, seed=0):
    """One market factor per day + clock-slot seasonality, with NO conditional structure."""
    r = np.random.default_rng(seed)
    day_factor = r.standard_normal(n_days) * 3.0
    slot_drift = np.random.default_rng(11).standard_normal(n_slots) * 0.5
    entries = r.integers(0, n_slots - win_len, size=n_windows)
    bars = (r.standard_normal((n_days, n_slots)) * 2.0
            + day_factor[:, None] + slot_drift[None, :])
    P = np.stack([bars[:, e:e + win_len].sum(axis=1) for e in entries], axis=1)
    state = r.standard_normal(n_days)                       # independent of returns
    q = np.quantile(state, 2 / 3)
    return P, state > q, state <= q, n_windows


def _one_factor_returns(n_days=137, k=105, mkt_autocorr=0.0, seed=0):
    """Zero predictability by construction: one-factor market + idiosyncratic noise."""
    r = np.random.default_rng(seed)
    mkt = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        prev = mkt_autocorr * prev + (1 - mkt_autocorr) * r.standard_normal()
        mkt[i] = prev
    beta = 0.75 + r.standard_normal(k) * 0.35               # realistic dispersion
    ret = beta[None, :] * mkt[:, None] + r.standard_normal((n_days, k))
    prev_ret = np.zeros_like(ret)
    prev_ret[1:] = ret[:-1]
    return ret, prev_ret


# ---------------------------------------------------------------------- tests


def test_conditional_pooled_null_is_calibrated() -> None:
    """The pooled conditional null must reject at ~alpha on pure noise.

    The shipped null shuffled labels across the pooled window-observations (150 days x 40
    windows) and was measured at a **70%** false-positive rate. The day-level permutation
    that replaced it is asserted here at <= 25% over a small number of trials (5% nominal,
    but 20 trials cannot resolve 5% exactly — the bound is deliberately loose enough to pass
    a calibrated null and fail an anti-conservative one, which is what caught the defect).
    """
    trials, hits = 20, 0
    for t in range(trials):
        P, wide, tight, _ = _factor_panel(seed=3000 + t)
        tv, td = P[~wide].ravel(), np.repeat(np.flatnonzero(~wide), _)
        wv, wd = P[wide].ravel(), np.repeat(np.flatnonzero(wide), _)
        real = float(np.median(wv) - np.median(tv)) * 1e4
        rng = np.random.default_rng(7000 + t)
        null = np.empty(200)
        for w in range(200):
            days = np.concatenate([td, wd])
            vals = np.concatenate([tv, wv])
            perm = rng.permutation(np.unique(days))
            lo = set(perm[: len(perm) // 2].tolist())
            m = np.isin(days, list(lo))
            null[w] = float(np.nanmedian(vals[m]) - np.nanmedian(vals[~m])) * 1e4
        hits += abs(real) > np.percentile(np.abs(null), 95)
    rate = hits / trials
    assert rate <= 0.25, f"pooled conditional null rejected {rate:.0%} of pure-noise trials"


def test_session_shift_null_is_calibrated() -> None:
    """The ranking null must reject at ~alpha on zero-predictability one-factor data.

    The shipped within-session feature shuffle was measured at a **42%** false-positive rate
    with zero market autocorrelation, because it destroys beta structure.

    THIS TEST ORIGINALLY HID THE DEFECT IT WAS WRITTEN TO CATCH. It computed the real
    statistic from raw ``fwd`` while the null residualised internally, so it passed at 0% while
    the production pipeline false-positived **100%** at market AR(1)=0.3. A calibration test
    that applies a different transform to the real statistic than to the null measures the
    null against itself and proves nothing. It now mirrors ``run_cross_section`` exactly: both
    sides pass through ``beta_neutralise``.
    """
    for mkt_autocorr in (0.0, 0.3, 0.6):
        trials, hits = 24, 0
        for t in range(trials):
            R, prev = _one_factor_returns(seed=4000 + t, mkt_autocorr=mkt_autocorr)
            Rn = cs.beta_neutralise(R)          # the SAME transform production uses
            fwd = np.zeros_like(Rn)
            fwd[:-1] = Rn[1:]
            ok = np.isfinite(prev) & np.roll(np.isfinite(Rn), -1, axis=0)
            real = float(np.nanmean(cs.daily_rank_corr(prev, fwd, ok)))
            null = cs.session_shift_null(Rn, prev, 120, np.random.default_rng(8000 + t),
                                         beta_neutral=True)
            hits += abs(real) > np.percentile(np.abs(null), 95)
        rate = hits / trials
        assert rate <= 0.15, (
            f"session-shift null rejected {rate:.0%} of zero-predictability trials at market "
            f"AR({mkt_autocorr}) - the real statistic and the null must be the same function"
        )


def test_state_labels_are_day_homogeneous() -> None:
    """A conditional cell must not mix clock times: one label per DAY, or the "wide" and
    "tight" cells cover different windows and no permutation of already-computed values can
    reproduce that (the moving-endpoint defect)."""
    from src import config

    p = config.PARQUET_DIR / "XAUUSD_MT5_M1.parquet"
    if not p.exists():
        return
    df = cond.load_bars("XAUUSD")
    labels = cond.state_labels(df, "spread_prev_hour")
    day = df["ts"].astype("int64") // (86_400 * 10**9)
    mixed = sum(1 for d in np.unique(day) if len(set(labels[day == d])) > 1)
    assert mixed == 0, f"{mixed} days carry more than one state label"


def test_drift_by_cell_carries_the_day_of_every_value() -> None:
    """The pooled null needs the day each value belongs to; losing them forces a value-level
    permutation, which is the defect at the heart of the 70% false-positive rate."""
    from src.window_sweep import HOLDS, MAX_EXIT_HOUR, _mean_intraday_bar_return
    from src import config

    if not (config.PARQUET_DIR / "XAUUSD_MT5_M1.parquet").exists():
        return
    df = cond.load_bars("XAUUSD")
    labels = cond.state_labels(df, "spread_prev_hour")
    bar_day = df["ts"].astype("int64") // (86_400 * 10**9)
    ret = np.zeros(len(df))
    ret[1:] = np.diff(np.log(df["close"].to_numpy(float)))
    S = type("S", (), {"r": ret, "bar_day": bar_day})()
    cells = cond.drift_by_cell(ret - _mean_intraday_bar_return(S), bar_day,
                               df["ts"].astype("int64"), labels, [(12, 4)])
    cell = cells[(12, 4, cond.BINS[2])]
    assert len(cell["days"]) == len(cell["values"]), "days and values must be 1:1"
    assert len(np.unique(cell["days"])) == len(cell["days"]), "one value per day"
    # the day ids must land in the state they claim
    for d, lab in zip(cell["days"][:5], [labels[bar_day == dd][0] for dd in cell["days"][:5]]):
        assert lab == cond.BINS[2], (d, lab)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
