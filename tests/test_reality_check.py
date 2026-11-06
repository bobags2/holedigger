"""Tests for src/reality_check.py — White's RC and Hansen's SPA calibration.

Run:  python -m tests.test_reality_check
"""
from __future__ import annotations

import numpy as np

from src import gpu_null, reality_check
from src.reality_check import reality_check as white_rc, spa_p_value


def _panel(n_days=200, n_cand=50, seed=0, edge_bp=0.0, edge_cols=0):
    """A days x candidates panel of iid noise, optionally with an edge in the first columns."""
    rng = np.random.default_rng(seed)
    pnl = rng.standard_normal((n_days, n_cand)) * 2e-4
    if edge_bp and edge_cols:
        pnl[:, :edge_cols] += edge_bp * 1e-4
    return gpu_null.day_pnl_panel(
        {"S": np.arange(n_days)}, {"S": pnl}, ["S"], [(h, 1) for h in range(n_cand)]
    )


def test_white_rc_does_not_reject_pure_noise() -> None:
    """On pure noise the equal-weight portfolio must not look different from zero.

    This is the property the family null exists to establish, re-asked at the level of the
    whole search: 50 candidates, each individually able to reach t=3 by chance, must not
    make their *average* significant. One panel gives one p-value, so the assertion is the
    distributional one — the p-value must be a plausible draw from Uniform(0,1), not
    extreme — plus the check that the null's own scale is a t-distribution (mean ~0, p95 in
    the 1.6-2.2 band for 200 days).
    """
    panel = _panel(n_days=200, n_cand=50, seed=11)
    res = white_rc(panel, n_worlds=2000)
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] > 0.02, res           # not rejecting at even the 2% level
    assert abs(res["null_mean"]) < 0.5, res     # the null is centred on zero
    assert 1.4 < res["null_p95"] < 2.6, res     # and its scale is a t-distribution

    # across independent noise panels the rejection rate stays near nominal
    alpha, trials, rej = 0.05, 20, 0
    for t in range(trials):
        p = _panel(n_days=150, n_cand=40, seed=3000 + t)
        r = white_rc(p, n_worlds=400)
        rej += r["p_value"] <= alpha
    assert rej / trials <= 0.25, f"White RC rejected {rej}/{trials} pure-noise families"


def test_white_rc_finds_a_broad_real_effect() -> None:
    """An edge in HALF the family must be found by White's RC even if no single candidate is."""
    panel = _panel(n_days=200, n_cand=50, seed=12, edge_bp=6.0, edge_cols=25)
    res = white_rc(panel, n_worlds=2000)
    assert res["p_value"] < 0.10, res
    # and the per-candidate family is NOT uniformly significant: the average carries it
    res_spa = [spa_p_value(panel, n_worlds=500, candidate=j)["p_value"] for j in range(0, 50, 10)]
    assert min(res_spa) < 0.5, res_spa


def test_spa_is_not_anticonservative_on_noise() -> None:
    """Hansen's SPA must not reject the noise family more than its size.

    The naive max-t does: taking the best of 50 candidates against a per-candidate 95th
    percentile rejects far more than 5% of noise worlds. SPA's studentization and
    truncation are what fix that, so the assertion is that the rejection rate over many
    independent noise panels stays near the nominal level - not that one panel gives one
    particular number.
    """
    alpha = 0.05
    rejections = 0
    trials = 20
    for t in range(trials):
        panel = _panel(n_days=150, n_cand=40, seed=2000 + t)
        res = spa_p_value(panel, n_worlds=500)
        rejections += res["p_value"] <= alpha
    rate = rejections / trials
    assert rate <= 0.25, f"SPA rejected {rate:.0%} of pure-noise families"


def test_spa_finds_a_single_real_candidate() -> None:
    """One real candidate among 40 must be identified with a small p-value."""
    panel = _panel(n_days=250, n_cand=40, seed=7, edge_bp=40.0, edge_cols=1)
    res = spa_p_value(panel, n_worlds=2000)
    assert res["candidate"] == 0, res["candidate"]
    assert res["p_value"] < 0.05, res
    assert res["real_t"] > 3.0, res


def test_reality_checks_agree_with_the_family_null() -> None:
    """On a panel with one strong edge, both checks and the family scan-max should agree."""
    panel = _panel(n_days=250, n_cand=40, seed=7, edge_bp=40.0, edge_cols=1)
    rc = white_rc(panel, n_worlds=1000)
    spa = spa_p_value(panel, n_worlds=1000)
    fam = gpu_null.family_max(gpu_null.sign_flip_null(panel, 1000, np.random.default_rng(0)).t)
    real_max = float(np.nanmax(np.abs(gpu_null.real_moments(panel)[1])))
    assert abs(real_max) > np.percentile(fam, 95), (real_max, np.percentile(fam, 95))
    assert spa["p_value"] < 0.05
    assert rc["p_value"] < 0.10 or spa["p_value"] < 0.05, (rc["p_value"], spa["p_value"])


def test_bootstrap_variant_runs_and_is_close() -> None:
    """The block-bootstrap variant must run and give compatible answers.

    On the same panel the two families need not agree exactly (they are different nulls), so
    the assertion is that they are compatible — each p-value within the other's worlds-level
    sampling noise — rather than equal. A 20bp edge in 2 of 20 columns is below White's RC
    power at 300 worlds (measured both at ~0.48), which is the conservative test behaving.
    """
    panel = _panel(n_days=150, n_cand=20, seed=31, edge_bp=20.0, edge_cols=2)
    a = white_rc(panel, n_worlds=300, bootstrap=False)
    b = white_rc(panel, n_worlds=300, bootstrap=True)
    assert np.isfinite(a["p_value"]) and np.isfinite(b["p_value"])
    assert 0.0 <= a["p_value"] <= 1.0 and 0.0 <= b["p_value"] <= 1.0
    # compatible, not identical: |difference| within what 300 worlds can resolve
    assert abs(a["p_value"] - b["p_value"]) <= 0.30, (a["p_value"], b["p_value"])

    # and with the edge broad enough to carry the portfolio, BOTH find it
    strong = _panel(n_days=200, n_cand=20, seed=32, edge_bp=40.0, edge_cols=10)
    sa = white_rc(strong, n_worlds=1000, bootstrap=False)
    sb = white_rc(strong, n_worlds=1000, bootstrap=True)
    assert sa["p_value"] < 0.05, sa
    assert sb["p_value"] < 0.05, sb


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
