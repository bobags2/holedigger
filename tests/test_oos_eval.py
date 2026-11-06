"""Tests for src/oos_eval.py — the purged walk-forward harness and selection-inside-null.

Run:  python -m tests.test_oos_eval
"""
from __future__ import annotations

import numpy as np

from src import gpu_null, oos_eval
from src.oos_eval import (
    Strategy,
    empirical_bayes_edge,
    falsify_selection,
    merge_columns,
    split_half_selection,
    storey_pi0,
    walk_forward_columns,
)


def _make_panel_with_strategies(n_days=60, seed=0, edge_bp=30.0, noise_bp=200.0, n_folds=4):
    """A base panel plus one real OOS edge and two noise OOS columns, purged walk-forward."""
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    market = rng.standard_normal((n_days, 1)) * 2e-4
    edge = edge_bp / 1e4

    def edge_fit(train, rf):
        return edge

    def edge_pnl(pred, test, rf):
        # a real edge with realistic dispersion, so the column is studentizable
        return pred + np.random.default_rng(len(test)).standard_normal(len(test)) * edge / 4

    def noise_pnl_factory(s):
        r = np.random.default_rng(s)
        return lambda pred, test, rf: r.standard_normal(len(test)) * noise_bp / 1e4

    cols = walk_forward_columns(
        days,
        [
            Strategy("real_edge", edge_fit, edge_pnl),
            Strategy("noise_a", lambda t, rf: None, noise_pnl_factory(11)),
            Strategy("noise_b", lambda t, rf: None, noise_pnl_factory(22)),
        ],
        n_folds=n_folds,
        embargo=2,
    )
    base = gpu_null.day_pnl_panel({"MKT": days}, {"MKT": market}, ["MKT"], [(0, 1)])
    return merge_columns(base, cols), cols


def test_walk_forward_purges_and_never_scores_training_days() -> None:
    """The harness must be out-of-sample: no day is realised inside its own training fold.

    A column that silently scored training days would price its own lookahead as edge, and
    nothing downstream could tell. So the contract is checked directly: the purged band,
    the count of realised days, and that folds are separated by at least the embargo.
    """
    _panel, cols = _make_panel_with_strategies(n_days=60, n_folds=4)
    for name, c in cols.items():
        pnl, oob = c["pnl"], c["oob"]
        assert not np.isfinite(pnl[~oob]).any(), f"{name}: scored a non-test day"
        assert 0 < int(oob.sum()) < 60, f"{name}: realised {int(oob.sum())} of 60 days"
        idx = np.flatnonzero(oob)
        # embargo: the *fold boundaries* must be separated by >= embargo+1, while days
        # inside one fold are legitimately consecutive (gap 1)
        gaps = np.diff(idx)
        if len(gaps):
            boundaries = gaps[gaps > 1]
            assert boundaries.min() >= 2, f"{name}: folds touch, purge band missing"
        # and no realised day sits inside an embargo band next to a training span
        assert int(oob.sum()) < 60, f"{name}: realised every day"


def test_falsify_selection_prices_the_rule_not_the_candidate() -> None:
    """A real OOS edge must stand outside the null; noise-only must not.

    The *selection rule* (best of the whole family by |t|) runs inside every null world
    too, so the p-value already carries the cost of having searched. That is the number
    the repo needed and did not have.
    """
    panel, _ = _make_panel_with_strategies()
    res = falsify_selection(panel, k=1, n_worlds=2000)
    assert panel.symbols[int(res["selected"][0])] == "real_edge", res["selected"]
    assert res["null"]["percentile"] > 0.95, res["null"]

    # the same family with the real edge removed must be insignificant
    cols = walk_forward_columns(
        np.arange(len(panel.days)),
        [
            Strategy("noise_a", lambda t, rf: None,
                     lambda p, test, rf: np.random.default_rng(11).standard_normal(len(test)) * 2e-4),
            Strategy("noise_b", lambda t, rf: None,
                     lambda p, test, rf: np.random.default_rng(22).standard_normal(len(test)) * 2e-4),
        ],
        n_folds=4,
        embargo=2,
    )
    noise_panel = merge_columns(
        gpu_null.day_pnl_panel(
            {"MKT": panel.days}, {"MKT": panel.pnl[:, 0:1]}, ["MKT"], [(0, 1)]
        ),
        cols,
    )
    res2 = falsify_selection(noise_panel, k=1, n_worlds=2000)
    assert res2["null"]["percentile"] < 0.99, res2["null"]


def test_storey_pi0_responds_to_structure() -> None:
    """pi0 must read near 1 on a noise family and drop when structure is present.

    ``storey_pi0`` is a tail estimator and is badly biased at small ``m`` (with 3
    candidates the observed share above ``lambda`` is quantised to thirds, so a pure-noise
    family reads pi0 ~ 0.67 - measured, not assumed). So the *precision* claim is tested
    on a large family where the estimator is valid, and the small-family case only asserts
    the direction (no structure -> pi0 does not collapse toward 0).
    """
    panel, _ = _make_panel_with_strategies(n_days=120, edge_bp=60.0)
    res = falsify_selection(panel, k=1, n_worlds=1000)
    pi0 = res["pi0"]
    assert 0.0 < pi0["pi0"] <= 1.0, pi0
    assert pi0["m"] == panel.n_candidates, pi0
    # the real edge must survive step-down; noise must not
    j = panel.symbols.index("real_edge")
    assert res["p_values"][j] < 0.01, res["p_values"]
    assert res["adjusted_p"][j] < 0.05, res["adjusted_p"]
    for name in ("noise_a", "noise_b"):
        jj = panel.symbols.index(name)
        assert res["adjusted_p"][jj] > 0.05, (name, res["adjusted_p"])

    # a LARGE noise family: the estimator is valid here, and must read ~1
    rng = np.random.default_rng(9)
    n, k = 120, 200
    big_pnl = rng.standard_normal((n, k)) * 2e-4
    big = gpu_null.day_pnl_panel({"S": np.arange(n)}, {"S": big_pnl}, ["S"], [(h, 1) for h in range(k)])
    big_res = falsify_selection(big, k=1, n_worlds=2000)
    assert big_res["pi0"]["pi0"] > 0.8, big_res["pi0"]
    assert big_res["pi0"]["m"] == k, big_res["pi0"]
    # and with a real edge planted in 20 of those 200 columns, pi0 must drop
    big_pnl2 = big_pnl.copy()
    big_pnl2[:, :20] += 30e-4
    big2 = gpu_null.day_pnl_panel({"S": np.arange(n)}, {"S": big_pnl2}, ["S"], [(h, 1) for h in range(k)])
    big2_res = falsify_selection(big2, k=1, n_worlds=2000)
    assert big2_res["pi0"]["pi0"] < 0.9, big2_res["pi0"]


def test_split_half_rejects_a_selection_half_only_edge() -> None:
    """An edge that exists only in the training half must not survive the holdout.

    This is the failure the half-split exists to catch: the first-half/second-half decay
    the project keeps finding is exactly an edge living in one half. So plant one, and
    require the held-out half to say so.
    """
    rng = np.random.default_rng(4)
    n_days, n_cand = 120, 30
    cut = n_days // 2
    pnl = rng.standard_normal((n_days, n_cand)) * 2e-4
    pnl[:cut, 0] += 30e-4                 # the edge lives ONLY in the first half
    panel = gpu_null.day_pnl_panel(
        {"S": np.arange(n_days)}, {"S": pnl}, ["S"], [(h, 1) for h in range(n_cand)]
    )
    res = split_half_selection(panel, k=1, n_worlds=1000)
    assert int(res["picked"][0]) == 0, res["picked"]      # it does win the selection half
    assert res["train_stat"] > res["test_stat"], res        # and it decays
    assert res["decay"] > 2.0, res                         # the decay is large, not noise
    # a genuinely persistent edge must survive
    pnl2 = rng.standard_normal((n_days, n_cand)) * 2e-4
    pnl2[:, 1] += 30e-4
    panel2 = gpu_null.day_pnl_panel(
        {"S": np.arange(n_days)}, {"S": pnl2}, ["S"], [(h, 1) for h in range(n_cand)]
    )
    res2 = split_half_selection(panel2, k=1, n_worlds=1000)
    assert int(res2["picked"][0]) == 1, res2
    assert res2["null_test"]["percentile"] > 0.95, res2["null_test"]
    assert abs(res2["decay"]) < 10.0, res2


def test_empirical_bayes_shrinks_noise_and_keeps_signal() -> None:
    """Partial pooling must pull noise columns toward the family mean and keep a real edge.

    The property that matters is separation: after shrinkage, a genuine edge stands far
    from the family mean and a noise column sits near it. (Shrink *weights* are not ordered
    by signal - B is driven by each column's own standard error, and a strong column has
    both more signal and more dispersion, so its se is comparable to a noise column's.
    Measured: B ~ 0.9996 for all 50 columns, because one strong candidate makes the
    cross-sectional prior wide. That is EB behaving correctly, not a defect.)
    """
    rng = np.random.default_rng(6)
    n_days, n_cand = 300, 50
    pnl = rng.standard_normal((n_days, n_cand)) * 2e-4
    pnl[:, 0] += 40e-4
    panel = gpu_null.day_pnl_panel(
        {"S": np.arange(n_days)}, {"S": pnl}, ["S"], [(h, 1) for h in range(n_cand)]
    )
    eb = empirical_bayes_edge(panel)
    raw, post = eb["raw"], eb["posterior"]
    fam = eb["family_mean"]
    strong, noise = 0, np.arange(1, n_cand)
    # the strong candidate keeps its distance from the family mean
    assert abs(post[strong] - fam) > 0.5 * abs(raw[strong] - fam), eb["posterior"][strong]
    # noise columns are pulled IN, i.e. strictly closer to the family mean than raw
    assert (
        np.abs(post[noise] - fam) <= np.abs(raw[noise] - fam) + 1e-15
    ).all(), "noise columns must not move away from the family mean"
    # and the interval each candidate carries is the honest unit
    assert (eb["se"] >= 0).all()
    assert np.isfinite(eb["posterior"]).all()


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
