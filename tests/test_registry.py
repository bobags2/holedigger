"""Tests for src/registry.py.

Run:  python -m tests.test_registry
"""
from __future__ import annotations

import pathlib
import tempfile

from src.registry import Registry


def test_register_idempotent_eval_gated() -> None:
    with tempfile.TemporaryDirectory() as d:
        reg = Registry(pathlib.Path(d) / "registry.jsonl")
        h = {"rule": "mom(1)", "symbol": "XAUUSD"}
        hid = reg.register(h, generator="tests", generator_version="0")
        assert reg.register(h, generator="tests", generator_version="0") == hid
        assert len(reg.load()) == 1

        reg.record_evaluation(hid, {"t": 1.23})
        assert len(reg.load()) == 2
        assert reg.evaluations(hid)[0]["metrics"]["t"] == 1.23

        other = reg.register({"rule": "rev(1)", "symbol": "XAUUSD"}, "tests", "0")
        assert other != hid

        try:
            reg.record_evaluation("deadbeef", {})
            raise AssertionError("expected ValueError for unregistered id")
        except ValueError:
            pass

        reg2 = Registry(reg.path)
        assert len(reg2.load()) == 3
        assert reg2.evaluations(hid)[0]["metrics"]["t"] == 1.23


def test_role_separates_discovery_from_confirmation() -> None:
    """A discovered hypothesis and a pre-registered one must not look alike.

    A search that produces a candidate carries the selection bias of the search. A
    hypothesis stated first and then tested does not. Only the second can support a claim
    on its own, so the registry has to record which is which.
    """
    from src.registry import CONFIRMATION, DISCOVERY

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(pathlib.Path(d) / "registry.jsonl")
        h = {"rule": "mom(1)", "symbol": "XAUUSD"}
        hid = reg.register(h, "tests", "0", role=DISCOVERY)
        rec = [r for r in reg.load() if r["id"] == hid][0]
        assert rec["role"] == DISCOVERY

        hid2 = reg.register({"rule": "stated first"}, "tests", "0", role=CONFIRMATION)
        assert reg.register({"rule": "stated first"}, "tests", "0", role=CONFIRMATION) == hid2

        try:
            reg.register(h, "tests", "0", role="whatever")
            raise AssertionError("expected ValueError for a bad role")
        except ValueError:
            pass


def test_evaluation_carries_data_and_clock_provenance() -> None:
    """An evaluation must say which data and which clock it priced.

    Before the DST repair and after it are not the same measurement; without the tag the
    two are silently comparable and a stale number can be re-read as current.
    """
    from src.registry import data_fingerprint

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(pathlib.Path(d) / "registry.jsonl")
        hid = reg.register({"rule": "z"}, "tests", "0")
        fp = data_fingerprint({"symbols": ["EURUSD"], "spans": "2025-11-19..2026-09-30",
                               "code": "abc123"})
        reg.record_evaluation(hid, {"t": -3.0}, data_fingerprint=fp, clock="stored-shift-3h")
        ev = reg.evaluations(hid)[0]
        assert ev["data_fingerprint"] == fp
        assert ev["clock"] == "stored-shift-3h"

        # a different fingerprint for different data, even with the same metrics
        fp2 = data_fingerprint({"symbols": ["EURUSD"], "spans": "x", "code": "abc123"})
        assert fp2 != fp


def test_withdrawal_is_append_only_and_filters_active() -> None:
    """Withdrawing marks an evaluation superseded without editing history."""
    from src.registry import data_fingerprint

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(pathlib.Path(d) / "registry.jsonl")
        hid = reg.register({"rule": "z"}, "tests", "0")
        reg.record_evaluation(hid, {"t": -6.07}, data_fingerprint="fpA", clock="stored")
        reg.record_evaluation(hid, {"t": -6.08}, data_fingerprint="fpB", clock="true-utc")
        first, second = reg.evaluations(hid)
        assert len(reg.load()) == 3          # 1 hypothesis + 2 evaluations
        reg.withdraw([first["seq"]], reason="calibrated against an invalid null")
        assert len(reg.load()) == 4          # appended, never edited
        active = reg.active_evaluations(hid)
        assert [a["seq"] for a in active] == [second["seq"]]
        s = reg.summary()
        assert s["withdrawn_evaluations"] == 1
        assert s["active_evaluations"] >= 1
        assert s["evaluations_by_clock"].get("stored") == 1
        assert s["evaluations_without_fingerprint"] == 0


def test_summary_prices_project_multiplicity() -> None:
    """The counts the review asked for, from one call."""
    from src.registry import DISCOVERY, data_fingerprint

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(pathlib.Path(d) / "registry.jsonl")
        h1 = reg.register({"a": 1}, "tests", "0", role=DISCOVERY)
        h2 = reg.register({"a": 2}, "tests", "0", role="confirmation")
        for i in range(3):
            reg.record_evaluation(h1, {"i": i}, data_fingerprint=data_fingerprint({"i": i}))
        reg.record_evaluation(h2, {"i": 9}, data_fingerprint="fp")
        s = reg.summary()
        assert s["hypotheses"] == 2
        assert s["hypotheses_by_role"][DISCOVERY] == 1
        assert s["evaluations"] == 4
        assert s["hypotheses_evaluated_more_than_once"] == 1
        assert s["distinct_data_fingerprints"] == 4


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
