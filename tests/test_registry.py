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


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
