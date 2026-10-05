"""Append-only hypothesis registry.

The discipline this encodes: a hypothesis is registered *before* it is evaluated,
so the number of things tried stays reconstructible from the log instead of from
memory. Nothing here evaluates anything; it only records.

Layout: one JSON object per line under ``artifacts/registry.jsonl``::

    {"seq": 0, "ts": ..., "kind": "hypothesis", "id": "...", "hypothesis": {...},
     "generator": "...", "generator_version": "..."}
    {"seq": 1, "ts": ..., "kind": "evaluation", "id": "...", "metrics": {...},
     "config": {...}}

Evaluating an unregistered id raises: that is the point.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self.load()
        self._seq = len(self._records)
        self._ids = {r["id"] for r in self._records if r.get("kind") == "hypothesis"}

    def _append(self, record: dict) -> None:
        record = {"seq": self._seq, "ts": time.time(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical(record) + "\n")
        self._seq += 1
        self._records.append(record)

    def register(self, hypothesis: dict, generator: str, generator_version: str) -> str:
        """Register a hypothesis; returns a stable content id. Idempotent."""
        hid = hashlib.sha256(
            canonical([hypothesis, generator, generator_version]).encode("utf-8")
        ).hexdigest()[:16]
        if hid not in self._ids:
            self._append(
                {
                    "kind": "hypothesis",
                    "id": hid,
                    "hypothesis": hypothesis,
                    "generator": generator,
                    "generator_version": generator_version,
                }
            )
            self._ids.add(hid)
        return hid

    def record_evaluation(self, hid: str, metrics: dict, config: dict | None = None) -> None:
        if hid not in self._ids:
            raise ValueError(f"unknown hypothesis id {hid!r}; register before evaluating")
        self._append({"kind": "evaluation", "id": hid, "metrics": metrics, "config": config or {}})

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def evaluations(self, hid: str) -> list[dict]:
        return [r for r in self._records if r.get("kind") == "evaluation" and r["id"] == hid]
