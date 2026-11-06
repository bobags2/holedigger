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


DISCOVERY = "discovery"        # found by a search; carrying selection bias by construction
CONFIRMATION = "confirmation"  # a pre-registered test of a stated hypothesis


def data_fingerprint(parts: dict) -> str:
    """Stable hash of exactly which data an evaluation priced.

    Evaluations made before the DST repair and evaluations made after it are not the same
    measurement, and the registry must be able to tell them apart without a human
    remembering. Pass whatever identifies the inputs: symbol list, panel spans, the clock
    the stamps were on, and the code version.
    """
    return hashlib.sha256(canonical(parts).encode("utf-8")).hexdigest()[:12]


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

    def register(
        self,
        hypothesis: dict,
        generator: str,
        generator_version: str,
        role: str = DISCOVERY,
    ) -> str:
        """Register a hypothesis; returns a stable content id. Idempotent.

        ``role`` separates a hypothesis a search produced (``discovery`` - it carries the
        selection bias of the search that found it) from one that was stated first and then
        tested (``confirmation``). Only the second can support a claim on its own; the first
        is a lead until something pre-registers it. Re-registering the same content with a
        different role keeps the id and appends the role change, so the history stays
        reconstructible.
        """
        if role not in (DISCOVERY, CONFIRMATION):
            raise ValueError(f"role must be {DISCOVERY!r}|{CONFIRMATION!r}, got {role!r}")
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
                    "role": role,
                }
            )
            self._ids.add(hid)
        return hid

    def record_evaluation(
        self,
        hid: str,
        metrics: dict,
        config: dict | None = None,
        *,
        data_fingerprint: str | None = None,
        clock: str | None = None,
        supersedes: list[str] | None = None,
    ) -> None:
        """Record an evaluation with the provenance that makes it interpretable later.

        ``data_fingerprint`` pins the exact data (symbols, spans, code) the numbers came
        from; ``clock`` records which clock the panel's stamps were on, so the evaluations
        made before the DST repair are identifiable instead of silently comparable to the
        ones made after; ``supersedes`` lists evaluation seq numbers this one replaces, which
        is what lets a withdrawn claim be marked rather than quietly re-measured.
        """
        if hid not in self._ids:
            raise ValueError(f"unknown hypothesis id {hid!r}; register before evaluating")
        rec = {
            "kind": "evaluation",
            "id": hid,
            "metrics": metrics,
            "config": config or {},
        }
        if data_fingerprint:
            rec["data_fingerprint"] = data_fingerprint
        if clock:
            rec["clock"] = clock
        if supersedes:
            rec["supersedes"] = list(supersedes)
        self._append(rec)

    def withdraw(self, seqs: list[int], reason: str) -> None:
        """Append withdrawal records for the named evaluation seqs. History is never edited.

        Withdrawing is the append-only counterpart of correcting: the old numbers stay where
        they are, marked superseded with a reason, so the count of things tried - and the
        count later found invalid - both stay reconstructible.
        """
        for s in seqs:
            self._append(
                {"kind": "withdrawal", "target_seq": int(s), "reason": reason}
            )

    def active_evaluations(self, hid: str) -> list[dict]:
        """Evaluations of ``hid`` that no withdrawal has superseded, newest first."""
        dead = {
            r["target_seq"]
            for r in self._records
            if r.get("kind") == "withdrawal"
        }
        dead |= {
            s
            for r in self._records
            if r.get("kind") == "evaluation"
            for s in (r.get("supersedes") or [])
        }
        alive = [
            r
            for r in self.evaluations(hid)
            if r["seq"] not in dead
        ]
        return sorted(alive, key=lambda r: -r["seq"])

    def summary(self) -> dict:
        """The project-level multiplicity the review asked to be able to price.

        Hypotheses split by role, evaluations split by clock and data fingerprint, and the
        withdrawal count - so "how many things were tried, on what data, under which clock,
        and how many were later found invalid" is one call instead of an archaeology project.
        """
        hyp = [r for r in self._records if r.get("kind") == "hypothesis"]
        ev = [r for r in self._records if r.get("kind") == "evaluation"]
        wd = [r for r in self._records if r.get("kind") == "withdrawal"]
        dead = {r["target_seq"] for r in wd} | {
            s for r in ev for s in (r.get("supersedes") or [])
        }
        by_role: dict[str, int] = {}
        for r in hyp:
            by_role[r.get("role", "unknown")] = by_role.get(r.get("role", "unknown"), 0) + 1
        by_clock: dict[str, int] = {}
        for r in ev:
            by_clock[r.get("clock", "untagged")] = by_clock.get(r.get("clock", "untagged"), 0) + 1
        return {
            "hypotheses": len(hyp),
            "hypotheses_by_role": by_role,
            "evaluations": len(ev),
            "evaluations_by_clock": by_clock,
            "distinct_data_fingerprints": len(
                {r["data_fingerprint"] for r in ev if r.get("data_fingerprint")}
            ),
            "evaluations_without_fingerprint": sum(
                1 for r in ev if not r.get("data_fingerprint")
            ),
            "withdrawn_evaluations": len(dead),
            "active_evaluations": len(ev) - len(dead),
            "hypotheses_evaluated_more_than_once": sum(
                1 for h in hyp if len(self.evaluations(h["id"])) > 1
            ),
        }

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def evaluations(self, hid: str) -> list[dict]:
        return [r for r in self._records if r.get("kind") == "evaluation" and r["id"] == hid]
