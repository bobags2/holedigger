"""Central paths and constants.

Source of truth for price data stays the cTrader bridge's ASCII history
(``../ctrader/history``). This project never writes there; parquet under
``data/`` is a derived cache.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT.parent / "ctrader" / "history"

DATA_DIR = REPO_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
AUDIT_DIR = ARTIFACTS_DIR / "audit"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.jsonl"

SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD", "US500", "NAS100", "USDX", "XTIUSD", "XAGUSD"]

# Universe expansion, verified against Fusion's live symbol list (274 symbols, 2026-10-05).
SOFT_COMMODITIES = [
    "COFARA", "COFROB", "CORN", "COTTON", "OJ", "SOYBEAN", "SUGAR", "SUGARRAW",
    "UKCOCOA", "USCOCOA", "WHEAT",
]
ENERGY_METALS = ["XNGUSD", "XBRUSD", "XCUUSD", "XALUSD", "XNIUSD", "XPBUSD", "XZNUSD", "XPTUSD", "XPDUSD"]
EXPANSION_SYMBOLS = SOFT_COMMODITIES + ENERGY_METALS

# Fallback curation start when no audit report exists. The audit measures a
# per-symbol `uniform_start` (first month of honestly-uniform M1) and overrides this.
CURATED_START = "2021-01-01"

SEED = 20261004


def ensure_dirs() -> None:
    for p in (DATA_DIR, PARQUET_DIR, ARTIFACTS_DIR, AUDIT_DIR):
        p.mkdir(parents=True, exist_ok=True)
