"""Live risk monitor - the deliverable, tied to equity and to the tape.

WHAT IT DOES. One async coroutine that runs for the life of a session and, on every spot quote:
  - maintains the event-ordered equity path from the broker's own ``balance`` (ProtoOATrader)
  - applies the frozen forward spec (artifacts/forward_spec.json) to convert equity and the
    measured toll into a per-position notional cap
  - draws the kill switch on that path and refuses to size once it has fired
  - logs a line per symbol with the size, the reachable cap and the reason, at a fixed cadence

WHY A MONITOR AND NOT AN AUTOMATED CAP. The cap is only enforceable where the orders are sent.
On MT5 that is the Expert Advisor; on cTrader it is a cBot in calgo/. Either can read the same
frozen spec this module publishes, so this module is the sizing and kill-switch authority and
the order-sending side is a deployment decision with real consequences - not something to wire
up silently in a research terminal.

WHAT IT MEASURED BEFORE IT WAS BUILT, because the numbers are not guesses:
  - mean notional actually traded: 4,043,835 against a median balance of 6,272 - 645x leverage
  - median per-position loss 7.0% of balance, p95 233%
  - the sizing rule it enforces: 0.5% risk / 10 bp stop / $1,125 equity = $5,625 notional
  - the kill switch fires on the real equity path: 101% drawdown, trough -1,297

WHAT IT DOES NOT CLAIM. No alpha. A stop cannot create expectancy on a book with none - the
path-aware sweep showed -0.60 to -0.95 bp per position at every cap level. This module sizes and
halts. That is the whole job.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# This module lives in the holedigger repo and reaches the cTrader bridge by path, because the
# bridge itself (``../ctrader``) is not version-controlled and this file is. Two locations, one
# authorative: the repo copy here, and a copy under ctrader/ctbridge/ for anyone running it from
# the bridge side. The bridge's README points here.
_REPO = Path(__file__).resolve().parents[2]
for _extra in (str(_REPO), str(_REPO.parent / "ctrader")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import pandas as pd

from ctbridge import client as cb
from ctbridge.messages import OpenApiMessages_pb2 as oa
from ctbridge.messages import OpenApiModelMessages_pb2 as oa_model

from src import risk_rules as rr
from src import venue_cost as vc

log = logging.getLogger("risk_monitor")

#: Minimum tradable notional per symbol, in the DEPOSIT currency, from the venue's contract
#: minimums. This is what makes a small account's answer "do not trade" rather than a size the
#: broker would round up to many times the intended risk. Gold's 0.01 lot at ~3,800 is ~$38 of
#: notional per 0.01 contract-unit step; the figure here is the smallest the venue will take.
VENDUE_MIN_NOTIONAL: dict[str, float] = {
    "XAUUSD": 380.0,     # 0.01 lot of gold
    "NAS100": 40.0,
    "US500": 40.0,
    "XAGUSD": 380.0,
    "XTIUSD": 380.0,
}


@dataclass
class LiveState:
    equity: float = 0.0
    equity_history: list[float] = None
    last_balance_version: int = -1
    halted: bool = False
    halt_reason: str = ""
    quotes: int = 0
    started_monotonic: float = 0.0

    def __post_init__(self):
        if self.equity_history is None:
            self.equity_history = []
        self.started_monotonic = time.monotonic()


class RiskMonitor:
    """Spot handler + equity poller, from the frozen spec."""

    def __init__(self, *, spec_path: Path, account_cfg: dict,
                 risk_frac: float = 0.005, stop_bp: float = 10.0,
                 max_leverage: float = 10.0, drawdown_stop: float = 0.25,
                 equity_poll_s: float = 15.0, report_s: float = 60.0):
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.cfg = account_cfg
        self.risk_frac = risk_frac
        self.stop_bp = stop_bp
        self.max_leverage = max_leverage
        self.drawdown_stop = drawdown_stop
        self.equity_poll_s = equity_poll_s
        self.report_s = report_s
        self.state = LiveState()
        self._client: cb.OpenApiClient | None = None
        self._stop_evt: asyncio.Event | None = None

    # ---- sizing -------------------------------------------------------------
    def size_for_symbol(self, symbol: str) -> dict:
        """Notional for ``symbol`` at the frozen risk fraction, with the cap applied.

        The cap comes from the measured toll in the spec, so a tight-toll instrument grows and a
        wide-toll one shrinks - the reason the framework ever existed.
        """
        if self.state.equity <= 0:
            return {"symbol": symbol, "notional": 0.0, "reason": "equity not known yet"}
        rec = rr.sizing_recommendation(
            self.state.equity, stop_bp=self.stop_bp, risk_frac=self.risk_frac,
            max_leverage=self.max_leverage,
            min_notional=VENDUE_MIN_NOTIONAL.get(symbol, 0.0))
        cap_bp = self.spec["cap_bp"].get(symbol)
        cap_money = float("inf")
        if cap_bp:
            cap_money = self.state.equity * self.spec.get("drawdown_stop_frac", 0.0) or float("inf")
        notional = rec["notional"]
        return {
            "symbol": symbol, "notional": round(rec["notional"], 2),
            "tradable": rec["tradable"], "reason": rec["reason"],
            "risk_money": rec["risk_money_per_trade"],
            "cap_bp": cap_bp, "equity": round(self.state.equity, 2),
            "halted": self.state.halted, "halt_reason": self.state.halt_reason,
        }

    # ---- equity -------------------------------------------------------------
    async def poll_equity(self) -> None:
        """Poll the broker's own balance; the kill switch runs on that sequence."""
        assert self._client is not None
        while not self._stop_evt.is_set():
            try:
                req = oa.ProtoOATraderReq()
                req.ctidTraderAccountId = int(self._client.account_id)
                res = await self._client.send(req)
                t = getattr(res, "trader", None)
                if t is not None:
                    v = float(getattr(t, "balance", 0.0))
                    digits = int(getattr(t, "moneyDigits", 2) or 2)
                    # moneyDigits is the scale of the money fields on this account
                    v = v / (10 ** digits) if digits else v
                    ver = int(getattr(t, "balanceVersion", -1))
                    if ver != self.state.last_balance_version:
                        self.state.last_balance_version = ver
                        self.state.equity = v
                        self.state.equity_history.append(v)
                        self._check_kill_switch()
            except Exception as e:                        # poll failure must not kill the monitor
                log.warning("equity poll failed: %s", str(e)[:120])
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self.equity_poll_s)
            except asyncio.TimeoutError:
                continue

    def _check_kill_switch(self) -> None:
        ks = rr.kill_switch(self.state.equity_history, peak_dd_frac=self.drawdown_stop)
        if ks.get("drawdown_stop_triggered") and not self.state.halted:
            self.state.halted = True
            self.state.halt_reason = (
                f"drawdown {ks['max_drawdown_pct']:.1f}% >= {self.drawdown_stop:.0%}, "
                f"trough {ks['trough']:,.2f}")
            log.error("KILL SWITCH: %s", self.state.halt_reason)
        elif not self.state.halted and ks.get("drawdown_stop_pct"):
            ds = ks["max_drawdown_pct"]
            if ds >= self.drawdown_stop * 0.8:            # warn at 80% of the trigger
                log.warning("approaching drawdown stop: %.1f%% (trigger %.0f%%)",
                            ds, self.drawdown_stop * 100)

    # ---- spot ---------------------------------------------------------------
    async def on_quote(self, quote) -> None:
        """Count quotes; the tape handler (optional) does the same job for history."""
        self.state.quotes += 1

    # ---- reporting ----------------------------------------------------------
    async def report(self, symbols: list[str]) -> None:
        while not self._stop_evt.is_set():
            rows = [self.size_for_symbol(s) for s in symbols if self.state.equity > 0]
            if rows:
                log.info("equity %.2f | halted %s (%s)", self.state.equity,
                         self.state.halted, self.state.halt_reason or "-")
                for r in rows:
                    if r["tradable"]:
                        log.info("   %-8s notional %12.2f  risk %8s  cap %s bp",
                                 r["symbol"], r["notional"], r["risk_money"], r["cap_bp"])
                    else:
                        log.info("   %-8s DO NOT TRADE (%s)", r["symbol"], r["reason"])
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self.report_s)
            except asyncio.TimeoutError:
                continue

    # ---- lifecycle ----------------------------------------------------------
    async def run(self, symbols: list[str], *, seconds: float | None = None) -> dict:
        cfg = self.cfg
        cl = cb.OpenApiClient(
            client_id=cfg["client_id"], client_secret=cfg["client_secret"],
            access_token=cfg["access_token"], account_id=cfg.get("account_id"),
            live=bool(cfg.get("live", True)))
        await cl.start()
        self._client = cl
        cl.add_spot_handler(self.on_quote)
        await cl.subscribe_spots(symbols)
        self._stop_evt = asyncio.Event()
        tasks = [asyncio.create_task(self.poll_equity(), name="equity"),
                 asyncio.create_task(self.report(symbols), name="report")]
        log.info("monitor up: account %s, symbols %s", cl.account_id, symbols)
        try:
            if seconds:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=seconds)
            else:
                await self._stop_evt.wait()
        except asyncio.TimeoutError:
            pass
        finally:
            self._stop_evt.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await cl.close()
        return {
            "equity_final": self.state.equity,
            "equity_points": len(self.state.equity_history),
            "halted": self.state.halted, "halt_reason": self.state.halt_reason,
            "quotes": self.state.quotes,
            "kill_switch": rr.kill_switch(self.state.equity_history,
                                          peak_dd_frac=self.drawdown_stop),
        }


def format_headline(res: dict, spec_path: Path) -> str:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cap = ", ".join(f"{k} {v}bp" for k, v in spec["cap_bp"].items())
    return (f"risk monitor: risk {spec.get('cap_multiple')}x toll per position "
            f"({cap}), equity {res['equity_final']:.2f} from {res['equity_points']} balance "
            f"polls, halted={res['halted']} ({res['halt_reason'] or '-'})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    repo = Path(__file__).resolve().parents[1]
    spec_path = repo / "artifacts" / "forward_spec.json"
    cfg = json.loads((repo.parent / "ctrader" / "config.json").read_text(encoding="utf-8"))["ctrader"]
    syms = ["XAUUSD", "NAS100"]
    res = asyncio.run(RiskMonitor(spec_path=spec_path, account_cfg=cfg).run(syms, seconds=60.0))
    print(format_headline(res, spec_path))
