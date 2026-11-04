"""MetaTrader 5 MCP client (JSON-RPC over HTTP, MCP protocol 2025-06-18).

Thin, boring, dependency-free. Session lifecycle (initialize -> notifications/initialized
-> tools/call) handled automatically; a 401 or session error re-initialises once.

Credentials come from the environment; never hardcode and never commit them:

    set HOLEDIGGER_MT5_URL=http://127.0.0.1:22346/mcp
    set HOLEDIGGER_MT5_KEY=<api key>

Trading tools (trade_send_*, trade_close_*, tester_*) are *not* wrapped on purpose: they
are consequential actions and require an explicit human instruction each time.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:22346/mcp"


class Mt5Error(RuntimeError):
    pass


class Mt5:
    """One MCP session against the MetaTrader 5 terminal."""

    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 120.0) -> None:
        self.url = url or os.environ.get("HOLEDIGGER_MT5_URL", DEFAULT_URL)
        self.key = key or os.environ.get("HOLEDIGGER_MT5_KEY", "")
        if not self.key:
            raise SystemExit("no MT5 key: set HOLEDIGGER_MT5_KEY (env) or pass key=")
        self.timeout = timeout
        self.session: str | None = None
        self._id = 0
        self.server_info: dict = {}

    # ------------------------------------------------------------------ transport
    def _post(self, body: dict, with_session: bool = True) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.key}",
            "MCP-Protocol-Version": "2025-06-18",
        }
        if with_session and self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                session = resp.headers.get("Mcp-Session-Id")
                if session:
                    self.session = session
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                self.session = None
            raise Mt5Error(f"HTTP {exc.code} from MT5 MCP") from exc
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("event:") or "\ndata:" in text or text.startswith("data:"):
            for line in text.splitlines():
                if line.startswith("data:"):
                    text = line[5:].strip()
                    break
        if not text:
            return {}
        return json.loads(text)

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    # ------------------------------------------------------------------ protocol
    def initialize(self) -> dict:
        self.session = None
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "holedigger", "version": "1.0"},
            },
        }
        result = self._post(body, with_session=False)["result"]
        self.server_info = result.get("serverInfo", {})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, )
        return result

    def tools(self) -> list[dict]:
        return self.call("tools/list", None, method=True)["tools"]

    def call(self, name: str, args: dict | None = None, method: bool = False) -> dict | list:
        """Call a tool (or a raw method when method=True). Returns parsed tool output."""
        m = name if method else "tools/call"
        params = {} if method else {"name": name, "arguments": args or {}}
        body = {"jsonrpc": "2.0", "id": self._next_id(), "method": m, "params": params}
        for attempt in (0, 1):
            if self.session is None:
                self.initialize()
            try:
                resp = self._post(body)
                break
            except Mt5Error:
                if attempt:
                    raise
                self.initialize()
        if "error" in resp:
            raise Mt5Error(f"{name}: {resp['error']}")
        if method:
            return resp.get("result", {})
        result = resp.get("result", {})
        if result.get("isError"):
            raise Mt5Error(f"{name}: tool error")
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except (json.JSONDecodeError, TypeError):
                    return block["text"]
        return result

    # ------------------------------------------------------------------ conveniences
    def workspace_info(self) -> dict:
        return self.call("get_workspace_info")

    def account_info(self) -> dict:
        return self.call("get_trading_account_info")

    def symbols(self) -> list:
        return self.call("get_marketwatch_symbols")

    def chart_history(self, symbol: str, period: str, datetime_from: str, datetime_to: str,
                      limit: int | None = None) -> dict | list:
        args = {"symbol": symbol, "period": period, "datetime_from": datetime_from, "datetime_to": datetime_to}
        if limit is not None:
            args["limit"] = limit
        return self.call("get_chart_history", args)

    def ticks_history(self, symbol: str, datetime_from: str, datetime_to: str,
                      limit: int | None = None) -> dict | list:
        args = {"symbol": symbol, "datetime_from": datetime_from, "datetime_to": datetime_to}
        if limit is not None:
            args["limit"] = limit
        return self.call("get_chart_ticks_history", args)
