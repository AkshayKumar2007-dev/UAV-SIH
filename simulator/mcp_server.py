"""MCP (Model Context Protocol) server bridging the UAV simulator to AI agents.

Any MCP-capable agent — Claude Desktop, `claude code`, Cursor, or a custom
client built on an MCP SDK — can connect to the running simulator and:

  * read live telemetry and status (get_status, get_telemetry)
  * chat with the AI flight copilot (ask_copilot — LLM when available,
    offline rules otherwise; advisory-only by design)
  * drive the dashboard-level demo controls (inject_fault, reset_sim)
  * enumerate the bundled scenarios (list_scenarios)

The bridge is deliberately a THIN CLIENT of the simulator's existing
WebSocket protocol (the same one the dashboard uses): it subscribes to the
10 Hz telemetry stream, caches the latest snapshot, and relays ai_query /
fault_inject / sim_reset messages. No change to the simulator core is
needed, and the advisory-only trust model is inherited: the copilot can
only answer and suggest; fault injection and reset are exactly the
dashboard's powers, not flight-control authority.

Transports
----------
  * stdio (default): newline-delimited JSON-RPC on stdin/stdout — this is
    how Claude Desktop and other MCP clients launch servers.
  * HTTP: `--http PORT` serves JSON-RPC over `POST /mcp`, plus an SSE
    endpoint (`GET /mcp`) for streamable-HTTP clients.

Only the Python standard library is used; `websockets` (already a project
dependency) talks to the simulator.

Run it alongside the simulator:

    python -m simulator.main                 # terminal 1
    python -m simulator.mcp_server           # terminal 2 (stdio, for MCP clients)
    python -m simulator.mcp_server --http 8767   # or over HTTP for network agents
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import deque

from simulator.scenario import list_bundled

# ---------------------------------------------------------------------- #
# MCP tool catalog                                                        #
# ---------------------------------------------------------------------- #
TOOLS = [
    {
        "name": "get_status",
        "description": "Current simulator status: connected flag, flight mode, "
                       "sim time, crash state, overall AI health score/status, "
                       "AI backend (LLM vs offline rules), engine running.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_telemetry",
        "description": "Latest live telemetry snapshot as JSON: flight data "
                       "(IAS, altitude, AGL, vertical speed, heading), engine "
                       "(RPM, power, CHT, EGT, oil pressure, fuel), GPS, wind, "
                       "AI assessment (alerts, suggestions, subsystem scores), "
                       "digital twin + ML diagnosis.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_copilot",
        "description": "Ask the AI flight assistant a question. It answers from "
                       "live telemetry via the LLM copilot (falls back to the "
                       "offline rules), and the turn is published to the same "
                       "chat transcript the dashboard shows (get_chat_log). It "
                       "can only advise — it cannot fly the UAV. Examples: "
                       "'status', 'how much fuel to reach home?', 'what should I "
                       "do right now?', 'engine temps', 'ml diagnose'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "Your question to the copilot."},
                "verbose": {"type": "boolean", "default": False,
                            "description": "Return JSON with the answer plus "
                                           "provenance (backend, live alert and "
                                           "suggestion counts) instead of plain "
                                           "answer text."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_ai_suggestions",
        "description": "Read the live 'AI suggestions' panel exactly as the "
                       "dashboard shows it: the current advisory suggestion "
                       "cards (title, detail, source alert), the active alert "
                       "set they were derived from, and overall health. "
                       "Read-only — the assistant can only suggest.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_chat_log",
        "description": "Read the recent AI flight-assistant chat transcript — "
                       "the same conversation shown in the dashboard chatbox, "
                       "newest last: sequence, sim time, who asked (dashboard or "
                       "mcp), the question and the answer. Read-only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100,
                          "default": 20,
                          "description": "Most-recent turns to return."},
            },
        },
    },
    {
        "name": "inject_fault",
        "description": "Inject a demo fault exactly like the dashboard FAULT "
                       "INJECT page: carb_ice, oil_leak, stress, seizure, or "
                       "clear_all. value is severity 0..1 (ignored for "
                       "seizure/clear_all).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fault": {"type": "string",
                          "enum": ["carb_ice", "oil_leak", "stress",
                                   "seizure", "clear_all"]},
                "value": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                          "default": 1.0},
            },
            "required": ["fault"],
        },
    },
    {
        "name": "reset_sim",
        "description": "Reset the simulator to the initial runway state "
                       "(same as pressing R or the dashboard RESET SIM).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_scenarios",
        "description": "List the bundled YAML/JSON scenario files (name, path, "
                       "description) that configure a flight.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_REPLY_TYPE = {
    "ai_query": "ai_reply",
    "fault_inject": "fault_ack",
    "sim_reset": "sim_reset_ack",
}


# ---------------------------------------------------------------------- #
# SimLink — websocket client to the running simulator                     #
# ---------------------------------------------------------------------- #
class SimLink:
    """Background websocket client: caches telemetry, relays requests.

    All asyncio state lives on one daemon thread (the pattern used by the
    telemetry server). `request()` is the thread-safe entry point used by
    the MCP transports.
    """

    def __init__(self, ws_url):
        self.ws_url = ws_url
        self._loop = None
        self._thread = None
        self._ws = None
        self._latest = None
        self._hello = None
        self._pending = {}          # request type -> asyncio future (loop thread only)
        self._chat = deque(maxlen=100)   # Q&A turns this bridge asked/answered
        self._connected = False
        self._stop = False

    # ------------------------- lifecycle ------------------------- #
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mcp-simlink")
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                lambda: self._loop.create_task(self._close_ws()))

    # -------------------------- status --------------------------- #
    @property
    def connected(self):
        return self._connected

    @property
    def latest(self):
        return self._latest

    @property
    def hello(self):
        return self._hello

    @property
    def chat_turns(self):
        return list(self._chat)

    # ------------------------- requests -------------------------- #
    def request(self, mtype, payload, timeout=20.0):
        """Send a request to the sim and wait for its ack/reply."""
        deadline = time.time() + 5.0      # give the link thread a moment
        while self._loop is None or self._ws is None:
            if time.time() > deadline:
                raise ConnectionError(
                    "simulator not connected (is it running on "
                    f"{self.ws_url}?)")
            time.sleep(0.05)
        fut = asyncio_run_coroutine_threadsafe(
            self._do_request(mtype, payload, timeout), self._loop)
        return fut.result(timeout + 5.0)

    # ------------------------- internals ------------------------- #
    def _run(self):
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._client())
        finally:
            self._connected = False

    async def _client(self):
        import asyncio
        from websockets.asyncio.client import connect
        while not self._stop:
            try:
                async with connect(self.ws_url, open_timeout=5.0,
                                   close_timeout=2.0) as ws:
                    self._ws = ws
                    self._connected = True
                    print(f"[mcp] sim link connected: {self.ws_url}",
                          file=sys.stderr, flush=True)
                    try:
                        # announce the bridge so the dashboard headers can
                        # show that chat + suggestions are MCP-published
                        await ws.send(json.dumps({"type": "client_hello",
                                                  "role": "mcp"}))
                    except Exception:
                        pass
                    try:
                        async for raw in ws:
                            self._on_message(raw)
                    except Exception:
                        pass
            except Exception as ex:
                if not self._stop:
                    print(f"[mcp] sim link: {ex}", file=sys.stderr, flush=True)
            self._connected = False
            if not self._stop:
                await asyncio.sleep(2.0)

    def _on_message(self, raw):
        try:
            m = json.loads(raw)
        except (ValueError, TypeError):
            return
        t = m.get("type")
        if t == "telemetry":
            self._latest = m.get("d")
        elif t == "hello":
            self._hello = m
        else:
            if t == "ai_reply":
                # cache our own Q&A turn so get_chat_log is immediate instead
                # of waiting for the next throttled telemetry packet
                self._chat.append({k: v for k, v in m.items() if k != "type"})
            for qtype, rtype in _REPLY_TYPE.items():
                if t == rtype:
                    fut = self._pending.get(qtype)
                    if fut is not None and not fut.done():
                        fut.set_result(m)
                    break

    async def _close_ws(self):
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        self._loop.stop()

    async def _do_request(self, mtype, payload, timeout):
        import asyncio
        fut = self._loop.create_future()
        self._pending[mtype] = fut
        try:
            ws = self._ws
            if ws is None:
                raise ConnectionError("simulator not connected")
            await ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(mtype, None)


def asyncio_run_coroutine_threadsafe(coro, loop):
    import asyncio
    return asyncio.run_coroutine_threadsafe(coro, loop)


# ---------------------------------------------------------------------- #
# McpBridge — JSON-RPC (MCP) request processor                            #
# ---------------------------------------------------------------------- #
class McpBridge:
    """Stateless JSON-RPC processor for the MCP protocol.

    `handle(raw_json)` returns a serialized response string, or None for
    notifications (which get no reply). Protocol: initialize, ping,
    tools/list, tools/call — enough for any standard MCP client.
    """

    def __init__(self, link=None):
        self.link = link

    # ------------------------ entry point ------------------------ #
    def handle(self, raw):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return self._error(None, -32700, "Parse error: invalid JSON")
        if not isinstance(msg, dict) or "method" not in msg:
            return self._error(None, -32600, "Invalid request")
        method = msg["method"]
        rid = msg.get("id")
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                return self._ok(rid, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "cybersparks-uav-sim",
                                   "version": "1.0.0"},
                })
            if method == "notifications/initialized":
                return None
            if method == "ping":
                return self._ok(rid, {})
            if method == "tools/list":
                return self._ok(rid, {"tools": TOOLS})
            if method == "tools/call":
                return self._call_tool(rid, params)
            return self._error(rid, -32601, f"Method not found: {method}")
        except Exception as ex:
            return self._error(rid, -32603, f"Internal error: {ex}")

    # ----------------------- tools/call -------------------------- #
    def _call_tool(self, rid, params):
        name = params.get("name") if isinstance(params, dict) else None
        args = params.get("arguments") or {} if isinstance(params, dict) else {}
        if name not in {t["name"] for t in TOOLS}:
            return self._error(rid, -32602, f"Unknown tool: {name}")
        try:
            text = self._run_tool(name, args)
            return self._ok(rid, {"content": [{"type": "text", "text": text}]})
        except Exception as ex:
            return self._ok(rid, {
                "content": [{"type": "text", "text": f"Error: {ex}"}],
                "isError": True,
            })

    def _run_tool(self, name, args):
        l = self.link
        if name == "get_status":
            if l is None or not l.connected:
                return json.dumps({
                    "connected": False,
                    "hint": "Start the simulator first: python -m simulator.main",
                }, indent=2)
            snap = l.latest or {}
            ai = snap.get("ai") or {}
            ov = ai.get("overall") or {}
            meta = snap.get("ai_meta") or {}
            return json.dumps({
                "connected": True,
                "mode": snap.get("mode"),
                "t_s": snap.get("t_s"),
                "crashed": (snap.get("crash") or {}).get("crashed", False),
                "overall_health": ov.get("score"),
                "overall_status": ov.get("status"),
                "ai_backend": "llm" if meta.get("llm") else "rules",
                "engine_running": (snap.get("engine_summary") or {}).get("running"),
            }, indent=2)

        if name == "get_telemetry":
            if l is None or not l.connected or l.latest is None:
                return ("Simulator not connected — no telemetry yet. "
                        "Start it with `python -m simulator.main`.")
            return json.dumps(l.latest, indent=2)

        if name == "ask_copilot":
            q = str((args or {}).get("question", "")).strip()
            if not q:
                raise ValueError("question is required")
            verbose = bool((args or {}).get("verbose", False))
            # source=mcp tags the turn in the shared transcript so the
            # dashboard chatbox mirrors this question alongside its own
            reply = l.request("ai_query",
                              {"type": "ai_query", "text": q[:500],
                               "source": "mcp"},
                              timeout=30.0)
            answer = str(reply.get("answer", "(no answer)"))
            if verbose:
                snap = l.latest or {}
                ai = snap.get("ai") or {}
                meta = snap.get("ai_meta") or {}
                return json.dumps({
                    "answer": answer,
                    "backend": "llm" if meta.get("llm") else "rules",
                    "mcp_bridge": bool(meta.get("mcp")),
                    "active_alerts": len(ai.get("alerts") or []),
                    "suggestions": len(ai.get("suggestions") or []),
                    "t_s": snap.get("t_s"),
                }, indent=2)
            return answer

        if name == "inject_fault":
            fault = str((args or {}).get("fault", ""))
            value = float((args or {}).get("value", 1.0))
            if fault not in ("carb_ice", "oil_leak", "stress", "seizure",
                             "clear_all"):
                raise ValueError(f"unknown fault: {fault}")
            reply = l.request("fault_inject",
                              {"type": "fault_inject", "fault": fault,
                               "value": value},
                              timeout=10.0)
            return f"fault_inject ack: {reply.get('fault')} = {reply.get('value')}"

        if name == "reset_sim":
            l.request("sim_reset", {"type": "sim_reset"}, timeout=10.0)
            return "sim_reset ack — aircraft back on the runway."

        if name == "list_scenarios":
            rows = []
            for sname, spath in list_bundled():
                rows.append({"name": sname, "path": spath})
            return json.dumps(rows, indent=2)

        if name == "get_ai_suggestions":
            if l is None or not l.connected or l.latest is None:
                return ("Simulator not connected — no suggestions yet. "
                        "Start it with `python -m simulator.main`.")
            snap = l.latest
            ai = snap.get("ai") or {}
            meta = snap.get("ai_meta") or {}
            return json.dumps({
                "t_s": snap.get("t_s"),
                "overall": ai.get("overall"),
                "advisory_only": ai.get("advisory_only", True),
                "backend": "llm" if meta.get("llm") else "rules",
                "suggestions": ai.get("suggestions") or [],
                "active_alerts": ai.get("alerts") or [],
            }, indent=2)

        if name == "get_chat_log":
            if l is None or not l.connected or l.latest is None:
                return ("Simulator not connected — no chat yet. "
                        "Start it with `python -m simulator.main`.")
            try:
                limit = int((args or {}).get("limit", 20))
            except (TypeError, ValueError):
                limit = 20
            limit = max(1, min(100, limit))
            # merge the shared transcript (telemetry) with the turns this
            # bridge itself asked, keyed by seq so neither is duplicated
            merged = {}
            for turn in ((l.latest or {}).get("ai_chat") or []):
                if isinstance(turn, dict) and "seq" in turn:
                    merged[turn["seq"]] = turn
            for turn in l.chat_turns:
                if isinstance(turn, dict) and "seq" in turn:
                    merged[turn["seq"]] = turn
            log = [merged[k] for k in sorted(merged)]
            return json.dumps(log[-limit:], indent=2)

        raise ValueError(f"unhandled tool: {name}")

    # ------------------------ json-rpc --------------------------- #
    @staticmethod
    def _ok(rid, result):
        return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result},
                          separators=(",", ":"))

    @staticmethod
    def _error(rid, code, message):
        return json.dumps({"jsonrpc": "2.0", "id": rid, "error": {
            "code": code, "message": message}}, separators=(",", ":"))


# ---------------------------------------------------------------------- #
# transports                                                              #
# ---------------------------------------------------------------------- #
def run_stdio(bridge):
    """Newline-delimited JSON-RPC over stdin/stdout (MCP stdio transport)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp = bridge.handle(line)
        if resp:
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()


def run_http(bridge, port):
    """JSON-RPC over HTTP: POST /mcp (request/response) + GET /mcp (SSE)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class McpHTTPServer(ThreadingHTTPServer):
        daemon_threads = True   # handler threads never block interpreter exit

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/mcp":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode("utf-8", "replace")
            resp = bridge.handle(body)
            if resp is None:                       # notification
                self.send_response(202)
                self.end_headers()
                return
            data = resp.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/mcp":
                self.send_error(404)
                return
            # streamable-HTTP: an SSE stream advertising the POST endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(b"event: endpoint\ndata: /mcp\n\n")
                self.wfile.flush()
                import time
                while True:
                    time.sleep(15)
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, *args):
            pass

    httpd = McpHTTPServer(("127.0.0.1", port), Handler)
    print(f"[mcp] HTTP JSON-RPC listening on http://127.0.0.1:{port}/mcp",
          file=sys.stderr, flush=True)
    return httpd


def main():
    parser = argparse.ArgumentParser(
        description="MCP server bridging the UAV simulator to AI agents.")
    parser.add_argument(
        "--connect", metavar="WS_URL", default="ws://127.0.0.1:8765",
        help="simulator websocket URL (default: ws://127.0.0.1:8765)")
    parser.add_argument(
        "--http", metavar="PORT", type=int, default=0,
        help="serve JSON-RPC over HTTP on this port instead of stdio")
    args = parser.parse_args()

    link = SimLink(args.connect)
    link.start()
    bridge = McpBridge(link)
    try:
        if args.http:
            httpd = run_http(bridge, args.http)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
        else:
            print("[mcp] stdio transport — speak MCP JSON-RPC on stdin.",
                  file=sys.stderr, flush=True)
            run_stdio(bridge)
    finally:
        link.stop()


if __name__ == "__main__":
    main()