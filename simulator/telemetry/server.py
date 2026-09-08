"""Telemetry bridge: broadcasts simulator state to the web dashboard.

  - WebSocket  ws://<host>:<ws_port>   telemetry stream @ TELEMETRY_HZ
  - HTTP       http://<host>:<http_port>  serves the dashboard page

Inbound traffic is LIMITED to {"type": "ai_query", "text": ...} — a question
for the AI assistant, answered with data. There is NO command channel: the
dashboard cannot send anything to the aircraft. Control inputs exist only on
the simulator window (keyboard/mouse).
"""

import asyncio
import json
import os
import threading
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from simulator.config import TELEMETRY_HZ, HOME, WORLD, PISTON_ENGINE

DASHBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def jsonable(obj, _depth=0):
    """Recursively convert numpy containers/scalars to plain JSON types."""
    if _depth > 8:
        return None
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist(), _depth + 1)
    if isinstance(obj, np.generic):
        return jsonable(obj.item(), _depth + 1)
    if isinstance(obj, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in obj.items()
                if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v, _depth + 1) for v in obj]
    return None  # arbitrary objects (terrain handles, etc.) are dropped


class _DashboardHTTP:
    def __init__(self, port):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html", "/dashboard.html"):
                    try:
                        with open(DASHBOARD_PATH, "rb") as f:
                            body = f.read()
                    except OSError:
                        self.send_error(500, "dashboard.html missing")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/healthz":
                    body = b'{"ok":true}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True, name="telemetry-http")

    def start(self):
        self.thread.start()


class TelemetryServer:
    """Threaded telemetry broadcaster + AI query relay (questions only)."""

    def __init__(self, host=None, ws_port=None, http_port=None, hz=None):
        import simulator.config as cfg
        t = cfg.TELEMETRY
        self.host = host or t["ws_host"]
        self.ws_port = ws_port or t["ws_port"]
        self.http_port = http_port or t["http_port"]
        self.hz = hz or TELEMETRY_HZ
        self._latest = None          # last jsonable snapshot (set from sim thread)
        self._clients = set()
        self._query_handler = None   # fn(text) -> str, executed on ws thread
        self._reset_handler = None   # fn() -> None, human-initiated sim reset
        self._loop = None
        self._started = False
        self.http = _DashboardHTTP(self.http_port)

    # ------------------------- sim-thread API ------------------------- #
    def start(self):
        if self._started:
            return
        self._started = True
        self.http.start()
        th = threading.Thread(target=self._run_loop, daemon=True, name="telemetry-ws")
        th.start()

    def publish(self, snapshot):
        """Called from the sim loop every render tick (throttled to hz)."""
        self._latest = jsonable(snapshot)

    def set_query_handler(self, fn):
        self._query_handler = fn

    def set_reset_handler(self, fn):
        """Reset is a SIMULATOR function (like restarting the program), invoked
        only by a human pressing R or clicking RESET SIM — never by the AI."""
        self._reset_handler = fn

    @property
    def url(self):
        return f"http://127.0.0.1:{self.http_port}"

    @property
    def latest_snapshot(self):
        return self._latest

    # ------------------------- ws internals --------------------------- #
    def _run_loop(self):
        from websockets.asyncio.server import serve
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(ws):
            self._clients.add(ws)
            try:
                hello = {
                    "type": "hello",
                    "home": HOME,
                    "nfz_all": jsonable(WORLD["no_fly_zones"]),
                    "engine_cfg": jsonable({
                        "fuel_full_L": PISTON_ENGINE["fuel_full_L"],
                        "max_rpm": PISTON_ENGINE["max_rpm"],
                        "max_power_kw": PISTON_ENGINE["max_power_kw"],
                        "cht_max_c": PISTON_ENGINE["cht_max_c"],
                        "egt_max_c": PISTON_ENGINE["egt_max_c"],
                    }),
                    "world": jsonable({
                        "airport": WORLD.get("airport", {}),
                        "mountains": WORLD.get("mountains", []),
                        "trees": WORLD.get("trees", []),
                        "buildings": WORLD.get("buildings", []),
                        "lake": WORLD.get("lake", None),
                        "roads": WORLD.get("roads", []),
                        "airstrips": WORLD.get("airstrips", []),
                    }),
                    "hz": self.hz,
                }
                await ws.send(json.dumps(hello))
                if self._latest is not None:
                    await ws.send(json.dumps({"type": "telemetry", "d": self._latest}))
                async for raw in ws:
                    self._handle_client_msg(ws, raw)
            except Exception:
                pass
            finally:
                self._clients.discard(ws)

        async def broadcaster():
            period = 1.0 / self.hz
            while True:
                await asyncio.sleep(period)
                if not self._clients or self._latest is None:
                    continue
                pkt = json.dumps({"type": "telemetry", "d": self._latest})
                for ws in list(self._clients):
                    try:
                        # a half-dead client (killed tab, vanished network)
                        # blocks a bare await forever — bound every send and
                        # drop the client on timeout so one zombie can't
                        # freeze the stream for every other dashboard
                        await asyncio.wait_for(ws.send(pkt), timeout=2.0)
                    except Exception:
                        self._clients.discard(ws)

        async def main():
            async with serve(handler, self.host, self.ws_port, max_size=1 << 20):
                print(f"[telemetry] ws://127.0.0.1:{self.ws_port}  |  "
                      f"dashboard: http://127.0.0.1:{self.http_port}")
                await broadcaster()

        try:
            self._loop.run_until_complete(main())
        except Exception as ex:
            print("[telemetry] server stopped:", ex)

    def _handle_client_msg(self, ws, raw):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        mtype = msg.get("type")
        if mtype == "ai_query":
            text = str(msg.get("text", ""))[:500]
            if self._query_handler is None:
                answer = "Assistant not connected to the simulator yet."
            else:
                try:
                    answer = str(self._query_handler(text))
                except Exception as ex:
                    answer = f"Assistant error: {ex}"
            pkt = json.dumps({"type": "ai_reply", "question": text, "answer": answer})
            asyncio.ensure_future(self._safe_send(ws, pkt))
        elif mtype == "sim_reset":
            # human-initiated simulator reset (R key / RESET SIM button)
            if self._reset_handler is not None:
                try:
                    self._reset_handler()
                    asyncio.ensure_future(self._safe_send(
                        ws, json.dumps({"type": "sim_reset_ack"})))
                except Exception as ex:
                    print("[telemetry] reset error:", ex)
        # NOTE: any other message type is ignored by design — the dashboard
        # has no way to command the aircraft.

    async def _safe_send(self, ws, pkt):
        try:
            await ws.send(pkt)
        except Exception:
            self._clients.discard(ws)
