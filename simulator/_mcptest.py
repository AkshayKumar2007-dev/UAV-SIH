"""Tests for the MCP server (simulator/mcp_server.py).

Part 1 — unit: the JSON-RPC processor against a fake sim link (no simulator
needed). Part 2 — end-to-end: spawn the real simulator headless on isolated
ports and drive it through the MCP bridge (stdio processor + HTTP transport).
No external network is used.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.mcp_server import (  # noqa: E402
    TOOLS, McpBridge, SimLink, run_http,
)

WS_PORT = 18765
HTTP_PORT = 18766


# ---------------------------------------------------------------------- #
# fake sim link for unit tests                                            #
# ---------------------------------------------------------------------- #
class FakeLink:
    connected = True
    latest = {
        "mode": "STAB", "t_s": 42.0,
        "ai": {"overall": {"score": 80, "status": "NOMINAL"},
               "suggestions": [{"id": "STRESS_WARN", "title": "Reduce power",
                                "detail": "Bring RPM below ~90%.",
                                "source_alert": "engine stress 50%"}],
               "alerts": [{"id": "STRESS_WARN", "sev": "WARN",
                           "msg": "Engine stress 50%.", "t_s": 41.0}],
               "advisory_only": True},
        "ai_meta": {"llm": False, "mcp": True},
        "ai_chat": [{"seq": 1, "t_s": 40.0, "source": "dashboard",
                     "question": "status", "answer": "Nominal."}],
        "engine_summary": {"running": True},
    }
    chat_turns = []

    def __init__(self, connected=True):
        self.connected = connected
        self.calls = []

    def request(self, mtype, payload, timeout=20.0):
        self.calls.append((mtype, payload))
        if mtype == "ai_query":
            return {"type": "ai_reply", "answer": "Fake copilot answer."}
        if mtype == "fault_inject":
            return {"type": "fault_ack", "fault": payload["fault"],
                    "value": payload.get("value")}
        if mtype == "sim_reset":
            return {"type": "sim_reset_ack"}
        raise AssertionError(f"unexpected request: {mtype}")


def _rpc(method, params=None, rid=1):
    return json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                       "params": params or {}})


def _call(bridge, name, arguments):
    resp = json.loads(bridge.handle(
        _rpc("tools/call", {"name": name, "arguments": arguments})))
    assert "error" not in resp, resp.get("error")
    return resp["result"]["content"][0]["text"]


# ---------------------------------------------------------------------- #
# part 1 — unit                                                           #
# ---------------------------------------------------------------------- #
def test_protocol():
    b = McpBridge(FakeLink())
    # initialize
    r = json.loads(b.handle(_rpc("initialize", {"protocolVersion": "2024-11-05"})))
    assert r["result"]["serverInfo"]["name"] == "cybersparks-uav-sim"
    assert r["result"]["capabilities"]["tools"] == {}
    # notification -> no response
    assert b.handle(_rpc("notifications/initialized")) is None
    # ping
    r = json.loads(b.handle(_rpc("ping")))
    assert r["result"] == {}
    # unknown method
    r = json.loads(b.handle(_rpc("bogus/method")))
    assert r["error"]["code"] == -32601
    # malformed JSON
    r = json.loads(b.handle("{not json"))
    assert r["error"]["code"] == -32700
    print("1. MCP protocol (initialize/ping/errors) OK")


def test_tools_list():
    b = McpBridge(FakeLink())
    r = json.loads(b.handle(_rpc("tools/list")))
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == {"get_status", "get_telemetry", "ask_copilot",
                     "inject_fault", "reset_sim", "list_scenarios",
                     "get_ai_suggestions", "get_chat_log"}, names
    assert len(r["result"]["tools"]) == len(TOOLS)
    print("2. tools/list OK")


def test_tools_with_sim():
    link = FakeLink()
    b = McpBridge(link)
    # get_status
    st = json.loads(_call(b, "get_status", {}))
    assert st["connected"] is True and st["mode"] == "STAB"
    assert st["ai_backend"] == "rules"
    # get_telemetry
    tel = json.loads(_call(b, "get_telemetry", {}))
    assert tel["t_s"] == 42.0 and "ai" in tel
    # ask_copilot (plain + verbose provenance)
    assert "Fake copilot answer." in _call(b, "ask_copilot", {"question": "status"})
    assert link.calls[-1][1].get("source") == "mcp"
    v = json.loads(_call(b, "ask_copilot", {"question": "status", "verbose": True}))
    assert v["answer"] == "Fake copilot answer." and v["backend"] == "rules"
    assert v["mcp_bridge"] is True and v["suggestions"] == 1
    # get_ai_suggestions / get_chat_log
    sug = json.loads(_call(b, "get_ai_suggestions", {}))
    assert sug["suggestions"][0]["title"] == "Reduce power", sug
    assert sug["advisory_only"] is True and sug["backend"] == "rules"
    chat = json.loads(_call(b, "get_chat_log", {}))
    assert chat[-1]["answer"] == "Nominal." and chat[-1]["source"] == "dashboard", chat
    assert json.loads(_call(b, "get_chat_log", {"limit": 1})) == chat[-1:]
    # inject_fault
    assert "carb_ice" in _call(b, "inject_fault", {"fault": "carb_ice", "value": 0.5})
    assert link.calls[-1][0] == "fault_inject" and link.calls[-1][1]["value"] == 0.5
    # reset
    assert "ack" in _call(b, "reset_sim", {})
    assert link.calls[-1][0] == "sim_reset"
    # ask_copilot without question -> tool error (isError, not json-rpc error)
    resp = json.loads(b.handle(_rpc("tools/call", {"name": "ask_copilot", "arguments": {}})))
    assert resp["result"]["isError"] is True
    # unknown tool -> json-rpc error
    resp = json.loads(b.handle(_rpc("tools/call", {"name": "nope", "arguments": {}})))
    assert resp["error"]["code"] == -32602
    print("3. tools/call with live link OK")


def test_tools_without_sim():
    link = FakeLink(connected=False)
    b = McpBridge(link)
    st = json.loads(_call(b, "get_status", {}))
    assert st["connected"] is False and "Start the simulator" in st["hint"]
    assert "not connected" in _call(b, "get_telemetry", {})
    assert "not connected" in _call(b, "get_ai_suggestions", {})
    assert "not connected" in _call(b, "get_chat_log", {})
    print("4. graceful behaviour with simulator down OK")


def test_list_scenarios():
    b = McpBridge(FakeLink())
    rows = json.loads(_call(b, "list_scenarios", {}))
    assert isinstance(rows, list) and len(rows) >= 1, rows
    assert all(r["name"] and r["path"] for r in rows)
    print(f"5. list_scenarios OK ({len(rows)} bundled)")


# ---------------------------------------------------------------------- #
# part 2 — end-to-end against the real simulator                          #
# ---------------------------------------------------------------------- #
def test_e2e():
    WRAPPER = f"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import simulator.config as cfg
cfg.TELEMETRY["ws_port"] = {WS_PORT}
cfg.TELEMETRY["http_port"] = {HTTP_PORT}
from simulator.main import main
main()
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", WRAPPER],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    def _drain():
        for _line in proc.stdout:
            pass
    threading.Thread(target=_drain, daemon=True).start()

    link = SimLink(f"ws://127.0.0.1:{WS_PORT}")
    link.start()
    try:
        deadline = time.time() + 90
        while time.time() < deadline and not (link.connected and link.latest):
            time.sleep(0.5)
        assert link.connected and link.latest, "sim never connected over WS"
        b = McpBridge(link)

        st = json.loads(_call(b, "get_status", {}))
        assert st["connected"] is True and st["mode"] == "STAB", st
        print(f"6a. e2e get_status OK (health {st['overall_status']})")

        tel = json.loads(_call(b, "get_telemetry", {}))
        for key in ("engine_sensors", "ai", "pitot_reading", "gps"):
            assert key in tel, key
        print(f"6b. e2e get_telemetry OK (t={tel['t_s']:.1f}s)")

        ans = _call(b, "ask_copilot", {"question": "status"})
        assert "Overall health" in ans or "Health" in ans, ans
        print("6c. e2e ask_copilot OK:", ans[:80])

        ack = _call(b, "inject_fault", {"fault": "carb_ice", "value": 0.9})
        assert "carb_ice" in ack, ack
        ack = _call(b, "inject_fault", {"fault": "clear_all"})
        assert "clear_all" in ack, ack
        print("6d. e2e inject_fault OK")

        ack = _call(b, "reset_sim", {})
        assert "ack" in ack, ack
        print("6e. e2e reset_sim OK")

        # live suggestion panel + shared chat transcript over MCP
        sug = json.loads(_call(b, "get_ai_suggestions", {}))
        assert "suggestions" in sug and sug["advisory_only"] is True, sug
        print(f"6g. e2e get_ai_suggestions OK ({len(sug['suggestions'])} cards)")

        _call(b, "ask_copilot", {"question": "engine temps"})
        chat = json.loads(_call(b, "get_chat_log", {"limit": 10}))
        assert any(t.get("source") == "mcp" and t.get("question") == "engine temps"
                   for t in chat), chat
        print(f"6h. e2e get_chat_log OK ({len(chat)} turns, MCP turn present)")

        meta = link.latest.get("ai_meta") or {}
        assert meta.get("mcp") is True, meta
        print("6i. e2e MCP bridge flagged live in telemetry OK")

        # ---- HTTP transport ----
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        hport = s.getsockname()[1]
        s.close()
        holder = []

        def _serve():
            httpd = run_http(b, hport)
            holder.append(httpd)
            httpd.serve_forever()

        th = threading.Thread(target=_serve, daemon=True)
        th.start()
        deadline = time.time() + 20
        while time.time() < deadline and not holder:
            time.sleep(0.1)
        assert holder, "HTTP server never started"
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{hport}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 7,
                                 "method": "tools/call",
                                 "params": {"name": "get_status",
                                             "arguments": {}}})
                .encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = json.loads(r.read())
            st2 = json.loads(resp["result"]["content"][0]["text"])
            assert st2["connected"] is True and st2["mode"] == "STAB", st2
            # SSE endpoint announces the message URL
            with urllib.request.urlopen(f"http://127.0.0.1:{hport}/mcp",
                                        timeout=5) as r:
                first = r.readline().decode().strip()
                second = r.readline().decode().strip()
            assert first == "event: endpoint", first
            assert second == "data: /mcp", second
        finally:
            holder[0].shutdown()
        print("6f. e2e HTTP transport OK")

        print("E2E MCP TESTS PASSED")
    finally:
        link.stop()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------- #
def main():
    test_protocol()
    test_tools_list()
    test_tools_with_sim()
    test_tools_without_sim()
    test_list_scenarios()
    test_e2e()
    print("MCP TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())