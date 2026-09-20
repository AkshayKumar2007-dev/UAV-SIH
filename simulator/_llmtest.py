"""Hermetic tests for the LLM copilot (ai_advisor/llm.py).

Covers the rules fallback, suggestion JSON parsing, the OpenAI-compatible
HTTP path against a tiny LOCAL fake endpoint, and the non-blocking
suggestion refresh. No external network is used.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.ai_advisor.llm import LlmCopilot  # noqa: E402

# ---------------------------------------------------------------------- #
# shared telemetry snapshot + assessment (HealthMonitor-format)           #
# ---------------------------------------------------------------------- #
SNAP = {
    "mode": "STAB",
    "t_s": 100.0,
    "state": {"pos_ned": [0.0, 0.0, -275.0]},
    "pitot_reading": {"ias_mps": 40.0, "baro_alt_msl_m": 500.0, "vsi_mps": 1.0},
    "gps": {"health": "OK", "satellites": 14, "hdop": 0.8,
            "groundspeed_mps": 40.0, "track_deg": 0.0},
    "engine_sensors": {"rpm": 4800, "cht_C": 128, "egt_C": 760, "oil_psi": 55,
                       "fuel_qty_L": 90, "fuel_flow_Lph": 9.0},
    "engine_summary": {"running": True, "power_kw": 60.0,
                       "health_flags": ["CHT_HIGH"], "faults": {}},
    "wind_ned": [3.0, 0.0, 0.0],
    "mission_count": 0,
}

ASSESS = {
    "overall": {"score": 72, "status": "CAUTION",
                "summary": "Engine hot but recovering."},
    "alerts": [{"id": "CHT_HIGH", "sev": "WARN",
                "msg": "CHT 128 C trending up 6 C/min."}],
    "new_alerts": [],
    "suggestions": [{"id": "CHT_HIGH", "title": "Reduce power",
                     "detail": "Cut throttle to cool CHT.",
                     "source_alert": "CHT 128 C trending up."}],
    "insights": {"endurance_min": 300, "fuel_flow_lph": 9.0,
                 "fuel_home_L": 12.0, "can_reach_home": True, "agl_m": 300,
                 "stall_margin_kt": 25, "stall_ias_kt": 40,
                 "headwind_mps": 2.0, "range_km": 120,
                 "cht_trend_c_per_min": 6.0},
    "subsystems": {"engine": {"score": 60, "notes": ["hot"]}},
    "advisory_only": True,
}

DOWN_URL = "http://127.0.0.1:1"   # nothing listens here; connect is refused


# ---------------------------------------------------------------------- #
# fake OpenAI-compatible endpoint                                         #
# ---------------------------------------------------------------------- #
class _FakeHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/models"):
            self._reply(200, {"data": [{"id": "fake-model"}]})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        msgs = body.get("messages", [])
        text = " ".join(str(m.get("content", "")) for m in msgs)
        if "JSON array" in text:
            content = ('[{"title": "Cut power", "detail": "CHT 128 C is '
                       'climbing 6 C/min toward the 150 C limit."}, '
                       '{"title": "Watch fuel", "detail": "Endurance is '
                       '5.0 h at current flow."}]')
        else:
            content = "Fake LLM answer grounded in telemetry."
        self._reply(200, {"choices": [{"message": {"content": content}}]})

    def _reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class FakeServer:
    def __init__(self):
        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0),
                                                     _FakeHandler)
        self.httpd.allow_reuse_address = True
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------- #
# tests                                                                  #
# ---------------------------------------------------------------------- #
def test_parse_suggestions():
    good = ('[{"title": "Reduce power", "detail": "CHT climbing."}, '
            '{"title": "Watch fuel", "detail": "5 h left."}]')
    assert LlmCopilot._parse_suggestions(good) == [
        {"id": "LLM", "title": "Reduce power", "detail": "CHT climbing.",
         "source_alert": "LLM-drafted"},
        {"id": "LLM", "title": "Watch fuel", "detail": "5 h left.",
         "source_alert": "LLM-drafted"},
    ]
    fenced = '```json\n[{"title": "A", "detail": "B"}]\n```'
    assert LlmCopilot._parse_suggestions(fenced)[0]["title"] == "A"
    prose = "Here are the cards:\n\n[{\"title\": \"A\", \"detail\": \"B\"}]\n\nHope that helps."
    assert LlmCopilot._parse_suggestions(prose)[0]["title"] == "A"
    assert LlmCopilot._parse_suggestions("no cards here") is None
    assert LlmCopilot._parse_suggestions('[{"detail": "missing title"}]') == []
    assert LlmCopilot._parse_suggestions("") is None
    print("1. suggestion JSON parsing OK")


def test_fallback_when_down():
    cop = LlmCopilot(base_url=DOWN_URL, model="fake", timeout_s=2)
    ans = cop.answer("status", SNAP, ASSESS)
    assert "Overall health" in ans, ans          # rule-engine output
    assert not cop.available
    assert cop.meta()["llm"] is False
    assert cop.meta()["status"] == "probed"
    print("2. rules fallback when LLM unreachable OK")


def test_rules_suggestions_when_down():
    cop = LlmCopilot(base_url=DOWN_URL, model="fake", timeout_s=2)
    cop.refresh_suggestions(ASSESS, SNAP)        # spawns draft thread
    deadline = time.time() + 10
    while time.time() < deadline and cop._sug_busy:
        time.sleep(0.05)
    assert not cop._sug_busy, "draft thread never finished"
    items = cop.refresh_suggestions(ASSESS, SNAP)
    assert not cop._sug["llm"]
    assert items and items[0]["title"] == "Reduce power", items
    print("3. rule suggestions kept when LLM down OK")


def test_llm_chat_path():
    with FakeServer() as srv:
        cop = LlmCopilot(base_url=srv.url, model="fake-model", timeout_s=5)
        ans = cop.answer("status", SNAP, ASSESS)
        assert "Fake LLM answer" in ans, ans
        assert cop.available
        assert cop.meta()["llm"] is True and cop.meta()["model"] == "fake-model"
        # probe is cached — a second call is still served
        ans2 = cop.answer("engine", SNAP, ASSESS)
        assert "Fake LLM answer" in ans2, ans2
    print("4. LLM chat path (fake OpenAI endpoint) OK")


def test_llm_suggestions_draft():
    with FakeServer() as srv:
        cop = LlmCopilot(base_url=srv.url, model="fake-model", timeout_s=5)
        first = cop.refresh_suggestions(ASSESS, SNAP)   # starts draft thread
        assert first == [] or first[0]["title"] == "Reduce power"  # rules first
        items = first
        deadline = time.time() + 10
        while time.time() < deadline:
            items = cop.refresh_suggestions(ASSESS, SNAP)
            if cop._sug.get("llm"):
                break
            time.sleep(0.05)
        assert cop._sug["llm"], "LLM suggestion draft never landed"
        assert items and items[0]["title"] == "Cut power", items
        assert items[0]["source_alert"] == "LLM-drafted"
        assert len(items) == 2
    print("5. LLM-drafted suggestions (non-blocking refresh) OK")


def test_context_builder():
    cop = LlmCopilot(base_url=DOWN_URL, model="fake")
    ctx = cop._context(SNAP, ASSESS)
    for needle in ("mode STAB", "IAS", "CHT 128 C", "stall margin",
                   "Overall health: 72/100", "ENGINE" if False else "Engine:"):
        assert needle in ctx, (needle, ctx)
    assert "Mission:" not in ctx                    # no mission loaded

    # The live sim hands the copilot the RAW snapshot, where wind_ned and
    # nfz_violations are numpy arrays (only the published copy is jsonable()).
    # Truth-testing an ndarray raises "the truth value ... is ambiguous" and
    # silently forced the whole copilot back to the rule engine whenever an
    # LLM endpoint was reachable, so this shape is pinned here.
    raw = dict(SNAP, wind_ned=np.array([3.0, 0.0, 0.0]),
               nfz_violations=[{"name": "R-401", "penetration_m": 120.0}])
    rctx = cop._context(raw, ASSESS)
    assert "Wind: 3.0 m/s from 180 deg" in rctx, rctx
    assert "BREACHING R-401" in rctx, rctx

    # zero-length containers must not produce spurious lines
    empty = dict(SNAP, wind_ned=np.zeros(3), nfz_violations=[])
    assert "NFZ:" not in cop._context(empty, ASSESS)
    print("6. telemetry context builder OK")


# ---------------------------------------------------------------------- #
def main():
    test_parse_suggestions()
    test_fallback_when_down()
    test_rules_suggestions_when_down()
    test_llm_chat_path()
    test_llm_suggestions_draft()
    test_context_builder()
    print("LLM TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())