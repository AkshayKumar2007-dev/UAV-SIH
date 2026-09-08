import os, sys, time, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulator.ai_advisor.monitor import HealthMonitor, PilotAssistant
from simulator.telemetry.server import TelemetryServer

def make_snap(t, **over):
    s = {
        "t_s": t,
        "state": {"pos_ned": np.array([500.0, 300.0, -120.0]),
                  "vel_body": np.array([28.0, 0, 0]),
                  "euler": np.array([0.05, 0.03, 0.2]),
                  "rates": np.zeros(3)},
        "aero": {"alpha_rad": 0.05, "beta_rad": 0.0, "V_air_mps": 58.0, "stall": False,
                 "CL": 0.45, "CD": 0.05, "q_dyn_Pa": 1080.0, "on_ground": False, "h_agl_m": 3450.0},
        "engine_summary": {"rpm": 4400.0, "running": True, "power_kw": 38.0, "fuel_L": 132.0,
                           "fuel_kg_remaining": 105.6, "fuel_flow_Lph": 14.2, "cht_C": 175.0,
                           "egt_C": 700.0, "oil_psi": 65.0, "advance_ratio": 0.6,
                           "carb_heat": False, "starter_active": False, "run_time_s": t,
                           "health_flags": []},
        "engine_sensors": {"rpm": 4401.0, "fuel_flow_Lph": 14.2, "fuel_qty_L": 132.0,
                           "cht_C": 175.0, "egt_C": 700.0, "oil_psi": 65.0, "health": "OK",
                           "raw_flags": []},
        "pitot_reading": {"ias_mps": 42.0, "tas_mps": 58.0, "baro_alt_msl_m": 3725.0,
                          "vsi_mps": 0.5, "health": "OK", "new_data": True},
        "gps": {"lat_deg": 28.613, "lon_deg": 77.230, "alt_msl_m": 3726.0, "vn_mps": 57.0,
                "ve_mps": 1.0, "vd_mps": 0.0, "groundspeed_mps": 57.0, "track_deg": 2.0,
                "hdop": 1.1, "satellites": 10, "health": "OK", "new_data": True},
        "nfz_violations": [], "nfz_nearest": {"name": "AIRPORT CTR", "margin_m": 300.0},
        "agl_m": 3450.0, "mode": "WAYPOINT",
        "mission_current_idx": 0, "mission_count": 3, "mission_dist_m": 800.0,
        "mission_bearing_deg": 45.0,
        "manual_controls": {"delta_a_deg": 0, "delta_e_deg": 0, "delta_r_deg": 0,
                            "delta_flap_deg": 0, "throttle": 0.55, "gear_down": True,
                            "brakes": False, "carb_heat": False, "delta_e_trim_deg": -2.5},
        "ap_debug": {}, "wind_ned": np.array([3.0, -1.0, 0.0]),
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(s.get(k), dict):
            s[k].update(v)
        else:
            s[k] = v
    return s

# ---- 0. free the telemetry ports (a leftover simulator would otherwise be
#          the one answering) ----
import subprocess as _sp
_sp.run(["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*simulator/main.py*' } | "
         "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"], capture_output=True)
time.sleep(1.0)

# ---- 1. HealthMonitor nominal run ----
mon = HealthMonitor()
snap = make_snap(0)
for i in range(400):           # 40 s of nominal data
    snap = make_snap(i * 0.1, engine_sensors={"cht_C": 175.0 + i * 0.05, "fuel_qty_L": 200.0 - i * 0.05})
    a = mon.update(snap)
assert a["advisory_only"] is True
assert a["overall"]["score"] > 60, f"nominal score too low: {a['overall']}"
assert not [al for al in a["alerts"] if al["sev"] == "CRIT"], "unexpected CRIT"
assert a["insights"]["endurance_min"] and a["insights"]["endurance_min"] > 20
print("1. nominal monitor OK  score=%.0f status=%s endurance=%.0fmin stall_margin=%.0fkt"
      % (a["overall"]["score"], a["overall"]["status"], a["insights"]["endurance_min"],
         a["insights"]["stall_margin_kt"]))

# ---- 2. degraded: stall + fuel low + NFZ breach ----
mon2 = HealthMonitor()
for i in range(300):
    s = make_snap(i * 0.1,
                  aero={"stall": True},
                  engine_summary={"fuel_L": 18.0, "health_flags": ["FUEL_LOW"]},
                  engine_sensors={"fuel_qty_L": 18.0, "oil_psi": 18.0, "cht_C": 215.0},
                  nfz_violations=[{"name": "R-401", "penetration_m": 50.0}],
                  agl_m=15.0)
    a2 = mon2.update(s)
assert a2["overall"]["status"] in ("WARNING", "CRITICAL"), a2["overall"]
sevs = {al["id"] for al in a2["alerts"]}
assert {"STALL", "FUEL_LOW", "OIL_LOW", "NFZ_BREACH", "TERRAIN"} <= sevs, sevs
assert all("pilot_action" not in sug for sug in a2["suggestions"]), "suggestions must be text-only"
assert all("pilot_action" not in al for al in a2["alerts"]), "alerts must not carry actions"
print("2. degraded monitor OK  score=%.0f alerts=%s" % (a2["overall"]["score"], sorted(sevs)))

# ---- 3. assistant Q&A ----
pa = PilotAssistant()
for q in ["status", "how much fuel do I have?", "engine", "am I near stall?", "wind?",
          "any no-fly zones?", "what should I do?", "who are you?", "gps ok?"]:
    ans = pa.answer(q, snap, a2)
    assert isinstance(ans, str) and len(ans) > 10
print("3. assistant OK  e.g. 'what should I do?' ->", pa.answer("what should I do?", snap, a2)[:110])

# ---- 4. telemetry server: publish + ws client + ai_query round trip ----
srv = TelemetryServer()
srv.set_query_handler(lambda txt: PilotAssistant().answer(txt, snap, a2))
srv.start()
for i in range(8):
    srv.publish({**make_snap(i * 0.1), "ai": a2})
    time.sleep(0.05)

import asyncio, websockets
async def client():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        hello = json.loads(await ws.wait_for(lambda: True) if False else await ws.recv())
        assert hello["type"] == "hello" and "nfz_all" in hello, hello
        got_tel = None
        for _ in range(50):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            if msg["type"] == "telemetry":
                got_tel = msg["d"]
                break
        assert got_tel and "ai" in got_tel and got_tel["ai"]["overall"]["score"] > 0, (
        f"got_tel={bool(got_tel)}, ai={'ai' in (got_tel or {})}, "
        f"overall={ (got_tel or {}).get('ai', {}).get('overall') if got_tel else None}")
        assert got_tel["ai"]["advisory_only"] is True
        await ws.send(json.dumps({"type": "ai_query", "text": "status"}))
        for _ in range(50):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            if msg["type"] == "ai_reply":
                return got_tel, msg
        raise AssertionError("no ai_reply")
tel, reply = asyncio.new_event_loop().run_until_complete(asyncio.wait_for(client(), timeout=10))
print("4. WS round-trip OK  packet keys:", sorted(tel.keys())[:8], "...")
print("   ai_reply:", reply["answer"][:90])

# ---- 5. HTTP dashboard ----
import urllib.request
html = urllib.request.urlopen("http://127.0.0.1:8766/").read().decode()
assert "UAV TELEMETRY" in html and "view3d" in html and "ai_query" in html
assert urllib.request.urlopen("http://127.0.0.1:8766/healthz").status == 200
print("5. HTTP dashboard OK  (%d bytes, 3D canvas present)" % len(html))

# ---- 6. no command channel: unknown msg types ignored ----
async def try_cmd():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "set_mode", "value": "RTH"}))
        await ws.send(json.dumps({"type": "pilot_cmd", "cmd": "shutdown"}))
        await ws.send("not json")
        # any telemetry continuing to arrive means server is healthy; no ack of commands
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
        assert msg["type"] in ("telemetry",)
asyncio.new_event_loop().run_until_complete(asyncio.wait_for(try_cmd(), timeout=10))
print("6. command-type messages ignored (no control channel) OK")

print("ALL ADAPTER TESTS PASSED")
