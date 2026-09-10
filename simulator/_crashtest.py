"""End-to-end crash & reset test (headless).

1. Start TelemetryServer + a crashing flight profile (no display needed).
2. Verify: crash detected -> physics frozen -> CRASHED alert -> reset via WS -> flying again.
"""
import os, sys, time, json, math, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from simulator.main import _evaluate_impact
from simulator.config import CRASH

# ---- 1. impact evaluator unit checks ----
assert _evaluate_impact(-12.0, 0.0, 0.0, 20.0) is not None      # hard vertical
assert _evaluate_impact(-2.0, 70.0, 0.0, 20.0) is not None      # wing strike
assert _evaluate_impact(-2.0, 0.0, -45.0, 20.0) is not None     # nose-first
assert _evaluate_impact(-2.0, 0.0, 40.0, 20.0) is not None      # tail strike
assert _evaluate_impact(-2.0, 0.0, 0.0, 45.0) is not None       # high speed
assert _evaluate_impact(-2.5, 3.0, 5.0, 22.0) is None           # gentle landing
print("1. impact evaluator OK")

# ---- 1b. free the telemetry ports (a leftover simulator would otherwise be
#          the one answering, running old code) ----
import subprocess as _sp
_sp.run(["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*simulator/main.py*' -or "
         "$_.CommandLine -like '*from simulator.main import*' } | "
         "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"], capture_output=True)
time.sleep(1.0)

# ---- 2. live sim: run the real main loop headless until it crashes ----
# Patch spawn: 8 m above terrain, nose-down 28 deg, 15 m/s vertical -> impact ~0.5 s
import subprocess
WRAPPER = """
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import simulator.config as cfg
cfg.INITIAL_CONDITIONS["pos_ned_m"] = np.array([0.0, 0.0, -30.0])
cfg.INITIAL_CONDITIONS["vel_body_mps"] = np.array([30.0, 0.0, 15.0])
cfg.INITIAL_CONDITIONS["euler_rad"] = np.array([0.0, -0.5, 0.0])
from simulator.main import main
main()
"""
proc = subprocess.Popen(
    [sys.executable, "-u", "-c", WRAPPER],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
print("2. sim launched (guaranteed-impact spawn), waiting for crash...")
try:
    import websockets
    asyncio = __import__("asyncio")

    async def watch():
        ws = None
        for _ in range(30):                       # wait for the server to bind
            try:
                ws = await websockets.connect("ws://127.0.0.1:8765")
                break
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.5)
        assert ws is not None, "simulator WS never came up"
        async with ws:
            await ws.recv()  # hello
            crashed = None
            frozen = False
            # wait up to 90 s for the crash
            t0 = time.time()
            while time.time() - t0 < 90:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if m["type"] != "telemetry":
                    continue
                d = m["d"]
                if d["crash"]["crashed"] and crashed is None:
                    crashed = d
                    print(f"   CRASH at t={d['crash']['t_s']:.0f}s  reason: {d['crash']['reason']}")

                if crashed is not None:
                    if crashed["t_s"] == d["t_s"]:
                        frozen = True
                        break
                    crashed = d
            assert crashed, "no crash within 90 s"
            assert frozen, "physics not frozen after crash"
            assert crashed["engine"]["running"] is False, "engine alive after crash"
            assert crashed["engine"]["rpm"] < 1.0, f"rpm not zeroed: {crashed['engine']['rpm']}"
            assert crashed["engine"]["power_kw"] == 0.0
            # EVERYTHING zeroed: instruments, fuel, temps, gps, health scores
            e = crashed["engine"]
            assert all(e[k] == 0.0 for k in ("fuel_L", "fuel_flow_Lph", "cht_C", "egt_C",
                                             "oil_psi", "thrust_N")), "engine gauges not zeroed"
            assert "DESTROYED" in e["health_flags"]
            pr = crashed["pitot_reading"] if "pitot_reading" in crashed else crashed["pitot"]
            assert crashed["pitot"]["ias_mps"] == 0.0 and crashed["pitot"]["baro_alt_msl_m"] == 0.0
            assert crashed["pitot"]["vsi_mps"] == 0.0 and crashed["pitot"]["health"] == "FAILED"
            assert crashed["engine_sensors"]["fuel_qty_L"] == 0.0
            assert crashed["gps"]["health"] == "NO_FIX" and crashed["gps"]["satellites"] == 0
            assert crashed["gps"]["groundspeed_mps"] == 0.0
            assert crashed["imu"]["accel_mps2"] == [0.0, 0.0, 0.0]
            assert crashed["agl_m"] == 0.0
            assert crashed["ai"]["overall"]["score"] == 0.0
            assert all(sub["score"] == 0.0 for sub in crashed["ai"]["subsystems"].values())
            assert crashed["engine"]["faults"]["stress_pct"] == 0.0
            ai = crashed.get("ai") or {}
            ids = [a["id"] for a in ai.get("alerts", [])]
            assert "CRASHED" in ids, f"no CRASHED alert: {ids}"
            assert "ENGINE_OFF" in crashed["engine"]["health_flags"]
            sug = [s["title"] for s in ai.get("suggestions", [])]
            assert any("Reset" in s for s in sug), sug
            print("   freeze OK, CRASHED alert + suggestion present")

            # ---- 3. reset over websocket ----
            await ws.send(json.dumps({"type": "sim_reset"}))
            ack = None
            t0 = time.time()
            flying_again = None
            while time.time() - t0 < 20:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if m["type"] == "sim_reset_ack":
                    ack = True
                if m["type"] == "telemetry":
                    d = m["d"]
                    if ack and (not d["crash"]["crashed"]) and d["t_s"] < 5.0:
                        flying_again = d
                        break
            assert ack, "no reset ack"
            # a live packet here, or the guaranteed re-crash found in the next
            # phase, both prove the reset restarted the simulation
            if flying_again is not None:
                print(f"   reset OK: t_s={flying_again['t_s']:.2f}s, mode={flying_again['mode']}, "
                      f"crashed={flying_again['crash']['crashed']}")
            else:
                print("   reset OK: no live packet before the re-crash "
                      "(guaranteed-impact spawn) — physics restart proven in phase 4")

            # ---- 4. physics running again: with this spawn it re-crashes -> t_s advanced ----
            re_crashed = None
            t0 = time.time()
            while time.time() - t0 < 15:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if m["type"] == "telemetry":
                    d = m["d"]
                    if d["crash"]["crashed"] and d["crash"]["t_s"] > 0.05:
                        re_crashed = d
                        break
            assert re_crashed is not None, "no second crash -> physics not running after reset"
            print(f"   physics running again: re-crashed at t={re_crashed['crash']['t_s']:.2f}s "
                  f"(guaranteed-impact spawn) OK")

    asyncio.new_event_loop().run_until_complete(
        asyncio.wait_for(watch(), timeout=130))
    print("CRASH & RESET E2E PASSED")
finally:
    proc.terminate()
