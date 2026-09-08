"""Systematic fault tests: abuse -> seizure, carb ice -> carb heat cure,
oil leak, and the AI monitor's detect/advise/verify/resolve messaging.

Fault timeline semantics (matches faults.py):
  - ice/leak onsets are scheduled in REAL seconds from flight start
  - ice/leak evolution uses the ENGINE aging clock (ENGINE_TIME_SCALE)
  - stress rates are per REAL second of abuse
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulator.config import DT_PHYS, PISTON_ENGINE, ENGINE_TIME_SCALE
from simulator.aircraft.piston_engine import PistonEngine
from simulator.ai_advisor.monitor import HealthMonitor, PilotAssistant


def run_engine(real_seconds, throttle_fn, carb_fn=lambda t: False, rho=1.19, V=58.0):
    """Steps real_seconds of physics; the engine AGING clock runs at
    ENGINE_TIME_SCALE, so each real second ages the engine 60 s."""
    eng = PistonEngine()
    n = int(real_seconds / DT_PHYS)
    for i in range(n):
        t = i * DT_PHYS
        eng.set_carb_heat(carb_fn(t))
        eng.step(throttle_fn(t), DT_PHYS, rho, V)
    return eng


def isolate(eng, abuse=False, ice=False, leak=False):
    eng.faults._ice_this_flight = ice
    eng.faults._leak_this_flight = leak
    if abuse or ice or leak:
        eng.faults._ice_delay = 0.0
        eng.faults._leak_delay = 0.0
    return eng


# ---- 1. sustained high RPM -> staged damage -> seizure (abuse only) ----
eng = isolate(PistonEngine(), abuse=True)
for i in range(int(30.0 / DT_PHYS)):          # 30 real-s at full power
    eng.step(1.0, DT_PHYS, 1.19, 58.0)
assert not eng.faults.seized and 20.0 < eng.faults.stress * 100 < 95.0, \
    "stress should build: %.3f" % eng.faults.stress
assert eng.running and eng.power_kw > 20.0
assert "ENGINE_DAMAGE" in eng.health_flags()
print("1a. abuse 30 s -> stress %.0f%%, power degraded OK" % (eng.faults.stress * 100))

eng = isolate(PistonEngine(), abuse=True)
for i in range(int(120.0 / DT_PHYS)):         # 120 real-s: past the limit
    eng.step(1.0, DT_PHYS, 1.19, 58.0)
assert eng.faults.seized and not eng.running, "sustained abuse must seize the engine"
assert "ENGINE_SEIZED" in eng.health_flags()
eng.command_starter(True)
eng.step(1.0, DT_PHYS, 1.19, 58.0)
assert not eng.running, "seized engine must not restart"
print("1b. sustained abuse -> seizure, no restart OK")

# ---- 2. gentle operation heals partial stress ----
eng2 = isolate(PistonEngine())
eng2.faults.stress = 0.5
for i in range(int(8.0 / DT_PHYS)):           # 8 real-s of gentle cruise
    eng2.step(0.45, DT_PHYS, 1.19, 58.0)
assert eng2.faults.stress < 0.2, "stress should decay gently: %.3f" % eng2.faults.stress
assert eng2.running
print("2. gentle operation heals stress OK (%.2f)" % eng2.faults.stress)

# ---- 3. carb ice builds, carb heat cures it ----
eng3 = isolate(PistonEngine(), ice=True)
for i in range(int(55.0 / DT_PHYS)):          # 55 real-s of icing
    eng3.step(0.55, DT_PHYS, 1.19, 58.0)
assert eng3.faults.ice > 0.2, "ice should build: %.3f" % eng3.faults.ice
assert "CARB_ICE" in eng3.health_flags()
pf_iced = eng3.faults.power_factor
for i in range(int(10.0 / DT_PHYS)):          # 10 real-s of carb heat
    eng3.set_carb_heat(True)
    eng3.step(0.55, DT_PHYS, 1.19, 58.0)
assert eng3.faults.ice < 0.05, "carb heat must cure ice: %.3f" % eng3.faults.ice
assert eng3.faults.power_factor > 0.99
assert "CARB_ICE" not in eng3.health_flags()
print("3. carb ice build + carb-heat cure OK (iced pf=%.2f -> %.2f)" % (pf_iced, eng3.faults.power_factor))

# ---- 4. oil leak drains quantity and pressure ----
eng4 = isolate(PistonEngine(), leak=True)
for i in range(int(130.0 / DT_PHYS)):         # 130 real-s of leaking
    eng4.step(0.7, DT_PHYS, 1.19, 58.0)
assert eng4.faults.leak_active and eng4.faults.oil_qty_pct < 70.0
assert "OIL_LEAK" in eng4.health_flags()
assert eng4.oil_psi < PISTON_ENGINE["oil_pressure_nominal_psi"] - 5.0, eng4.oil_psi
print("4. oil leak OK (qty %.0f%% -> oil %.1f psi)" % (eng4.faults.oil_qty_pct, eng4.oil_psi))

# ---- 5. AI monitor: detect -> advise remedy -> verify -> resolve ----
def snap(t, **over):
    f = over.pop("faults", {})
    s = {
        "t_s": t, "mode": "MANUAL",
        "state": {"pos_ned": [0, 0, -3450.0], "euler": [0.0, 0.03, 0.2], "rates": [0, 0, 0],
                  "vel_body": [58.0, 0, 0]},
        "aero": {"alpha_rad": 0.05, "stall": False, "V_air_mps": 58.0, "CL": 0.45, "CD": 0.05,
                 "q_dyn_Pa": 1080.0},
        "engine_summary": {"running": True, "rpm": 4400.0, "power_kw": 38.0, "fuel_L": 132.0,
                           "fuel_kg_remaining": 105.6, "fuel_flow_Lph": 14.2, "cht_C": 175.0,
                           "egt_C": 700.0, "oil_psi": 65.0, "advance_ratio": 0.6,
                           "carb_heat": over.pop("carb_heat", False), "starter_active": False,
                           "run_time_s": t, "health_flags": [], "faults": f},
        "engine_sensors": {"rpm": 4401.0, "fuel_flow_Lph": 14.2, "fuel_qty_L": 132.0,
                           "cht_C": 175.0, "egt_C": 700.0, "oil_psi": 65.0, "health": "OK",
                           "raw_flags": []},
        "pitot_reading": {"ias_mps": 42.0, "tas_mps": 58.0, "baro_alt_msl_m": 3725.0,
                          "vsi_mps": 0.0, "health": "OK", "new_data": True},
        "gps": {"health": "OK", "hdop": 1.1, "satellites": 10, "groundspeed_mps": 57.0,
                "track_deg": 0.0, "lat_deg": 28.6, "lon_deg": 77.2},
        "nfz_violations": [], "nfz_nearest": {"name": "X", "margin_m": 500.0},
        "agl_m": 3450.0,
        "manual_controls": {"delta_a_deg": 0, "delta_e_deg": -2.5, "delta_r_deg": 0,
                            "delta_e_trim_deg": -2.5, "delta_flap_deg": 0.0, "throttle": 0.6,
                            "gear_down": True, "brakes": False, "carb_heat": False},
        "crash": {"crashed": False},
    }
    s.update(over)
    return s

mon = HealthMonitor()
fired = {}
for i in range(600):
    st = min(70.0, i * 0.15) if i < 400 else max(10.0, 70.0 - (i - 400) * 0.2)
    a = mon.update(snap(i * 0.1, faults={"stress_pct": st, "ice_pct": 0.0, "oil_qty_pct": 100.0,
                                         "power_factor": 1.0 - 0.6 * st / 100.0}))
    for al in a["alerts"]:
        if al.get("_just_emitted"):
            fired.setdefault(al["id"], al["msg"])
assert "STRESS_WARN" in fired, sorted(fired)
assert "STRESS_EASING" in fired, sorted(fired)
print("5a. stress warn + recovery confirmation OK:", fired["STRESS_EASING"][:70])

mon2 = HealthMonitor()
fired2 = {}
for i in range(500):
    t = i * 0.1
    ice = min(45.0, max(0.0, (t - 5) * 1.5)) if t < 33 else max(0.0, 45.0 - (t - 33) * 4.0)
    carb = t > 33
    a2 = mon2.update(snap(t, faults={"stress_pct": 0.0, "ice_pct": ice, "oil_qty_pct": 100.0,
                                     "power_factor": 1.0}, carb_heat=carb))
    for al in a2["alerts"]:
        if al.get("_just_emitted"):
            fired2.setdefault(al["id"], al["msg"])
assert "CARB_ICE_WARN" in fired2, sorted(fired2)
assert "CARB_HEAT_APPLIED" in fired2, sorted(fired2)
assert "CARB_ICE_CLEARED" in fired2, sorted(fired2)
print("5b. ice detect -> remedy ack -> resolved OK:", fired2["CARB_ICE_CLEARED"][:70])

mon3 = HealthMonitor()
fired3 = {}
for i in range(900):
    q = max(5.0, 100.0 - i * 0.11)
    a3 = mon3.update(snap(i * 0.1, faults={"stress_pct": 0.0, "ice_pct": 0.0,
                                           "oil_qty_pct": q, "power_factor": 1.0,
                                           "leak_active": True}))
    for al in a3["alerts"]:
        if al.get("_just_emitted"):
            fired3[al["id"]] = al["msg"]
assert "OIL_LEAK" in fired3
assert "Land NOW" in fired3["OIL_LEAK"] or "land NOW" in fired3["OIL_LEAK"], fired3["OIL_LEAK"]
print("5c. oil leak escalation OK:", fired3["OIL_LEAK"][:60])

pa = PilotAssistant()
ans = pa.answer("what is wrong with the engine?", snap(10.0, faults={"stress_pct": 60.0,
                "ice_pct": 40.0, "oil_qty_pct": 55.0, "power_factor": 0.7, "leak_active": True},
                carb_heat=False), {"alerts": [], "suggestions": [], "insights": {}, "subsystems": {}})
assert "stress" in ans and "ice" in ans and "oil" in ans
print("5d. assistant fault Q&A OK:", ans[:80])

print("ALL FAULT TESTS PASSED")
