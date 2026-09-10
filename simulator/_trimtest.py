"""Headless MANUAL level-cruise trim test (no pygame / no telemetry).

Starts the airframe in the air at cruise speed in still air, hands off the
controls (fixed elevator trim, fixed throttle), and checks that the
airframe holds a stable cruise within a reasonable envelope.

Why airborne instead of a runway takeoff: from a standing start the
aircraft needs ~30 s just to reach rotation speed, so a 6 s run could
never reach cruise; the only way it used to "fly" in that window was via
bogus low-speed aero coefficients (alpha wrapping to +/-180 deg when the
relative flow comes from behind, which extrapolated the linear Cm/CL
curves to absurd values and pitched the parked aircraft over). Those
coefficients are now folded about +/-90 deg in aerodynamics.py, and this
test isolates the airframe's steady-state trim behaviour directly.

Still air (zero wind) keeps the check deterministic: crosswind
lateral-directional response is _stabtest's job, not a trim check's.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
from simulator.config import (
    G, DT_PHYS, HOME, GUIDANCE, INITIAL_CONDITIONS, AIRFRAME, WIND,
)
from simulator.aircraft.rigid_body import RigidBody6DOF, rot_ned_b
from simulator.aircraft.aerodynamics import compute_forces_and_moments
from simulator.aircraft.piston_engine import PistonEngine
from simulator.environment.atmosphere import isa_atmosphere, WindField
from simulator.environment.world import WorldTerrain, WaypointMission
from simulator.sensors.gps import GPS
from simulator.sensors.imu import IMU
from simulator.sensors.pitot import PitotStatic
from simulator.sensors.engine_sensors import EngineSensors
from simulator.flight_controller.autopilot import Autopilot

# Still air: zero the wind field so the trim check is deterministic.
WIND["speed_north_mps"] = 0.0
WIND["speed_east_mps"] = 0.0
WIND["gust_amplitude_mps"] = 0.0
WIND["turbulence_intensity"] = 0.0

CRUISE_IAS_MPS = GUIDANCE["cruise_ias_mps"]
START_ALT_MSL_M = 500.0

rigid = RigidBody6DOF()
# Airborne start: over the airfield at 500 m MSL, at cruise speed, nose
# slightly up near the trimmed attitude, wings level.
rigid.state["pos_ned"] = np.array(
    [0.0, 0.0, -(START_ALT_MSL_M - HOME["alt_msl_m"])])
rigid.state["vel_body"] = np.array([CRUISE_IAS_MPS, 0.0, 0.0])
rigid.state["euler"] = np.array([0.0, math.radians(2.3), 0.0])
rigid.state["rates"] = np.zeros(3)

eng = PistonEngine()
wind = WindField()
terr = WorldTerrain()
mission = WaypointMission()
gps = GPS()
imu_s = IMU()
pit = PitotStatic()
es = EngineSensors()
ap = Autopilot()
ap.target_alt_msl_m = float(HOME["alt_msl_m"] - INITIAL_CONDITIONS["pos_ned_m"][2])
ap.target_hdg_deg = 0.0
ap.target_ias_mps = GUIDANCE["cruise_ias_mps"]

print(f"Test 1: {300} ticks MANUAL level cruise (no stick, trim -2.5, thr 0.5, still air)")
last_aero = {"alpha_rad": 0.0, "beta_rad": 0.0, "V_air_mps": 0.0, "stall": False, "CL": 0.0, "CD": 0.0, "q_dyn_Pa": 0.0, "on_ground": False, "h_agl_m": 0.0}
last_rho = 1.225; last_V = 0.0
start_alt = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
alt_lo = alt_hi = start_alt
max_pitch = 0.0

for i in range(300):
    wind.step(DT_PHYS)
    atm = isa_atmosphere(rigid.altitude_msl_m)
    rho = atm["rho_kgm3"]
    Rnb = rot_ned_b(*rigid.state["euler"])
    w_ned = wind.wind_at(rigid.altitude_msl_m, rigid.state["pos_ned"])
    wb = Rnb @ w_ned
    Vae = float(np.linalg.norm(rigid.state["vel_body"] - wb))
    thr = eng.step(0.5, DT_PHYS, rho, Vae)
    pit.step(DT_PHYS, last_V, last_rho, rigid.altitude_msl_m)
    manual = {
        "delta_a_deg": 0.0, "delta_e_deg": 0.0, "delta_r_deg": 0.0,
        "delta_flap_deg": 0.0, "gear_down": False, "brakes": False,
        "carb_heat": False, "delta_e_trim_deg": -2.5, "throttle": 0.5,
    }
    sensors = {"pitot": pit.reading(), "gps": gps.reading(), "imu": imu_s.reading(), "engine": es.reading(eng.summary())}
    ctrl, dbg = ap.step(DT_PHYS, rigid.state, last_aero, sensors, manual, mission)
    # pure airframe: aileron/rudder/elevator from MANUAL input only (AP output discarded)
    manual_controls_only = {
        "delta_a_deg": 0.0,
        "delta_e_deg": 0.0 + manual["delta_e_trim_deg"],
        "delta_r_deg": 0.0,
        "delta_flap_deg": 0.0,
        "gear_down": False,
        "brakes": False,
    }
    agl0 = max(0.0, (HOME["alt_msl_m"] - rigid.state["pos_ned"][2])
               - terr.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1]))
    aero = compute_forces_and_moments(rigid.state, manual_controls_only, rho, w_ned, rigid.mass_kg,
                                      thrust_N=thr, agl_m=agl0)
    rigid.step(aero["F_body"], aero["M_body"], DT_PHYS)
    hT = terr.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1])
    acA = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
    agl = acA - hT
    if agl < 0.0:
        rigid.state["pos_ned"][2] += agl
        rigid.state["vel_body"][2] = min(rigid.state["vel_body"][2], 0.0)
    imu_s.step(DT_PHYS, aero["F_body"]/rigid.mass_kg, rigid.state["rates"])
    gps.step(DT_PHYS, rigid.state["pos_ned"], rigid.state["vel_body"], rigid.state["euler"])
    rigid.update_mass(AIRFRAME["mass_dry_kg"] + eng.fuel_kg_remaining)
    last_aero = aero; last_rho = rho; last_V = aero["V_air_mps"]
    alt_lo = min(alt_lo, acA); alt_hi = max(alt_hi, acA)
    max_pitch = max(max_pitch, abs(math.degrees(rigid.state["euler"][1])))

p = rigid.state["pos_ned"]; eu = rigid.state["euler"]; v = rigid.state["vel_body"]
print(f"  Final: roll={math.degrees(eu[0]):+.1f}°  pitch={math.degrees(eu[1]):+.1f}°  yaw={math.degrees(eu[2]):+.1f}°")
print(f"  Alt MSL: {HOME['alt_msl_m']-p[2]:.0f}m (start {start_alt:.0f}m, swing {alt_lo:.0f}-{alt_hi:.0f}m)  Vtas: {aero['V_air_mps']*1.94:.1f}kt  AoA: {math.degrees(aero['alpha_rad']):+.1f}°")
print(f"  CL={aero['CL']:.3f}  CD={aero['CD']:.3f}  stall={aero['stall']}  RPM={eng.rpm:.0f}  thrust={thr:.0f}N")
stable = (
    abs(math.degrees(eu[0])) < 15.0
    and abs(math.degrees(eu[1])) < 15.0
    and alt_hi - start_alt < 40.0
    and start_alt - alt_lo < 40.0
    and not aero["stall"]
)
print("  RESULT:", "PASS (stable level-cruise trim)" if stable else "ADJUST NEEDED (diverged beyond acceptable envelope)")
sys.exit(0 if stable else 1)