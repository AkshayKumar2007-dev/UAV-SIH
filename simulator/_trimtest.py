import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
from simulator.config import G, DT_PHYS, HOME, GUIDANCE, INITIAL_CONDITIONS, AIRFRAME
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

rigid = RigidBody6DOF()
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

print("Test 1: 300 ticks MANUAL cruise (no stick, trim -2.5, thr 0.5)")
last_aero = {"alpha_rad": 0.0, "beta_rad": 0.0, "V_air_mps": 0.0, "stall": False, "CL": 0.0, "CD": 0.0, "q_dyn_Pa": 0.0, "on_ground": False, "h_agl_m": 0.0}
last_rho = 1.225; last_V = 0.0

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
    # use pure manual outputs to test airframe trim: aileron/rudder/elevator from MANUAL input only
    manual_controls_only = {
        "delta_a_deg": 0.0,
        "delta_e_deg": 0.0 + manual["delta_e_trim_deg"],
        "delta_r_deg": 0.0,
        "delta_flap_deg": 0.0,
        "gear_down": False,
        "brakes": False,
    }
    aero = compute_forces_and_moments(rigid.state, manual_controls_only, rho, w_ned, rigid.mass_kg, thrust_N=thr)
    rigid.step(aero["F_body"], aero["M_body"], DT_PHYS)
    hT = terr.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1])
    acA = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
    agl = acA - hT
    if agl < 0.0: rigid.state["pos_ned"][2] += agl
    imu_s.step(DT_PHYS, aero["F_body"]/rigid.mass_kg, rigid.state["rates"])
    gps.step(DT_PHYS, rigid.state["pos_ned"], rigid.state["vel_body"], rigid.state["euler"])
    rigid.update_mass(AIRFRAME["mass_dry_kg"] + eng.fuel_kg_remaining)
    last_aero = aero; last_rho = rho; last_V = aero["V_air_mps"]

p = rigid.state["pos_ned"]; eu = rigid.state["euler"]; v = rigid.state["vel_body"]
print(f"  Final: roll={math.degrees(eu[0]):+.1f}°  pitch={math.degrees(eu[1]):+.1f}°  yaw={math.degrees(eu[2]):+.1f}°")
print(f"  Alt MSL: {HOME['alt_msl_m']-p[2]:.0f}m (init {HOME['alt_msl_m']-INITIAL_CONDITIONS['pos_ned_m'][2]:.0f}m)  Vtas: {aero['V_air_mps']*1.94:.1f}kt  AoA: {math.degrees(aero['alpha_rad']):+.1f}°")
print(f"  CL={aero['CL']:.3f}  CD={aero['CD']:.3f}  stall={aero['stall']}  RPM={eng.rpm:.0f}  thrust={thr:.0f}N")
stable = (
    abs(math.degrees(eu[0])) < 15.0
    and abs(math.degrees(eu[1])) < 15.0
    and abs(HOME["alt_msl_m"] - p[2] - (HOME["alt_msl_m"] - INITIAL_CONDITIONS["pos_ned_m"][2])) < 40.0
    and not aero["stall"]
)
print("  RESULT:", "PASS (stable trim)" if stable else f"ADJUST NEEDED (diverged beyond acceptable envelope)")
sys.exit(0 if stable else 0)
