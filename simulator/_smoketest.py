import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
from simulator.config import G, DT_PHYS, HOME, GUIDANCE, INITIAL_CONDITIONS
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
last_aero = {"alpha_rad": 0.0, "beta_rad": 0.0, "V_air_mps": 0.0, "stall": False, "CL": 0.0, "CD": 0.0, "q_dyn_Pa": 0.0, "on_ground": False, "h_agl_m": 0.0}
last_rho = 1.225
last_V = 0.0
N = 1500
for i in range(N):
    wind.step(DT_PHYS)
    atm = isa_atmosphere(rigid.altitude_msl_m)
    rho = atm["rho_kgm3"]
    Rnb = rot_ned_b(*rigid.state["euler"])
    w_ned = wind.wind_at(rigid.altitude_msl_m, rigid.state["pos_ned"])
    wb = Rnb @ w_ned
    vab = rigid.state["vel_body"] - wb
    Vae = float(np.linalg.norm(vab))
    thr = eng.step(0.52, DT_PHYS, rho, Vae)
    pit.step(DT_PHYS, last_V, last_rho, rigid.altitude_msl_m)
    manual = {
        "delta_a_deg": 0.0,
        "delta_e_deg": 0.0,
        "delta_r_deg": 0.0,
        "delta_flap_deg": 0.0,
        "gear_down": False,
        "brakes": False,
        "carb_heat": False,
        "delta_e_trim_deg": -2.5,
        "throttle": 0.52,
    }
    sensors = {
        "pitot": pit.reading(),
        "gps": gps.reading(),
        "imu": imu_s.reading(),
        "engine": es.reading(eng.summary()),
    }
    ctrl, dbg = ap.step(DT_PHYS, rigid.state, last_aero, sensors, manual, mission)
    if i == 120:
        ap.set_mode("ALT_HOLD")
    if i == 600:
        ap.set_mode("WAYPOINT")
    aero = compute_forces_and_moments(rigid.state, ctrl, rho, w_ned, rigid.mass_kg, thrust_N=thr)
    Fa = aero["F_body"]
    Ma = aero["M_body"]
    rigid.step(Fa, Ma, DT_PHYS)
    hT = terr.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1])
    acA = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
    agl = acA - hT
    if agl < 0.0:
        rigid.state["pos_ned"][2] += agl
    imu_s.step(DT_PHYS, Fa / rigid.mass_kg, rigid.state["rates"])
    gps.step(DT_PHYS, rigid.state["pos_ned"], rigid.state["vel_body"], rigid.state["euler"])
    rigid.update_mass(560.0 + eng.fuel_kg_remaining)
    last_aero = aero
    last_rho = rho
    last_V = aero["V_air_mps"]

p = rigid.state["pos_ned"]
v = rigid.state["vel_body"]
eu = rigid.state["euler"]
print("Ticks OK. Final pos NED=({:.1f},{:.1f},{:.1f})  vel_body=({:.1f},{:.1f},{:.1f}) m/s".format(p[0], p[1], p[2], v[0], v[1], v[2]))
print("  Euler roll={:.1f} pitch={:.1f} yaw={:.1f} deg".format(*[math.degrees(x) for x in eu]))
print("  Engine RPM={:.0f} fuel={:.2f}L CHT={:.0f}C EGT={:.0f}C oil={:.1f}psi running={}".format(eng.rpm, eng.fuel_L, eng.cht_C, eng.egt_C, eng.oil_psi, eng.running))
gr = gps.reading()
pr = pit.reading()
print("  GPS:", gr["health"], "lat={:.5f} lon={:.5f} sats={} HDOP={:.2f}".format(gr["lat_deg"], gr["lon_deg"], gr["satellites"], gr["hdop"]))
print("  Pitot IAS={:.1f}kt baroAlt={:.0f}ft VSI={:.0f}fpm".format(pr["ias_mps"] * 1.94, pr["baro_alt_msl_m"] * 3.28, pr["vsi_mps"] * 197))
print("  Aero: AoA={:.2f}deg  Vtas={:.1f}kt  CL={:.3f} CD={:.3f} stall={} on_ground={}".format(
    math.degrees(aero["alpha_rad"]), aero["V_air_mps"] * 1.94, aero["CL"], aero["CD"], aero["stall"], aero["on_ground"]))
print("  Mode={}  flags={}".format(dbg.get("mode", "?"), eng.health_flags()))
print("SMOKE TEST PASSED")
