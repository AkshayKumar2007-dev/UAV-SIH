"""Headless autopilot stability test (no pygame / no telemetry).

Engages STAB at t=5s and flies 200 s; then WAYPOINT for 120 s.
Prints pass/fail metrics: roll, pitch, altitude error, airspeed, stalls.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

I_PROP_KGM2 = 0.009
P_FACTOR_K = 0.018

def prop_gyro_pf(thrust_N, rpm, alpha, rates, prop_diam):
    omega = rpm * 2 * math.pi / 60.0
    L = I_PROP_KGM2 * omega
    p, q, r = rates
    return np.array([0.0, -r * L, q * L]) + np.array(
        [0.0, 0.0, -P_FACTOR_K * thrust_N * math.sin(alpha) * (prop_diam / 2)])

def run(name, engage_mode, engage_t, total_t, clear_mission=False, expect_capture=False):
    rigid = RigidBody6DOF()
    eng = PistonEngine()
    wind = WindField()
    terr = WorldTerrain()
    mission = WaypointMission()
    gps = GPS(); imu_s = IMU(); pit = PitotStatic(); es = EngineSensors()
    ap = Autopilot()
    eng.faults._ice_delay = 1e9     # keep the stability test deterministic
    eng.faults._leak_delay = 1e9
    if clear_mission:
        mission.clear()
    ap.target_alt_msl_m = float(HOME["alt_msl_m"] - INITIAL_CONDITIONS["pos_ned_m"][2])
    ap.target_hdg_deg = 0.0
    ap.target_ias_mps = GUIDANCE["cruise_ias_mps"]

    last_aero = {"alpha_rad": 0.0, "beta_rad": 0.0, "V_air_mps": 0.0, "stall": False,
                 "CL": 0.0, "CD": 0.0, "q_dyn_Pa": 0.0, "on_ground": False, "h_agl_m": 0.0}
    last_V = 0.0; last_rho = 1.225
    manual = {"delta_a_deg": 0.0, "delta_e_deg": -2.5, "delta_r_deg": 0.0, "delta_flap_deg": 0.0,
              "gear_down": False, "brakes": False, "carb_heat": False,
              "delta_e_trim_deg": -2.5, "throttle": 0.55}
    N = int(total_t / DT_PHYS)
    t = 0.0
    cap_alt = None
    max_roll = max_pitch = 0.0; min_ias = 1e9; max_ias = 0.0
    stalls = 0; agl_min = 1e9
    alt_errs = []
    airborne_seen = False
    settled = int((engage_t + 75) / DT_PHYS)   # takeoff + climb-out complete
    for i in range(N):
        wind.step(DT_PHYS)
        atm = isa_atmosphere(rigid.altitude_msl_m); rho = atm["rho_kgm3"]
        Rnb = rot_ned_b(*rigid.state["euler"])
        w_ned = wind.wind_at(rigid.altitude_msl_m, rigid.state["pos_ned"])
        vab = rigid.state["vel_body"] - Rnb @ w_ned
        Vae = float(np.linalg.norm(vab))
        pit.step(DT_PHYS, last_V, last_rho, rigid.altitude_msl_m)
        sensors = {"pitot": pit.reading(), "gps": gps.reading(), "imu": imu_s.reading(),
                   "engine": es.reading(eng.summary())}
        if t >= engage_t and ap.mode != engage_mode:
            ap.set_mode(engage_mode)
        ctrl, dbg = ap.step(DT_PHYS, rigid.state, last_aero, sensors, manual, mission)
        thr = eng.step(ctrl["throttle"], DT_PHYS, rho, Vae)
        aero = compute_forces_and_moments(rigid.state, ctrl, rho, w_ned, rigid.mass_kg, thrust_N=thr)
        Ma = aero["M_body"] + prop_gyro_pf(thr, eng.rpm, aero["alpha_rad"], rigid.state["rates"], 0.66)
        rigid.step(aero["F_body"], Ma, DT_PHYS)
        hT = terr.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1])
        acA = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
        agl = acA - hT
        if agl < 0.0:
            rigid.state["pos_ned"][2] += agl
            rigid.state["vel_body"][2] = min(rigid.state["vel_body"][2], 0.0)
        # gear: wheels hold the attitude on the ground
        if agl < 0.6:
            rigid.state["rates"][0] *= 0.55
            rigid.state["rates"][1] *= 0.55
            rigid.state["euler"][0] *= 0.82
            if pit.reading()["ias_mps"] < 8.0:
                rigid.state["euler"][1] = float(np.clip(rigid.state["euler"][1], -0.06, 0.16))
        rigid.update_mass(560.0 + eng.fuel_kg_remaining)
        imu_s.step(DT_PHYS, aero["F_body"] / rigid.mass_kg, rigid.state["rates"])
        gps.step(DT_PHYS, rigid.state["pos_ned"], rigid.state["vel_body"], rigid.state["euler"])
        rigid.update_mass(560.0 + eng.fuel_kg_remaining)
        last_aero = aero; last_rho = rho; last_V = aero["V_air_mps"]
        t += DT_PHYS

        if t >= engage_t:
            if expect_capture:
                cap_alt = mission.current()["alt_msl_m"] if mission.count() else cap_alt
            elif cap_alt is None:
                cap_alt = ap.target_alt_msl_m
            if agl > 15.0:
                airborne_seen = True
            roll = abs(math.degrees(rigid.state["euler"][0]))
            pitch = abs(math.degrees(rigid.state["euler"][1]))
            ias = pit.reading()["ias_mps"]
            max_roll = max(max_roll, roll); max_pitch = max(max_pitch, pitch)
            if airborne_seen:
                min_ias = min(min_ias, ias); max_ias = max(max_ias, ias)
                if agl > 3.0:
                    agl_min = min(agl_min, agl)   # exclude touch-and-go moments
            if aero["stall"]:
                stalls += 1
            if i >= settled:
                alt_errs.append(abs(acA - cap_alt))
    ae = np.array(alt_errs)
    n_cap = sum(mission.captured)
    print(f"== {name}: engage {engage_mode} @ t={engage_t}s, flew to {total_t}s")
    print(f"   max|roll| {max_roll:6.1f} deg | max|pitch| {max_pitch:6.1f} deg | "
          f"IAS {min_ias:5.1f}-{max_ias:5.1f} m/s | min AGL {agl_min:6.1f} m | stall ticks {stalls}")
    if len(ae):
        print(f"   alt err after settle: mean {ae.mean():6.1f} m | max {ae.max():6.1f} m")
    if expect_capture:
        print(f"   waypoints captured: {n_cap}/{len(mission.items)}")
    roll_lim = 15.0 if (engage_mode == "STAB" and not expect_capture) else 38.0
    alt_lim = 120.0 if (engage_mode == "STAB" and not expect_capture) else 160.0
    ok = (max_roll < roll_lim and max_pitch < 30 and stalls == 0
          and (not airborne_seen or min_ias > 20)
          and (not len(ae) or ae.mean() < alt_lim)
          and (not expect_capture or n_cap >= 1))
    print(f"   => {'PASS' if ok else 'FAIL'}")
    return ok

ok1 = run("STABLE-HOLD", "STAB", 5.0, 205.0, clear_mission=True)
ok2 = run("STABLE-NAV", "STAB", 5.0, 205.0, expect_capture=True)
ok3 = run("WAYPOINT", "WAYPOINT", 5.0, 205.0, expect_capture=True)
print("ALL PASS" if (ok1 and ok2 and ok3) else "SOME FAILED")
