import os
import sys
import math
import time
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import (
    SIM_PHYSICS_HZ, DT_PHYS, SIM_RENDER_HZ, TELEMETRY_HZ, HOME, G,
    GUIDANCE, WORLD, AERO, AIRFRAME, PISTON_ENGINE, INITIAL_CONDITIONS, CRASH,
    ENGINE_TIME_SCALE,
)
from simulator.aircraft.rigid_body import RigidBody6DOF, rot_ned_b
from simulator.aircraft.aerodynamics import compute_forces_and_moments
from simulator.aircraft.piston_engine import PistonEngine
from simulator.environment.atmosphere import isa_atmosphere, WindField
from simulator.environment.world import WorldTerrain, NoFlyZones, WaypointMission
from simulator.sensors.gps import GPS
from simulator.sensors.imu import IMU
from simulator.sensors.pitot import PitotStatic
from simulator.sensors.engine_sensors import EngineSensors
from simulator.flight_controller.autopilot import Autopilot
from simulator.ai_advisor.monitor import HealthMonitor, PilotAssistant
from simulator.ai_advisor.twin import EngineDigitalTwin
from simulator.telemetry.server import TelemetryServer
from simulator.ui.input_handler import InputHandler
from simulator.ui.renderer import Renderer


I_PROP_KGM2 = 0.009  # 2-blade propeller inertia (rough)
P_FACTOR_K = 0.018   # asymmetric blade thrust moment arm (m) scaled


def _obstacle_hit(pos_ned, alt_msl, obstacles):
    """Returns the obstacle dict the aircraft is inside (below its top), or None."""
    for ob in obstacles:
        if alt_msl >= ob["top"]:
            continue
        dn = pos_ned[0] - ob["n"]
        de = pos_ned[1] - ob["e"]
        if dn * dn + de * de < ob["r"] * ob["r"]:
            return ob
    return None


def _evaluate_impact(vs_impact_mps, bank_deg, pitch_deg, ias_mps):
    """Returns a crash reason string, or None if the airframe survived the
    terrain contact (i.e. it was a landing)."""
    if vs_impact_mps < CRASH["vs_limit_mps"]:
        return f"hard impact ({vs_impact_mps:.0f} m/s vertical)"
    if abs(bank_deg) > CRASH["bank_limit_deg"]:
        return f"wing strike (bank {abs(bank_deg):.0f} deg)"
    if pitch_deg < CRASH["pitch_down_limit_deg"]:
        return f"nose-first impact (pitch {pitch_deg:.0f} deg)"
    if pitch_deg > CRASH["pitch_up_limit_deg"]:
        return f"tail strike (pitch +{pitch_deg:.0f} deg)"
    if ias_mps > CRASH["ias_limit_mps"]:
        return f"high-speed terrain impact ({ias_mps * 1.94384:.0f} kt)"
    return None


def _prop_gyro_and_pfactor(thrust_N, rpm, alpha_rad, rates_body_radps, prop_diam_m):
    omega_prop = rpm * 2.0 * math.pi / 60.0
    L = I_PROP_KGM2 * omega_prop
    p, q, r = rates_body_radps
    M_gyro = np.array([0.0, -r * L, q * L])
    M_pf = np.array([0.0, 0.0, -P_FACTOR_K * thrust_N * math.sin(alpha_rad) * (prop_diam_m / 2.0)])
    return M_gyro + M_pf


def _zero_sim_snapshot(s):
    """After a crash every live reading is zeroed — the wreck is inert.
    Position/orientation survive so the map and 3D scene show the site."""
    s["pitot"] = s["pitot_reading"] = {
        "ias_mps": 0.0, "tas_mps": 0.0, "baro_alt_msl_m": 0.0, "vsi_mps": 0.0,
        "health": "FAILED", "new_data": False,
    }
    zero_faults = {"stress_pct": 0.0, "ice_pct": 0.0, "oil_qty_pct": 0.0,
                   "leak_active": False, "seized": True, "power_factor": 0.0}
    zero_engine = {
        "rpm": 0.0, "running": False, "thrust_N": 0.0, "power_kw": 0.0,
        "fuel_L": 0.0, "fuel_kg_remaining": 0.0, "fuel_flow_Lph": 0.0,
        "cht_C": 0.0, "egt_C": 0.0, "oil_psi": 0.0, "advance_ratio": 0.0,
        "carb_heat": False, "starter_active": False, "run_time_s": 0.0,
        "health_flags": ["ENGINE_OFF", "DESTROYED"], "faults": zero_faults,
    }
    s["engine"] = s["engine_summary"] = zero_engine
    s["engine_sensors"] = {"rpm": 0.0, "fuel_flow_Lph": 0.0, "fuel_qty_L": 0.0,
                           "cht_C": 0.0, "egt_C": 0.0, "oil_psi": 0.0,
                           "health": "FAILED", "raw_flags": ["DESTROYED"]}
    g = dict(s.get("gps") or {})
    g.update({"vn_mps": 0.0, "ve_mps": 0.0, "vd_mps": 0.0, "groundspeed_mps": 0.0,
              "satellites": 0, "hdop": 0.0, "health": "NO_FIX", "new_data": False})
    s["gps"] = g
    s["imu"] = {"accel_mps2": [0.0, 0.0, 0.0], "gyro_radps": [0.0, 0.0, 0.0],
                "temp_c": 0.0, "bias_accel_mps2": [0.0, 0.0, 0.0],
                "bias_gyro_radps": [0.0, 0.0, 0.0], "health": "FAILED", "new_data": False}
    s["agl_m"] = 0.0


def _screen_click_to_ned(pos, ac_pos_ned, scale):
    W, H = pygame.display.get_surface().get_size()
    mw = W // 2 - 20
    x0, y0 = 10, 10
    cx = x0 + mw // 2
    cy = y0 + (H - 120) // 2
    sx, sy = pos
    e = (sx - cx) / scale + ac_pos_ned[1]
    n = -(sy - cy) / scale + ac_pos_ned[0]
    return n, e


def main():
    pygame.init()

    rigid = RigidBody6DOF()
    engine = PistonEngine()
    wind = WindField()
    terrain = WorldTerrain()
    no_fly = NoFlyZones()
    mission = WaypointMission()
    gps = GPS()
    imu = IMU()
    pitot = PitotStatic()
    eng_sensors = EngineSensors()
    ap = Autopilot()
    initial_alt_msl = HOME["alt_msl_m"] - INITIAL_CONDITIONS["pos_ned_m"][2]
    ap.target_alt_msl_m = float(initial_alt_msl)
    ap.target_hdg_deg = 0.0
    ap.target_ias_mps = GUIDANCE["cruise_ias_mps"]
    ap.set_mode("STAB")                       # auto-takeoff from the runway
    mission.home_ned = np.array([0.0, 0.0, 0.0])
    inp = InputHandler()
    renderer = Renderer()

    # AI flight assistant (suggestions only — no control path) + telemetry page
    advisor = HealthMonitor()
    assistant = PilotAssistant()
    twin = EngineDigitalTwin()
    server = TelemetryServer()

    def _on_ai_query(text):
        snap = server.latest_snapshot
        if not snap:
            return "No telemetry received yet — start flying first."
        return assistant.answer(text, snap, snap.get("ai") or {})

    server.set_query_handler(_on_ai_query)
    server.start()
    renderer.log_advisory("INFO", "[ATC] UAV-01 cleared for takeoff — runway 36.", 0.0)
    print("=" * 64)
    print("  Telemetry dashboard (3D):  http://127.0.0.1:8766")
    print("  WebSocket stream:          ws://127.0.0.1:8765")
    print("  AI assistant: SUGGESTIONS ONLY — it cannot fly the UAV.")
    print("=" * 64)

    last_time = time.time()
    accumulator = 0.0
    render_accum = 0.0
    tel_accum = 0.0
    t_s = 0.0
    frames_rendered = 0
    fps_t = time.time()
    last_fps = 60.0

    last_agl = terrain.agl_at(rigid.state["pos_ned"])
    last_V_air = rigid.airspeed_body_mps
    last_rho = isa_atmosphere(rigid.altitude_msl_m)["rho_kgm3"]
    last_aero = {"alpha_rad": 0.0, "beta_rad": 0.0, "V_air_mps": 0.0, "stall": False, "CL": 0.0, "CD": 0.0, "q_dyn_Pa": 0.0, "on_ground": False, "h_agl_m": 0.0}
    ap_debug = {"mode": ap.mode}
    crash_state = {"crashed": False, "reason": "", "t_s": 0.0}
    prev_alt_msl = rigid.altitude_msl_m
    last_touchdown_t = -1e9
    was_airborne = False    # True only after the UAV has actually been flying
    last_airborne = True

    def _do_reset():
        """Full simulator reset (pilot presses R or clicks RESET SIM on the
        dashboard). Restores spawn state, engine, sensors, mission and AI history."""
        nonlocal rigid, engine, gps, imu, pitot, eng_sensors, mission, ap, twin
        nonlocal t_s, accumulator, tel_accum, last_aero, last_V_air, last_rho, last_agl
        nonlocal prev_alt_msl, last_touchdown_t, ap_debug
        rigid = RigidBody6DOF()
        engine = PistonEngine()
        gps = GPS()
        imu = IMU()
        pitot = PitotStatic()
        eng_sensors = EngineSensors()
        mission = WaypointMission()
        mission.home_ned = np.array([0.0, 0.0, 0.0])
        ap = Autopilot()
        ap.target_alt_msl_m = float(HOME["alt_msl_m"] - INITIAL_CONDITIONS["pos_ned_m"][2])
        ap.target_hdg_deg = 0.0
        ap.target_ias_mps = GUIDANCE["cruise_ias_mps"]
        ap.set_mode("STAB")                   # reset: auto-takeoff again
        advisor.reset()
        twin = EngineDigitalTwin()
        crash_state["crashed"] = False
        crash_state["reason"] = ""
        crash_state["t_s"] = 0.0
        t_s = 0.0
        accumulator = 0.0
        tel_accum = 0.0
        last_aero = {"alpha_rad": 0.0, "beta_rad": 0.0, "V_air_mps": 0.0, "stall": False,
                     "CL": 0.0, "CD": 0.0, "q_dyn_Pa": 0.0, "on_ground": False, "h_agl_m": 0.0}
        last_V_air = rigid.airspeed_body_mps
        last_rho = isa_atmosphere(rigid.altitude_msl_m)["rho_kgm3"]
        last_agl = terrain.agl_at(rigid.state["pos_ned"])
        prev_alt_msl = rigid.altitude_msl_m
        last_touchdown_t = -1e9
        was_airborne = False
        last_airborne = True
        ap_debug = {"mode": ap.mode}
        inp.throttle = 0.55

    # WS thread only flags the request; the reset itself runs on the main thread
    reset_requests = []
    server.set_reset_handler(lambda: reset_requests.append(1))

    running = True
    sim_state_for_render = {}

    while running:
        now = time.time()
        frame_dt = min(now - last_time, 0.1)
        last_time = now
        accumulator += frame_dt
        render_accum += frame_dt

        inp.process_events(frame_dt)
        if inp.quit:
            running = False
            break
        if inp.paused:
            time.sleep(0.02)
            continue
        if inp.screenshot:
            try:
                pygame.image.save(renderer.screen, f"screenshot_{int(t_s)}.png")
                print(f"Saved screenshot at t={t_s:.0f}s")
            except Exception as ex:
                print("screenshot error", ex)

        if inp.mode_toggle is not None:
            m = ap.set_mode(inp.mode_toggle)
            renderer.log_advisory("INFO", f"Mode → {m}", t_s)
        if inp.waypoint_next and mission.count() > 0:
            mission.advance()
            renderer.log_advisory("INFO", f"Advance → WP {mission.current_index+1}", t_s)
        if inp.map_toggle:
            renderer.show_map = not renderer.show_map
        if inp.hud_toggle:
            renderer.show_hud = not renderer.show_hud
        if inp.mission_clear:
            mission.clear()
            renderer.log_advisory("WARN", "Mission cleared", t_s)
        if inp.waypoint_add is not None:
            n_m, e_m = _screen_click_to_ned(inp.waypoint_add, rigid.state["pos_ned"], renderer.map_zoom)
            alt = float(max(HOME["alt_msl_m"] - rigid.state["pos_ned"][2] + 50, HOME["alt_msl_m"] + 80))
            mission.add_waypoint(n_m, e_m, alt)
            renderer.log_advisory("INFO", f"Added WP {mission.count()} @ ({n_m:.0f}, {e_m:.0f}, {alt:.0f}m)", t_s)

        if inp.starter_pulse and not engine.running:
            engine.command_starter(True)
            renderer.log_advisory("INFO", "Starter cranking...", t_s)

        if inp.reset or reset_requests:
            reset_requests.clear()
            _do_reset()
            renderer.log_advisory("INFO", "Simulator reset — on the runway, cleared for takeoff.", t_s)

        while accumulator >= DT_PHYS:
            accumulator -= DT_PHYS
            if crash_state["crashed"]:
                continue   # wreck frozen: drain the accumulator, simulate nothing
            t_s += DT_PHYS
            wind.step(DT_PHYS)
            engine.set_carb_heat(inp.carb_heat)

            manual = inp.controls_dict()

            atm = isa_atmosphere(rigid.altitude_msl_m)
            rho = atm["rho_kgm3"]
            wind_ned = wind.wind_at(rigid.altitude_msl_m, rigid.state["pos_ned"])

            Rnb = rot_ned_b(*rigid.state["euler"])
            wind_body_est = Rnb @ wind_ned
            V_body_air_est = rigid.state["vel_body"] - wind_body_est
            V_air_est = float(np.linalg.norm(V_body_air_est))
            alpha_est = math.atan2(V_body_air_est[2], V_body_air_est[0]) if V_body_air_est[0] > 0.5 else 0.0

            pitot.step(DT_PHYS, last_V_air, last_rho, rigid.altitude_msl_m)

            sensor_data = {
                "pitot": pitot.reading(),
                "gps": gps.reading(),
                "imu": imu.reading(),
                "engine": eng_sensors.reading(engine.summary()),
                "on_ground": last_agl < 1.0,
            }
            controls, ap_debug = ap.step(DT_PHYS, rigid.state, last_aero, sensor_data, manual, mission)
            ap_debug["mode"] = ap.mode

            # the AP's throttle command drives the engine (pilot lever in MANUAL)
            thrust_N = engine.step(controls["throttle"], DT_PHYS, rho, V_air_est)

            F_applied = np.zeros(3)
            M_applied = np.zeros(3)
            total_mass = rigid.mass_kg
            aero_res = compute_forces_and_moments(
                rigid.state, controls, rho, wind_ned, total_mass, thrust_N=thrust_N,
            )
            F_applied += aero_res["F_body"]
            M_applied += aero_res["M_body"]

            M_gyr_pf = _prop_gyro_and_pfactor(
                thrust_N, engine.rpm, aero_res["alpha_rad"], rigid.state["rates"], AERO["prop_diam_m"],
            )
            M_applied += M_gyr_pf

            rigid.step(F_applied, M_applied, DT_PHYS)

            h_terrain_msl = terrain.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1])
            ac_alt_msl = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
            agl = ac_alt_msl - h_terrain_msl
            if agl < 0.0:
                # impact evaluation ONLY on an airborne-to-ground transition:
                # the takeoff roll has continuous wheel contact and must never
                # be classified as a crash
                vs_impact = (ac_alt_msl - prev_alt_msl) / DT_PHYS
                pos_n = rigid.state["pos_ned"][0]
                pos_e = rigid.state["pos_ned"][1]
                slope = max(
                    abs(h_terrain_msl - terrain.altitude_msl_at(pos_n + 12.0, pos_e)),
                    abs(h_terrain_msl - terrain.altitude_msl_at(pos_n, pos_e + 12.0)),
                ) / 12.0
                reason = None
                if was_airborne:
                    reason = _evaluate_impact(
                        vs_impact,
                        math.degrees(rigid.state["euler"][0]),
                        math.degrees(rigid.state["euler"][1]),
                        pitot.reading()["ias_mps"],
                    )
                    if reason is None and slope > 0.35 and pitot.reading()["ias_mps"] > 15.0:
                        reason = "flew into rising terrain"
                    if reason is None and terrain.is_water(pos_n, pos_e):
                        reason = "ditched in the lake"
                if reason is not None:
                    crash_state["crashed"] = True
                    crash_state["reason"] = reason
                    crash_state["t_s"] = t_s
                    rigid.state["pos_ned"][2] += agl   # settle into the terrain
                    rigid.state["vel_body"][:] = 0.0
                    rigid.state["rates"][:] = 0.0
                    engine.kill()
                    renderer.log_advisory("CRIT", f"CRASH — {reason}. Press R to reset.", t_s)
                elif was_airborne:
                    if t_s - last_touchdown_t > 5.0 and vs_impact < -0.8:
                        renderer.log_advisory(
                            "INFO", f"Touchdown at {vs_impact:.1f} m/s — airframe intact.", t_s)
                        last_touchdown_t = t_s
                    was_airborne = False
                if not crash_state["crashed"]:
                    rigid.state["pos_ned"][2] += agl
                    if rigid.state["vel_body"][2] > 0:
                        rigid.state["vel_body"][1:] *= 0.995
                    rigid.state["vel_body"][2] = min(rigid.state["vel_body"][2], 0.0)
                    v_n = rigid.state["vel_body"][0]
                    if abs(v_n) < 0.5 and manual.get("brakes", False):
                        rigid.state["rates"][:] *= 0.96
                        rigid.state["vel_body"][0] *= 0.92

            # obstacles: trees, buildings, hangar, ATC tower
            if (not crash_state["crashed"]) and agl < 60.0:
                ob = _obstacle_hit(rigid.state["pos_ned"], ac_alt_msl, terrain.obstacles)
                if ob is not None:
                    crash_state["crashed"] = True
                    crash_state["reason"] = f"struck a {ob['kind']}"
                    crash_state["t_s"] = t_s
                    rigid.state["vel_body"][:] = 0.0
                    rigid.state["rates"][:] = 0.0
                    engine.kill()
                    renderer.log_advisory("CRIT", f"CRASH — {crash_state['reason']}. Press R to reset.", t_s)

            # gear: on the ground the wheels hold the attitude and damp rates —
            # otherwise wind-driven aero forces tumble the parked aircraft.
            # the pitch limit applies only when parked (ias < 8): once rolling,
            # the nose must be free to rotate for takeoff
            if agl < 0.6:
                rigid.state["rates"][0] *= 0.55
                rigid.state["rates"][1] *= 0.55
                rigid.state["euler"][0] *= 0.82
                if pitot.reading()["ias_mps"] < 8.0:
                    rigid.state["euler"][1] = float(np.clip(rigid.state["euler"][1], -0.06, 0.16))

            mass_kg = rigid.mass_kg
            rigid.update_mass(mass_kg)

            # true accel body for IMU: F/m + omega × v - R g
            F_aero_engine = F_applied
            omega = rigid.state["rates"]
            v_b = rigid.state["vel_body"]
            omega_cross_v = np.cross(omega, v_b)
            true_a_body = (F_aero_engine / rigid.mass_kg) - omega_cross_v
            # Actually true "accelerometer" sees F_aero_engine/mass (no gravity),
            # because accelerometers measure specific force.
            imu.step(DT_PHYS, F_aero_engine / rigid.mass_kg, rigid.state["rates"])

            gps.step(
                DT_PHYS, rigid.state["pos_ned"], rigid.state["vel_body"], rigid.state["euler"],
            )

            # total mass follows fuel burn-off
            rigid.update_mass(AIRFRAME["mass_dry_kg"] + engine.fuel_kg_remaining)

            last_aero = aero_res
            last_rho = rho
            last_V_air = aero_res["V_air_mps"]
            last_agl = agl
            prev_alt_msl = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]

        if render_accum >= 1.0 / SIM_RENDER_HZ:
            render_accum = 0.0
            frames_rendered += 1
            now2 = time.time()
            if now2 - fps_t > 0.5:
                last_fps = frames_rendered / (now2 - fps_t)
                frames_rendered = 0
                fps_t = now2

            ac_alt_msl = HOME["alt_msl_m"] - rigid.state["pos_ned"][2]
            h_terrain_msl = terrain.altitude_msl_at(rigid.state["pos_ned"][0], rigid.state["pos_ned"][1])
            agl = max(0.0, ac_alt_msl - h_terrain_msl)
            mission_dist = mission.distance_to_current(rigid.state["pos_ned"])
            mission_brg = mission.bearing_to_current_deg(rigid.state["pos_ned"])
            nfz_v = no_fly.violations(rigid.state["pos_ned"], ac_alt_msl)
            nfz_n = no_fly.nearest_boundary(rigid.state["pos_ned"])

            eng_sum = engine.summary()

            sim_s = {
                "t_s": t_s,
                "fps": float(last_fps),
                "state": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in rigid.state.items()},
                "aero": last_aero,
                "engine": eng_sum,
                "engine_summary": eng_sum,
                "engine_sensors": eng_sensors.reading(eng_sum),
                "pitot": pitot.reading(),
                "pitot_reading": pitot.reading(),
                "gps": gps.reading(),
                "imu": imu.reading(),
                "world_terrain": terrain,
                "mission_items": list(mission.items),
                "mission_captured": list(mission.captured),
                "mission_current_idx": mission.current_index,
                "mission_count": mission.count(),
                "mission_dist_m": mission_dist,
                "mission_bearing_deg": mission_brg,
                "no_fly_zones": nfz_v,
                "nfz_violations": nfz_v,
                "nfz_nearest": nfz_n,
                "agl_m": agl,
                "manual_controls": inp.controls_dict(),
                "ap_debug": ap_debug,
                "mode": ap.mode,
                "on_ground": last_agl < 1.0,
                "crash": dict(crash_state),
            }

            # ATC radio chatter at departure / arrival
            if not crash_state["crashed"]:
                agl_now = sim_s["agl_m"]
                if (not last_airborne) and agl_now > 8.0 and t_s > 2.0:
                    renderer.log_advisory("INFO",
                                          "[ATC] UAV-01 airborne — departure approved, radar contact.", t_s)
                    last_airborne = True
                elif last_airborne and agl_now < 0.5:
                    renderer.log_advisory("INFO",
                                          "[ATC] UAV-01 back on the ground — welcome back.", t_s)
                    last_airborne = False

            # a crashed aircraft is completely dead: every reading zeroed
            if crash_state["crashed"]:
                _zero_sim_snapshot(sim_s)

            if agl > 0.6:
                was_airborne = True
            sim_s["wind_ned"] = wind.wind_at(rigid.altitude_msl_m, rigid.state["pos_ned"])

            # AI advisor + telemetry page @ 10 Hz (suggestions only — no control path)
            tel_accum += 1.0 / SIM_RENDER_HZ
            if tel_accum >= 1.0 / TELEMETRY_HZ:
                tel_accum = 0.0
                es_read = sensor_data["engine"]
                twin.step(1.0 / TELEMETRY_HZ * ENGINE_TIME_SCALE, 1.0 / TELEMETRY_HZ,
                          es_read, manual["throttle"], last_agl < 1.0,
                          carb_heat=manual["carb_heat"])
                sim_s["twin"] = twin.summary()
                if crash_state["crashed"]:
                    assessment = advisor.dead_assessment(t_s, crash_state["reason"])
                else:
                    assessment = advisor.update(sim_s)
                sim_s["ai"] = assessment
                for a in assessment["new_alerts"]:
                    renderer.log_advisory(a["sev"], a["msg"], t_s)
                server.publish({k: v for k, v in sim_s.items() if k != "world_terrain"})

            renderer.draw(sim_s)

    pygame.quit()
    print("Simulator exited.")


if __name__ == "__main__":
    main()
