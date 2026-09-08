import math
import numpy as np
from simulator.config import PID as PID_CFG, GUIDANCE, AIRFRAME
from simulator.flight_controller.pid import PID
from simulator.flight_controller.guidance import L1Guidance


MODES = ["MANUAL", "STAB", "ALT_HOLD", "HDG_HOLD", "WAYPOINT", "RTH"]


def _wrap_deg(d):
    d = d % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _pid_cfg(name):
    cfg = dict(PID_CFG[name])
    if "out_lim_deg" in cfg:
        cfg["out_lim"] = cfg.pop("out_lim_deg")
    return cfg


class Autopilot:
    def __init__(self):
        self.mode = "MANUAL"
        self.roll_pid = PID(**_pid_cfg("roll_att"))
        self.pitch_pid = PID(**_pid_cfg("pitch_att"))
        self.yaw_pid = PID(**_pid_cfg("yaw_damper"))
        self.hdg_pid = PID(**_pid_cfg("hdg_hold"))
        self.alt_pid = PID(**_pid_cfg("alt_hold"))
        self.climb_pid = PID(**_pid_cfg("climb_rate"))
        self.aspd_pid = PID(**_pid_cfg("airspeed_pitch"))
        self.thr_pid = PID(**_pid_cfg("throttle_ias"))
        self.guidance = L1Guidance()

        self.target_hdg_deg = 0.0
        self.target_alt_msl_m = 200.0
        self.target_ias_mps = GUIDANCE["cruise_ias_mps"]
        self.rth_armed = False
        self._next_wp_idx_cached = 0
        self._last_from_wp = None
        self._debug = {}
        self._to_active = False     # auto-takeoff in progress
        self._to_done = False       # takeoff completed since last mode change
        self._field_alt = None
        self._vsi_filt = None       # low-passed vertical speed (raw baro VSI is very noisy)
        self._vsi_thr = None        # heavier filter for the throttle loop
        self._ias_filt = None
        self._stab_locked = False   # STAB captures heading/alt once on engage

    def set_mode(self, mode):
        if mode in MODES and mode != self.mode:
            for p in [self.roll_pid, self.pitch_pid, self.yaw_pid, self.hdg_pid, self.alt_pid, self.climb_pid, self.aspd_pid, self.thr_pid]:
                p.reset()
            self.mode = mode
            self._stab_locked = False
            self._to_done = False
        return self.mode

    def step(self, dt, state, aero, sensors, manual_controls, mission):
        pos = state["pos_ned"]
        euler = state["euler"]
        rates = state["rates"]
        phi_deg = math.degrees(euler[0])
        theta_deg = math.degrees(euler[1])
        psi_deg = (math.degrees(euler[2]) + 360.0) % 360.0
        p_rps, q_rps, r_rps = rates

        pitot_read = sensors.get("pitot", {})
        gps_read = sensors.get("gps", {})
        ias_raw = pitot_read.get("ias_mps", aero["V_air_mps"])
        ias_raw = max(ias_raw, 0.0)
        if self._ias_filt is None:
            self._ias_filt = ias_raw
        self._ias_filt += (ias_raw - self._ias_filt) * min(1.0, dt / 0.35)
        ias_mps = self._ias_filt
        alt_msl = pitot_read.get("baro_alt_msl_m", None)
        if alt_msl is None:
            from simulator.config import HOME
            alt_msl = HOME["alt_msl_m"] - pos[2]
        vsi_raw = pitot_read.get("vsi_mps", None)
        if vsi_raw is None:
            vsi_raw = 0.0
        if self._vsi_filt is None:
            self._vsi_filt = vsi_raw
        # raw baro VSI differentiates 0.9 m altitude noise over 50 ms — filter heavily
        self._vsi_filt += (vsi_raw - self._vsi_filt) * min(1.0, dt / 0.8)
        vsi = self._vsi_filt
        if self._vsi_thr is None:
            self._vsi_thr = vsi_raw
        self._vsi_thr += (vsi_raw - self._vsi_thr) * min(1.0, dt / 2.5)
        gnd_v_ned = np.array([gps_read.get("vn_mps", 0.0), gps_read.get("ve_mps", 0.0), gps_read.get("vd_mps", 0.0)])
        if np.linalg.norm(gnd_v_ned) < 0.5:
            from simulator.aircraft.rigid_body import rot_b_ned
            gnd_v_ned = rot_b_ned(*euler) @ state["vel_body"]
        gspd = math.sqrt(gnd_v_ned[0] ** 2 + gnd_v_ned[1] ** 2)

        mode = self.mode
        controls = dict(manual_controls)

        if mode == "STAB":
            # one-key stable autopilot: with a mission loaded it flies THROUGH
            # the waypoints marked on the map (stabilized); without one it
            # captures and holds the current heading + altitude.
            if not self._stab_locked:
                self.target_hdg_deg = psi_deg
                self.target_alt_msl_m = float(alt_msl)
                self.target_ias_mps = GUIDANCE["cruise_ias_mps"]
                self._stab_locked = True
            mode = "WAYPOINT" if mission.count() > 0 else "HDG_HOLD"

        if mode == "MANUAL":
            return controls, self._debug_dict(aero, ias_mps, alt_msl, gspd, mission)

        # ---- auto-takeoff: engaged on the ground in any AP mode (also
        #      re-arms after a touch-and-go so STAB keeps the UAV flying) ----
        on_ground = bool(sensors.get("on_ground", False))
        if (on_ground and not self._to_active
                and ias_mps < GUIDANCE["rotate_ias_mps"] + 5.0):
            self._to_active = True
            self._field_alt = float(alt_msl)
            self.target_alt_msl_m = self._field_alt + GUIDANCE["min_agl_after_takeoff_m"]
        if self._to_active:
            throttle_cmd = 1.0
            pitch_cmd, roll_cmd = 0.0, 0.0
            on_g = sensors.get("on_ground", False)
            if on_g:
                # wheel-borne: the altitude loop would wind up on the ground
                self.alt_pid.reset()
                if ias_mps >= GUIDANCE["rotate_ias_mps"]:
                    pitch_cmd = GUIDANCE["takeoff_pitch_deg"]   # rotate
            else:
                hdg_err = _wrap_deg(self.target_hdg_deg - psi_deg)
                roll_cmd = float(np.clip(self.hdg_pid.update(psi_deg + hdg_err, psi_deg, dt),
                                         -12.0, 12.0))
                if ias_mps < GUIDANCE["rotate_ias_mps"] + 6.0:
                    pitch_cmd = 1.5                             # accelerate, level
                else:
                    # established at flying speed: climb to the min-altitude
                    # target right away — gating the climb on cruise speed made
                    # the aircraft level-accelerate for kilometres downstream
                    pitch_cmd, vs_cmd = self.alt_loop(alt_msl, vsi, ias_mps, dt)
                    throttle_cmd = self.throttle_loop(self.target_ias_mps, ias_mps,
                                                      vs_cmd, self._vsi_thr, dt)
                    if alt_msl >= self.target_alt_msl_m:
                        self._to_active = False
                        self._to_done = True
            inner = self._inner_loops(roll_cmd, pitch_cmd,
                                      self._turn_coord(phi_deg, r_rps, ias_mps, dt),
                                      phi_deg, theta_deg, p_rps, q_rps, r_rps, ias_mps, dt)
            controls.update(inner)
            controls["throttle"] = float(throttle_cmd)
            dbg = {"roll_cmd_deg": float(roll_cmd), "pitch_cmd_deg": float(pitch_cmd),
                   "hdg_target_deg": float(self.target_hdg_deg),
                   "alt_target_msl_m": float(self.target_alt_msl_m),
                   "ias_target_mps": float(self.target_ias_mps),
                   "takeoff": self._to_active}
            return controls, dbg

        roll_cmd_deg = 0.0
        pitch_cmd_deg = 0.0
        throttle_cmd = float(manual_controls.get("throttle", 0.5))
        rudder_cmd_deg = 0.0   # yaw loop below tracks the coordinated turn rate

        if mode in ("ALT_HOLD",):
            roll_cmd_deg = float(manual_controls.get("delta_a_deg", 0.0)) * 0.75
            pitch_cmd_deg, vs_cmd = self.alt_loop(alt_msl, vsi, ias_mps, dt)
            throttle_cmd = self.throttle_loop(self.target_ias_mps, ias_mps, vs_cmd, self._vsi_thr, dt)

        elif mode == "HDG_HOLD":
            hdg_err = _wrap_deg(self.target_hdg_deg - psi_deg)
            hdg_cmd_r = self.hdg_pid.update(psi_deg + hdg_err, psi_deg, dt)
            hdg_cmd_r = np.clip(hdg_cmd_r, -GUIDANCE["max_bank_deg_aps"], GUIDANCE["max_bank_deg_aps"])
            roll_cmd_deg = hdg_cmd_r
            pitch_cmd_deg, vs_cmd = self.alt_loop(alt_msl, vsi, ias_mps, dt)
            throttle_cmd = self.throttle_loop(self.target_ias_mps, ias_mps, vs_cmd, self._vsi_thr, dt)

        elif mode in ("WAYPOINT", "RTH"):
            if mode == "RTH":
                from simulator.config import HOME
                home_wp = {"n_m": 0.0, "e_m": 0.0, "alt_msl_m": GUIDANCE["rth_alt_msl_m"]}
                dist_home = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                if dist_home < GUIDANCE["loiter_radius_m"] + 30.0:
                    bank, hdg_t, xtk, cap, dist, prog = self.guidance.loiter(pos, gnd_v_ned, [0.0, 0.0], GUIDANCE["loiter_radius_m"], direction=1)
                else:
                    bank, hdg_t, xtk, cap, dist, prog = self.guidance.direct_to(pos, gnd_v_ned, home_wp)
                target_alt = GUIDANCE["rth_alt_msl_m"]
                target_spd = GUIDANCE["cruise_ias_mps"]
            else:
                if mission.count() == 0:
                    mode = "ALT_HOLD"
                    roll_cmd_deg = 0.0
                    pitch_cmd_deg, vs_cmd = self.alt_loop(alt_msl, vsi, ias_mps, dt)
                    throttle_cmd = self.throttle_loop(self.target_ias_mps, ias_mps, vs_cmd, self._vsi_thr, dt)
                    inner = self._inner_loops(roll_cmd_deg, pitch_cmd_deg, rudder_cmd_deg, phi_deg, theta_deg, p_rps, q_rps, r_rps, ias_mps, dt)
                    controls.update(inner)
                    controls["throttle"] = throttle_cmd
                    return controls, self._debug_dict(aero, ias_mps, target_alt, gspd, mission, xtrack=0.0, hdg_tgt=0.0, bank_tgt=0.0, leg_progress=0.0)

                current = mission.current()
                idx = mission.current_index
                n_wp = mission.count()
                next_wp = mission.items[(idx + 1) % n_wp] if n_wp > 1 else current

                if self._last_from_wp is None or self._next_wp_idx_cached != idx:
                    if idx == 0 and not any(mission.captured):
                        # route not yet started: the first leg departs from home
                        self._last_from_wp = {"n_m": float(mission.home_ned[0]),
                                              "e_m": float(mission.home_ned[1]),
                                              "alt_msl_m": current.get("alt_msl_m", alt_msl)}
                    else:
                        self._last_from_wp = mission.items[(idx - 1) % n_wp] if n_wp > 1 else                             {"n_m": pos[0], "e_m": pos[1], "alt_msl_m": current.get("alt_msl_m", alt_msl)}
                    self._next_wp_idx_cached = idx

                bank, hdg_t, xtk, cap, dist, prog = self.guidance.track_waypoint(pos, gnd_v_ned, self._last_from_wp, current)
                if cap:
                    mission.advance()
                    self._last_from_wp = current
                    self._next_wp_idx_cached = mission.current_index

                target_alt = current.get("alt_msl_m", alt_msl)
                target_spd = GUIDANCE["cruise_ias_mps"]

            roll_cmd_deg = bank
            self.target_hdg_deg = hdg_t
            pitch_cmd_deg, vs_cmd = self.alt_loop(alt_msl, vsi, ias_mps, dt, override_target=target_alt)
            throttle_cmd = self.throttle_loop(target_spd, ias_mps, vs_cmd, self._vsi_thr, dt)

        inner = self._inner_loops(
            roll_cmd_deg, pitch_cmd_deg, rudder_cmd_deg + self._turn_coord(phi_deg, r_rps, ias_mps, dt),
            phi_deg, theta_deg, p_rps, q_rps, r_rps, ias_mps, dt,
        )
        controls.update(inner)
        controls["throttle"] = float(throttle_cmd)
        dbg = {
            "roll_cmd_deg": float(roll_cmd_deg),
            "pitch_cmd_deg": float(pitch_cmd_deg),
            "hdg_target_deg": float(self.target_hdg_deg),
            "alt_target_msl_m": float(self.target_alt_msl_m) if mode != "WAYPOINT" else float(target_alt),
            "ias_target_mps": float(self.target_ias_mps),
        }
        return controls, dbg

    def alt_loop(self, alt_msl, vsi, ias_mps, dt, override_target=None):
        """Returns (pitch_cmd_deg, vs_cmd_mps); vs_cmd feeds the throttle
        energy feed-forward so the AP can actually climb/descend."""
        tgt = self.target_alt_msl_m if override_target is None else float(override_target)
        vs_cmd = self.alt_pid.update(tgt, alt_msl, dt)
        vs_cmd = float(np.clip(vs_cmd, -6.0, 6.0))
        pitch_alt = self.climb_pid.update(vs_cmd, vsi, dt)
        pitch_spd = self.aspd_pid.update(self.target_ias_mps, ias_mps, dt)
        w_alt = 0.75
        pitch_cmd = w_alt * pitch_alt + (1.0 - w_alt) * pitch_spd
        return float(np.clip(pitch_cmd, -18.0, 18.0)), vs_cmd

    def throttle_loop(self, ias_tgt, ias_mps, vs_cmd, vsi, dt):
        """Airspeed hold + direct climb-power feed-forward: without it a fast
        aircraft below altitude keeps the throttle at idle and can never climb
        back (energy deadlock)."""
        vs_term = float(np.clip(0.25 * (vs_cmd - vsi), -0.7, 0.7))
        thr = self.thr_pid.update(ias_tgt, ias_mps, dt) + vs_term
        return float(np.clip(thr, 0.05, 1.0))

    def _turn_coord(self, phi_deg, r_rps, ias_mps, dt):
        """Yaw-rate controller: rudder drives the yaw rate toward the value a
        coordinated turn at the current bank would produce (0 when level).
        Cn_delta_r < 0, so positive PID output must become NEGATIVE rudder."""
        ideal_yaw = 9.80665 * math.tan(math.radians(phi_deg)) / max(ias_mps, 8.0)
        ideal_deg = math.degrees(ideal_yaw)
        return -self.yaw_pid.update(ideal_deg, math.degrees(r_rps), dt)

    def _inner_loops(self, roll_cmd, pitch_cmd, rudder_cmd, phi, theta, p, q, r, V, dt):
        ail = self.roll_pid.update(roll_cmd, phi, dt)
        ail = np.clip(ail, -AIRFRAME["delta_a_max_deg"], AIRFRAME["delta_a_max_deg"])

        # Cm_delta_e < 0 (positive elevator = nose down), so a PID asking for
        # more pitch must command NEGATIVE elevator. Trim is already baked into
        # the manual elevator position (InputHandler), the AP replaces surfaces.
        elev = -self.pitch_pid.update(pitch_cmd, theta, dt) + 0.35 * math.degrees(q)
        elev = np.clip(elev, -AIRFRAME["delta_e_max_deg"], AIRFRAME["delta_e_max_deg"])

        rud = float(np.clip(rudder_cmd, -AIRFRAME["delta_r_max_deg"], AIRFRAME["delta_r_max_deg"]))

        return {
            "delta_a_deg": float(ail),
            "delta_e_deg": float(elev),
            "delta_r_deg": float(rud),
        }

    def _debug_dict(self, aero, ias, alt, gspd, mission, xtrack=0.0, hdg_tgt=0.0, bank_tgt=0.0, leg_progress=0.0):
        return {
            "mode": self.mode,
            "alpha_deg": math.degrees(aero.get("alpha_rad", 0.0)),
            "beta_deg": math.degrees(aero.get("beta_rad", 0.0)),
            "ias_mps": float(ias),
            "alt_msl_m": float(alt),
            "gspd_mps": float(gspd),
            "CL": float(aero.get("CL", 0.0)),
            "CD": float(aero.get("CD", 0.0)),
            "stall": bool(aero.get("stall", False)),
            "xtrack_m": float(xtrack),
            "hdg_tgt_deg": float(hdg_tgt),
            "bank_tgt_deg": float(bank_tgt),
            "leg_progress": float(leg_progress),
            "mission_idx": int(mission.current_index if mission else 0),
            "mission_count": int(mission.count() if mission else 0),
        }
