"""AI Health Monitor & Pilot Assistant (SUGGESTIONS ONLY).

Architecture guarantee — the AI can never override the UAV:
  1. HealthMonitor receives telemetry snapshots (plain dicts) and returns
     assessments + textual suggestions. It imports nothing from the flight
     controller and holds no reference to controls, actuators or the
     autopilot.
  2. Suggestions are TEXT ONLY. There is no action channel, no command
     queue, no button that forwards anything to the aircraft. The only
     control path in the whole system is the pilot's keyboard/mouse in the
     simulator window.

Everything is offline and deterministic: a rule-based expert system with
trend analysis (linear regression over sliding windows) — no network calls.
"""

import math
import numpy as np
from collections import deque

from simulator.config import (
    PISTON_ENGINE, AERO, AIRFRAME, GUIDANCE, G,
)

SEV_ORDER = {"INFO": 0, "WARN": 1, "CRIT": 2}
STATUS_BY_SCORE = [(80, "NOMINAL"), (60, "CAUTION"), (35, "WARNING"), (0, "CRITICAL")]

KT = 1.94384


def _status_for_score(score):
    for thr, name in STATUS_BY_SCORE:
        if score >= thr:
            return name
    return "CRITICAL"


def _slope_per_min(hist, window_s=30.0):
    """Least-squares slope of (t, v) pairs over the trailing window, per minute."""
    if len(hist) < 4:
        return None
    t_now = hist[-1][0]
    pts = [(t, v) for t, v in hist if t_now - t <= window_s]
    if len(pts) < 4:
        return None
    t = np.array([p[0] for p in pts])
    v = np.array([p[1] for p in pts])
    if float(t.max() - t.min()) < 4.0:
        return None
    A = np.vstack([t, np.ones_like(t)]).T
    m, _ = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(m * 60.0)


class HealthMonitor:
    """Scores subsystem health, detects trends, emits alerts + suggestions."""

    def __init__(self):
        self._hist = {k: deque(maxlen=600) for k in
                      ("cht", "egt", "fuel", "ias", "alt", "rpm", "oil", "vs")}
        self._alert_last = {}     # code -> t of last emit
        self._active = {}         # code -> alert dict (persists while condition holds)
        self._prev_faults = None  # previous-step fault snapshot for edge detection
        self._iced_latch = False  # latched while carb ice is significant
        self._vs_f = None         # monitor-side smoothed vertical speed
        self._hist_t = None
        self._last_dist_home = None
        self.coolDown = {"INFO": 8.0, "WARN": 6.0, "CRIT": 4.0, "RESOLVED": 6.0}

    def dead_assessment(self, t_s, reason="terrain impact"):
        """Post-crash assessment: every subsystem at zero, only the CRASHED
        alert and the reset suggestion remain."""
        zero_ss = self._ss(0.0, ["destroyed"])
        return {
            "t_s": t_s,
            "overall": {"score": 0.0, "status": "CRITICAL",
                        "summary": "Aircraft destroyed — all telemetry dead."},
            "subsystems": {k: dict(zero_ss) for k in
                           ("engine", "energy", "flight", "nav", "sensors")},
            "insights": {"endurance_s": None, "endurance_min": None, "range_km": None,
                         "fuel_flow_lph": 0.0, "dist_home_km": None, "fuel_home_L": None,
                         "time_home_min": None, "can_reach_home": None, "stall_ias_kt": 0.0,
                         "stall_margin_kt": None, "cht_trend_c_per_min": None,
                         "egt_trend_c_per_min": None, "fuel_trend_lph": None,
                         "headwind_mps": None, "agl_m": 0.0},
            "alerts": [{"id": "CRASHED", "sev": "CRIT",
                        "msg": (f"AIRCRAFT CRASHED — {reason}. "
                                "Press R in the simulator (or RESET SIM on the dashboard) to fly again."),
                        "t_s": t_s}],
            "new_alerts": [],
            "suggestions": [{"id": "CRASHED", "title": "Reset the simulator",
                             "detail": "Press R in the simulator window or click RESET SIM on the dashboard.",
                             "source_alert": "aircraft destroyed"}],
            "advisory_only": True,
        }

    def reset(self):
        """Clear all history and alerts when the simulator resets."""
        for dq in self._hist.values():
            dq.clear()
        self._alert_last.clear()
        self._active.clear()
        self._last_dist_home = None
        self._vs_f = None
        self._hist_t = None

    # ------------------------------------------------------------------ #
    def update(self, s):
        """s: telemetry snapshot dict. Returns assessment dict (JSON-safe)."""
        t = float(s.get("t_s", 0.0))
        self._push_history(t, s)
        insights = self._compute_insights(s)
        subsystems = self._score_subsystems(s, insights)
        overall, status = self._overall(subsystems)
        self._evaluate_alerts(t, s, insights)
        active = sorted(self._active.values(),
                        key=lambda a: -SEV_ORDER.get(a["sev"], 0))
        suggestions = self._build_suggestions(s, insights, active)
        return {
            "t_s": t,
            "overall": {"score": overall, "status": status,
                        "summary": self._summary_line(subsystems, insights)},
            "subsystems": subsystems,
            "insights": insights,
            "alerts": list(active),
            "new_alerts": [a for a in active if a.get("_just_emitted")],
            "suggestions": suggestions,
            "advisory_only": True,
        }

    # ------------------------------------------------------------------ #
    def _push_history(self, t, s):
        eng_s = s.get("engine_sensors", s.get("engine_summary", {}))
        pit = s.get("pitot_reading", s.get("pitot", {}))
        vs_raw = float(pit.get("vsi_mps", 0.0))
        if self._vs_f is None:
            self._vs_f = vs_raw
        elif self._hist_t is not None:
            dt_h = max(0.001, min(2.0, t - self._hist_t))
            self._vs_f += (vs_raw - self._vs_f) * min(1.0, dt_h / 0.8)
        self._hist_t = t
        self._hist["cht"].append((t, float(eng_s.get("cht_C", 0.0))))
        self._hist["egt"].append((t, float(eng_s.get("egt_C", 0.0))))
        self._hist["fuel"].append((t, float(eng_s.get("fuel_qty_L", 0.0))))
        self._hist["ias"].append((t, float(pit.get("ias_mps", 0.0))))
        self._hist["alt"].append((t, float(pit.get("baro_alt_msl_m", 0.0))))
        self._hist["rpm"].append((t, float(eng_s.get("rpm", 0.0))))
        self._hist["oil"].append((t, float(eng_s.get("oil_psi", 0.0))))
        self._hist["vs"].append((t, float(pit.get("vsi_mps", 0.0))))

    def _compute_insights(self, s):
        eng_s = s.get("engine_sensors", s.get("engine_summary", {}))
        eng_t = s.get("engine_summary", {})
        pit = s.get("pitot_reading", s.get("pitot", {}))
        gps = s.get("gps", {})
        agl = float(s.get("agl_m", 0.0))

        ff = max(float(eng_s.get("fuel_flow_Lph", 0.0)), 0.0)
        fuel = max(float(eng_s.get("fuel_qty_L", 0.0)), 0.0)
        gspd = max(float(gps.get("groundspeed_mps", 0.0)), 0.0)
        endurance_s = (fuel / ff * 3600.0) if ff > 0.05 else float("inf")

        dist_home = math.hypot(*_pos_xy(s))
        if gspd > 2.0 and ff > 0.05:
            fuel_home = dist_home / gspd * ff / 3600.0
            time_home = dist_home / gspd
            can_home = fuel > fuel_home * 1.25   # 25% reserve
        else:
            fuel_home, time_home, can_home = None, None, None

        rho = 1.225
        m = AIRFRAME["mass_dry_kg"] + eng_t.get("fuel_kg_remaining", 4.5)
        cl_max = AERO["CL0"] + AERO["CL_alpha_per_rad"] * AERO["alpha_stall_rad"]
        vs_true = math.sqrt(2.0 * m * G / max(rho * AIRFRAME["S_wing_m2"] * cl_max, 1e-6))
        stall_ias = vs_true * KT

        wind = s.get("wind_ned")
        headwind = None
        if wind is not None:
            psi = math.radians(math.degrees(s["state"]["euler"][2]) % 360.0)
            track = math.atan2(wind[1], wind[0])
            headwind = float(-(wind[0] * math.cos(psi) + wind[1] * math.sin(psi)))

        return {
            "endurance_s": endurance_s if math.isfinite(endurance_s) else None,
            "endurance_min": endurance_s / 60.0 if math.isfinite(endurance_s) else None,
            "range_km": gspd * endurance_s / 1000.0 if math.isfinite(endurance_s) and gspd > 1 else None,
            "fuel_flow_lph": ff,
            "dist_home_km": dist_home / 1000.0,
            "fuel_home_L": fuel_home,
            "time_home_min": time_home / 60.0 if time_home is not None else None,
            "can_reach_home": can_home,
            "stall_ias_kt": stall_ias,
            "stall_margin_kt": pit.get("ias_mps", 0.0) * KT - stall_ias,
            "cht_trend_c_per_min": _slope_per_min(self._hist["cht"]),
            "egt_trend_c_per_min": _slope_per_min(self._hist["egt"]),
            "fuel_trend_lph": -_slope_per_min(self._hist["fuel"]) * 60.0
                              if _slope_per_min(self._hist["fuel"]) is not None else None,
            "headwind_mps": headwind,
            "agl_m": agl,
        }

    def _score_subsystems(self, s, ins):
        eng_t = s.get("engine_summary", {})
        eng_s = s.get("engine_sensors", eng_t)
        pit = s.get("pitot_reading", s.get("pitot", {}))
        gps = s.get("gps", {})
        aero = s.get("aero", {})
        flags = set(eng_t.get("health_flags", []))

        # --- engine ---
        sc = 100.0
        notes = []
        cht_frac = max(0.0, (eng_s.get("cht_C", 0) - PISTON_ENGINE["cht_nominal_c"]) /
                       max(PISTON_ENGINE["cht_max_c"] - PISTON_ENGINE["cht_nominal_c"], 1))
        egt_frac = max(0.0, (eng_s.get("egt_C", 0) - PISTON_ENGINE["egt_nominal_c"]) /
                       max(PISTON_ENGINE["egt_max_c"] - PISTON_ENGINE["egt_nominal_c"], 1))
        sc -= 35.0 * cht_frac
        sc -= 20.0 * egt_frac
        if eng_t.get("running", False) and eng_s.get("oil_psi", 55) < PISTON_ENGINE["oil_pressure_min_psi"] + 8:
            sc -= 45.0
            notes.append("oil pressure low")
        if "RPM_FLUCTUATION" in flags:
            sc -= 20.0
            notes.append("RPM rough")
        if "ENGINE_OFF" in flags:
            sc = 0.0
            notes.append("engine off")
        if cht_frac > 0.75:
            notes.append("CHT high")
        if egt_frac > 0.8:
            notes.append("EGT high")
        engine = self._ss(sc, notes)

        # --- energy (fuel) ---
        fuel = eng_s.get("fuel_qty_L", 0.0)
        frac = fuel / PISTON_ENGINE["fuel_full_L"]
        sc = 100.0 * min(1.0, frac / 0.25)
        notes = []
        if frac < 0.17:
            notes.append("fuel low")
        end_warn = PISTON_ENGINE.get("endurance_warn_min", 8.0)
        if ins.get("endurance_min") is not None and ins["endurance_min"] < end_warn:
            sc = min(sc, 45.0)
            notes.append(f"endurance {ins['endurance_min']:.0f} min")
        if ins.get("can_reach_home") is False:
            sc = min(sc, 30.0)
            notes.append("cannot reach home on remaining fuel")
        energy = self._ss(sc, notes)

        # --- flight envelope ---
        ias_kt = pit.get("ias_mps", 0.0) * KT
        sc = 100.0
        notes = []
        stall_margin = ins["stall_margin_kt"]
        if aero.get("stall"):
            sc = 10.0
            notes.append("STALL")
        elif stall_margin < 5:
            sc = min(sc, 30.0)
            notes.append(f"only {stall_margin:.0f} kt above stall")
        elif stall_margin < 10:
            sc = min(sc, 65.0)
        vs = self._vs_f if self._vs_f is not None else pit.get("vsi_mps", 0.0)
        if vs < -6.0:
            sc = min(sc, 40.0)
            notes.append(f"sink {vs*196.85:.0f} fpm")
        bank = abs(math.degrees(s["state"]["euler"][0]))
        if bank > GUIDANCE["max_bank_deg_aps"] + 10:
            sc = min(sc, 50.0)
            notes.append(f"bank {bank:.0f} deg")
        agl = ins["agl_m"]
        on_gnd = bool(s.get("on_ground", False))
        if agl < 50 and not on_gnd:
            sc = min(sc, 55.0)
            notes.append(f"AGL {agl:.0f} m")
        if agl < 20:
            sc = min(sc, 20.0)
        flight = self._ss(sc, notes)

        # --- navigation ---
        sc = 100.0
        notes = []
        nfz = s.get("nfz_violations", [])
        near = s.get("nfz_nearest")
        if nfz:
            sc = 0.0
            notes.append(f"inside {nfz[0]['name']}")
        elif near and near["margin_m"] < 150:
            sc = min(sc, 55.0)
            notes.append(f"NFZ {near['name']} in {near['margin_m']:.0f} m")
        if gps.get("health") != "OK":
            sc = min(sc, 35.0)
            notes.append("GPS " + str(gps.get("health")))
        elif gps.get("hdop", 1.0) > 4.0 or gps.get("satellites", 10) < 6:
            sc = min(sc, 65.0)
            notes.append("GPS degraded")
        nav = self._ss(sc, notes)

        # --- sensors ---
        sc = 100.0
        notes = []
        if gps.get("health") != "OK":
            sc -= 40.0
            notes.append("GPS out")
        ias = pit.get("ias_mps", 0.0)
        if ias < 1.0 and not eng_t.get("on_ground", False) and s.get("mode") != "MANUAL":
            pass  # pitot always reports; keep placeholder for future plausibility checks
        sensors = self._ss(sc, notes)

        return {
            "engine": engine,
            "energy": energy,
            "flight": flight,
            "nav": nav,
            "sensors": sensors,
        }

    @staticmethod
    def _ss(score, notes):
        score = float(np.clip(score, 0.0, 100.0))
        status = _status_for_score(score)
        return {"score": round(score, 1), "status": status, "notes": notes}

    @staticmethod
    def _overall(subsystems):
        w = {"engine": 0.30, "energy": 0.20, "flight": 0.30, "nav": 0.15, "sensors": 0.05}
        score = sum(subsystems[k]["score"] * w[k] for k in w)
        for k in w:
            if subsystems[k]["status"] == "CRITICAL":
                score = min(score, 25.0)
            elif subsystems[k]["status"] == "WARNING":
                score = min(score, 55.0)
        return round(score, 1), _status_for_score(score)

    @staticmethod
    def _summary_line(subsystems, ins):
        worst = min(subsystems.items(), key=lambda kv: kv[1]["score"])
        line = f"Weakest system: {worst[0].upper()} ({worst[1]['score']:.0f})"
        if ins.get("endurance_min") is not None:
            line += f" | endurance {ins['endurance_min']:.0f} min"
        if ins.get("stall_margin_kt") is not None:
            line += f" | stall margin {ins['stall_margin_kt']:.0f} kt"
        return line

    # ------------------------------------------------------------------ #
    def _emit(self, t, code, sev, msg, suggestion=None, condition=True):
        """Register alert while `condition` holds; respects per-code cooldown."""
        if condition:
            last = self._alert_last.get(code, -1e9)
            if t - last >= self.coolDown.get(sev, 6.0):
                self._alert_last[code] = t
                self._active[code] = {"id": code, "sev": sev, "msg": msg,
                                      "t_s": t, "_just_emitted": True,
                                      "suggestion": suggestion}
                return
            if code in self._active:
                self._active[code]["_just_emitted"] = False
        else:
            self._active.pop(code, None)

    def _evaluate_alerts(self, t, s, ins):
        eng_t = s.get("engine_summary", {})
        eng_s = s.get("engine_sensors", eng_t)
        pit = s.get("pitot_reading", s.get("pitot", {}))
        gps = s.get("gps", {})
        aero = s.get("aero", {})
        flags = set(eng_t.get("health_flags", []))
        agl = ins["agl_m"]

        self._emit(t, "STALL", "CRIT", "STALL — lower the nose, full throttle, level wings.",
                   condition=bool(aero.get("stall")))
        self._emit(t, "SPEED_LOW", "WARN",
                   f"Airspeed low ({pit.get('ias_mps',0)*KT:.0f} kt) — stall risk, add power.",
                   suggestion={"title": "Add power", "detail": "Increase throttle to recover energy."},
                   condition=pit.get("ias_mps", 50) * KT < ins["stall_ias_kt"] + 8 and agl > 5)
        vs_f = self._vs_f if self._vs_f is not None else pit.get("vsi_mps", 0.0)
        self._emit(t, "SINK", "WARN", f"High sink rate ({vs_f*196.85:.0f} fpm).",
                   condition=vs_f < -6.5)

        # fuel endurance watch: the resource that actually limits the flight
        end_min = ins.get("endurance_min")
        end_warn = PISTON_ENGINE.get("endurance_warn_min", 8.0)
        end_crit = PISTON_ENGINE.get("endurance_crit_min", 4.0)
        if end_min is not None and eng_t.get("running", False):
            if end_min < end_crit:
                self._emit(t, "ENDURANCE_LOW", "CRIT",
                           f"Only {end_min / 60:.1f} h of flight time left — land now.",
                           {"title": "Land now", "detail": "Find a clearing; fuel to home check on the panel."},
                           condition=True)
            elif end_min < end_warn:
                self._emit(t, "ENDURANCE_LOW", "WARN",
                           f"Endurance {end_min:.0f} min — plan your return now.",
                           {"title": "Return to home",
                            "detail": "Switch to RTH to land before the tank runs dry."},
                           condition=True)
            else:
                self._emit(t, "ENDURANCE_LOW", "WARN", "", condition=False)
                self._emit(t, "ENDURANCE_LOW", "CRIT", "", condition=False)
        else:
            self._emit(t, "ENDURANCE_LOW", "WARN", "", condition=False)
            self._emit(t, "ENDURANCE_LOW", "CRIT", "", condition=False)
        self._emit(t, "TERRAIN", "CRIT", f"Terrain — AGL {agl:.0f} m.",
                   condition=agl < 25 and not s.get("on_ground", False) and s.get("t_s", 0) > 15)
        self._emit(t, "BANK", "WARN",
                   f"Excessive bank ({abs(math.degrees(s['state']['euler'][0])):.0f} deg) — level wings.",
                   condition=abs(math.degrees(s["state"]["euler"][0])) > 50)

        if "FUEL_DEPLETED" in flags:
            self._emit(t, "FUEL_OUT", "CRIT", "Fuel depleted — engine out. Trim for best glide.",
                       condition=True)
        elif "FUEL_LOW" in flags:
            rth = {"title": "Return to home",
                   "detail": "Fuel below reserve; switch to RTH mode to keep a glide margin."}
            self._emit(t, "FUEL_LOW", "WARN",
                       f"Fuel low ({eng_s.get('fuel_qty_L',0):.1f} L, "
                       f"{(ins.get('endurance_min') or 0):.0f} min) — RTH recommended.", rth, True)
        else:
            self._emit(t, "FUEL_LOW", "WARN", "", condition=False)
            self._emit(t, "FUEL_OUT", "CRIT", "", condition=False)

        cht = eng_s.get("cht_C", 0)
        self._emit(t, "CHT_HIGH", "WARN",
                   f"CHT {cht:.0f} C (trend {ins.get('cht_trend_c_per_min') or 0:+.0f} C/min) — reduce power.",
                   suggestion={"title": "Reduce power", "detail": "Cut throttle to bring CHT down."},
                   condition=cht > PISTON_ENGINE["cht_max_c"] - 25)
        egt = eng_s.get("egt_C", 0)
        carb = {"title": "Apply carb heat", "detail": "Enriches mixture; helps EGT and carb icing."}
        self._emit(t, "EGT_HIGH", "WARN", f"EGT {egt:.0f} C — enrich mixture / reduce power.",
                   carb, condition=egt > PISTON_ENGINE["egt_max_c"] - 60)
        self._emit(t, "OIL_LOW", "CRIT",
                   f"Oil pressure {eng_s.get('oil_psi',0):.0f} psi — land ASAP.",
                   {"title": "Return to home",
                    "detail": "Engine failure possible; get over home before it quits."},
                   condition=eng_t.get("running", False) and eng_s.get("oil_psi", 55) < PISTON_ENGINE["oil_pressure_min_psi"])
        self._emit(t, "RPM_ROUGH", "WARN", "RPM fluctuating — possible carb ice; apply carb heat.",
                   carb, condition="RPM_FLUCTUATION" in flags)
        if "ENGINE_OFF" in flags and s.get("t_s", 0) > 20:
            restart = {"title": "Attempt restart",
                       "detail": "Engage starter if altitude permits; else trim for best glide."}
            self._emit(t, "ENGINE_OFF", "CRIT",
                       f"Engine off — best glide ~{GUIDANCE['cruise_ias_mps']*KT:.0f} kt. Restart or prepare dead-stick.",
                       restart, True)
        else:
            self._emit(t, "ENGINE_OFF", "CRIT", "", condition=False)

        nfz = s.get("nfz_violations", [])
        near = s.get("nfz_nearest")
        if nfz:
            self._emit(t, "NFZ_BREACH", "CRIT",
                       f"NFZ breach: {nfz[0]['name']} — exit immediately.", condition=True)
        else:
            self._emit(t, "NFZ_BREACH", "CRIT", "", condition=False)
        if near is not None and near["margin_m"] < 120 and not nfz:
            self._emit(t, "NFZ_NEAR", "WARN",
                       f"Approaching NFZ {near['name']} ({near['margin_m']:.0f} m).", condition=True)
        else:
            self._emit(t, "NFZ_NEAR", "WARN", "", condition=False)

        # pilot fighting the autopilot: stick deflected while an AP mode is active
        crashed = bool((s.get("crash") or {}).get("crashed"))
        mc = s.get("manual_controls", {})
        stick_elev = mc.get("delta_e_deg", 0.0) - mc.get("delta_e_trim_deg", 0.0)
        fighting = (not crashed) and s.get("mode") != "MANUAL" and (
            abs(mc.get("delta_a_deg", 0.0)) > 8.0
            or abs(stick_elev) > 6.0
            or abs(mc.get("delta_r_deg", 0.0)) > 10.0)
        self._emit(t, "PILOT_INPUT", "INFO",
                   "Autopilot is flying — your stick/throttle input is ignored. Press 1 for MANUAL.",
                   {"title": "Take manual control",
                    "detail": "Press 1 in the simulator window to switch to MANUAL."},
                   condition=fighting)

        # ---- systematic engine faults: detect -> advise remedy -> verify -> resolve ----
        f = (s.get("engine_summary") or {}).get("faults") or {}
        mc0 = s.get("manual_controls", {})
        stress = float(f.get("stress_pct", 0.0))
        ice = float(f.get("ice_pct", 0.0))
        oil_qty = float(f.get("oil_qty_pct", 100.0))
        seized = bool(f.get("seized", False))
        pf = float(f.get("power_factor", 1.0))

        if stress > 75.0:
            self._emit(t, "STRESS_CRIT", "CRIT",
                       f"Severe engine damage — stress {stress:.0f}%, power -{(1-pf)*100:.0f}%. "
                       "Expect failure — land as soon as possible.",
                       {"title": "Reduce power", "detail": "Throttle back below ~75% RPM immediately."},
                       condition=True)
        elif stress > 45.0:
            self._emit(t, "STRESS_WARN", "WARN",
                       f"Engine stress {stress:.0f}% — sustained high RPM/temps. "
                       "Reduce throttle now to avoid power loss.",
                       {"title": "Reduce power", "detail": "Bring RPM below ~90% to let the engine cool."},
                       condition=True)
        else:
            self._emit(t, "STRESS_WARN", "WARN", "", condition=False)
            self._emit(t, "STRESS_CRIT", "CRIT", "", condition=False)

        if ice > 30.0:
            self._emit(t, "CARB_ICE_WARN", "WARN",
                       f"Carb icing {ice:.0f}% — RPM rough, power -{(1-pf)*100:.0f}%. "
                       "Apply carb heat (H).",
                       {"title": "Apply carb heat", "detail": "Press H — ice melts in ~15 s."},
                       condition=True)
        else:
            self._emit(t, "CARB_ICE_WARN", "WARN", "", condition=False)

        if f.get("leak_active") and oil_qty < 70.0 and not seized:
            if oil_qty < 25.0:
                self._emit(t, "OIL_LEAK", "CRIT",
                           f"Oil nearly gone ({oil_qty:.0f}%) — engine failure imminent. Land NOW.",
                           {"title": "Land immediately", "detail": "Reduce power to slow the loss and glide in."},
                           condition=True)
            else:
                self._emit(t, "OIL_LEAK", "WARN",
                           f"Oil leak — quantity {oil_qty:.0f}%. Reduce power to slow the loss; plan a landing.",
                           {"title": "Reduce power & plan landing",
                            "detail": "Lower throttle slows the leak; get over a landing spot."},
                           condition=True)
        else:
            self._emit(t, "OIL_LEAK", "WARN", "", condition=False)

        if seized:
            self._emit(t, "ENGINE_SEIZED", "CRIT",
                       "ENGINE SEIZED — permanent failure. Trim for best glide and force-land.",
                       {"title": "Force land", "detail": "Best glide speed, pick a clearing, gear as needed."},
                       condition=True)
        else:
            self._emit(t, "ENGINE_SEIZED", "CRIT", "", condition=False)

        # ---- resolution confirmations (edge-triggered) ----
        prev = self._prev_faults or {}
        was_carb = prev.get("carb_heat")
        carb_now = s.get("engine_summary", {}).get("carb_heat")
        if was_carb is False and carb_now and ice > 10.0:
            self._emit(t, "CARB_HEAT_APPLIED", "RESOLVED",
                       f"Carb heat applied — ice melting, ~{ice / 5.5:.0f} s to clear. Watching it…",
                       condition=True)
        if ice > 20.0:
            self._iced_latch = True
        if self._iced_latch and ice < 5.0:
            self._iced_latch = False
            self._emit(t, "CARB_ICE_CLEARED", "RESOLVED",
                       "Carb ice fully cleared — RPM steady, power restored. Problem resolved.",
                       condition=True)
        if (prev.get("stress", 0.0) >= 45.0) and 5.0 < stress < 45.0:
            self._emit(t, "STRESS_EASING", "RESOLVED",
                       f"Engine stress easing ({stress:.0f}% and falling) — power recovering. Well handled.",
                       condition=True)
        if (prev.get("oil_qty", 100.0) > 90.0) and oil_qty <= 90.0:
            self._emit(t, "OIL_LEAK_CONFIRMED", "WARN",
                       f"Oil quantity dropping fast ({oil_qty:.0f}%) — leak confirmed. "
                       "This one cannot be fixed in the air: land before it seizes.",
                       condition=True)
        self._prev_faults = {"carb_heat": carb_now, "ice": ice, "stress": stress,
                             "oil_qty": oil_qty, "seized": seized}

        # ---- digital twin prognostics (estimated hidden states) ----
        tw = s.get("twin") or {}
        crashed_now = bool((s.get("crash") or {}).get("crashed", False))
        if tw and not crashed_now:
            rul = tw.get("rul_h")
            if rul is not None and rul < 0.5:
                self._emit(t, "TWIN_RUL", "CRIT",
                           f"Twin predicts engine failure in {rul:.1f} h "
                           f"(±{(tw.get('rul_hi', 0) or 0) - rul:.1f}). Land before it quits.",
                           {"title": "Land before failure",
                            "detail": "Twin wear estimate is approaching the failure threshold."},
                           condition=True)
            elif rul is not None and rul < 1.5:
                self._emit(t, "TWIN_RUL", "WARN",
                           f"Twin prognostics: {rul:.1f} h to predicted power loss at this power profile.",
                           {"title": "Reduce power", "detail": "Lower RPM slows the twin's wear trend."},
                           condition=True)
            else:
                self._emit(t, "TWIN_RUL", "WARN", "", condition=False)
                self._emit(t, "TWIN_RUL", "CRIT", "", condition=False)
            if not tw.get("synced", True) and not crashed_now:
                self._emit(t, "TWIN_DIVERGENCE", "WARN",
                           f"Twin divergence {tw.get('divergence', 0):.2f} — measurements don't match "
                           "the engine model. A sensor may be lying.",
                           condition=True)
            else:
                self._emit(t, "TWIN_DIVERGENCE", "WARN", "", condition=False)

        # ---- ML layer: learned diagnosis + failure risk ----
        # The ML models are trained offline on simulator fault scenarios and
        # keep adapting online. Thresholds are deliberately higher than the
        # classifier's arg-max: only confident, sustained signatures alert.
        # The first minute is skipped — cold-start/warm-up transients are in
        # neither model's training distribution.
        ml = (tw or {}).get("ml") or {}
        if ml.get("available") and not crashed_now and s.get("t_s", 0) > 60:
            risk = float(ml.get("fail_risk_60s") or 0.0)
            if risk > 0.60:
                self._emit(t, "ML_RISK", "CRIT",
                           f"ML failure risk {risk * 100:.0f}% — model expects an engine "
                           "failure within ~1 min. Get down now.",
                           {"title": "Land immediately",
                            "detail": "Learned risk model sees a failure signature — arrive over a landing spot."},
                           condition=True)
            elif risk > 0.35:
                self._emit(t, "ML_RISK", "WARN",
                           f"ML failure risk {risk * 100:.0f}% over the next minute — reduce power and "
                           "stay in glide range.",
                           {"title": "Reduce power",
                            "detail": "Lower RPM/temps slow every degradation channel the model tracks."},
                           condition=True)
            else:
                self._emit(t, "ML_RISK", "CRIT", "", condition=False)
                self._emit(t, "ML_RISK", "WARN", "", condition=False)

            dx, conf = ml.get("class"), float(ml.get("conf") or 0.0)
            probs = ml.get("probs") or {}
            ice_p = float(probs.get("carb_ice", 0.0))
            leak_p = float(probs.get("oil_leak", 0.0))
            if ice_p > 0.55 and "RPM_FLUCTUATION" not in flags:
                self._emit(t, "ML_ICE", "WARN",
                           f"ML signature: carb icing pattern ({ice_p * 100:.0f}% confidence) — RPM not "
                           "rough yet. Carb heat now beats waiting.",
                           carb, condition=True)
            else:
                self._emit(t, "ML_ICE", "WARN", "", condition=False)
            if leak_p > 0.60:
                self._emit(t, "ML_LEAK", "WARN",
                           f"ML signature: oil-system loss pattern ({leak_p * 100:.0f}% confidence) — "
                           "watch pressure.",
                           {"title": "Plan a landing",
                            "detail": "The learned model recognises an oil-loss pattern; plan a landing early."},
                           condition="OIL_LEAK" not in flags)
            else:
                self._emit(t, "ML_LEAK", "WARN", "", condition=False)

        # aircraft destroyed
        crash = s.get("crash") or {}
        if crash.get("crashed"):
            self._emit(t, "CRASHED", "CRIT",
                       f"AIRCRAFT CRASHED — {crash.get('reason', 'terrain impact')}. "
                       "Press R in the simulator (or RESET SIM on the dashboard) to fly again.",
                       {"title": "Reset the simulator",
                        "detail": "Press R in the simulator window or click RESET SIM on the dashboard."},
                       condition=True)
        else:
            self._emit(t, "CRASHED", "CRIT", "", condition=False)
        self._emit(t, "GPS_LOST", "WARN", f"GPS {gps.get('health')} — HDOP {gps.get('hdop',0):.1f}.",
                   condition=gps.get("health") != "OK")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_suggestions(s, ins, active):
        """Top textual suggestions, derived from active alerts. Text only —
        nothing here can reach the aircraft; the pilot decides and acts."""
        out, seen = [], set()
        for a in active:
            sug = a.get("suggestion")
            if sug and sug["title"] not in seen:
                out.append({"id": a["id"], "title": sug["title"], "detail": sug["detail"],
                            "source_alert": a["msg"]})
                seen.add(sug["title"])
            if len(out) >= 4:
                break
        return out


def _pos_xy(s):
    pos = s.get("state", {}).get("pos_ned", [0.0, 0.0, 0.0])
    return float(pos[0]), float(pos[1])


class PilotAssistant:
    """Rule-based chat assistant. Answers are grounded in the live telemetry.

    It can only ANSWER and SUGGEST — it has no ability to command the aircraft.
    """

    def __init__(self):
        self.last_answers = deque(maxlen=50)

    # ------------------------------------------------------------------ #
    def answer(self, text, s, assessment):
        if not s or not assessment:
            return "No telemetry received yet — start the simulator first."
        q = text.lower().strip()
        ins = assessment["insights"]
        subs = assessment["subsystems"]
        kt = lambda v: v * KT
        ft = lambda v: v * 3.28084

        def has(*words):
            return any(w in q for w in words)

        if has("endurance", "how long", "fuel", "range", "fly for"):
            e = ins.get("endurance_min")
            if e is None:
                return ("Fuel flow is ~0 (engine idle or off). Fuel on board: "
                        f"{s.get('engine_sensors',{}).get('fuel_qty_L',0):.2f} L.")
            rng = ins.get("range_km")
            home = (f" Fuel needed to reach home: {ins['fuel_home_L']:.2f} L — "
                    + ("OK with reserve." if ins.get("can_reach_home")
                       else "WARNING: not enough for home with reserve!")) \
                   if ins.get("fuel_home_L") is not None else ""
            e_txt = f"{e:.0f} min" if e < 150 else f"{e / 60:.1f} h"
            return (f"Fuel {s.get('engine_sensors',{}).get('fuel_qty_L',0):.0f} L at "
                    f"{ins['fuel_flow_lph']:.1f} L/h → endurance ≈ {e_txt}"
                    + (f", range ≈ {rng:.0f} km." if rng else ".") + home)

        if has("status", "health", "how is", "how's", "score", "condition"):
            ov = assessment["overall"]
            worst = min(subs.items(), key=lambda kv: kv[1]["score"])
            notes = "; ".join(worst[1]["notes"]) or "no issues"
            return (f"Overall health {ov['score']:.0f}/100 ({ov['status']}). "
                    f"Weakest: {worst[0].upper()} — {notes}. {ov['summary']}")

        if has("resolve", "resolved", "fixed", "repair", "what is wrong", "what's wrong", "problem"):
            f = (s.get("engine_summary") or {}).get("faults") or {}
            parts = []
            stress = float(f.get("stress_pct", 0.0))
            ice = float(f.get("ice_pct", 0.0))
            oil_qty = float(f.get("oil_qty_pct", 100.0))
            if f.get("seized"):
                parts.append("ENGINE SEIZED — permanent failure, only a forced landing now.")
            if stress > 5:
                parts.append(f"mechanical stress {stress:.0f}% (power -{(1-f.get('power_factor',1))*100:.0f}%) — "
                             + ("reducing: recovering." if assessment and any(a["id"]=="STRESS_EASING" for a in assessment.get("alerts",[]))
                                else "remedy: cut throttle below ~90% RPM"))
            if ice > 5:
                parts.append(f"carb ice {ice:.0f}% — remedy: carb heat "
                             + ("applied, melting." if s.get("engine_summary",{}).get("carb_heat") else "(press H)"))
            if f.get("leak_active"):
                parts.append(f"oil leak — {oil_qty:.0f}% left, not fixable in flight — land before it seizes.")
            if not parts:
                return "No active problems — engine clean, no faults detected. I'll flag anything the moment it starts."
            return "Active issues: " + "; ".join(parts) +                    ". I track each one and confirm here as soon as it's resolved."

        if has("stress", "damage", "abuse", "overheat"):
            f = (s.get("engine_summary") or {}).get("faults") or {}
            st = float(f.get("stress_pct", 0.0))
            if f.get("seized"):
                return "The engine is seized — sustained abuse cooked it. Best glide and force-land now."
            return (f"Mechanical stress {st:.0f}% (power factor {f.get('power_factor',1):.2f}). "
                    + ("It heals slowly below ~78% RPM — hold power down and I'll confirm recovery."
                       if st > 5 else "Clean. Keep RPM out of the red for long stretches and it stays this way."))

        if has("icing", "carb ice", "carburetor", "carburettor"):
            f = (s.get("engine_summary") or {}).get("faults") or {}
            ice = float(f.get("ice_pct", 0.0))
            ch = s.get("engine_summary", {}).get("carb_heat")
            if ice < 2:
                return "No carb ice right now. I keep watching — rough RPM would show it early."
            return (f"Carb ice {ice:.0f}%. Carb heat is {'ON — melting, a few more seconds.' if ch else 'OFF — press H to apply it'}. "
                    f"Full ice costs up to {45:.0f}% power.")

        if has("oil"):
            f = (s.get("engine_summary") or {}).get("faults") or {}
            q = float(f.get("oil_qty_pct", 100.0))
            if not f.get("leak_active") and q > 95:
                return "Oil quantity and pressure nominal — no leak detected."
            rate = 100.0 / 130.0 * (1.0 - 0.4 * s.get("manual_controls", {}).get("throttle", 0.55))
            eta = q / max(rate, 0.01)
            return (f"Oil leak in progress: {q:.0f}% remaining (~{eta/60:.0f} min at this power). "
                    + ("Cutting throttle slows it; landing before empty is the only fix." if q > 25
                       else "Nearly dry — expect seizure any moment, get it on the ground."))


        if has("engine", "motor", "rpm", "prop"):
            es = s.get("engine_sensors", {})
            et = s.get("engine_summary", {})
            return (f"Engine {'RUNNING' if et.get('running') else 'OFF'}: "
                    f"{es.get('rpm',0):.0f} RPM, {et.get('power_kw',0):.1f} kW, "
                    f"CHT {es.get('cht_C',0):.0f} C, EGT {es.get('egt_C',0):.0f} C, "
                    f"oil {es.get('oil_psi',0):.0f} psi, FF {es.get('fuel_flow_Lph',0):.2f} L/h."
                    + (f" Flags: {', '.join(et.get('health_flags', []))}." if et.get('health_flags') else ""))

        if has("cht", "temp", "heat", "egt"):
            tr = ins.get("cht_trend_c_per_min")
            return (f"CHT {s.get('engine_sensors',{}).get('cht_C',0):.0f} C "
                    f"(limit {PISTON_ENGINE['cht_max_c']:.0f}), "
                    f"EGT {s.get('engine_sensors',{}).get('egt_C',0):.0f} C "
                    f"(limit {PISTON_ENGINE['egt_max_c']:.0f})."
                    + (f" CHT trend {tr:+.0f} C/min." if tr is not None else ""))

        if has("oil"):
            return (f"Oil pressure {s.get('engine_sensors',{}).get('oil_psi',0):.1f} psi "
                    f"(nominal {PISTON_ENGINE['oil_pressure_nominal_psi']:.0f}, "
                    f"min {PISTON_ENGINE['oil_pressure_min_psi']:.0f}).")

        if has("stall", "speed margin", "slow"):
            return (f"IAS {kt(s.get('pitot_reading',{}).get('ias_mps',0)):.0f} kt; "
                    f"estimated stall {ins['stall_ias_kt']:.0f} kt → margin "
                    f"{ins['stall_margin_kt']:.0f} kt. Keep ≥ 10 kt above stall, "
                    f"especially in turns (load factor raises stall speed).")

        if has("alt", "height", "agl", "high", "climb", "descend"):
            return (f"Baro altitude {ft(s.get('pitot_reading',{}).get('baro_alt_msl_m',0)):.0f} ft MSL, "
                    f"AGL {ins['agl_m']:.0f} m, VS {s.get('pitot_reading',{}).get('vsi_mps',0)*196.85:+.0f} fpm.")

        if has("gps", "navigation", "hdop", "satellite"):
            g = s.get("gps", {})
            return (f"GPS {g.get('health')}: {g.get('satellites')} sats, HDOP {g.get('hdop',0):.2f}, "
                    f"groundspeed {g.get('groundspeed_mps',0)*KT:.0f} kt, track {g.get('track_deg',0):.0f} deg.")

        if has("nfz", "no-fly", "nofly", "restricted", "airspace"):
            v = s.get("nfz_violations", [])
            n = s.get("nfz_nearest")
            if v:
                return f"BREACHING {v[0]['name']} — penetration {v[0]['penetration_m']:.0f} m. Exit immediately!"
            if n:
                return (f"Clear of NFZs. Nearest: {n['name']} boundary in {n['margin_m']:.0f} m.")
            return "No no-fly zones defined."

        if has("wind", "gust", "turbulen"):
            hw = ins.get("headwind_mps")
            w = s.get("wind_ned")
            if w is None:
                return "No wind data in this snapshot."
            spd = math.hypot(w[0], w[1])
            frm = (math.degrees(math.atan2(-w[1], -w[0])) + 360.0) % 360.0
            comp = f" {'Headwind' if (hw or 0) > 0 else 'Tailwind'} component {abs(hw):.1f} m/s." if hw is not None else ""
            return f"Wind {spd:.1f} m/s from {frm:.0f} deg.{comp}"

        if has("mission", "waypoint", "route", "wp"):
            idx = s.get("mission_current_idx", 0)
            cnt = s.get("mission_count", 0)
            if cnt == 0:
                return "No mission loaded. Click on the map (sim window) or use the dashboard to add waypoints."
            d = s.get("mission_dist_m")
            b = s.get("mission_bearing_deg", 0)
            return (f"Waypoint {idx+1}/{cnt}: {d:.0f} m away, bearing {b:.0f} deg. "
                    f"Mode is {s.get('mode')}.")

        if has("suggest", "should i", "what now", "advice", "help me", "recommend"):
            sug = assessment.get("suggestions", [])
            if not sug:
                alerts = assessment.get("alerts", [])
                if alerts:
                    return ("No direct actions to suggest — watch: "
                            + "; ".join(a["msg"] for a in alerts[:3]))
                return "All systems nominal. No action needed — keep the aircraft within envelope."
            lines = [f"{i+1}. {x['title']} — {x['detail']}." for i, x in enumerate(sug)]
            return "My suggestions (you decide and act — I never fly the aircraft): " + " ".join(lines)

        if has("ml", "machine learning", "neural", "diagnos", "trained", "learn"):
            tw = s.get("twin") or {}
            ml = tw.get("ml") or {}
            if not ml.get("available"):
                return ("The ML models aren't loaded (run simulator/ai_advisor/train_ml.py once) — "
                        "I'm still monitoring with the physics twin and rules.")
            probs = ml.get("probs", {})
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            risk = float(ml.get("fail_risk_60s") or 0.0)
            line = ", ".join(f"{k} {v * 100:.0f}%" for k, v in top)
            return (f"ML diagnosis: {line}. Failure risk within ~60 s: {risk * 100:.0f}%. "
                    f"Behaviour model R² ≈ 0.99 on normal response; it has adapted online to "
                    f"{ml.get('learned_samples', 0)} live samples of this engine. "
                    "Diagnosis is a probability, not a verdict — cross-check the gauges.")

        if has("who are you", "can you", "override", "control", "fly"):
            return ("I'm the AI flight assistant. I monitor engine, energy, flight envelope, "
                    "navigation and sensors in real time and I give suggestions — but I "
                    "never override the UAV. I have no connection to the controls; you fly.")

        if has("hello", "hi", "hey"):
            ov = assessment["overall"]
            return (f"Hello! Monitoring live — overall health {ov['score']:.0f}/100 ({ov['status']}). "
                    f"Ask me about fuel/endurance, engine temps, stall margin, GPS, NFZs, wind, or say 'suggest'.")

        return ("I can answer about: fuel & endurance, engine/RPM/CHT/EGT/oil, stall & speeds, "
                "altitude, GPS, no-fly zones, wind, mission waypoints, 'ml diagnose', or say 'suggest'. "
                "Note: I only advise — I never override the UAV.")

    # ------------------------------------------------------------------ #
