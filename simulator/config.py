import math
import numpy as np

G = 9.80665
RHO_SL = 1.225
P_SL = 101325.0
T_SL = 288.15
LAPSE_RATE = 0.0065
R_AIR = 287.058

SIM_PHYSICS_HZ = 50
DT_PHYS = 1.0 / SIM_PHYSICS_HZ
SIM_RENDER_HZ = 60
TELEMETRY_HZ = 10

TELEMETRY = {
    "ws_host": "127.0.0.1",
    "ws_port": 8765,
    "http_port": 8766,
}

HOME = {
    "lat_deg": 28.6129,
    "lon_deg": 77.2295,
    "alt_msl_m": 225.0,
}

# Spawn: on the runway threshold, at rest, lined up on runway 36.
# z = HOME.alt - field elevation (airport plateau ~247.3 m MSL)
INITIAL_CONDITIONS = {
    "pos_ned_m": np.array([-370.0, 0.0, -22.3]),
    "vel_body_mps": np.array([0.0, 0.0, 0.0]),
    "euler_rad": np.array([0.0, 0.02, 0.0]),
    "rates_radps": np.array([0.0, 0.0, 0.0]),
}

# Demo time compression: engine AGING (fuel burn, thermal states, wear, icing,
# leak) runs at this multiple of real time so a MALE-length mission arc is
# visible in a short demo. Flight dynamics stay real-time. The AI reports
# endurance in honest engine-hours (e.g. "16 h"), which then drain 60x faster.
ENGINE_TIME_SCALE = 60.0

# MALE-class reference platform (TAPAS / TB2 / Rotax-914 heavy-fuel class):
# ~750 kg MTOW, 14 m span, 86 kW turbocharged piston, 240 L fuel, 20 h endurance.
AIRFRAME = {
    "mass_dry_kg": 560.0,
    "fuel_full_kg": 240.0 * 0.8,
    "Ixx": 420.0,
    "Iyy": 900.0,
    "Izz": 1250.0,
    "Ixz": 10.0,
    "S_wing_m2": 13.5,
    "b_span_m": 14.0,
    "c_bar_m": 0.964,
    "x_cg_m": 0.02,
    "delta_a_max_deg": 20.0,
    "delta_e_max_deg": 25.0,
    "delta_r_max_deg": 30.0,
    "delta_flap_max_deg": 40.0,
    "gear_drag_coeff": 0.018,
    "flap_delta_CD": 0.06,
    "flap_delta_CL": 0.5,
    "brake_friction_coeff": 0.25,
    "wheel_rr_coeff": 0.015,
}

AERO = {
    "CL0": 0.18,
    "CL_alpha_per_rad": 5.2,
    "CL_alpha_stall_per_rad": 0.8,
    "alpha_stall_rad": math.radians(14.0),
    "alpha_neg_stall_rad": math.radians(-10.0),
    "CD0": 0.028,
    "K_induced": 1.0 / (math.pi * 0.78 * (AIRFRAME["b_span_m"] ** 2 / AIRFRAME["S_wing_m2"])),
    "Cm0": -0.015,
    "Cm_alpha_per_rad": -0.75,
    "Cm_q_per_radps": -8.0,
    "CL_q_per_radps": 2.5,
    "Cl_p_per_radps": -0.48,
    "Cl_beta_per_rad": -0.09,
    "Cl_delta_a_per_rad": 0.24,
    "Cn_r_per_radps": -0.10,
    "Cn_beta_per_rad": 0.14,
    "Cn_delta_r_per_rad": -0.16,
    "Cn_p_per_radps": 0.02,
    "CY_beta_per_rad": -0.58,
    "CY_delta_r_per_rad": 0.22,
    "Cm_delta_e_per_rad": -1.1,
    "CL_delta_e_per_rad": 0.35,
    "prop_diam_m": 1.90,
    "engine_mount_offset_z_m": -0.05,
}

PISTON_ENGINE = {
    "idle_rpm": 1400.0,
    "max_rpm": 5800.0,
    "rpm_tau_s": 0.5,
    "max_power_kw": 86.0,
    "prop_eff_coeff": 0.135,
    "prop_power_coeff": 0.037,
    "bsfc_kg_per_kwh": 0.30,          # heavy-fuel EFI piston at ~55% power
    "fuel_full_L": 240.0,
    "fuel_density_kg_per_L": 0.80,    # Jet-A / heavy fuel
    "cht_nominal_c": 150.0,
    "cht_max_c": 240.0,
    "cht_tau_s": 2400.0,
    "egt_nominal_c": 680.0,
    "egt_max_c": 880.0,
    "egt_tau_s": 300.0,
    "oil_pressure_nominal_psi": 65.0,
    "oil_pressure_min_psi": 30.0,
    "oil_tau_s": 240.0,
    "starter_torque_s": 3.0,
    # AI endurance-watch thresholds (honest engine-hours based)
    "endurance_warn_min": 90.0,
    "endurance_crit_min": 45.0,
}

WIND = {
    "speed_north_mps": 4.0,
    "speed_east_mps": -2.5,
    "gust_amplitude_mps": 2.0,
    "turbulence_intensity": 0.08,
    "wind_shear_exp": 0.14,
}

WORLD = {
    "terrain_radius_n_m": 14000,
    "terrain_base_alt_msl_m": HOME["alt_msl_m"],
    "terrain_bumps_m": 35.0,
    "no_fly_zones": [
        {"center_ned": (800.0, 600.0), "radius_m": 250.0, "floor_m": 0.0, "ceil_m": 800.0, "name": "AIRPORT CTR"},
        {"center_ned": (-1500.0, 1200.0), "radius_m": 400.0, "floor_m": 0.0, "ceil_m": 1200.0, "name": "RESTRICTED R-401"},
        {"center_ned": (6500.0, -5200.0), "radius_m": 700.0, "floor_m": 0.0, "ceil_m": 1500.0, "name": "TOWN CTR"},
    ],
    "default_waypoints": [
        {"n_m": 1200.0, "e_m": 400.0, "alt_msl_m": 400.0},
        {"n_m": 2400.0, "e_m": -800.0, "alt_msl_m": 450.0},
        {"n_m": 500.0, "e_m": -2200.0, "alt_msl_m": 420.0},
    ],
    "airport": {
        "runway": {"n": 0.0, "e": 0.0, "heading_deg": 0.0, "length_m": 800.0, "width_m": 30.0},
        "hangar": {"n": -150.0, "e": -75.0, "w_m": 36.0, "d_m": 24.0, "h_m": 10.0},
        "tower": {"n": 90.0, "e": -85.0, "radius_m": 4.5, "h_m": 26.0},
        "flatten_radius_m": 900.0,
        "flatten_inner_m": 450.0,
    },
    "mountains": [
        {"n": -2200.0, "e": -1500.0, "radius_m": 650.0, "h_m": 360.0},
        {"n": 3200.0, "e": 2600.0, "radius_m": 600.0, "h_m": 320.0},
        {"n": -1800.0, "e": 3400.0, "radius_m": 800.0, "h_m": 430.0},
        {"n": 2900.0, "e": -3100.0, "radius_m": 550.0, "h_m": 300.0},
        {"n": -5200.0, "e": 2400.0, "radius_m": 900.0, "h_m": 520.0},
        {"n": -4600.0, "e": 4400.0, "radius_m": 750.0, "h_m": 460.0},
        {"n": 5200.0, "e": 6800.0, "radius_m": 850.0, "h_m": 560.0},
        {"n": 6800.0, "e": -7200.0, "radius_m": 950.0, "h_m": 620.0},
        {"n": -9800.0, "e": -6200.0, "radius_m": 1100.0, "h_m": 680.0},
    ],
    "lake": {"n": -4200.0, "e": 3600.0, "radius_m": 900.0},
    "airstrips": [
        {"n": -8200.0, "e": 6400.0, "heading_deg": 30.0, "length_m": 1100.0, "width_m": 38.0,
         "name": "NORTH STRIP"},
    ],
    "roads": [
        [[0.0, -120.0], [700.0, -60.0], [1200.0, 350.0], [1700.0, 1000.0]],
        [[60.0, 150.0], [1800.0, -700.0], [3800.0, -2800.0], [5200.0, -4300.0], [6300.0, -5100.0]],
    ],
}

SENSORS = {
    "gps": {
        "rate_hz": 5,
        "pos_sigma_m": 2.5,
        "alt_sigma_m": 4.5,
        "vel_sigma_mps": 0.25,
        "hdop_nominal": 1.2,
        "hdop_max": 6.0,
        "dropout_prob_per_sec": 0.0004,
        "min_dropout_s": 2.0,
        "max_dropout_s": 10.0,
    },
    "imu": {
        "rate_hz": 50,
        "accel_noise_std_mps2": 0.03,
        "gyro_noise_std_radps": 0.0003,
        "accel_bias_stability_mps2": 0.004,
        "gyro_bias_stability_radps": 0.00008,
        "bias_tau_s": 400.0,
    },
    "pitot": {
        "rate_hz": 20,
        "ias_noise_std_mps": 0.4,
        "baro_alt_noise_std_m": 0.9,
        "vsi_noise_std_mps": 0.15,
        "position_error_mps": 1.0,
    },
    "engine": {
        "rpm_noise_std_pct": 0.3,
        "fuel_flow_noise_std_Lph": 0.15,
        "cht_noise_std_c": 1.5,
        "egt_noise_std_c": 6.0,
        "oil_p_noise_std_psi": 0.6,
        "fuel_qty_noise_std_L": 0.08,
    },
}

PID = {
    "roll_att": {"kp": 1.2, "ki": 0.1, "kd": 0.15, "i_lim": 8.0, "out_lim_deg": 18.0},
    "pitch_att": {"kp": 1.2, "ki": 0.25, "kd": 0.2, "i_lim": 8.0, "out_lim_deg": 18.0},
    "yaw_damper": {"kp": 1.2, "ki": 0.0, "kd": 0.05, "i_lim": 0.0, "out_lim_deg": 12.0},
    "hdg_hold": {"kp": 1.6, "ki": 0.05, "kd": 0.0, "i_lim": 12.0, "out_lim_deg": 25.0},
    "alt_hold": {"kp": 0.08, "ki": 0.035, "kd": 0.0, "i_lim": 120.0, "out_lim_deg": 5.0},
    "climb_rate": {"kp": 1.8, "ki": 0.15, "kd": 0.0, "i_lim": 20.0, "out_lim_deg": 12.0},
    "airspeed_pitch": {"kp": 0.35, "ki": 0.05, "kd": 0.0, "i_lim": 8.0, "out_lim_deg": 12.0},
    "throttle_ias": {"kp": 0.09, "ki": 0.03, "kd": 0.0, "i_lim": 30.0, "out_lim": 1.0},
}

FAULTS = {
    "high_rpm_frac": 0.92,          # RPM above this fraction of max = abuse
    # stress rates are per REAL second (a normal takeoff is safe; holding max
    # power for minutes is what kills the engine)
    "stress_per_s_high_rpm": 1.0 / 100.0,
    "stress_per_s_overtemp": 1.0 / 180.0,
    "stress_per_s_low_oil": 1.0 / 25.0,
    "stress_decay_per_s": 0.30 / 60.0,     # heals slowly when flown gently
    "stress_power_penalty": 0.6,           # power factor = 1 - penalty * stress
    "overtemp_margin_c": 5.0,
    # ice/leak rates are per REAL second (consistent with the stress rates):
    # a full-ice or empty-oil arc therefore plays out over ~2 minutes once active
    "ice_rate_per_s": 1.0 / 8400.0,        # carb ice build-up while window active
    "ice_start_s_range": (90.0, 240.0),    # randomized onset, REAL seconds into the flight
    "ice_chance": 0.75,                    # probability the icing fault occurs this flight
    "ice_melt_per_s": 1.0 / 1080.0,        # with carb heat applied
    "ice_power_loss_frac": 0.45,           # max power loss at full ice
    "oil_leak_delay_s_range": (240.0, 600.0),   # randomized onset, REAL seconds
    "oil_leak_chance": 0.5,                # probability of a leak this flight
    "oil_leak_rate_per_s": 1.0 / 7800.0,   # fraction of oil quantity per second
    "oil_leak_throttle_relief": 0.4,       # low throttle slows the leak by this share
}

CRASH = {
    "vs_limit_mps": -6.0,          # harder vertical impact than this destroys the aircraft
    "bank_limit_deg": 55.0,        # wing strike
    "pitch_down_limit_deg": -30.0, # nose-first impact
    "pitch_up_limit_deg": 25.0,    # tail strike
    "ias_limit_mps": 34.0,         # high-speed terrain contact
}

GUIDANCE = {
    "L1_period_s": 15.0,
    "L1_damping": 0.85,
    "loiter_radius_m": 400.0,
    "rth_alt_msl_m": 3650.0,
    "waypoint_capture_radius_m": 250.0,
    "min_bank_deg_for_turn": 10.0,
    "max_bank_deg_aps": 25.0,
    "approach_ias_mps": 34.0,
    "cruise_ias_mps": 42.0,
    "climb_ias_mps": 38.0,
    "descend_ias_mps": 40.0,
    "rotate_ias_mps": 32.0,
    "takeoff_pitch_deg": 10.0,
    "min_agl_after_takeoff_m": 120.0,
}

UI = {
    "screen_w_px": 1280,
    "screen_h_px": 800,
    "map_px_per_m": 0.08,
    "hud_font_size": 16,
    "panel_font_size": 14,
    "sky_color": (135, 180, 220),
    "ground_color": (85, 115, 70),
    "accent_color": (255, 210, 60),
    "warn_color": (240, 120, 30),
    "crit_color": (220, 40, 40),
    "ok_color": (60, 200, 90),
}


# --------------------------------------------------------------------- #
# Scenery: deterministic forests and a small village (single source of
# truth for physics collision, the chase view and the dashboard scene).
# --------------------------------------------------------------------- #
import math as _math
import random as _random

_scenery_rng = _random.Random(20240905)
_ap = WORLD["airport"]
_rw, _hg, _tw = _ap["runway"], _ap["hangar"], _ap["tower"]


def _in_keepout(n, e):
    if abs(n) < 550 and abs(e) < 130:          # runway strip
        return True
    if abs(n - _hg["n"]) < 70 and abs(e - _hg["e"]) < 70:
        return True
    if _math.hypot(n - _tw["n"], e - _tw["e"]) < 60:
        return True
    return False


_trees = []
_woods = [(900.0, 1500.0, 60), (-1500.0, 2600.0, 55), (2600.0, -1800.0, 50),
          (-3600.0, -500.0, 65), (4200.0, 900.0, 60), (-700.0, -3400.0, 55),
          (5200.0, 3200.0, 60), (-6800.0, -2600.0, 70), (8600.0, -1600.0, 55),
          (-2600.0, 7800.0, 60), (7600.0, 4400.0, 50), (-9200.0, 1200.0, 55),
          (1200.0, 6800.0, 50), (9400.0, 6200.0, 45), (-11600.0, -1200.0, 45)]
for _wn, _we, _count in _woods:
    for _ in range(_count):
        _ang = _scenery_rng.uniform(0.0, 2.0 * _math.pi)
        _dist = _math.sqrt(_scenery_rng.uniform(0.0, 1.0)) * 900.0
        _n, _e = _wn + _dist * _math.cos(_ang), _we + _dist * _math.sin(_ang)
        if _in_keepout(_n, _e):
            continue
        _trees.append({"n": round(_n, 1), "e": round(_e, 1),
                       "h_m": round(_scenery_rng.uniform(9.0, 20.0), 1),
                       "r_m": round(_scenery_rng.uniform(2.5, 5.0), 1)})
for _ in range(240):                            # scattered singles across the region
    _ang = _scenery_rng.uniform(0.0, 2.0 * _math.pi)
    _dist = _math.sqrt(_scenery_rng.uniform(700.0 ** 2, 13000.0 ** 2))
    _n, _e = _dist * _math.cos(_ang), _dist * _math.sin(_ang)
    if _in_keepout(_n, _e):
        continue
    _trees.append({"n": round(_n, 1), "e": round(_e, 1),
                   "h_m": round(_scenery_rng.uniform(9.0, 19.0), 1),
                   "r_m": round(_scenery_rng.uniform(2.5, 5.0), 1)})
for _ in range(60):                             # near-airport woods
    _ang = _scenery_rng.uniform(0.0, 2.0 * _math.pi)
    _dist = _scenery_rng.uniform(320.0, 820.0)
    _n, _e = _dist * _math.cos(_ang), _dist * _math.sin(_ang)
    if _in_keepout(_n, _e):
        continue
    _trees.append({"n": round(_n, 1), "e": round(_e, 1),
                   "h_m": round(_scenery_rng.uniform(8.0, 16.0), 1),
                   "r_m": round(_scenery_rng.uniform(2.5, 4.5), 1)})
WORLD["trees"] = _trees

_village = []
for _i in range(10):
    _n = 1700.0 + (_i % 5) * 62.0 - 124.0 + _scenery_rng.uniform(-9.0, 9.0)
    _e = 1100.0 + (_i // 5) * 74.0 - 37.0 + _scenery_rng.uniform(-9.0, 9.0)
    _village.append({"n": round(_n, 1), "e": round(_e, 1),
                     "w_m": round(_scenery_rng.uniform(14.0, 26.0), 1),
                     "d_m": round(_scenery_rng.uniform(12.0, 22.0), 1),
                     "h_m": round(_scenery_rng.uniform(9.0, 26.0), 1)})
_town = []
for _i in range(28):                            # town blocks
    _n = 6300.0 + (_i % 7) * 90.0 - 270.0 + _scenery_rng.uniform(-12.0, 12.0)
    _e = -5400.0 + (_i // 7) * 105.0 - 160.0 + _scenery_rng.uniform(-12.0, 12.0)
    _town.append({"n": round(_n, 1), "e": round(_e, 1),
                  "w_m": round(_scenery_rng.uniform(16.0, 30.0), 1),
                  "d_m": round(_scenery_rng.uniform(14.0, 26.0), 1),
                  "h_m": round(_scenery_rng.uniform(12.0, 34.0), 1)})
_industrial = []
for _i in range(7):                             # industrial sheds by the town
    _n = 5600.0 + _i * 120.0 - 360.0 + _scenery_rng.uniform(-15.0, 15.0)
    _e = -4600.0 + _scenery_rng.uniform(-40.0, 40.0)
    _industrial.append({"n": round(_n, 1), "e": round(_e, 1),
                        "w_m": round(_scenery_rng.uniform(40.0, 70.0), 1),
                        "d_m": round(_scenery_rng.uniform(25.0, 40.0), 1),
                        "h_m": round(_scenery_rng.uniform(11.0, 16.0), 1)})
WORLD["buildings"] = _village + _town + _industrial
