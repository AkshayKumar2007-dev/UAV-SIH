import numpy as np
from simulator.config import AERO, AIRFRAME
from simulator.aircraft.rigid_body import rot_ned_b


def _lift_coeff(alpha_rad, delta_e_rad, delta_flap_rad, q_hat):
    a_s = AERO["alpha_stall_rad"]
    a_neg_s = AERO["alpha_neg_stall_rad"]
    cl_a = AERO["CL_alpha_per_rad"]
    cl_a_stall = AERO["CL_alpha_stall_per_rad"]
    delta_cl_flap = (delta_flap_rad / np.radians(AIRFRAME["delta_flap_max_deg"])) * AIRFRAME["flap_delta_CL"]
    delta_cl_elev = AERO["CL_delta_e_per_rad"] * delta_e_rad
    delta_cl_q = AERO["CL_q_per_radps"] * q_hat

    if alpha_rad > a_s:
        cl_max = AERO["CL0"] + cl_a * a_s + delta_cl_flap + delta_cl_q
        cl = cl_max + cl_a_stall * (alpha_rad - a_s)
    elif alpha_rad < a_neg_s:
        cl_min_neg = AERO["CL0"] + cl_a * a_neg_s + delta_cl_flap + delta_cl_q
        cl = cl_min_neg - cl_a_stall * (a_neg_s - alpha_rad)
    else:
        cl = AERO["CL0"] + cl_a * alpha_rad + delta_cl_flap + delta_cl_elev + delta_cl_q

    return float(cl)


def _drag_coeff(cl, delta_flap_rad, gear_down, brakes_on):
    cd = AERO["CD0"] + AERO["K_induced"] * cl * cl
    cd += (delta_flap_rad / np.radians(AIRFRAME["delta_flap_max_deg"])) * AIRFRAME["flap_delta_CD"]
    if gear_down:
        cd += AIRFRAME["gear_drag_coeff"]
    if brakes_on:
        cd += 0.02
    return float(cd)


def compute_forces_and_moments(state, controls, rho_kgm3, wind_ned_mps, mass_kg, thrust_N=0.0):
    pos, vel_b, euler, rates = state["pos_ned"], state["vel_body"], state["euler"], state["rates"]
    p, q, r = rates
    phi, theta, psi = euler

    R_ned_b = rot_ned_b(phi, theta, psi)
    wind_body = R_ned_b @ wind_ned_mps
    v_air_b = vel_b - wind_body
    u_air, v_air, w_air = v_air_b
    V_air = float(np.linalg.norm(v_air_b))
    V_safe = max(V_air, 1.0)

    alpha = float(np.arctan2(w_air, u_air)) if abs(u_air) > 0.3 else 0.0
    beta = float(np.arcsin(np.clip(v_air / V_safe, -1.0, 1.0)))

    q_dyn = 0.5 * rho_kgm3 * V_air * V_air
    S = AIRFRAME["S_wing_m2"]
    b = AIRFRAME["b_span_m"]
    c = AIRFRAME["c_bar_m"]

    p_hat = p * b / (2.0 * V_safe)
    q_hat = q * c / (2.0 * V_safe)
    r_hat = r * b / (2.0 * V_safe)

    d_a = np.radians(controls["delta_a_deg"])
    d_e = np.radians(controls["delta_e_deg"])
    d_r = np.radians(controls["delta_r_deg"])
    d_flap = np.radians(controls["delta_flap_deg"])

    CL = _lift_coeff(alpha, d_e, d_flap, q_hat)
    stall = alpha > AERO["alpha_stall_rad"] or alpha < AERO["alpha_neg_stall_rad"]
    CD = _drag_coeff(CL, d_flap, controls.get("gear_down", False), controls.get("brakes", False))
    CY = AERO["CY_beta_per_rad"] * beta + AERO["CY_delta_r_per_rad"] * d_r

    Cl = (
        AERO["Cl_p_per_radps"] * p_hat
        + AERO["Cl_beta_per_rad"] * beta
        + AERO["Cl_delta_a_per_rad"] * d_a
    )
    Cm = (
        AERO["Cm0"]
        + AERO["Cm_alpha_per_rad"] * alpha
        + AERO["Cm_delta_e_per_rad"] * d_e
        + AERO["Cm_q_per_radps"] * q_hat
    )
    Cn = (
        AERO["Cn_r_per_radps"] * r_hat
        + AERO["Cn_beta_per_rad"] * beta
        + AERO["Cn_delta_r_per_rad"] * d_r
        + AERO["Cn_p_per_radps"] * p_hat
    )

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    L = q_dyn * S * CL
    D = q_dyn * S * CD
    Y = q_dyn * S * CY
    Fx_aero = -D * ca * cb + Y * ca * sb + L * sa
    Fy_aero = -D * sb + Y * cb
    Fz_aero = -D * sa * cb + Y * sa * sb - L * ca
    F_aero_body = np.array([Fx_aero, Fy_aero, Fz_aero])

    F_thrust_body = np.array([thrust_N, 0.0, 0.0])
    F_body = F_aero_body + F_thrust_body

    M_aero = np.array([q_dyn * S * b * Cl, q_dyn * S * c * Cm, q_dyn * S * b * Cn])

    prop_offset_z = AERO["engine_mount_offset_z_m"]
    M_thrust = np.array([0.0, -thrust_N * prop_offset_z, 0.0])

    M_body = M_aero + M_thrust

    h_agl = max(0.0, -pos[2])
    on_ground = h_agl < 0.3
    if on_ground:
        N = mass_kg * 9.80665 * np.cos(theta)
        rr_force = -np.sign(vel_b[0]) * AIRFRAME["wheel_rr_coeff"] * N
        brake_force = 0.0
        if controls.get("brakes", False) and abs(vel_b[0]) < 60:
            brake_force = -np.sign(vel_b[0]) * AIRFRAME["brake_friction_coeff"] * N
        F_body[0] += rr_force + brake_force
        if pos[2] > -0.05:
            F_body[2] += N * 0.9

    return {
        "F_body": F_body,
        "M_body": M_body,
        "alpha_rad": alpha,
        "beta_rad": beta,
        "V_air_mps": V_air,
        "q_dyn_Pa": q_dyn,
        "CL": CL,
        "CD": CD,
        "stall": bool(stall),
        "on_ground": on_ground,
        "h_agl_m": h_agl,
    }
