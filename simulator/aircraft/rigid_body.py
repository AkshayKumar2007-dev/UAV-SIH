import numpy as np
from simulator.config import AIRFRAME, INITIAL_CONDITIONS, G


def rot_b_ned(phi, theta, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    R = np.array([
        [ct * c_psi, sp * st * c_psi - cp * s_psi, cp * st * c_psi + sp * s_psi],
        [ct * s_psi, sp * st * s_psi + cp * c_psi, cp * st * s_psi - sp * c_psi],
        [-st, sp * ct, cp * ct],
    ])
    return R


def rot_ned_b(phi, theta, psi):
    return rot_b_ned(phi, theta, psi).T


def euler_kinematics(phi, theta):
    sp, cp = np.sin(phi), np.cos(phi)
    ct = np.cos(theta)
    safe_ct = ct if abs(ct) > 1e-6 else 1e-6
    tt = np.tan(theta) if abs(ct) > 1e-6 else 1e6
    sec_t = 1.0 / safe_ct
    H = np.array([
        [1.0, sp * tt, cp * tt],
        [0.0, cp, -sp],
        [0.0, sp * sec_t, cp * sec_t],
    ])
    return H


class RigidBody6DOF:
    def __init__(self):
        self.mass_kg = AIRFRAME["mass_dry_kg"] + AIRFRAME["fuel_full_kg"]
        Ixx, Iyy, Izz, Ixz = AIRFRAME["Ixx"], AIRFRAME["Iyy"], AIRFRAME["Izz"], AIRFRAME["Ixz"]
        self.I = np.array([
            [Ixx, 0.0, -Ixz],
            [0.0, Iyy, 0.0],
            [-Ixz, 0.0, Izz],
        ])
        self.I_inv = np.linalg.inv(self.I)
        self.state = {
            "pos_ned": INITIAL_CONDITIONS["pos_ned_m"].copy(),
            "vel_body": INITIAL_CONDITIONS["vel_body_mps"].copy(),
            "euler": INITIAL_CONDITIONS["euler_rad"].copy(),
            "rates": INITIAL_CONDITIONS["rates_radps"].copy(),
        }
        self._deriv_cache = None

    def update_mass(self, new_total_mass_kg):
        ratio = new_total_mass_kg / self.mass_kg if self.mass_kg > 0 else 1.0
        self.mass_kg = max(0.01, new_total_mass_kg)
        self.I = self.I * ratio
        self.I_inv = np.linalg.inv(self.I)

    def _derivatives(self, state, F_applied_body, M_applied_body):
        phi, theta, psi = state["euler"]
        vel_b = state["vel_body"]
        omega = state["rates"]

        R_ned_b = rot_ned_b(phi, theta, psi)
        g_ned = np.array([0.0, 0.0, G])
        g_body = R_ned_b @ g_ned

        omega_cross_v = np.cross(omega, vel_b)
        acc_body = (F_applied_body / self.mass_kg) + g_body - omega_cross_v

        H = euler_kinematics(phi, theta)
        euler_dot = H @ omega

        I_omega = self.I @ omega
        omega_cross_Iomega = np.cross(omega, I_omega)
        omega_dot = self.I_inv @ (M_applied_body - omega_cross_Iomega)

        R_b_ned = rot_b_ned(phi, theta, psi)
        dpos_ned = R_b_ned @ vel_b

        return {
            "pos_ned": dpos_ned,
            "vel_body": acc_body,
            "euler": euler_dot,
            "rates": omega_dot,
        }

    def _add(self, s_a, s_b, scale=1.0):
        return {k: s_a[k] + scale * s_b[k] for k in s_a}

    def step(self, F_applied_body_N, M_applied_body_Nm, dt):
        s = self.state
        k1 = self._derivatives(s, F_applied_body_N, M_applied_body_Nm)
        s2 = self._add(s, k1, dt / 2.0)
        k2 = self._derivatives(s2, F_applied_body_N, M_applied_body_Nm)
        s3 = self._add(s, k2, dt / 2.0)
        k3 = self._derivatives(s3, F_applied_body_N, M_applied_body_Nm)
        s4 = self._add(s, k3, dt)
        k4 = self._derivatives(s4, F_applied_body_N, M_applied_body_Nm)

        for k in s:
            s[k] = s[k] + (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k])

        # safety clamps: keep a bad transient from exploding the integrator
        s["rates"][:] = np.clip(s["rates"], -8.0, 8.0)
        s["vel_body"][:] = np.clip(s["vel_body"], -160.0, 160.0)

        s["euler"][1] = np.clip(s["euler"][1], -1.55, 1.55)
        eul = s["euler"]
        if eul[0] > np.pi:
            eul[0] -= 2.0 * np.pi
        elif eul[0] < -np.pi:
            eul[0] += 2.0 * np.pi
        if eul[2] > np.pi:
            eul[2] -= 2.0 * np.pi
        elif eul[2] < -np.pi:
            eul[2] += 2.0 * np.pi

    @property
    def altitude_msl_m(self):
        from simulator.config import HOME
        return HOME["alt_msl_m"] - self.state["pos_ned"][2]

    @property
    def airspeed_body_mps(self):
        return float(np.linalg.norm(self.state["vel_body"]))

    @property
    def groundspeed_mps(self):
        R_b_ned = rot_b_ned(*self.state["euler"])
        v_ned = R_b_ned @ self.state["vel_body"]
        return float(np.linalg.norm(v_ned[:2]))

    @property
    def alpha_rad(self):
        u, w = self.state["vel_body"][0], self.state["vel_body"][2]
        if abs(u) < 0.5 and abs(w) < 0.5:
            return 0.0
        return float(np.arctan2(w, u))

    @property
    def beta_rad(self):
        u, v, w = self.state["vel_body"]
        V = np.linalg.norm(self.state["vel_body"])
        if V < 0.5:
            return 0.0
        return float(np.arcsin(np.clip(v / max(V, 1e-6), -1.0, 1.0)))
