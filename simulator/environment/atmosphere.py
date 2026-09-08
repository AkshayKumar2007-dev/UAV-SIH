import math
import numpy as np
from simulator.config import (
    G, R_AIR, T_SL, P_SL, LAPSE_RATE, RHO_SL, WIND,
)


def isa_atmosphere(alt_msl_m):
    h = max(alt_msl_m, 0.0)
    if h < 11000.0:
        T = T_SL - LAPSE_RATE * h
        p_ratio = (T / T_SL) ** (G / (LAPSE_RATE * R_AIR))
    else:
        T_tropo = T_SL - LAPSE_RATE * 11000.0
        T = T_tropo
        p_tropo = (T_tropo / T_SL) ** (G / (LAPSE_RATE * R_AIR))
        p_ratio = p_tropo * math.exp(-G * (h - 11000.0) / (R_AIR * T_tropo))
    P = P_SL * p_ratio
    rho = P / (R_AIR * max(T, 1.0))
    return {"T_K": T, "P_Pa": P, "rho_kgm3": rho, "a_mps": math.sqrt(1.4 * R_AIR * max(T, 1.0))}


class WindField:
    def __init__(self):
        self.t = 0.0
        self._noise_state = np.zeros(3)

    def step(self, dt):
        self.t += dt
        tau = 2.5
        wn = WIND["turbulence_intensity"] * 3.2
        self._noise_state += (np.random.randn(3) * wn - self._noise_state) * (dt / tau)

    def wind_at(self, alt_msl_m, pos_ned=None):
        h_ref = 10.0
        h_ratio = max(alt_msl_m / h_ref, 1e-4)
        shear = h_ratio ** WIND["wind_shear_exp"]
        base_N = WIND["speed_north_mps"] * shear
        base_E = WIND["speed_east_mps"] * shear
        base_D = 0.0

        gust_N = WIND["gust_amplitude_mps"] * math.sin(self.t * 0.7 + 0.3)
        gust_E = WIND["gust_amplitude_mps"] * 0.7 * math.sin(self.t * 0.55 + 1.1)
        gust_D = WIND["gust_amplitude_mps"] * 0.3 * math.sin(self.t * 0.9 + 2.5)

        turb = self._noise_state * WIND["turbulence_intensity"] * 2.5

        v_north = base_N + gust_N + turb[0]
        v_east = base_E + gust_E + turb[1]
        v_down = base_D + gust_D + turb[2]
        return np.array([v_north, v_east, v_down])

    def summary(self):
        w = self.wind_at(100.0)
        return {
            "wind_n_mps": float(w[0]),
            "wind_e_mps": float(w[1]),
            "wind_d_mps": float(w[2]),
            "wind_speed_mps": float(np.linalg.norm(w)),
            "wind_dir_from_deg": float((math.degrees(math.atan2(-w[1], -w[0])) + 360.0) % 360.0),
        }
