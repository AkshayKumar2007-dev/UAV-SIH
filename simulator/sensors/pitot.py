import math
import numpy as np
from simulator.config import SENSORS, P_SL, T_SL, LAPSE_RATE, G, R_AIR, HOME, INITIAL_CONDITIONS


def _alt_to_pressure(alt_msl):
    T = T_SL - LAPSE_RATE * max(alt_msl, 0)
    return P_SL * (T / T_SL) ** (G / (LAPSE_RATE * R_AIR))


def _pressure_to_alt(p):
    ratio = (p / P_SL) ** ((LAPSE_RATE * R_AIR) / G)
    return (T_SL - T_SL * ratio) / LAPSE_RATE


class PitotStatic:
    def __init__(self):
        cfg = SENSORS["pitot"]
        self.rate_hz = cfg["rate_hz"]
        self.dt = 1.0 / self.rate_hz
        self._timer = 0.0
        self._new_data = False
        self._ias_mps = 0.0
        self._tas_mps = 0.0
        # baro initializes at the spawn altitude so the first AP samples are sane
        self._baro_alt_msl_m = float(HOME["alt_msl_m"] - INITIAL_CONDITIONS["pos_ned_m"][2])
        self._vsi_mps = 0.0
        self._last_baro_alt = float(self._baro_alt_msl_m)
        self.health = "OK"

    def step(self, dt, V_air_true_mps, rho_true_kgm3, alt_msl_true_m):
        self._timer += dt
        self._new_data = False
        if self._timer >= self.dt:
            self._timer -= self.dt
            cfg = SENSORS["pitot"]
            q_true = 0.5 * rho_true_kgm3 * V_air_true_mps ** 2
            ias_true = math.sqrt(2.0 * q_true / 1.225)
            ias_meas = ias_true + np.random.randn() * cfg["ias_noise_std_mps"] + cfg["position_error_mps"]
            self._ias_mps = float(max(0.0, ias_meas))
            self._tas_mps = float(V_air_true_mps + np.random.randn() * cfg["ias_noise_std_mps"])

            p_true = _alt_to_pressure(alt_msl_true_m)
            sigma = max(cfg["baro_alt_noise_std_m"], 0.01)
            p_noisy = _alt_to_pressure(alt_msl_true_m + np.random.randn() * sigma)
            baro_alt = _pressure_to_alt(p_noisy)
            if self._last_baro_alt is None:
                vsi = 0.0
            else:
                vsi = (baro_alt - self._last_baro_alt) / self.dt
            self._vsi_mps = vsi + np.random.randn() * cfg["vsi_noise_std_mps"]
            self._last_baro_alt = baro_alt
            self._baro_alt_msl_m = float(baro_alt)
            self._new_data = True

    @property
    def new_data(self):
        return self._new_data

    def reading(self):
        return {
            "ias_mps": float(self._ias_mps),
            "tas_mps": float(self._tas_mps),
            "baro_alt_msl_m": float(self._baro_alt_msl_m),
            "vsi_mps": float(self._vsi_mps),
            "health": self.health,
            "new_data": bool(self._new_data),
        }
