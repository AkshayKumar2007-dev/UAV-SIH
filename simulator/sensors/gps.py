import math
import numpy as np
from simulator.config import SENSORS, HOME, R_AIR, T_SL, G


class GPS:
    def __init__(self):
        cfg = SENSORS["gps"]
        self.rate_hz = cfg["rate_hz"]
        self.dt = 1.0 / self.rate_hz
        self._timer = 0.0
        self._new_data = False
        self._dropout = False
        self._dropout_timer = 0.0
        self._next_dropout_t = 0.0
        self._last_valid = None
        self._hdop = cfg["hdop_nominal"]
        self.lat_deg = HOME["lat_deg"]
        self.lon_deg = HOME["lon_deg"]
        self.alt_msl_m = HOME["alt_msl_m"]
        self.vn_mps = 0.0
        self.ve_mps = 0.0
        self.vd_mps = 0.0
        self.groundspeed_mps = 0.0
        self.track_deg = 0.0
        self.satellites = 10
        self.health = "OK"

    def _ned_to_ll(self, n_m, e_m, d_m):
        R = 6378137.0
        lat = HOME["lat_deg"] + math.degrees(n_m / R)
        lon = HOME["lon_deg"] + math.degrees(e_m / (R * math.cos(math.radians(HOME["lat_deg"]))))
        alt = HOME["alt_msl_m"] - d_m
        return lat, lon, alt

    def step(self, dt, pos_ned, vel_body, euler):
        self._timer += dt
        self._new_data = False
        if self._dropout:
            self._dropout_timer -= dt
            if self._dropout_timer <= 0:
                self._dropout = False
                self.health = "OK"
            else:
                return
        else:
            p_drop_per_step = SENSORS["gps"]["dropout_prob_per_sec"] * dt
            if np.random.rand() < p_drop_per_step:
                cfg = SENSORS["gps"]
                self._dropout = True
                self._dropout_timer = float(np.random.uniform(cfg["min_dropout_s"], cfg["max_dropout_s"]))
                self.health = "NO_FIX"
                return

        if self._timer >= self.dt:
            self._timer -= self.dt
            n, e, d = pos_ned
            lat_true, lon_true, alt_true = self._ned_to_ll(n, e, d)
            cfg = SENSORS["gps"]
            self._hdop = float(np.clip(cfg["hdop_nominal"] + np.random.randn() * 0.25, 0.6, cfg["hdop_max"]))
            k = self._hdop / 1.2
            self.lat_deg = lat_true + np.random.randn() * (cfg["pos_sigma_m"] * k) / 111320.0
            self.lon_deg = lon_true + np.random.randn() * (cfg["pos_sigma_m"] * k) / (111320.0 * math.cos(math.radians(lat_true)))
            self.alt_msl_m = alt_true + np.random.randn() * cfg["alt_sigma_m"] * k

            from simulator.aircraft.rigid_body import rot_b_ned
            v_ned = rot_b_ned(*euler) @ vel_body
            self.vn_mps = v_ned[0] + np.random.randn() * cfg["vel_sigma_mps"]
            self.ve_mps = v_ned[1] + np.random.randn() * cfg["vel_sigma_mps"]
            self.vd_mps = v_ned[2] + np.random.randn() * cfg["vel_sigma_mps"]
            self.groundspeed_mps = math.sqrt(self.vn_mps ** 2 + self.ve_mps ** 2)
            self.track_deg = (math.degrees(math.atan2(self.ve_mps, self.vn_mps)) + 360.0) % 360.0
            self.satellites = int(max(4, 10 + int(np.round(np.random.randn() * 1.5))))
            self._new_data = True
            self.health = "OK"

    @property
    def new_data(self):
        return self._new_data

    @property
    def hdop(self):
        return self._hdop

    def reading(self):
        return {
            "lat_deg": float(self.lat_deg),
            "lon_deg": float(self.lon_deg),
            "alt_msl_m": float(self.alt_msl_m),
            "vn_mps": float(self.vn_mps),
            "ve_mps": float(self.ve_mps),
            "vd_mps": float(self.vd_mps),
            "groundspeed_mps": float(self.groundspeed_mps),
            "track_deg": float(self.track_deg),
            "hdop": float(self._hdop),
            "satellites": int(self.satellites),
            "health": self.health,
            "new_data": bool(self._new_data),
        }
