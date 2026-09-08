import math
import numpy as np
from simulator.config import HOME, WORLD


class WorldTerrain:
    def __init__(self):
        self.base_alt = WORLD["terrain_base_alt_msl_m"]
        self.radius = WORLD["terrain_radius_n_m"]
        self.bumps = WORLD["terrain_bumps_m"]
        self.mountains = WORLD.get("mountains", [])
        self.airport = WORLD.get("airport", {})
        self.lake = WORLD.get("lake", None)
        self.airstrips = WORLD.get("airstrips", [])
        self._flat_alt = None
        self._strip_flats = {}
        self._lake_level = None
        self._obstacles = None

    # ------------------------------------------------------------------ #
    def _natural_height(self, n_m, e_m):
        """Original rolling-terrain formula (before mountains/airport)."""
        r = math.sqrt(n_m * n_m + e_m * e_m) / max(self.radius, 1.0)
        bowl = self.bumps * 0.6 * max(r - 0.4, 0.0) ** 2
        h1 = self.bumps * 0.55 * math.sin(n_m * 0.0012 + 0.4) * math.cos(e_m * 0.0015 - 0.1)
        h2 = self.bumps * 0.35 * math.sin(n_m * 0.004 + 1.8) * math.cos(e_m * 0.003 + 0.7)
        h3 = self.bumps * 0.22 * math.sin((n_m + e_m) * 0.0028 + 2.3)
        return float(self.base_alt + bowl + h1 + h2 + h3)

    def altitude_msl_at(self, n_m, e_m):
        alt = self._natural_height(n_m, e_m)

        # mountains: gaussian peaks
        for m in self.mountains:
            d = math.hypot(n_m - m["n"], e_m - m["e"])
            if d < m["radius_m"] * 3.0:
                alt += m["h_m"] * math.exp(-2.2 * (d / m["radius_m"]) ** 2)

        # smooth flatten zones: airport plateau + outlying dirt strips
        zones = []
        fr = self.airport.get("flatten_radius_m", 0.0)
        if fr > 0.0:
            if self._flat_alt is None:
                self._flat_alt = self._natural_height(0.0, 0.0)
            zones.append((0.0, 0.0, fr, self.airport.get("flatten_inner_m", fr * 0.5),
                          self._flat_alt))
        for st in self.airstrips:
            key = (st["n"], st["e"])
            if key not in self._strip_flats:
                self._strip_flats[key] = self._natural_height(st["n"], st["e"])
            zones.append((st["n"], st["e"], 550.0, 250.0, self._strip_flats[key]))

        for (zn, ze, fr, fi, flat_alt) in zones:
            d = math.hypot(n_m - zn, e_m - ze)
            if d < fr:
                t = max(0.0, min(1.0, (fr - d) / max(fr - fi, 1.0)))
                t = t * t * (3.0 - 2.0 * t)
                alt = alt * (1.0 - t) + flat_alt * t

        # lake basin: depress the ground to the water level
        if self.lake:
            d = math.hypot(n_m - self.lake["n"], e_m - self.lake["e"])
            if d < self.lake["radius_m"] * 1.15:
                if self._lake_level is None:
                    self._lake_level = self._natural_height(self.lake["n"], self.lake["e"]) - 6.0
                t = max(0.0, min(1.0, (self.lake["radius_m"] * 1.15 - d) /
                                      (self.lake["radius_m"] * 0.35)))
                t = t * t * (3.0 - 2.0 * t)
                alt = alt * (1.0 - t) + self._lake_level * t
        return float(alt)

    def is_water(self, n_m, e_m):
        """Inside the lake surface circle (water level = lake bed + depth)."""
        if not self.lake:
            return False
        d = math.hypot(n_m - self.lake["n"], e_m - self.lake["e"])
        return d < self.lake["radius_m"] * 0.98

    # kept for compatibility with older call sites
    def agl_at(self, pos_ned):
        n, e, d = pos_ned
        alt_ac_above_msl = self.base_alt - d
        terrain_msl = self.altitude_msl_at(n, e)
        return float(max(alt_ac_above_msl - terrain_msl, 0.0))

    # ------------------------------------------------------------------ #
    @property
    def obstacles(self):
        """Collision list: trees, buildings, hangar and the ATC tower."""
        if self._obstacles is None:
            obs = []
            for t in WORLD.get("trees", []):
                obs.append({"n": t["n"], "e": t["e"], "r": t["r_m"] * 0.55,
                            "top": self.altitude_msl_at(t["n"], t["e"]) + t["h_m"],
                            "kind": "tree"})
            for b in WORLD.get("buildings", []):
                obs.append({"n": b["n"], "e": b["e"],
                            "r": 0.8 * math.hypot(b["w_m"], b["d_m"]) / 2.0,
                            "top": self.altitude_msl_at(b["n"], b["e"]) + b["h_m"],
                            "kind": "building"})
            ap = self.airport
            hg = ap.get("hangar")
            if hg:
                obs.append({"n": hg["n"], "e": hg["e"],
                            "r": 0.8 * math.hypot(hg["w_m"], hg["d_m"]) / 2.0,
                            "top": self.altitude_msl_at(hg["n"], hg["e"]) + hg["h_m"],
                            "kind": "hangar"})
            tw = ap.get("tower")
            if tw:
                obs.append({"n": tw["n"], "e": tw["e"], "r": tw["radius_m"],
                            "top": self.altitude_msl_at(tw["n"], tw["e"]) + tw["h_m"],
                            "kind": "control tower"})
            self._obstacles = obs
        return self._obstacles


class NoFlyZones:
    def __init__(self):
        self.zones = [dict(z) for z in WORLD["no_fly_zones"]]

    def violations(self, pos_ned, alt_msl_m):
        hits = []
        for z in self.zones:
            c_n, c_e = z["center_ned"]
            dz_n = pos_ned[0] - c_n
            dz_e = pos_ned[1] - c_e
            dist = math.sqrt(dz_n * dz_n + dz_e * dz_e)
            inside_xy = dist < z["radius_m"]
            inside_z = z["floor_m"] <= (HOME["alt_msl_m"] - pos_ned[2]) <= z["ceil_m"]
            if inside_xy and inside_z:
                hits.append({
                    "name": z["name"],
                    "distance_to_center_m": dist,
                    "radius_m": z["radius_m"],
                    "penetration_m": z["radius_m"] - dist,
                })
        return hits

    def nearest_boundary(self, pos_ned):
        nearest = None
        min_margin = float("inf")
        for z in self.zones:
            c_n, c_e = z["center_ned"]
            dz_n = pos_ned[0] - c_n
            dz_e = pos_ned[1] - c_e
            dist = math.sqrt(dz_n * dz_n + dz_e * dz_e)
            margin = dist - z["radius_m"]
            if margin < min_margin:
                min_margin = margin
                nearest = {"name": z["name"], "margin_m": float(margin), "radius_m": z["radius_m"]}
        return nearest


class WaypointMission:
    def __init__(self):
        self.items = [dict(wp) for wp in WORLD["default_waypoints"]]
        self.current_index = 0
        self.captured = [False] * len(self.items)
        self.home_ned = np.array([0.0, 0.0, 0.0])

    def count(self):
        return len(self.items)

    def current(self):
        if self.count() == 0:
            return None
        return self.items[self.current_index]

    def advance(self):
        if self.count() == 0:
            return
        self.captured[self.current_index] = True
        self.current_index = (self.current_index + 1) % self.count()

    def reset(self):
        self.current_index = 0
        self.captured = [False] * self.count()

    def add_waypoint(self, n_m, e_m, alt_msl_m):
        self.items.append({"n_m": float(n_m), "e_m": float(e_m), "alt_msl_m": float(alt_msl_m)})
        self.captured.append(False)

    def clear(self):
        self.items = []
        self.captured = []
        self.current_index = 0

    def distance_to_current(self, pos_ned):
        wp = self.current()
        if wp is None:
            return None
        dn = wp["n_m"] - pos_ned[0]
        de = wp["e_m"] - pos_ned[1]
        return math.sqrt(dn * dn + de * de)

    def bearing_to_current_deg(self, pos_ned):
        wp = self.current()
        if wp is None:
            return 0.0
        dn = wp["n_m"] - pos_ned[0]
        de = wp["e_m"] - pos_ned[1]
        return (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0
