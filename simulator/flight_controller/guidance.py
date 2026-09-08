import math
import numpy as np
from simulator.config import GUIDANCE, G


def _wrap_deg(d):
    d = d % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class L1Guidance:
    def __init__(self):
        self.tau = GUIDANCE["L1_period_s"]
        self.xi = GUIDANCE["L1_damping"]
        self.capture = GUIDANCE["waypoint_capture_radius_m"]
        self.max_bank_rad = math.radians(GUIDANCE["max_bank_deg_aps"])

    def L1_distance(self, groundspeed_mps):
        omega = 2.0 * math.pi / self.tau
        return max(80.0, groundspeed_mps / omega * 2.0)

    def track_waypoint(self, pos_ned, vg_ned, wp_from, wp_to):
        pA = np.array([wp_from["n_m"], wp_from["e_m"]])
        pB = np.array([wp_to["n_m"], wp_to["e_m"]])
        pP = np.array([pos_ned[0], pos_ned[1]])
        vg = np.array([vg_ned[0], vg_ned[1]])
        vg_mag = float(np.linalg.norm(vg))
        L1 = self.L1_distance(vg_mag)

        AB = pB - pA
        AB_len = float(np.linalg.norm(AB))
        if AB_len < 1e-3:
            return self._direct_to(pos_ned, vg, wp_to)
        ABu = AB / AB_len
        AP = pP - pA
        along = float(np.dot(AP, ABu))
        if along >= AB_len:
            # past the waypoint: the on-leg L1 reference point would sit even
            # further along the extended line and steer away from B forever —
            # fly direct to the waypoint so the capture ring can trigger
            return self._direct_to(pos_ned, vg, wp_to)
        cross = float(AP[1] * ABu[0] - AP[0] * ABu[1])
        cross_track = cross
        xtrack_v = -cross_track
        if xtrack_v > L1:
            xtrack_v = L1
        elif xtrack_v < -L1:
            xtrack_v = -L1

        ref_point = pA + ABu * max(along + math.sqrt(max(L1 ** 2 - xtrack_v ** 2, 0.0)), 0.0)
        dv = ref_point - pP
        dist_ref = float(np.linalg.norm(dv))
        if dist_ref < 1.0:
            heading_target = math.degrees(math.atan2(ABu[1], ABu[0]))
        else:
            heading_target = math.degrees(math.atan2(dv[1], dv[0]))
        dist_to_B = float(np.linalg.norm(pB - pP))
        captured = dist_to_B < self.capture
        bank = self._compute_bank(pos_ned, vg, heading_target, vg_mag)
        return bank, float(heading_target), float(cross_track), captured, dist_to_B, along / max(AB_len, 1e-3)

    def direct_to(self, pos_ned, vg_ned, wp):
        vg = np.array([vg_ned[0], vg_ned[1]])
        return self._direct_to(pos_ned, vg, wp)

    def _direct_to(self, pos_ned, vg2, wp):
        pP = np.array([pos_ned[0], pos_ned[1]])
        pB = np.array([wp["n_m"], wp["e_m"]])
        dv = pB - pP
        dist = float(np.linalg.norm(dv))
        heading_target = math.degrees(math.atan2(dv[1], dv[0])) if dist > 1.0 else 0.0
        vg_mag = float(np.linalg.norm(vg2))
        captured = dist < self.capture
        bank = self._compute_bank(pos_ned, vg2, heading_target, vg_mag)
        return bank, float(heading_target), 0.0, captured, dist, 0.0

    def loiter(self, pos_ned, vg_ned, center_ned, radius_m, direction=1):
        pP = np.array([pos_ned[0], pos_ned[1]])
        pC = np.array([center_ned[0], center_ned[1]])
        vg = np.array([vg_ned[0], vg_ned[1]])
        vg_mag = max(float(np.linalg.norm(vg)), 1.0)
        to_center = pC - pP
        d = float(np.linalg.norm(to_center))
        to_center_u = to_center / d if d > 1e-3 else np.array([1.0, 0.0])
        if direction >= 0:
            perp = np.array([-to_center_u[1], to_center_u[0]])
        else:
            perp = np.array([to_center_u[1], -to_center_u[0]])
        L1 = self.L1_distance(vg_mag)
        on_circle_point = pC + radius_m * perp
        ref = on_circle_point - L1 * perp
        dv = ref - pP
        heading_target = math.degrees(math.atan2(dv[1], dv[0]))
        bank = self._compute_bank(pos_ned, vg, heading_target, vg_mag)
        dist_to_radius = abs(d - radius_m)
        captured = dist_to_radius < 30.0
        return bank, float(heading_target), float(d - radius_m), captured, d, dist_to_radius

    def _compute_bank(self, pos_ned, vg, heading_target_deg, vg_mag):
        if vg_mag < 1.0:
            return 0.0
        psi = math.degrees(math.atan2(vg[1], vg[0]))
        err = _wrap_deg(heading_target_deg - psi)
        err_rad = math.radians(err)
        L1 = self.L1_distance(vg_mag)
        a_cmd = 2.0 * vg_mag * vg_mag / L1 * math.sin(err_rad)
        g_lim = G * 3.0
        a_cmd = np.clip(a_cmd, -g_lim, g_lim)
        phi_rad = math.atan2(a_cmd, G)
        phi_rad = np.clip(phi_rad, -self.max_bank_rad, self.max_bank_rad)
        return float(math.degrees(phi_rad))
