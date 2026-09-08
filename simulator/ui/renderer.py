import math
import pygame
import numpy as np
from simulator.config import UI, AIRFRAME, PISTON_ENGINE, GUIDANCE, HOME, WORLD
from simulator.ui.chase_view import ChaseView


def _mps_to_kt(v):
    return v * 1.94384


def _m_to_ft(m):
    return m * 3.28084


def _c_to_f(c):
    return c * 9 / 5 + 32


def _wrap_360(d):
    return d % 360.0


class Renderer:
    def __init__(self):
        pygame.display.set_caption("Piston UAV Flight Simulator v1.0")
        self.screen = pygame.display.set_mode((UI["screen_w_px"], UI["screen_h_px"]))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas", UI["panel_font_size"], bold=False)
        self.font_med = pygame.font.SysFont("consolas", UI["hud_font_size"], bold=True)
        self.font_big = pygame.font.SysFont("consolas", UI["hud_font_size"] + 8, bold=True)
        self.map_zoom = UI["map_px_per_m"]
        self._terrain_cache = None
        self._advisory_log = []
        self.chase = ChaseView()
        self.show_map = True
        self.show_hud = True

    def log_advisory(self, sev, text, t_s):
        self._advisory_log.append((t_s, sev, text))
        if len(self._advisory_log) > 6:
            self._advisory_log.pop(0)

    def _ned_to_screen(self, n, e, center_n, center_e, cx, cy, scale):
        sx = cx + (e - center_e) * scale
        sy = cy - (n - center_n) * scale
        return int(sx), int(sy)

    def draw(self, sim_state):
        scr = self.screen
        W, H = UI["screen_w_px"], UI["screen_h_px"]
        try:
            self.chase.draw(scr, sim_state)
        except Exception as ex:
            scr.fill(UI["sky_color"])
            self._draw_background_ground(scr, W, H, sim_state)
            err = self.font_small.render(f"chase view off ({type(ex).__name__})", True, (255, 80, 80))
            scr.blit(err, (W // 2 - 80, 4))

        if self.show_map:
            self._draw_map(scr, sim_state)
        if self.show_hud:
            self._draw_attitude(scr, sim_state)
            self._draw_speed_tape(scr, sim_state)
            self._draw_alt_tape(scr, sim_state)
            self._draw_vsi_tape(scr, sim_state)
            self._draw_hsi_heading(scr, sim_state)
        self._draw_engine_panel(scr, sim_state)
        self._draw_mode_and_controls(scr, sim_state)
        self._draw_advisories(scr, sim_state)
        self._draw_crash_banner(scr, sim_state)
        self._draw_fps(scr, sim_state)

        pygame.display.flip()

    def _draw_crash_banner(self, scr, s):
        cr = s.get("crash") or {}
        if not cr.get("crashed"):
            return
        W, H = UI["screen_w_px"], UI["screen_h_px"]
        banner = pygame.Surface((W, 150), pygame.SRCALPHA)
        banner.fill((120, 8, 8, 200))
        scr.blit(banner, (0, H // 2 - 75))
        pygame.draw.line(scr, UI["crit_color"], (0, H // 2 - 75), (W, H // 2 - 75), 3)
        pygame.draw.line(scr, UI["crit_color"], (0, H // 2 + 75), (W, H // 2 + 75), 3)
        t1 = self.font_big.render(f"CRASHED — {cr.get('reason', 'terrain impact')}", True, (255, 255, 255))
        t2 = self.font_med.render("Press R to reset the simulator", True, (255, 220, 120))
        scr.blit(t1, (W // 2 - t1.get_width() // 2, H // 2 - 45))
        scr.blit(t2, (W // 2 - t2.get_width() // 2, H // 2 + 10))

    def _draw_background_ground(self, scr, W, H, s):
        try:
            pitch = s["state"]["euler"][1]
            roll = s["state"]["euler"][0]
        except Exception:
            pitch = 0.0
            roll = 0.0
        horizon = int(H // 2 - math.degrees(pitch) * 6.0)
        pygame.draw.rect(scr, (80, 120, 72), (0, horizon, W, H - horizon))
        pygame.draw.line(scr, (40, 70, 40), (0, horizon), (W, horizon), 2)

    def _draw_map(self, scr, s):
        W, H = UI["screen_w_px"], UI["screen_h_px"]
        mw = W // 2 - 20
        mh = H - 120
        x0 = 10
        y0 = 10
        pygame.draw.rect(scr, (230, 225, 200), (x0, y0, mw, mh))
        pygame.draw.rect(scr, (0, 0, 0), (x0, y0, mw, mh), 2)

        pos = s["state"]["pos_ned"]
        psi = s["state"]["euler"][2]
        cx = x0 + mw // 2
        cy = y0 + mh // 2
        scale = self.map_zoom

        terr = s["world_terrain"]
        nfz = s.get("no_fly_zones", [])
        for z in WORLD["no_fly_zones"]:
            sx, sy = self._ned_to_screen(z["center_ned"][0], z["center_ned"][1], pos[0], pos[1], cx, cy, scale)
            r = int(z["radius_m"] * scale)
            if 0 <= sx < W and 0 <= sy < H:
                red = (220, 60, 60, 60)
                sr = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
                pygame.draw.circle(sr, (220, 60, 60, 70), (r, r), r, 0)
                pygame.draw.circle(sr, (180, 30, 30, 200), (r, r), r, 2)
                scr.blit(sr, (sx - r, sy - r))
                t = self.font_small.render(z["name"], True, (140, 10, 10))
                scr.blit(t, (sx - t.get_width() // 2, sy - 8))

        # waypoints
        mission = s.get("mission_items", [])
        mission_captured = s.get("mission_captured", [])
        if len(mission) > 1:
            pts = []
            for i, wp in enumerate(mission):
                sx, sy = self._ned_to_screen(wp["n_m"], wp["e_m"], pos[0], pos[1], cx, cy, scale)
                pts.append((sx, sy))
            # close loop
            pygame.draw.lines(scr, (100, 100, 100), False, pts, 1)

        for i, wp in enumerate(mission):
            sx, sy = self._ned_to_screen(wp["n_m"], wp["e_m"], pos[0], pos[1], cx, cy, scale)
            cap = mission_captured[i] if i < len(mission_captured) else False
            col = (30, 140, 30) if cap else (40, 40, 180)
            if i == s.get("mission_current_idx", 0) and len(mission) > 0:
                pygame.draw.circle(scr, (240, 200, 40), (sx, sy), 10, 2)
            pygame.draw.rect(scr, col, (sx - 4, sy - 4, 8, 8))
            num = self.font_small.render(str(i + 1), True, (0, 0, 0))
            scr.blit(num, (sx + 6, sy - 8))

        # home
        hsx, hsy = self._ned_to_screen(0, 0, pos[0], pos[1], cx, cy, scale)
        pygame.draw.polygon(scr, (120, 40, 40), [(hsx, hsy - 10), (hsx - 8, hsy + 7), (hsx + 8, hsy + 7)])
        ht = self.font_small.render("H", True, (255, 255, 255))
        scr.blit(ht, (hsx - ht.get_width() // 2, hsy - 8))

        # aircraft
        ac_len = 24
        c = math.cos(psi)
        sn = math.sin(psi)
        tip_x = cx + ac_len * c
        tip_y = cy - ac_len * sn
        left_x = cx + (-0.6 * c - 0.5 * sn) * ac_len
        left_y = cy - (-0.6 * sn - 0.5 * c) * ac_len
        right_x = cx + (-0.6 * c + 0.5 * sn) * ac_len
        right_y = cy - (-0.6 * sn + 0.5 * c) * ac_len
        tail_x = cx + (-0.9 * c) * ac_len
        tail_y = cy - (-0.9 * sn) * ac_len
        pygame.draw.polygon(
            scr, (10, 10, 10),
            [(int(tip_x), int(tip_y)), (int(left_x), int(left_y)),
             (int(tail_x), int(tail_y)), (int(right_x), int(right_y))],
        )

        # scale bar
        bar_w_m = 1000
        bar_w_px = int(bar_w_m * scale)
        pygame.draw.rect(scr, (0, 0, 0), (x0 + 14, y0 + mh - 24, bar_w_px, 6))
        for tk in range(5):
            tx = x0 + 14 + tk * bar_w_px // 4
            pygame.draw.line(scr, (0, 0, 0), (tx, y0 + mh - 28), (tx, y0 + mh - 18), 2)
        lbl = self.font_small.render(f"{bar_w_m} m", True, (0, 0, 0))
        scr.blit(lbl, (x0 + 14 + bar_w_px + 6, y0 + mh - 32))

        title = self.font_med.render("MAP (NED) - click to add waypoint", True, (0, 0, 0))
        scr.blit(title, (x0 + 6, y0 + 2))

    def _draw_attitude(self, scr, s):
        W = UI["screen_w_px"]
        H = UI["screen_h_px"]
        size = 220
        x0 = W - size - 30
        y0 = 160
        cx = x0 + size // 2
        cy = y0 + size // 2

        eul = s["state"]["euler"]
        phi_deg = math.degrees(eul[0])
        theta_deg = math.degrees(eul[1])
        half = size // 2

        sr = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(sr, (30, 60, 120, 255), (half, half), half - 2)
        pygame.draw.circle(sr, (0, 0, 0, 0), (half, half), half - 2, 0)

        roll = math.radians(-phi_deg)
        cr, sr_ = math.cos(roll), math.sin(roll)
        for pitch_tick in range(-60, 61, 10):
            y_off = (pitch_tick - theta_deg) * 4
            y1 = half + y_off
            if y1 < 0 or y1 > size:
                continue
            len1 = 60 if pitch_tick % 20 == 0 else 30
            x1a, y1a = half - len1, y1
            x2a, y2a = half + len1, y1
            p1 = (cr * (x1a - half) - sr_ * (y1a - half) + half,
                  sr_ * (x1a - half) + cr * (y1a - half) + half)
            p2 = (cr * (x2a - half) - sr_ * (y2a - half) + half,
                  sr_ * (x2a - half) + cr * (y2a - half) + half)
            if pitch_tick % 20 == 0 and pitch_tick != 0:
                col = (255, 255, 220)
                t = self.font_small.render(f"{pitch_tick:+d}", True, col)
                tx = cr * (half + len1 + 4 - half) - sr_ * (y1 - half) + half
                ty = sr_ * (half + len1 + 4 - half) + cr * (y1 - half) + half
                pygame.draw.line(sr, col, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 2)
            else:
                col = (200, 200, 200)
                pygame.draw.line(sr, col, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)

        pygame.draw.polygon(sr, (110, 160, 70, 255), [
            (half - size, half + 1), (half + size, half + 1),
            (half + size, size + 10), (half - size, size + 10)
        ])

        pygame.draw.circle(scr, (0, 0, 0), (cx, cy), half, 4)

        # roll pointer
        for tick in range(-60, 61, 10):
            ang = math.radians(-tick + 90)
            r_out = half - 6
            x1 = cx + r_out * math.cos(ang)
            y1 = cy + r_out * math.sin(ang)
            r_in = half - (18 if tick % 30 == 0 else 12)
            x2 = cx + r_in * math.cos(ang)
            y2 = cy + r_in * math.sin(ang)
            col = UI["accent_color"] if tick == 0 else (255, 255, 255)
            pygame.draw.line(scr, col, (int(x1), int(y1)), (int(x2), int(y2)), 2 if tick % 30 == 0 else 1)

        top_ang = math.radians(-(-phi_deg) + 90)
        tp1 = (cx + (half - 20) * math.cos(top_ang), cy + (half - 20) * math.sin(top_ang))
        tp2 = (cx + (half - 2) * math.cos(top_ang), cy + (half - 2) * math.sin(top_ang))
        pygame.draw.line(scr, UI["warn_color"], (int(tp1[0]), int(tp1[1])), (int(tp2[0]), int(tp2[1])), 4)

        # wings
        wl = 30
        pygame.draw.polygon(scr, UI["accent_color"], [
            (cx - 40, cy), (cx - 8, cy), (cx - 14, cy - 4),
        ])
        pygame.draw.polygon(scr, UI["accent_color"], [
            (cx + 40, cy), (cx + 8, cy), (cx + 14, cy - 4),
        ])
        pygame.draw.rect(scr, UI["accent_color"], (cx - 3, cy - 2, 6, 4))

        # fixed rect on top to clip rotated
        cover = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(cover, (0, 0, 0, 0), (0, 0, size, size))
        mask = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (half, half), half - 4)
        sr.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        scr.blit(sr, (x0, y0))

        lbl = self.font_med.render("ATTITUDE", True, (0, 0, 0))
        scr.blit(lbl, (x0 + 6, y0 - 20))
        p_t = self.font_small.render(f"θ={theta_deg:+6.1f}°  φ={phi_deg:+6.1f}°", True, (0, 0, 0))
        scr.blit(p_t, (x0 + 6, y0 + size + 4))

    def _draw_speed_tape(self, scr, s):
        ias = s.get("pitot", {}).get("ias_mps", 0.0)
        ias_kt = _mps_to_kt(ias)
        x0 = W = UI["screen_w_px"] // 2 + 20
        W_scr, H_scr = UI["screen_w_px"], UI["screen_h_px"]
        x0 = (W_scr // 2) + 30
        y0 = 160
        h = 240
        w = 70
        pygame.draw.rect(scr, (220, 220, 220), (x0, y0, w, h))
        pygame.draw.rect(scr, (0, 0, 0), (x0, y0, w, h), 2)
        center_y = y0 + h // 2
        for kt in range(0, 220, 10):
            delta_kt = kt - ias_kt
            y = center_y + delta_kt * 2.0
            if y0 <= y <= y0 + h:
                long_l = kt % 20 == 0
                le = w - (30 if long_l else 18)
                pygame.draw.line(scr, (0, 0, 0), (le, int(y)), (x0 + w - 2, int(y)), 2 if long_l else 1)
                if long_l:
                    t = self.font_small.render(f"{kt}", True, (0, 0, 0))
                    scr.blit(t, (x0 + 4, int(y) - 8))
        pygame.draw.polygon(scr, (220, 40, 40), [
            (x0 + w + 1, center_y - 6), (x0 + w - 18, center_y), (x0 + w + 1, center_y + 6),
        ])
        t = self.font_big.render(f"{ias_kt:5.0f}", True, (220, 40, 40))
        box = pygame.Surface((t.get_width() + 10, t.get_height() + 4), pygame.SRCALPHA)
        pygame.draw.rect(box, (255, 255, 255, 200), (0, 0, box.get_width(), box.get_height()), border_radius=4)
        scr.blit(box, (x0 + w - t.get_width() - 18, center_y - t.get_height() // 2 - 2))
        scr.blit(t, (x0 + w - t.get_width() - 13, center_y - t.get_height() // 2))
        tt = self.font_small.render("IAS kt", True, (0, 0, 0))
        scr.blit(tt, (x0 + 6, y0 + h + 4))

    def _draw_alt_tape(self, scr, s):
        W_scr = UI["screen_w_px"]
        x0 = W_scr - 120
        y0 = 160
        h = 240
        w = 80
        alt = s.get("pitot", {}).get("baro_alt_msl_m", 0.0)
        alt_ft = _m_to_ft(alt)
        pygame.draw.rect(scr, (220, 220, 220), (x0, y0, w, h))
        pygame.draw.rect(scr, (0, 0, 0), (x0, y0, w, h), 2)
        center_y = y0 + h // 2
        for ft in range(0, 15001, 100):
            delta = ft - alt_ft
            y = center_y - delta * 0.2
            if y0 <= y <= y0 + h:
                long_l = ft % 500 == 0
                le = x0 + (28 if long_l else 42)
                pygame.draw.line(scr, (0, 0, 0), (le, int(y)), (x0 + w - 2, int(y)), 2 if long_l else 1)
                if long_l:
                    t = self.font_small.render(f"{int(ft // 100)}", True, (0, 0, 0))
                    scr.blit(t, (x0 + 4, int(y) - 8))
        pygame.draw.polygon(scr, (40, 40, 200), [
            (x0 - 1, center_y - 6), (x0 + 18, center_y), (x0 - 1, center_y + 6),
        ])
        t = self.font_big.render(f"{alt_ft:5.0f}", True, (40, 40, 200))
        box = pygame.Surface((t.get_width() + 10, t.get_height() + 4), pygame.SRCALPHA)
        pygame.draw.rect(box, (255, 255, 255, 200), (0, 0, box.get_width(), box.get_height()), border_radius=4)
        scr.blit(box, (x0 + 2, center_y - t.get_height() // 2 - 2))
        scr.blit(t, (x0 + 7, center_y - t.get_height() // 2))
        tt = self.font_small.render("ALT ft", True, (0, 0, 0))
        scr.blit(tt, (x0 + 6, y0 + h + 4))

    def _draw_vsi_tape(self, scr, s):
        W_scr = UI["screen_w_px"]
        x0 = W_scr - 40
        y0 = 160
        h = 240
        w = 30
        vsi_mps = s.get("pitot", {}).get("vsi_mps", 0.0)
        vsi_fpm = vsi_mps * 196.85
        pygame.draw.rect(scr, (230, 230, 230), (x0, y0, w, h))
        pygame.draw.rect(scr, (0, 0, 0), (x0, y0, w, h), 2)
        cy = y0 + h // 2
        for v in range(-2000, 2001, 500):
            y = cy - v * 0.04
            if y0 < y < y0 + h:
                pygame.draw.line(scr, (0, 0, 0), (x0 + 4, int(y)), (x0 + w - 4, int(y)), 1)
                if v != 0:
                    t = self.font_small.render(f"{v//100:+d}", True, (0, 0, 0))
                    scr.blit(t, (x0 - 28, int(y) - 8))
        bar_h = int(np.clip(abs(vsi_fpm) * 0.04, 0, h // 2 - 10))
        if vsi_fpm >= 0:
            pygame.draw.rect(scr, (50, 180, 70), (x0 + 6, cy - bar_h, w - 12, bar_h))
        else:
            pygame.draw.rect(scr, (200, 60, 60), (x0 + 6, cy, w - 12, bar_h))
        pygame.draw.line(scr, (0, 0, 0), (x0, cy), (x0 + w, cy), 1)
        tt = self.font_small.render("VS fpm/100", True, (0, 0, 0))
        scr.blit(tt, (x0 - 28, y0 + h + 4))

    def _draw_hsi_heading(self, scr, s):
        W_scr = UI["screen_w_px"]
        cx = W_scr - 240
        cy = 480
        R = 70
        psi = math.degrees(s["state"]["euler"][2])
        psi = _wrap_360(psi)
        pygame.draw.circle(scr, (240, 240, 240), (cx, cy), R, 0)
        pygame.draw.circle(scr, (0, 0, 0), (cx, cy), R, 2)
        for deg in range(0, 360, 10):
            ang = math.radians(deg - psi + 90)
            r1 = R - 4
            r2 = R - (18 if deg % 30 == 0 else 10)
            x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
            x2, y2 = cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)
            pygame.draw.line(scr, (0, 0, 0), (int(x1), int(y1)), (int(x2), int(y2)), 2 if deg % 30 == 0 else 1)
            if deg % 90 == 0:
                lbl = {0: "N", 90: "E", 180: "S", 270: "W"}[deg]
                tx, ty = cx + (R - 30) * math.cos(ang), cy + (R - 30) * math.sin(ang)
                t = self.font_med.render(lbl, True, (200, 40, 40) if lbl == "N" else (0, 0, 0))
                scr.blit(t, (int(tx) - t.get_width() // 2, int(ty) - t.get_height() // 2))
            elif deg % 30 == 0:
                tx, ty = cx + (R - 30) * math.cos(ang), cy + (R - 30) * math.sin(ang)
                t = self.font_small.render(f"{deg//10}", True, (0, 0, 0))
                scr.blit(t, (int(tx) - t.get_width() // 2, int(ty) - t.get_height() // 2))
        heading = s.get("ap_debug", {}).get("hdg_target_deg", psi)
        hdg_err = _wrap_360(heading - psi)
        ang_h = math.radians(-hdg_err + 90)
        xh = cx + (R - 10) * math.cos(ang_h)
        yh = cy + (R - 10) * math.sin(ang_h)
        pygame.draw.circle(scr, UI["accent_color"], (int(xh), int(yh)), 5)
        # lubber line
        pygame.draw.polygon(scr, (220, 40, 40), [
            (cx - 6, cy - R + 4), (cx + 6, cy - R + 4), (cx, cy - R + 16),
        ])
        t = self.font_big.render(f"{psi:03.0f}°", True, (0, 0, 0))
        scr.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))
        lbl = self.font_small.render("HDG / HSI", True, (0, 0, 0))
        scr.blit(lbl, (cx - 40, cy + R + 6))

    def _draw_engine_panel(self, scr, s):
        W_scr = UI["screen_w_px"]
        H_scr = UI["screen_h_px"]
        x0 = W_scr // 2 + 30
        y0 = 580
        w = 460
        h = H_scr - y0 - 10
        pygame.draw.rect(scr, (230, 230, 230), (x0, y0, w, h))
        pygame.draw.rect(scr, (0, 0, 0), (x0, y0, w, h), 2)
        eng = s.get("engine", {})
        es = s.get("engine_sensors", eng)
        r, c = 0, 0
        ff_max = PISTON_ENGINE["bsfc_kg_per_kwh"] * PISTON_ENGINE["max_power_kw"] /             PISTON_ENGINE["fuel_density_kg_per_L"] * 1.3
        fields = [
            ("RPM", f"{es.get('rpm',0):.0f}", 0, PISTON_ENGINE["max_rpm"]),
            ("PWR kW", f"{eng.get('power_kw',0):.1f}", 0, PISTON_ENGINE["max_power_kw"]),
            ("FUEL L", f"{es.get('fuel_qty_L',0):.1f}", 0, PISTON_ENGINE["fuel_full_L"]),
            ("FF L/h", f"{es.get('fuel_flow_Lph',0):.1f}", 0, ff_max),
            ("CHT °C", f"{es.get('cht_C',0):.0f}", 40, PISTON_ENGINE["cht_max_c"]),
            ("EGT °C", f"{es.get('egt_C',0):.0f}", 100, PISTON_ENGINE["egt_max_c"]),
            ("OIL psi", f"{es.get('oil_psi',0):.1f}", 0, 90.0),
            ("J (adv)", f"{eng.get('advance_ratio',0):.2f}", 0, 1.8),
        ]
        col_w = w // 2 - 16
        row_h = 30
        for i, (name, val, mn, mx) in enumerate(fields):
            col = i % 2
            r_idx = i // 2
            bx = x0 + 10 + col * (col_w + 8)
            by = y0 + 24 + r_idx * row_h
            nm = self.font_small.render(f"{name}", True, (0, 0, 0))
            scr.blit(nm, (bx, by))
            vv = self.font_med.render(f"{val}", True, (0, 0, 0))
            scr.blit(vv, (bx + col_w - vv.get_width() - 24, by - 2))
            bar_x = bx
            bar_y = by + 20
            bar_wid = col_w - 4
            pygame.draw.rect(scr, (255, 255, 255), (bar_x, bar_y, bar_wid, 5))
            try:
                val_num = float(val.split()[0])
            except Exception:
                val_num = 0.0
            frac = np.clip((val_num - mn) / max(mx - mn, 1e-6), 0, 1)
            pct = frac
            if "CHT" in name and pct > 0.85 or "EGT" in name and pct > 0.85 or \
               "OIL" in name and pct < 0.3 and eng.get("running", True):
                col_bar = UI["crit_color"]
            elif pct > 0.75 or ("FUEL L" in name and pct < 0.2):
                col_bar = UI["warn_color"]
            else:
                col_bar = UI["ok_color"]
            pygame.draw.rect(scr, col_bar, (bar_x, bar_y, int(bar_wid * frac), 5), 0)
        title = self.font_med.render("ENGINE / SYSTEMS", True, (0, 0, 0))
        scr.blit(title, (x0 + 8, y0 + 2))
        flags = eng.get("health_flags", [])
        if flags:
            ft = self.font_small.render("FLAGS: " + ", ".join(flags), True, UI["crit_color"])
            scr.blit(ft, (x0 + 10, y0 + h - 22))

    def _draw_mode_and_controls(self, scr, s):
        W_scr = UI["screen_w_px"]
        x0 = 10
        y0 = UI["screen_h_px"] - 120
        w = W_scr // 2 - 20
        h = 110
        pygame.draw.rect(scr, (240, 240, 230), (x0, y0, w, h))
        pygame.draw.rect(scr, (0, 0, 0), (x0, y0, w, h), 2)
        title = self.font_med.render("MODE / CONTROLS", True, (0, 0, 0))
        scr.blit(title, (x0 + 8, y0 + 2))
        mode = s.get("ap_debug", {}).get("mode", "MANUAL")
        t = self.font_big.render(mode, True, UI["crit_color"] if mode == "MANUAL" else UI["ok_color"])
        scr.blit(t, (x0 + 12, y0 + 26))
        targets = s.get("ap_debug", {})
        info = [
            f"TGT HDG {targets.get('hdg_target_deg',0):.0f}°",
            f"TGT ALT {_m_to_ft(targets.get('alt_target_msl_m',0)):.0f} ft",
            f"TGT IAS {_mps_to_kt(targets.get('ias_target_mps',0)):.0f} kt",
        ]
        for i, ln in enumerate(info):
            st = self.font_small.render(ln, True, (0, 0, 0))
            scr.blit(st, (x0 + 150 + i * 100, y0 + 32))
        man = s.get("manual_controls", {})
        lines = [
            "A/D=roll  W/S=pitch  Q/E=rudder  ↑↓=throt  F=flaps  G=gear  SPACE/B=brakes",
            "1..5=MAN/ALT/HDG/WP/RTH  Z=STABLE AP (flies WPs)  H=carb heat  J=starter  L=next WP  C=clear WPs  P=pause",
            "M=map on/off  V=HUD/attitude on/off  R=reset after crash",
            f"Trim {man.get('delta_e_trim_deg',0):+.1f}°  Flap {man.get('delta_flap_deg',0):.0f}°  Gear={'DOWN' if man.get('gear_down') else 'UP'}  CarbHT={'ON' if man.get('carb_heat') else 'OFF'}",
            f"WP idx {s.get('mission_current_idx',0)+1}/{s.get('mission_count',0)}  GPS {s.get('gps',{}).get('health','?')} HDOP={s.get('gps',{}).get('hdop',0):.1f} SATs={s.get('gps',{}).get('satellites',0)}",
        ]
        for i, ln in enumerate(lines):
            lt = self.font_small.render(ln, True, (0, 0, 0))
            scr.blit(lt, (x0 + 10, y0 + 60 + i * 14))

    def _draw_advisories(self, scr, s):
        x0 = 10
        y0 = UI["screen_h_px"] - 140
        t_s = s.get("t_s", 0)
        for i, (at, sev, txt) in enumerate(reversed(self._advisory_log)):
            age = t_s - at
            if age > 15:
                continue
            alpha = np.clip(1.0 - age / 15, 0.2, 1.0)
            col_map = {"INFO": UI["ok_color"], "WARN": UI["warn_color"], "CRIT": UI["crit_color"], "RESOLVED": UI["ok_color"]}
            col = col_map.get(sev, (0, 0, 0))
            lt = self.font_small.render(f"[{sev}] {txt}", True, col)
            sfc = pygame.Surface((lt.get_width() + 6, lt.get_height()), pygame.SRCALPHA)
            sfc.fill((255, 255, 255, int(alpha * 220)))
            scr.blit(sfc, (x0 + 2, y0 - (i + 1) * 16 - 2))
            scr.blit(lt, (x0 + 5, y0 - (i + 1) * 16))

    def _draw_fps(self, scr, s):
        t_s = s.get("t_s", 0)
        fps = s.get("fps", 0.0)
        agl = s.get("agl_m", 0.0)
        t = self.font_small.render(f"FPS {fps:.0f} | T+{t_s:.0f}s | AGL {agl:.0f} m", True, (0, 0, 0))
        scr.blit(t, (8, 2))
