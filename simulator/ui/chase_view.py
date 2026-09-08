"""Software-rendered 3D chase view for the pygame window.

Renders the UAV flying through the environment: shaded terrain from the same
analytic function as the physics world, sky/sun, waypoint pillars, no-fly
cylinders and an animated aircraft model with prop disc. Pure numpy +
pygame drawing (no OpenGL), rendered at half resolution and smooth-scaled.
"""

import math
import random
import numpy as np
import pygame

from simulator.config import HOME, WORLD, AERO

BASE_ALT = HOME["alt_msl_m"]
LIGHT_DIR = np.array([-0.45, 0.82, 0.36])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)

MSCALE = 2.5


def _box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box. z is given "up-positive" and negated here because the
    body frame z axis points DOWN (aerospace convention). Face winding is
    reversed so outward normals (used for shading) stay outward."""
    n0, n1 = -z1, -z0
    v = [(x0, y0, n0), (x1, y0, n0), (x1, y1, n0), (x0, y1, n0),
         (x0, y0, n1), (x1, y0, n1), (x1, y1, n1), (x0, y1, n1)]
    f = [(3, 2, 1, 0), (7, 6, 5, 4), (4, 5, 1, 0), (5, 6, 2, 1), (6, 7, 3, 2), (7, 4, 0, 3)]
    return v, f


class ChaseView:
    HALF_WORLD = 7000.0     # terrain mesh extent (m)
    NG = 60                 # grid cells per side
    FOG_DIST = 3800.0

    def __init__(self):
        from simulator.environment.world import WorldTerrain
        self.terrain = WorldTerrain()
        self._build_terrain()
        self._build_model()
        self._build_scene()
        self._cam = None          # smoothed camera position (E,Up,N)
        self._last_t = None
        self._half = None
        self._sky_h = -1
        self._sky_surf = None
        self._haze_surf = None
        self._frame_ms = 0.0

    # --------------------------- static geometry --------------------------- #
    def _build_terrain(self):
        ng, half = self.NG, self.HALF_WORLD
        step = 2.0 * half / (ng - 1)
        verts = np.zeros((ng * ng, 3))
        k = 0
        for i in range(ng):
            n = -half + i * step
            for j in range(ng):
                e = -half + j * step
                verts[k] = (e, self.terrain.altitude_msl_at(n, e) - BASE_ALT, n)
                k += 1
        self.t_verts = verts
        idx = np.arange(ng * ng).reshape(ng, ng)
        a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
        c = idx[1:, 1:].ravel();  d = idx[1:, :-1].ravel()
        self.t_quads = np.stack([a, b, c, d], axis=1)
        _rng = random.Random(4242)
        va = verts[a]; vb = verts[b]; vd = verts[d]; vc = verts[c]
        u = vb - va; w = vd - va
        nrm = np.cross(w, u)          # (North × East) → Up-pointing normals
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        self.t_nrm = nrm
        self.t_centers = (va + vc) / 2.0
        ndl = np.maximum(0.0, nrm @ LIGHT_DIR)
        h = (va[:, 1] + vc[:, 1]) / 2.0
        t = np.clip((h + 28.0) / 75.0, 0.0, 1.0)
        shade = 0.62 + 0.5 * ndl
        lo = np.array([58, 92, 52]); hi = np.array([150, 140, 92])
        cols = (lo + t[:, None] * (hi - lo)) * shade[:, None]
        # farmland patchwork: subtle per-quad tint variation on low ground
        prng = np.random.RandomState(20240905)
        patch = 0.90 + 0.20 * prng.rand(len(cols))[:, None]
        low_mask = (h < 45.0)[:, None]
        cols = np.where(low_mask, cols * patch, cols)
        # snow caps on the high mountains
        snow = np.clip((h - 260.0) / 90.0, 0.0, 1.0)[:, None]
        snow_col = np.array([235, 240, 246])
        cols = cols * (1 - snow) + snow_col * snow
        self.t_cols = np.clip(cols, 0, 255).astype(np.uint8)

    def _build_model(self):
        parts = [
            _box(1.9, -1.9, -0.32, 0.32, -0.30, 0.30) + ((224, 147, 31),),
            _box(0.45, -0.35, -1.85, 1.85, -0.06, 0.10) + ((255, 176, 46),),
            _box(-1.65, -2.25, -0.75, 0.75, 0.28, 0.40) + ((255, 176, 46),),
            _box(-1.70, -2.30, -0.05, 0.05, 0.30, 1.05) + ((210, 134, 26),),
            _box(2.35, 1.60, -0.10, 0.10, -0.10, 0.10) + ((124, 85, 16),),
        ]
        all_v = []
        faces = []
        for v, f, col in parts:
            base = len(all_v)
            all_v.extend(v)
            for q in f:
                faces.append((tuple(base + i for i in q), col))
        self.m_verts0 = np.array(all_v) * MSCALE
        self.m_faces = faces

    # ---------------------------- static scenery --------------------------- #
    def _build_scene(self):
        """Runway, hangar, ATC tower, village buildings and trees in world coords."""
        ap = WORLD.get("airport", {})
        rw = ap.get("runway", {})
        self._flat = self.terrain.altitude_msl_at(0.0, 0.0) - BASE_ALT

        # runway slab (dark) + centreline dashes (white)
        hw, hl = rw.get("width_m", 30.0) / 2.0, rw.get("length_m", 800.0) / 2.0
        z = self._flat + 0.25
        self.runway_pts = [(rw["e"] - hw, z, rw["n"] - hl), (rw["e"] + hw, z, rw["n"] - hl),
                           (rw["e"] + hw, z, rw["n"] + hl), (rw["e"] - hw, z, rw["n"] + hl)]
        self.runway_dashes = []
        for k in range(-4, 5):
            n0 = k * 80.0 - 20.0
            self.runway_dashes.append(((rw["e"], z + 0.05, n0), (rw["e"], z + 0.05, n0 + 40.0)))
        # threshold stripes: 4 white bars in from each end
        self.runway_marks = []
        for end_n, sign in ((rw["n"] - hl + 12.0, 1.0), (rw["n"] + hl - 34.0, -1.0)):
            for k in range(4):
                e0 = rw["e"] - 11.0 + k * 6.0
                n0 = end_n if sign > 0 else end_n + 22.0
                self.runway_marks.append([
                    (e0, z + 0.04, n0), (e0 + 4.0, z + 0.04, n0),
                    (e0 + 4.0, z + 0.04, n0 + 22.0), (e0, z + 0.04, n0 + 22.0)])
        # painted designators: "36" south end, "18" north end
        self.runway_labels = [
            ((rw["e"], z + 0.06, rw["n"] - hl + 55.0), "36"),
            ((rw["e"], z + 0.06, rw["n"] + hl - 55.0), "18"),
        ]

        # solid boxes: hangar + village (world-space faces with outward normals)
        self._scene_faces = []

        def add_box(e_c, n_c, w, d, z0, z1, col):
            e0, e1 = e_c - w / 2, e_c + w / 2
            n0, n1 = n_c - d / 2, n_c + d / 2
            v = [(e0, z0, n0), (e1, z0, n0), (e1, z0, n1), (e0, z0, n1),
                 (e0, z1, n0), (e1, z1, n0), (e1, z1, n1), (e0, z1, n1)]
            faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            for q in faces:
                pts = [v[i] for i in q]
                u = np.array(pts[1]) - np.array(pts[0])
                wv = np.array(pts[3]) - np.array(pts[0])
                nrm = np.cross(u, wv)
                nl = np.linalg.norm(nrm) or 1.0
                lam = 0.62 + 0.38 * max(0.0, float(np.dot(nrm / nl, LIGHT_DIR)))
                self._scene_faces.append((pts, tuple(int(c * lam) for c in col)))

        hg = ap.get("hangar", {})
        if hg:
            add_box(hg["e"], hg["n"], hg["w_m"], hg["d_m"],
                    self.terrain.altitude_msl_at(hg["n"], hg["e"]) - BASE_ALT,
                    self.terrain.altitude_msl_at(hg["n"], hg["e"]) - BASE_ALT + hg["h_m"],
                    (150, 158, 170))
        for b in WORLD.get("buildings", []):
            add_box(b["e"], b["n"], b["w_m"], b["d_m"],
                    self.terrain.altitude_msl_at(b["n"], b["e"]) - BASE_ALT,
                    self.terrain.altitude_msl_at(b["n"], b["e"]) - BASE_ALT + b["h_m"],
                    (168, 150, 128))

        # ATC tower: shaft cylinder + cab box
        self._tower_quads = []
        tw = ap.get("tower", {})
        self._tower_top = None
        if tw:
            tb = self.terrain.altitude_msl_at(tw["n"], tw["e"]) - BASE_ALT
            segs = 10
            z0, z1 = tb, tb + tw["h_m"] * 0.62
            ring0, ring1 = [], []
            for s in range(segs):
                a = 2 * math.pi * s / segs
                ring0.append((tw["e"] + math.cos(a) * tw["radius_m"], z0, tw["n"] + math.sin(a) * tw["radius_m"]))
                ring1.append((tw["e"] + math.cos(a) * tw["radius_m"] * 1.5, z1, tw["n"] + math.sin(a) * tw["radius_m"] * 1.5))
            for s in range(segs):
                s2 = (s + 1) % segs
                self._tower_quads.append(([ring0[s], ring0[s2], ring1[s2], ring1[s]], (172, 178, 188)))
            add_box(tw["e"], tw["n"], tw["radius_m"] * 3.2, tw["radius_m"] * 3.2,
                    z1, z1 + tw["h_m"] * 0.38, (90, 120, 160))
            self._tower_top = (tw["e"], z1 + tw["h_m"] * 0.38 + 4.0, tw["n"])

        # trees: (n, e, base_up, top_up, r, shade)
        self.trees = []
        trng = random.Random(99)
        for t in WORLD.get("trees", []):
            base = self.terrain.altitude_msl_at(t["n"], t["e"]) - BASE_ALT
            sh = trng.uniform(0.8, 1.15)
            self.trees.append((t["n"], t["e"], base, base + t["h_m"], t["r_m"],
                               (int(40 * sh), int(104 * sh), int(48 * sh))))

        # clouds: fixed world-placed puffy sprites at 380-720 m
        crng = random.Random(777)
        self.clouds = []
        for i in range(24):
            cn = crng.uniform(-6500, 6500)
            ce = crng.uniform(-6500, 6500)
            alt = crng.uniform(380.0, 720.0)
            size = crng.uniform(150.0, 360.0)
            self.clouds.append((ce, alt, cn, size, self._make_cloud_sprite(900 + i)))

        # lake surface polygon + road strips (world coords, drawn with the scene)
        self.lake_pts = []
        if WORLD.get("lake"):
            lk = WORLD["lake"]
            lvl = self.terrain.altitude_msl_at(lk["n"], lk["e"])
            for s_i in range(24):
                a = 2 * math.pi * s_i / 24
                self.lake_pts.append((lk["e"] + math.cos(a) * lk["radius_m"] * 0.98,
                                      lvl + 0.3,
                                      lk["n"] + math.sin(a) * lk["radius_m"] * 0.98))
        self.road_quads = []
        for road in WORLD.get("roads", []):
            prev_pt = None
            for k in range(len(road) - 1):
                n0, e0 = road[k]
                n1, e1 = road[k + 1]
                seg_len = math.hypot(n1 - n0, e1 - e0)
                steps = max(1, int(seg_len / 90.0))
                for s_i in range(steps):
                    a0 = s_i / steps
                    a1 = (s_i + 1) / steps
                    na, ea = n0 + (n1 - n0) * a0, e0 + (e1 - e0) * a0
                    nb, eb = n0 + (n1 - n0) * a1, e0 + (e1 - e0) * a1
                    dnb, deb = (n1 - n0) / seg_len, (e1 - e0) / seg_len
                    perp_n, perp_e = -deb, dnb
                    za = self.terrain.altitude_msl_at(na, ea) - BASE_ALT + 0.2
                    zb = self.terrain.altitude_msl_at(nb, eb) - BASE_ALT + 0.2
                    self.road_quads.append((
                        [(ea + perp_e * 3.5, za, na + perp_n * 3.5),
                         (eb + perp_e * 3.5, zb, nb + perp_n * 3.5),
                         (eb - perp_e * 3.5, zb, nb - perp_n * 3.5),
                         (ea - perp_e * 3.5, za, na - perp_n * 3.5)], (66, 66, 72)))

        # windsock near the runway threshold
        self.windsock = {"e": 60.0, "n": 250.0, "h": 8.0}

    def _make_cloud_sprite(self, seed):
        rng = random.Random(seed)
        w, h = 256, 110
        sfc = pygame.Surface((w, h), pygame.SRCALPHA)
        for i in range(8):
            ccx = rng.uniform(45, w - 45)
            ccy = rng.uniform(h * 0.42, h * 0.72)
            rx = rng.uniform(32, 62)
            ry = rx * rng.uniform(0.42, 0.58)
            a = rng.randint(85, 140)
            pygame.draw.ellipse(sfc, (252, 252, 255, a), (ccx - rx, ccy - ry, rx * 2, ry * 2))
        # flat-ish underside shading
        pygame.draw.ellipse(sfc, (205, 214, 228, 70), (52, h * 0.62, w - 104, h * 0.30))
        return sfc

    def _draw_clouds(self, half, hw, hh, cam, yaw, pit, f, cx, cy):
        for (ce, alt, cn, size, spr) in self.clouds:
            p = self._proj_point(np.array([ce, alt, cn]), cam, yaw, pit, f, cx, cy)
            if not p:
                continue
            wpx = size * f / p[2]
            if wpx < 6 or wpx > hw * 2.5:
                continue
            if p[0] < -wpx or p[0] > hw + wpx or p[1] < -wpx or p[1] > hh + wpx:
                continue
            spr_scaled = pygame.transform.smoothscale(spr, (int(wpx), int(wpx * 110 / 256)))
            half.blit(spr_scaled, (p[0] - wpx / 2, p[1] - wpx * 55 / 256))

    def _draw_windsock(self, half, s, cam, yaw, pit, f, cx, cy):
        wnd = s.get("wind_ned")
        if wnd is None:
            return
        wn, we = float(wnd[0]), float(wnd[1])
        spd = math.hypot(wn, we)
        ws = self.windsock
        base = self.terrain.altitude_msl_at(ws["n"], ws["e"]) - BASE_ALT
        p0 = self._proj_point(np.array([ws["e"], base, ws["n"]]), cam, yaw, pit, f, cx, cy)
        p1 = self._proj_point(np.array([ws["e"], base + ws["h"], ws["n"]]), cam, yaw, pit, f, cx, cy)
        if not p0 or not p1:
            return
        pygame.draw.line(half, (220, 220, 228), (p0[0], p0[1]), (p1[0], p1[1]), 2)
        if spd > 0.05:
            dn, de = wn / spd, we / spd
            strength = min(1.0, spd / 8.0)
            droop = (1.0 - strength) * 1.6
            top = np.array([ws["e"], base + ws["h"], ws["n"]])
            tip = top + np.array([de * 2.4, -droop - 0.25, dn * 2.4])
            perp = np.array([-dn, 0.0, de]) * 0.32
            a = self._proj_point(top + perp, cam, yaw, pit, f, cx, cy)
            b = self._proj_point(tip + perp * 0.3, cam, yaw, pit, f, cx, cy)
            c = self._proj_point(tip - perp * 0.3, cam, yaw, pit, f, cx, cy)
            d = self._proj_point(top - perp, cam, yaw, pit, f, cx, cy)
            if a and b and c and d:
                pygame.draw.polygon(half, (255, 140, 40), [a[:2], b[:2], c[:2], d[:2]])

    # ------------------------------ helpers -------------------------------- #
    def _to_enu(self, pos_ned, alt_off=0.0):
        return np.array([pos_ned[1], -pos_ned[2] + alt_off, pos_ned[0]])

    def _rot_body_ned(self, eul):
        phi, th, psi = eul
        cph, sph = math.cos(phi), math.sin(phi)
        cth, sth = math.cos(th), math.sin(th)
        cps, sps = math.cos(psi), math.sin(psi)
        return np.array([
            [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
            [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
            [-sth, sph * cth, cph * cth],
        ])

    def _project_all(self, pts, cam, yaw, pit, f, cx, cy):
        """Vectorized projection; returns (sx, sy, depth) arrays + valid mask."""
        cyw, syw = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pit), math.sin(pit)
        q = pts - cam
        x1 = cyw * q[:, 0] - syw * q[:, 2]
        z1 = syw * q[:, 0] + cyw * q[:, 2]
        y2 = cp * q[:, 1] + sp * z1
        z2 = -sp * q[:, 1] + cp * z1
        # verts behind the near plane are clamped (not dropped) so close quads
        # fan outward and never leave holes in the mesh
        safe = np.where(z2 < 3.0, 3.0, z2)
        valid = z2 > 3.0
        return cx + f * x1 / safe, cy - f * y2 / safe, z2, valid

    # -------------------------------- draw --------------------------------- #
    def draw(self, surf, s):
        t0 = pygame.time.get_ticks()
        W, H = surf.get_size()
        hw, hh = max(W // 2, 160), max(H // 2, 120)
        if self._half is None or self._half.get_size() != (hw, hh):
            self._half = pygame.Surface((hw, hh))
        half = self._half
        f = 0.5 * min(hw, hh) / math.tan(math.radians(31))

        pos = s["state"]["pos_ned"]; eul = s["state"]["euler"]
        ac = self._to_enu(pos)
        psi = float(eul[2])
        fwd = np.array([math.sin(psi), 0.0, math.cos(psi)])
        cam_goal = ac - fwd * 18.0 + np.array([0.0, 6.0, 0.0])

        t_s = float(s.get("t_s", 0.0))
        dt = 0.05 if self._last_t is None else max(0.001, min(0.2, t_s - self._last_t))
        self._last_t = t_s
        if self._cam is None:
            self._cam = cam_goal.copy()
        k = 1.0 - math.exp(-dt * 3.5)
        self._cam += (cam_goal - self._cam) * k
        cam = self._cam.copy()
        ground_min = self.terrain.altitude_msl_at(pos[0], pos[1]) - BASE_ALT + 2.5
        if cam[1] < ground_min:
            cam[1] = ground_min

        look = ac + fwd * 7.0 + np.array([0.0, 1.2, 0.0])
        d = look - cam
        yaw = math.atan2(d[0], d[2])
        pit = -math.atan2(d[1], math.hypot(d[0], d[2]))
        cx, cy = hw / 2.0, hh * 0.46

        self._draw_sky(half, hw, hh, cam, yaw, pit, f, cx, cy)

        scene_entries, scene_labels = self._collect_scene(cam, yaw, pit, f, cx, cy, hw, hh)
        self._draw_windsock(half, s, cam, yaw, pit, f, cx, cy)
        sx, sy, dep, ok = self._project_all(self.t_verts, cam, yaw, pit, f, cx, cy)
        self._draw_terrain(half, hw, hh, sx, sy, dep, ok, cam, scene_entries)
        self._draw_trees(half, hw, hh, cam, yaw, pit, f, cx, cy)
        self._draw_clouds(half, hw, hh, cam, yaw, pit, f, cx, cy)
        self._draw_haze(half, hw, hh, cy - f * math.tan(pit))

        R_ned = self._rot_body_ned(eul)
        # body->NED then NED->ENU permutation: E=ned1, Up=-ned2, N=ned0
        R_pn = np.array([
            R_ned[1], 
            -R_ned[2],
            R_ned[0],
        ])
        self._draw_nfz(half, cam, yaw, pit, f, cx, cy)
        self._draw_waypoints(half, s, cam, yaw, pit, f, cx, cy)
        labels = self._draw_aircraft(half, ac, R_pn, s, cam, yaw, pit, f, cx, cy)

        if s.get("crash", {}).get("crashed"):
            self._draw_wreck(half, ac, cam, yaw, pit, f, cx, cy, labels)

        pygame.transform.smoothscale(half, (W, H), surf)
        labels.extend(getattr(self, "_frame_labels", []))
        for (lx, ly, txt, col, small) in labels:
            font = pygame.font.SysFont("consolas", 13 if small else 15, bold=not small)
            surf.blit(font.render(txt, True, col), (lx, ly))

        self._frame_ms = 0.9 * self._frame_ms + 0.1 * (pygame.time.get_ticks() - t0)

    def _collect_scene(self, cam, yaw, pit, f, cx, cy, hw, hh):
        """Runway, boxes and tower as painter entries merged with the terrain.
        Returns (entries, labels); entries are (depth, kind, data, color)."""
        entries = []
        self._frame_labels = []

        # runway slab
        pts, dep, ok = [], 0.0, True
        for p in self.runway_pts:
            pp = self._proj_point(np.asarray(p), cam, yaw, pit, f, cx, cy)
            if pp is None:
                ok = False
                break
            pts.append((pp[0], pp[1]))
            dep += pp[2]
        if ok:
            entries.append((dep / 4.0, "poly", pts, (56, 58, 64)))
            for a, b in self.runway_dashes:
                pa = self._proj_point(np.asarray(a), cam, yaw, pit, f, cx, cy)
                pb = self._proj_point(np.asarray(b), cam, yaw, pit, f, cx, cy)
                if pa and pb:
                    entries.append(((pa[2] + pb[2]) / 2 + 0.05, "line",
                                    [(pa[0], pa[1]), (pb[0], pb[1])], (235, 235, 235)))

        # solid boxes (hangar, village, tower cab)
        for pts3, col in self._scene_faces:
            pp = [self._proj_point(np.asarray(p), cam, yaw, pit, f, cx, cy) for p in pts3]
            if any(p is None for p in pp):
                continue
            scr = [(p[0], p[1]) for p in pp]
            xs = [p[0] for p in pp]; ys = [p[1] for p in pp]
            if max(xs) < -60 or min(xs) > hw + 60 or max(ys) < -60 or min(ys) > hh + 60:
                continue
            entries.append((sum(p[2] for p in pp) / 4.0, "poly", scr, col))

        # lake surface
        if self.lake_pts:
            pp = [self._proj_point(np.asarray(p), cam, yaw, pit, f, cx, cy) for p in self.lake_pts]
            if not any(p is None for p in pp):
                entries.append((sum(p[2] for p in pp) / len(pp), "poly",
                                [(p[0], p[1]) for p in pp], (42, 110, 168)))
        # roads
        for quad, col in self.road_quads:
            pp = [self._proj_point(np.asarray(p), cam, yaw, pit, f, cx, cy) for p in quad]
            if any(p is None for p in pp):
                continue
            xs = [p[0] for p in pp]; ys = [p[1] for p in pp]
            if max(xs) < -60 or min(xs) > hw + 60 or max(ys) < -60 or min(ys) > hh + 60:
                continue
            entries.append((sum(p[2] for p in pp) / 4.0, "poly", [(p[0], p[1]) for p in pp], col))

        # threshold stripes
        for mark in self.runway_marks:
            pp = [self._proj_point(np.asarray(p), cam, yaw, pit, f, cx, cy) for p in mark]
            if any(p is None for p in pp):
                continue
            entries.append((sum(p[2] for p in pp) / 4.0 + 0.03, "poly",
                            [(p[0], p[1]) for p in pp], (232, 232, 232)))
        for (world_pt, txt) in self.runway_labels:
            pp = self._proj_point(np.asarray(world_pt), cam, yaw, pit, f, cx, cy)
            if pp and 0 <= pp[0] < hw and 0 <= pp[1] < hh:
                self._frame_labels.append((int(pp[0] * 2) - 10, int(pp[1] * 2) - 8,
                                           txt, (240, 240, 240), False))

        # tower shaft quads + ATC label
        for quad, col in self._tower_quads:
            pp = [self._proj_point(np.asarray(p), cam, yaw, pit, f, cx, cy) for p in quad]
            if any(p is None for p in pp):
                continue
            scr = [(p[0], p[1]) for p in pp]
            entries.append((sum(p[2] for p in pp) / 4.0, "poly", scr, col))
        if self._tower_top is not None:
            tp = self._proj_point(np.asarray(self._tower_top), cam, yaw, pit, f, cx, cy)
            if tp and 0 <= tp[0] < hw and 0 <= tp[1] < hh:
                self._frame_labels.append((int(tp[0] * 2) - 14, int(tp[1] * 2) - 8,
                                           "ATC", (140, 190, 255), False))
        return entries, self._frame_labels

    def _draw_trees(self, half, hw, hh, cam, yaw, pit, f, cx, cy):
        items = []
        fog2 = self.FOG_DIST * self.FOG_DIST
        for (n, e, b_up, t_up, r, shade) in self.trees:
            dx = e - cam[0]
            dz = n - cam[2]
            if dx * dx + dz * dz > fog2:
                continue
            pb = self._proj_point(np.array([e, b_up, n]), cam, yaw, pit, f, cx, cy)
            pt = self._proj_point(np.array([e, t_up, n]), cam, yaw, pit, f, cx, cy)
            if not pb or not pt:
                continue
            if (pb[0] < -40 and pt[0] < -40) or (pb[0] > hw + 40 and pt[0] > hw + 40):
                continue
            if (pb[1] < -40 and pt[1] < -40) or (pb[1] > hh + 40 and pt[1] > hh + 40):
                continue
            items.append((pb[2], pb, pt, r, shade))
        items.sort(key=lambda it: -it[0])
        for d, pb, pt, r, shade in items:
            wpx = max(1.5, r * f / d)
            trunk_top = (pb[0] + (pt[0] - pb[0]) * 0.3, pb[1] + (pt[1] - pb[1]) * 0.3)
            pygame.draw.line(half, (74, 52, 30), (pb[0], pb[1]), trunk_top, 1)
            # two-tier pine: wide base triangle + narrower top triangle
            mid = (pb[0] * 0.25 + pt[0] * 0.75, pb[1] * 0.25 + pt[1] * 0.75)
            pygame.draw.polygon(half, shade, [
                (pt[0], pt[1]), (pb[0] - wpx * 0.62, mid[1]), (pb[0] + wpx * 0.62, mid[1])])
            pygame.draw.polygon(half, shade, [
                (mid[0], mid[1]), (pb[0] - wpx, pb[1]), (pb[0] + wpx, pb[1])])

    def _draw_wreck(self, half, ac, cam, yaw, pit, f, cx, cy, labels):
        """Smoke column, flame and CRASH SITE label over the wreck."""
        tnow = pygame.time.get_ticks() / 1000.0
        for i in range(6):
            ph = (tnow * 0.45 + i / 6.0) % 1.0
            up = ac[1] + 2.0 + ph * 55.0
            drift = math.sin(tnow * 0.7 + i) * ph * 12.0
            p = self._proj_point(np.array([ac[0] + drift, up, ac[2] + drift * 0.6]),
                                 cam, yaw, pit, f, cx, cy)
            if not p:
                continue
            r = int(max(2.0, f * (2.0 + ph * 9.0) / p[2]))
            g = int(70 + 90 * ph)
            pygame.draw.circle(half, (g, g, g), (int(p[0]), int(p[1])), r)
        pf = self._proj_point(np.array([ac[0], ac[1] + 1.5, ac[2]]), cam, yaw, pit, f, cx, cy)
        if pf:
            fl = max(2.0, f * (1.2 + 0.5 * math.sin(tnow * 23.0)) / pf[2])
            pygame.draw.circle(half, (255, 120, 30), (int(pf[0]), int(pf[1])), int(fl))
        pl = self._proj_point(np.array([ac[0], ac[1] + 70.0, ac[2]]), cam, yaw, pit, f, cx, cy)
        if pl:
            labels.append((int(pl[0] * 2) - 40, int(pl[1] * 2) - 8,
                           "CRASH SITE", (255, 150, 150), False))

    def _draw_sky(self, half, hw, hh, cam, yaw, pit, f, cx, cy):
        horizon = cy - f * math.tan(pit)
        if self._sky_h != hh or self._sky_surf is None:
            self._sky_surf = pygame.Surface((1, hh))
            self._sky_h = hh
        top = (10, 22, 48); mid = (48, 92, 150); low = (150, 190, 225)
        hy = int(max(0, min(hh - 1, horizon)))
        for y in range(hh):
            if y < hy:
                u = y / max(hy, 1)
                col = (int(top[0] + (mid[0] - top[0]) * u),
                       int(top[1] + (mid[1] - top[1]) * u),
                       int(top[2] + (mid[2] - top[2]) * u))
            else:
                # below the horizon: distant-ground haze (fills mesh-edge gaps)
                u = (y - hy) / max(hh - hy, 1)
                col = (int(150 - 55 * u), int(190 - 60 * u), int(215 - 90 * u))
            self._sky_surf.fill(col, (0, y, 1, 1))
        pygame.transform.scale(self._sky_surf, (hw, hh), half)

        # sun
        az, el = 2.15, 0.55
        sd = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        q = sd * 9000.0
        cyw, syw = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pit), math.sin(pit)
        x1 = cyw * q[0] - syw * q[2]; z1 = syw * q[0] + cyw * q[2]
        y2 = cp * q[1] + sp * z1;    z2 = -sp * q[1] + cp * z1
        if z2 > 0:
            sxp = cx + f * x1 / z2; syp = cy - f * y2 / z2
            if -50 < sxp < hw + 50 and -50 < syp < hh + 50:
                pygame.draw.circle(half, (255, 244, 200), (int(sxp), int(syp)), 13)
                halo = pygame.Surface((56, 56), pygame.SRCALPHA)
                pygame.draw.circle(halo, (255, 240, 180, 60), (28, 28), 26)
                half.blit(halo, (sxp - 28, syp - 28))

    def _draw_terrain(self, half, hw, hh, sx, sy, dep, ok, cam, extra_entries=()):
        quads = self.t_quads
        cols = self.t_cols
        depq = dep[quads]
        okq = ok[quads].all(axis=1)
        davg = depq.mean(axis=1)
        to_cam = cam[None, :] - self.t_centers
        facing = np.einsum("ij,ij->i", self.t_nrm, to_cam) > 0.0
        xs = sx[quads]; ys = sy[quads]
        off = (xs.max(axis=1) < -30) | (xs.min(axis=1) > hw + 30) | \
              (ys.max(axis=1) < -30) | (ys.min(axis=1) > hh + 30)
        mask = facing & okq & (davg < self.FOG_DIST) & ~off
        order = np.nonzero(mask)[0]
        order = order[np.argsort(-davg[order])]
        fog = np.clip(1.0 - davg / self.FOG_DIST, 0.0, 1.0)
        merged = []
        for qi in order:
            i0, i1, i2, i3 = quads[qi]
            fz = fog[qi]
            c = cols[qi]
            col = (int(c[0] * fz + 96 * (1 - fz)),
                   int(c[1] * fz + 130 * (1 - fz)),
                   int(c[2] * fz + 160 * (1 - fz)))
            merged.append((davg[qi], "poly", ((sx[i0], sy[i0]), (sx[i1], sy[i1]),
                                               (sx[i2], sy[i2]), (sx[i3], sy[i3])), col))
        merged.extend(extra_entries)
        merged.sort(key=lambda it: -it[0])
        poly = pygame.draw.polygon
        line = pygame.draw.line
        for it in merged:
            if it[1] == "poly":
                poly(half, it[3], it[2])
            else:
                line(half, it[3], it[2][0], it[2][1], 2)

    def _draw_haze(self, half, hw, hh, horizon_y):
        if self._haze_surf is None:
            sfc = pygame.Surface((1, 90), pygame.SRCALPHA)
            for y in range(90):
                a = int(150 * (1.0 - y / 89.0))
                sfc.fill((150, 190, 225, a), (0, y, 1, 1))
            self._haze_surf = sfc
        half.blit(pygame.transform.scale(self._haze_surf, (hw, 90)), (0, int(horizon_y) - 10))

    def _proj_point(self, p, cam, yaw, pit, f, cx, cy):
        cyw, syw = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pit), math.sin(pit)
        q = p - cam
        x1 = cyw * q[0] - syw * q[2]; z1 = syw * q[0] + cyw * q[2]
        y2 = cp * q[1] + sp * z1;    z2 = -sp * q[1] + cp * z1
        if z2 < 3.0:
            return None
        return (cx + f * x1 / z2, cy - f * y2 / z2, z2)

    def _draw_waypoints(self, half, s, cam, yaw, pit, f, cx, cy):
        for i, wp in enumerate(s.get("mission_items", [])):
            n, e = wp["n_m"], wp["e_m"]
            zb = self.terrain.altitude_msl_at(n, e) - BASE_ALT
            zt = max(wp["alt_msl_m"] - BASE_ALT, zb + 18.0)
            pb = self._proj_point(np.array([e, zb, n]), cam, yaw, pit, f, cx, cy)
            pt = self._proj_point(np.array([e, zt, n]), cam, yaw, pit, f, cx, cy)
            if not pb or not pt:
                continue
            cur = i == s.get("mission_current_idx", 0)
            col = (62, 207, 111) if (s.get("mission_captured") or [False] * 99)[i] else \
                  (255, 210, 74) if cur else (91, 140, 255)
            pygame.draw.line(half, col, (pb[0], pb[1]), (pt[0], pt[1]), 3 if cur else 2)
            r = 6
            pygame.draw.polygon(half, col, [
                (pt[0], pt[1] - r), (pt[0] + r, pt[1]), (pt[0], pt[1] + r), (pt[0] - r, pt[1])])
            if cur:
                pygame.draw.circle(half, col, (int(pt[0]), int(pt[1])), 13, 2)
            ft = pygame.font.SysFont("consolas", 12, bold=True)
            half.blit(ft.render(str(i + 1), True, (235, 245, 255)), (pt[0] + 8, pt[1] - 16))

    def _draw_nfz(self, half, cam, yaw, pit, f, cx, cy):
        for z in WORLD["no_fly_zones"]:
            cn, ce = z["center_ned"]
            r = z["radius_m"]
            zb = self.terrain.altitude_msl_at(cn, ce) - BASE_ALT - 4.0
            zt = max(z["ceil_m"] - BASE_ALT, zb + 60.0)
            segs = 18
            ring_t = [self._proj_point(np.array([ce + math.cos(2 * math.pi * s_ / segs) * r,
                                                 zt,
                                                 cn + math.sin(2 * math.pi * s_ / segs) * r]),
                                       cam, yaw, pit, f, cx, cy) for s_ in range(segs)]
            ring_b = [self._proj_point(np.array([ce + math.cos(2 * math.pi * s_ / segs) * r,
                                                 zb,
                                                 cn + math.sin(2 * math.pi * s_ / segs) * r]),
                                       cam, yaw, pit, f, cx, cy) for s_ in range(segs)]
            pts_t = [(p[0], p[1]) for p in ring_t if p]
            if len(pts_t) > 2:
                sfc = pygame.Surface((half.get_width(), half.get_height()), pygame.SRCALPHA)
                pygame.draw.polygon(sfc, (255, 70, 70, 34), pts_t)
                half.blit(sfc, (0, 0))
                pygame.draw.polygon(half, (255, 90, 90), pts_t, 2)
            for a, b in zip(ring_t, ring_b):
                if a and b:
                    pygame.draw.line(half, (185, 75, 75), (a[0], a[1]), (b[0], b[1]), 1)
            name = pygame.font.SysFont("consolas", 12, bold=True).render(
                z["name"], True, (255, 150, 150))
            if pts_t:
                top_y = min(p[1] for p in pts_t)
                cxm = sum(p[0] for p in pts_t) / len(pts_t)
                half.blit(name, (cxm - name.get_width() / 2, top_y - 16))

    def _draw_aircraft(self, half, ac, R, s, cam, yaw, pit, f, cx, cy):
        mv = self.m_verts0 @ R.T + ac
        sx, sy, dep, ok = self._project_all(mv, cam, yaw, pit, f, cx, cy)
        rpm = float(s.get("engine", {}).get("rpm", 0.0))
        faces = []
        for ids, col in self.m_faces:
            if not all(ok[i] for i in ids):
                continue
            u = mv[ids[1]] - mv[ids[0]]; w = mv[ids[3]] - mv[ids[0]]
            nrm = np.cross(u, w)
            nn = np.linalg.norm(nrm)
            if nn < 1e-9:
                continue
            lam = 0.75 + 0.35 * max(0.0, (nrm @ LIGHT_DIR) / nn)
            pts = [(sx[i], sy[i]) for i in ids]
            d = sum(dep[i] for i in ids) / 4.0
            faces.append((d, pts, (min(255, int(col[0] * lam)),
                                   min(255, int(col[1] * lam)),
                                   min(255, int(col[2] * lam)))))
        faces.sort(key=lambda it: -it[0])
        for _, pts, col in faces:
            pygame.draw.polygon(half, col, pts)

        # prop disc at the nose (body-x forward through the rotation matrix)
        nose = ac + R @ np.array([2.6 * MSCALE, 0.0, 0.0])
        pp = self._proj_point(nose, cam, yaw, pit, f, cx, cy)
        labels = []
        if pp and rpm > 100:
            rpx = max(3.0, f * 0.5 * MSCALE / pp[2])
            sfc = pygame.Surface((int(rpx * 2 + 4), int(rpx * 2 + 4)), pygame.SRCALPHA)
            pygame.draw.circle(sfc, (200, 200, 210, 60), (int(rpx + 2), int(rpx + 2)), int(rpx))
            half.blit(sfc, (pp[0] - rpx - 2, pp[1] - rpx - 2))
            ang = rpm * 0.0006 * (pygame.time.get_ticks() % 100000)
            for blade in (0, math.pi):
                bx = pp[0] + math.cos(ang + blade) * rpx
                by = pp[1] + math.sin(ang + blade) * rpx * 0.32
                pygame.draw.line(half, (60, 60, 66), (pp[0], pp[1]), (bx, by), 2)

        # callout
        pu = self._proj_point(ac + np.array([0, 2.4 * MSCALE, 0]), cam, yaw, pit, f, cx, cy)
        if pu:
            ex, ey = pu[0] + 22, pu[1] - 26
            pygame.draw.line(half, (255, 210, 74), (pu[0], pu[1]), (ex, ey), 1)
            ias = s.get("pitot_reading", {}).get("ias_mps", 0.0) * 1.94384
            alt = s.get("pitot_reading", {}).get("baro_alt_msl_m", 0.0) * 3.28084
            agl = s.get("agl_m", 0.0)
            labels.append((int(ex * 2), int(ey * 2 - 10),
                           f"IAS {ias:.0f}kt  ALT {alt:.0f}ft  AGL {agl:.0f}m", (255, 224, 130), False))
        labels.append((8, 8, "CHASE CAM  [M: map on/off]", (200, 220, 245), True))
        labels.append((8, 24, f"view {self._frame_ms:.1f} ms", (140, 160, 190), True))
        return labels
