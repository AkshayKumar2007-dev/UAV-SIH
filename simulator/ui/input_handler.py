import math
import pygame
import numpy as np
from simulator.config import UI, AIRFRAME, GUIDANCE, PISTON_ENGINE, HOME, WORLD


class InputHandler:
    def __init__(self):
        self.stick_roll = 0.0
        self.stick_pitch = 0.0
        self.pedal_rudder = 0.0
        self.throttle = 0.55
        self.flap_deg = 0.0
        self.gear_down = True
        self.brakes = False
        self.carb_heat = False
        self.starter_pulse = False
        self.elev_trim_deg = -2.5
        self.mode_toggle = None
        self.set_hdg = None
        self.set_alt = None
        self.waypoint_add = None
        self.waypoint_next = False
        self.mission_clear = False
        self.map_toggle = None
        self.hud_toggle = None
        self.reset = False
        self.quit = False
        self.screenshot = False
        self.paused = False

    def _smooth(self, cur, target, step):
        if target > cur:
            return min(cur + step, target)
        elif target < cur:
            return max(cur - step, target)
        return cur

    def process_events(self, dt):
        target_roll = 0.0
        target_pitch = 0.0
        target_rudder = 0.0
        thr_step = 0.0
        self.mode_toggle = None
        self.set_hdg = None
        self.set_alt = None
        self.waypoint_add = None
        self.waypoint_next = False
        self.mission_clear = False
        self.map_toggle = None
        self.hud_toggle = None
        self.reset = False
        self.starter_pulse = False
        self.screenshot = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            target_roll -= AIRFRAME["delta_a_max_deg"]
        if keys[pygame.K_d]:
            target_roll += AIRFRAME["delta_a_max_deg"]
        if keys[pygame.K_w]:
            target_pitch += AIRFRAME["delta_e_max_deg"] * 0.8   # push: nose down
        if keys[pygame.K_s]:
            target_pitch -= AIRFRAME["delta_e_max_deg"] * 0.8   # pull: nose up
        if keys[pygame.K_q]:
            target_rudder -= AIRFRAME["delta_r_max_deg"]
        if keys[pygame.K_e]:
            target_rudder += AIRFRAME["delta_r_max_deg"]
        if keys[pygame.K_UP]:
            thr_step = 0.6
            target_thr = min(1.0, self.throttle + thr_step * dt)
        elif keys[pygame.K_DOWN]:
            thr_step = 0.6
            target_thr = max(0.0, self.throttle - thr_step * dt)
        else:
            target_thr = self.throttle
        self.throttle = target_thr
        if keys[pygame.K_f]:
            self.flap_deg = self._smooth(self.flap_deg, AIRFRAME["delta_flap_max_deg"], 40.0 * dt)
        elif self.flap_deg > 0.0:
            self.flap_deg = self._smooth(self.flap_deg, 0.0, 30.0 * dt)

        if keys[pygame.K_t]:
            self.elev_trim_deg += 0.8 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else -0.8
            self.elev_trim_deg = float(np.clip(self.elev_trim_deg, -15.0, 15.0))

        self.stick_roll = self._smooth(self.stick_roll, target_roll, AIRFRAME["delta_a_max_deg"] * 3.2 * dt)
        self.stick_pitch = self._smooth(self.stick_pitch, target_pitch, AIRFRAME["delta_e_max_deg"] * 2.8 * dt)
        self.pedal_rudder = self._smooth(self.pedal_rudder, target_rudder, AIRFRAME["delta_r_max_deg"] * 2.5 * dt)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.quit = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.quit = True
                elif ev.key == pygame.K_g:
                    self.gear_down = not self.gear_down
                elif ev.key == pygame.K_SPACE:
                    self.brakes = True
                elif ev.key == pygame.K_b:
                    self.brakes = not self.brakes
                elif ev.key == pygame.K_h:
                    self.carb_heat = not self.carb_heat
                elif ev.key == pygame.K_j:
                    self.starter_pulse = True
                elif ev.key == pygame.K_1:
                    self.mode_toggle = "MANUAL"
                elif ev.key == pygame.K_2:
                    self.mode_toggle = "ALT_HOLD"
                elif ev.key == pygame.K_3:
                    self.mode_toggle = "HDG_HOLD"
                elif ev.key == pygame.K_4:
                    self.mode_toggle = "WAYPOINT"
                elif ev.key == pygame.K_5:
                    self.mode_toggle = "RTH"
                elif ev.key == pygame.K_l:
                    self.waypoint_next = True
                elif ev.key == pygame.K_c:
                    self.mission_clear = True
                elif ev.key == pygame.K_m:
                    self.map_toggle = True
                elif ev.key == pygame.K_r:
                    self.reset = True
                elif ev.key == pygame.K_v:
                    self.hud_toggle = True
                elif ev.key == pygame.K_z:
                    self.mode_toggle = "STAB"
                elif ev.key == pygame.K_p:
                    self.paused = not self.paused
                elif ev.key == pygame.K_F12:
                    self.screenshot = True
                elif ev.key == pygame.K_LEFTBRACKET:
                    self.elev_trim_deg = float(np.clip(self.elev_trim_deg - 0.8, -15.0, 15.0))
                elif ev.key == pygame.K_RIGHTBRACKET:
                    self.elev_trim_deg = float(np.clip(self.elev_trim_deg + 0.8, -15.0, 15.0))
                elif ev.key == pygame.K_TAB:
                    pass
            elif ev.type == pygame.KEYUP:
                if ev.key == pygame.K_SPACE and not keys[pygame.K_b]:
                    self.brakes = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self.waypoint_add = ev.pos
        return self

    def controls_dict(self):
        return {
            "delta_a_deg": float(self.stick_roll),
            # elevator position = stick + trim wheel (trim is part of the surface)
            "delta_e_deg": float(self.stick_pitch + self.elev_trim_deg),
            "delta_r_deg": float(self.pedal_rudder),
            "delta_flap_deg": float(self.flap_deg),
            "throttle": float(self.throttle),
            "gear_down": bool(self.gear_down),
            "brakes": bool(self.brakes),
            "carb_heat": bool(self.carb_heat),
            "delta_e_trim_deg": float(self.elev_trim_deg),
        }
