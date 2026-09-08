"""Systematic engine fault model.

Three independent fault channels feed a cumulative engine-stress state:

  1. Abuse damage — sustained high RPM, overtemperature or oil starvation
     accumulates `stress` (0..1). Stress costs power proportionally and a
     full tank of stress seizes the engine permanently.
  2. Carburettor icing — builds up during a randomized window per flight
     when carb heat is off; melts when the pilot applies carb heat.
  3. Oil leak — randomized onset; quantity drains (slower at low throttle)
     and oil pressure tracks quantity until the engine runs dry.

Everything here only ever DEGRADES the engine — it cannot touch flight
controls, so the AI-advisory architecture is unaffected.
"""

import math
import random

from simulator.config import FAULTS, ENGINE_TIME_SCALE


class FaultSystem:
    def __init__(self):
        self.stress = 0.0            # 0..1 cumulative mechanical damage
        self.seized = False          # permanent engine death
        self.ice = 0.0               # 0..1 carburettor ice
        self.oil_qty_pct = 100.0     # oil quantity
        self.leak_active = False
        self.power_factor = 1.0      # applied to engine power this step
        self.rpm_wobble_rpm = 0.0    # roughness added to rpm this step

        self._t = 0.0               # engine aging clock
        self._t_real = 0.0          # real-time clock
        # fault onsets are scheduled in REAL seconds so a fault never fires at
        # the very start of a flight — and some flights have no fault at all
        self._ice_delay = random.uniform(*FAULTS["ice_start_s_range"])
        self._leak_delay = random.uniform(*FAULTS["oil_leak_delay_s_range"])
        self._ice_this_flight = random.random() < FAULTS["ice_chance"]
        self._leak_this_flight = random.random() < FAULTS["oil_leak_chance"]
        self._wobble_phase = random.uniform(0.0, 2.0 * math.pi)
        self._wobble_freq = random.uniform(7.0, 11.0)

    # ------------------------------------------------------------------ #
    def step(self, dt_e, dt_real, rpm_frac, throttle, cht_C, oil_psi, carb_heat,
             engine_running, cht_max_c, oil_min_psi):
        self._t += dt_e
        self._t_real += dt_real
        self.power_factor = 0.0 if self.seized else 1.0
        self.rpm_wobble_rpm = 0.0
        if self.seized or not engine_running:
            return

        # ---- 1. abuse -> stress ----
        rate = 0.0
        hi = FAULTS["high_rpm_frac"]
        if rpm_frac > hi:
            depth = min(1.0, (rpm_frac - hi) / max(1.0 - hi, 1e-6))
            rate += FAULTS["stress_per_s_high_rpm"] * (0.4 + 0.6 * depth)
        if cht_C > cht_max_c - FAULTS["overtemp_margin_c"]:
            rate += FAULTS["stress_per_s_overtemp"]
        if oil_psi < oil_min_psi + 5.0 and self._t_real > 1.0:   # skip startup transient
            rate += FAULTS["stress_per_s_low_oil"]
        if rate > 0.0:
            self.stress = min(1.0, self.stress + rate * dt_real)
        elif rpm_frac < hi - 0.12 and cht_C < cht_max_c - FAULTS["overtemp_margin_c"] - 10.0:
            self.stress = max(0.0, self.stress - FAULTS["stress_decay_per_s"] * dt_e)

        if self.stress >= 1.0:
            self.seized = True
            return

        # ---- 2. carb ice ----
        if carb_heat:
            self.ice = max(0.0, self.ice - FAULTS["ice_melt_per_s"] * dt_e)
        elif self._ice_this_flight and self._t_real > self._ice_delay:
            self.ice = min(1.0, self.ice + FAULTS["ice_rate_per_s"] * dt_e)

        # ---- 3. oil leak ----
        if not self.leak_active and self._leak_this_flight and self._t_real > self._leak_delay:
            self.leak_active = True
        if self.leak_active:
            relief = 1.0 - FAULTS["oil_leak_throttle_relief"] * max(0.0, min(1.0, 1.0 - throttle))
            self.oil_qty_pct = max(0.0, self.oil_qty_pct - 100.0 * FAULTS["oil_leak_rate_per_s"] * relief * dt_e)

        # ---- effects ----
        pf_abuse = 1.0 - FAULTS["stress_power_penalty"] * self.stress
        pf_ice = 1.0 - FAULTS["ice_power_loss_frac"] * max(0.0, self.ice - 0.35) / 0.65
        self.power_factor = max(0.05, pf_abuse * pf_ice)

        rough = 0.8 * self.ice + 0.5 * max(0.0, self.stress - 0.4)
        self.rpm_wobble_rpm = rough * 220.0 * math.sin(2.0 * math.pi * self._wobble_freq * self._t + self._wobble_phase)

    # ------------------------------------------------------------------ #
    def summary(self):
        return {
            "stress_pct": round(self.stress * 100.0, 1),
            "ice_pct": round(self.ice * 100.0, 1),
            "oil_qty_pct": round(self.oil_qty_pct, 1),
            "leak_active": bool(self.leak_active),
            "seized": bool(self.seized),
            "power_factor": round(self.power_factor, 3),
        }
