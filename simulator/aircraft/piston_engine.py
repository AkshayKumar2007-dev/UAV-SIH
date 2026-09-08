import numpy as np
from simulator.config import PISTON_ENGINE, AERO, ENGINE_TIME_SCALE
from simulator.aircraft.faults import FaultSystem


class PistonEngine:
    def __init__(self):
        cfg = PISTON_ENGINE
        self.rpm = cfg["idle_rpm"]
        self.running = True
        self.faults = FaultSystem()
        self.fuel_L = cfg["fuel_full_L"]
        self._wobble_offset = 0.0      # reported-rpm roughness offset
        self.cht_C = 50.0
        self.egt_C = 150.0
        self.oil_psi = 0.0
        self.run_time_s = 0.0
        self.starter_active = False
        self.starter_timer_s = 0.0
        self.carb_heat = False
        self._t_real = 0.0
        self._power_kw = 0.0
        self._fuel_flow_Lph = 0.0
        self._thrust_N = 0.0
        self._advance_ratio = 0.0
        self._rpm_window = []
        self._rpm_window_t = 0.0

    def command_starter(self, active):
        self.starter_active = bool(active)

    def kill(self):
        """Crash handling: engine dead, gauges spool to zero immediately."""
        self.running = False
        self.starter_active = False
        self.rpm = 0.0
        self._power_kw = 0.0
        self._fuel_flow_Lph = 0.0
        self._thrust_N = 0.0

    def set_carb_heat(self, on):
        self.carb_heat = bool(on)

    def _fuel_used_kg(self):
        return PISTON_ENGINE["fuel_full_L"] * PISTON_ENGINE["fuel_density_kg_per_L"] - self.fuel_kg_remaining

    @property
    def fuel_kg_remaining(self):
        return self.fuel_L * PISTON_ENGINE["fuel_density_kg_per_L"]

    @property
    def fuel_flow_Lph(self):
        return self._fuel_flow_Lph

    @property
    def power_kw(self):
        return self._power_kw

    @property
    def thrust_N(self):
        return self._thrust_N

    def step(self, throttle_cmd, dt, rho_kgm3, V_air_mps):
        cfg = PISTON_ENGINE
        throttle = float(np.clip(throttle_cmd, 0.0, 1.0))
        # engine AGING clock: fuel burn, thermal states, wear and faults run on
        # compressed time (MALE mission arc in a short demo); RPM response and
        # the starter stay real-time.
        dt_e = dt * ENGINE_TIME_SCALE
        self.run_time_s += dt_e
        self._t_real += dt
        Dp = AERO["prop_diam_m"]
        A_prop = np.pi * (Dp / 2.0) ** 2

        if self.faults.seized:
            self.starter_active = False
            self.starter_timer_s = 0.0
        if self.starter_active and not self.running:
            self.starter_timer_s += dt
            self.rpm += (cfg["idle_rpm"] * 0.8 - self.rpm) * (1.0 - np.exp(-dt / 0.15))
            if self.rpm > 0.9 * cfg["idle_rpm"] and self.starter_timer_s > cfg["starter_torque_s"]:
                self.running = True
                self.starter_active = False
                self.starter_timer_s = 0.0
        elif self.starter_active and self.running:
            self.starter_active = False
            self.starter_timer_s = 0.0

        if self.fuel_L <= 0.0 and self.running:
            self.running = False

        if not self.running:
            tau_windmilling = V_air_mps / 45.0
            rpm_target = cfg["idle_rpm"] * 0.25 + tau_windmilling
            self.rpm += (rpm_target - self.rpm) * (1.0 - np.exp(-dt / 1.5))
            self._power_kw = 0.0
            self._fuel_flow_Lph = 0.0
        else:
            rpm_target = cfg["idle_rpm"] + (cfg["max_rpm"] - cfg["idle_rpm"]) * (throttle ** 0.85)
            self.rpm += (rpm_target - self.rpm) * (1.0 - np.exp(-dt / cfg["rpm_tau_s"]))
            rpm_frac = np.clip((self.rpm - cfg["idle_rpm"]) / max(cfg["max_rpm"] - cfg["idle_rpm"], 1), 0.0, 1.0)
            self._power_kw = cfg["max_power_kw"] * rpm_frac * (0.4 + 0.6 * throttle)
            fuel_used_kg_per_h = cfg["bsfc_kg_per_kwh"] * self._power_kw
            self._fuel_flow_Lph = fuel_used_kg_per_h / cfg["fuel_density_kg_per_L"]
            self.fuel_L = max(0.0, self.fuel_L - self._fuel_flow_Lph * (dt_e / 3600.0))

        n_rps = self.rpm / 60.0
        if n_rps > 0.1 and Dp > 0:
            J = V_air_mps / (n_rps * Dp) if n_rps * Dp > 0 else 0.0
            self._advance_ratio = J
            Ct0 = cfg["prop_power_coeff"] * 4.2
            Ct = max(0.0, Ct0 * (1.0 - J / 1.8))
            self._thrust_N = Ct * rho_kgm3 * n_rps ** 2 * Dp ** 4 * cfg["prop_eff_coeff"]
            if not self.running:
                self._thrust_N *= 0.15
        else:
            self._advance_ratio = 0.0
            self._thrust_N = 0.0

        # systematic faults: stress / carb ice / oil leak
        rpm_frac_now = np.clip((self.rpm - cfg["idle_rpm"]) / max(cfg["max_rpm"] - cfg["idle_rpm"], 1), 0.0, 1.0)
        self.faults.step(dt_e, dt, rpm_frac_now, throttle, self.cht_C, self.oil_psi,
                         self.carb_heat, self.running,
                         cfg["cht_max_c"], cfg["oil_pressure_min_psi"])
        self._power_kw *= self.faults.power_factor
        self._fuel_flow_Lph *= self.faults.power_factor
        if self.faults.seized and self.running:
            self.running = False   # permanent mechanical failure

        load_frac = self._power_kw / max(cfg["max_power_kw"], 0.001) if self.running else 0.0
        cht_target = cfg["cht_nominal_c"] + 0.9 * load_frac * (cfg["cht_max_c"] - cfg["cht_nominal_c"])
        cht_target += 10.0 * self.faults.stress
        if self.carb_heat:
            cht_target += 12.0
        self.cht_C += (cht_target - self.cht_C) * (1.0 - np.exp(-dt_e / cfg["cht_tau_s"]))
        egt_target = cfg["egt_nominal_c"] + load_frac * 220.0 - (0.5 - 0.3 * abs(throttle - 0.65))
        self.egt_C += (egt_target - self.egt_C) * (1.0 - np.exp(-dt_e / cfg["egt_tau_s"]))
        oil_qty_factor = min(1.0, self.faults.oil_qty_pct / 70.0)
        oil_target = cfg["oil_pressure_nominal_psi"] * oil_qty_factor if self.running else 0.0
        self.oil_psi += (oil_target - self.oil_psi) * (1.0 - np.exp(-dt_e / cfg["oil_tau_s"]))

        # roughness is applied to the REPORTED rpm only — never integrated
        # into the engine state (integrating it made the engine wander)
        self._wobble_offset = self.faults.rpm_wobble_rpm if self.running else 0.0

        self._rpm_window_t += dt
        if len(self._rpm_window) > 200:
            self._rpm_window.pop(0)
        self._rpm_window.append(float(self.rpm + self._wobble_offset))

        return self._thrust_N

    def health_flags(self):
        flags = []
        if not self.running:
            flags.append("ENGINE_OFF")
        if self.fuel_L < PISTON_ENGINE["fuel_full_L"] * 0.17:
            flags.append("FUEL_LOW")
        if self.fuel_L <= 0.0:
            flags.append("FUEL_DEPLETED")
        if self.cht_C > PISTON_ENGINE["cht_max_c"] - 20.0:
            flags.append("CHT_HIGH")
        if self.egt_C > PISTON_ENGINE["egt_max_c"] - 50.0:
            flags.append("EGT_HIGH")
        if self.running and self.oil_psi < PISTON_ENGINE["oil_pressure_min_psi"]:
            flags.append("OIL_PRESS_LOW")
        if (
            self.running
            and self.run_time_s > 10.0
            and len(self._rpm_window) > 40
            and np.std(self._rpm_window) > 120.0
        ):
            flags.append("RPM_FLUCTUATION")
        f = self.faults
        if f.seized:
            flags.append("ENGINE_SEIZED")
        elif f.stress > 0.25:
            flags.append("ENGINE_DAMAGE")
        if f.ice > 0.25:
            flags.append("CARB_ICE")
        if f.leak_active:
            flags.append("OIL_LEAK")
        return flags

    def summary(self):
        return {
            "rpm": float(self.rpm + self._wobble_offset),
            "running": bool(self.running),
            "thrust_N": float(self._thrust_N),
            "power_kw": float(self._power_kw),
            "fuel_L": float(self.fuel_L),
            "fuel_kg_remaining": float(self.fuel_kg_remaining),
            "fuel_flow_Lph": float(self._fuel_flow_Lph),
            "cht_C": float(self.cht_C),
            "egt_C": float(self.egt_C),
            "oil_psi": float(self.oil_psi),
            "advance_ratio": float(self._advance_ratio),
            "carb_heat": bool(self.carb_heat),
            "starter_active": bool(self.starter_active),
            "run_time_s": float(self.run_time_s),
            "health_flags": self.health_flags(),
            "faults": self.faults.summary(),
        }
