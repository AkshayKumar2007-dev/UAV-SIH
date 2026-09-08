"""Engine Digital Twin estimator.

A parallel software model of the aero piston engine. It is fed ONLY by the
noisy sensor stream (RPM, CHT, EGT, oil pressure) plus the commanded
throttle — never by the simulator's internal ground truth. It maintains a
mirror of the engine's hidden internal states (mechanical wear, carb
fouling, oil-system health, effective power) and corrects that mirror
against smoothed measurement residuals — an EKF-inspired sensor-fusion
loop with bounded, noise-rejecting updates.

Estimation structure (the key design):
  psi_model  = oil_pressure_nominal * oil_health      (the twin's BELIEF)
  res_oil    = psi_measured - psi_model               (measured - believed)
  oil_health += K * tanh(res_oil / 12) * dt           (deficit -> health down)

Same pattern for CHT (residual vs the clean-engine thermal target drives
wear) and EGT (drives fouling). The blended mirror readings are display
state only — they never feed the residuals.

Outputs:
  - estimated hidden states (0..1): wear, fouling, oil health, power
  - remaining-useful-life (RUL) prognosis in engine-hours with a
    confidence band from the trend noise
  - a divergence metric flagging unmodelled faults or lying sensors

All rates on the engine AGING clock (ENGINE_TIME_SCALE) so the twin lives
in the same mission-time as the fault physics.
"""

import math
from collections import deque

import numpy as np

from simulator.config import PISTON_ENGINE, ENGINE_TIME_SCALE
from simulator.ai_advisor.ml_runtime import MLEngineMonitor

E_S = ENGINE_TIME_SCALE   # engine-seconds per real second


class EngineDigitalTwin:
    def __init__(self):
        # ---- hidden state estimates (what real sensors can't measure) ----
        self.wear = 0.0            # mechanical wear / thermal damage 0..1
        self.fouling = 0.0         # carb / induction fouling 0..1
        self.oil_health = 1.0      # oil-system health 1..0
        self.power_est = 1.0       # estimated effective power fraction

        # ---- twin mirror of the measurable engine state (display) ----
        self.rpm_t = float(PISTON_ENGINE["idle_rpm"])
        self.cht_t = float(PISTON_ENGINE["cht_nominal_c"])
        self.egt_t = float(PISTON_ENGINE["egt_nominal_c"])
        self.oil_psi_t = float(PISTON_ENGINE["oil_pressure_nominal_psi"])

        # ---- estimation internals ----
        self._res_cht_s = 0.0      # smoothed CHT residual
        self._res_oil_s = 0.0      # smoothed oil-pressure residual
        self._res_egt_s = 0.0      # smoothed EGT residual
        self._rpm_sm = 0.0         # fast-smoothed rpm (for roughness)
        self._rpm_var = 0.0        # rpm variance estimate (roughness power)
        self._synced = True
        self._divergence = 0.0
        self._confidence = 1.0
        self._first = True         # snap the mirror to the first measurement

        # ---- prognostics ----
        self.rul_h = None
        self.rul_lo = None
        self.rul_hi = None
        self._wear_hist = deque(maxlen=600)    # 60 s of real time at 10 Hz
        self._foul_hist = deque(maxlen=600)

        # ML layer: behaviour regression + fault classifier + failure-risk
        # model, trained offline on simulator scenarios and continuing to
        # learn online from the live sensor stream
        self.ml = MLEngineMonitor()

        self._t_real = 0.0

    # ------------------------------------------------------------------ #
    def step(self, dt_e, dt_real, sensors, throttle, on_ground=False, carb_heat=False):
        """dt_e: engine-aging seconds (compressed); dt_real: real seconds.
        sensors: the noisy engine sensor dict (rpm, cht_C, egt_C, oil_psi)."""
        self._t_real += dt_real
        c = max(dt_real, 1e-3)
        self.ml.update(dt_real, sensors, throttle, carb_heat)

        rpm_m = float(sensors.get("rpm", self.rpm_t))
        cht_m = float(sensors.get("cht_C", self.cht_t))
        egt_m = float(sensors.get("egt_C", self.egt_t))
        psi_m = float(sensors.get("oil_psi", self.oil_psi_t))

        idle, mx = PISTON_ENGINE["idle_rpm"], PISTON_ENGINE["max_rpm"]
        rpm_cmd = idle + (mx - idle) * (max(0.0, min(1.0, throttle)) ** 0.85)

        if self._first:
            # cold start: the mirror begins where the engine actually is
            self._first = False
            self.rpm_t, self.cht_t = rpm_m, cht_m
            self.egt_t, self.oil_psi_t = egt_m, psi_m
            self._rpm_sm = rpm_m
        rpm_ratio = rpm_m / max(rpm_cmd, 1.0)

        # ---------- 1. twin mirror propagation (display state) -------------
        # the PISTON_ENGINE thermal taus are already in ENGINE-seconds
        tau_r = PISTON_ENGINE["rpm_tau_s"] * E_S
        self.rpm_t += (rpm_cmd - self.rpm_t) * (1.0 - math.exp(-dt_e / tau_r))
        load = self.power_est * (0.4 + 0.6 * max(0.0, min(1.0, throttle))) * \
            max(0.0, min(1.0, (self.rpm_t - idle) / max(mx - idle, 1.0)))
        cht_model = (PISTON_ENGINE["cht_nominal_c"] +
                     0.9 * load * (PISTON_ENGINE["cht_max_c"] -
                                   PISTON_ENGINE["cht_nominal_c"]))
        psi_model = PISTON_ENGINE["oil_pressure_nominal_psi"] * max(0.05, self.oil_health)
        egt_model = PISTON_ENGINE["egt_nominal_c"] + 220.0 * load
        self.cht_t += (cht_model - self.cht_t) * (1.0 - math.exp(
            -dt_e / PISTON_ENGINE["cht_tau_s"]))
        self.oil_psi_t += (psi_model - self.oil_psi_t) * (1.0 - math.exp(
            -dt_e / PISTON_ENGINE["oil_tau_s"]))
        self.egt_t += (egt_model - self.egt_t) * (1.0 - math.exp(
            -dt_e / PISTON_ENGINE["egt_tau_s"]))
        self.rpm_t += (rpm_m - self.rpm_t) * 0.25

        # ---------- 2. model-based residuals -------------------------------
        # res_cht: measured hotter than the clean-engine prediction => damage
        res_cht = cht_m - cht_model
        # res_oil: measured psi below the believed psi => oil-system loss
        res_oil = psi_m - psi_model
        # res_egt: measured EGT above predicted => combustion fouling
        res_egt = egt_m - egt_model

        # ---------- 3. smoothed residuals (real-time EMAs, tau 8 s) --------
        kf = 1.0 - math.exp(-c / 15.0)
        self._res_cht_s += (res_cht - self._res_cht_s) * kf
        self._res_oil_s += (res_oil - self._res_oil_s) * kf
        self._res_egt_s += (res_egt - self._res_egt_s) * kf

        # ---------- 4. rpm roughness (variance around the smooth rpm) ------
        ks = 1.0 - math.exp(-c / 0.35)
        self._rpm_sm += (rpm_m - self._rpm_sm) * ks
        dv = rpm_m - self._rpm_sm
        kv = 1.0 - math.exp(-c / 5.0)
        self._rpm_var += (dv * dv - self._rpm_var) * kv
        sigma = math.sqrt(max(self._rpm_var, 0.0))

        # ---------- 5. bounded health corrections (per real second) --------
        # Signs by construction: measured CHT/EGT ABOVE the clean-engine
        # prediction => wear/fouling up; measured OIL PSI below believed =>
        # oil-system health down.
        if self._t_real >= 8.0:      # let the mirror initialize before correcting
            self.wear = min(1.0, max(0.0,
                self.wear + 0.050 * math.tanh(self._res_cht_s / 25.0) * dt_real))
            self.oil_health = min(1.0, max(0.0,
                self.oil_health - 0.050 *
                min(1.0, max(0.0, (self.oil_psi_t - psi_m)) / 25.0) * dt_real))
            rough_term = 0.0 if on_ground else max(0.0, sigma - 10.0) / 50.0
            hot_term = max(0.0, self._res_egt_s) / 80.0
            drive = 0.020 * math.tanh(rough_term + hot_term) * dt_real
            if drive == 0.0:
                drive = -0.010 * dt_real          # clean running burns deposits off
            self.fouling = min(1.0, max(0.0,
                self.fouling + drive))
        else:
            # oil-system health estimate: normalized measured psi (leak watch)
            self.oil_health = min(1.0, max(0.05, psi_m /
                max(PISTON_ENGINE["oil_pressure_nominal_psi"], 1.0)))
        self.power_est = min(1.2, max(0.05,
            self.power_est + (rpm_ratio - self.power_est) *
            (1.0 - math.exp(-c / 4.0))))

        # ---------- 6. divergence (twin vs measured) -----------------------
        # a cold-soaked engine takes ~1 min to warm into the model's operating
        # band — the divergence is only accumulated once warmed up
        warm = cht_m > 0.75 * max(cht_model, 60.0)   # near operating temp
        div = (abs(self._res_cht_s) / 45.0 + abs(psi_model - psi_m) / 20.0 +
               abs(self._res_egt_s) / 80.0)
        if warm:
            self._divergence = self._divergence * 0.95 + div * 0.05
        else:
            self._divergence *= 0.9          # decay toward sync while warming
        self._synced = self._divergence < 1.5   # tolerant of warm-up/ice transients
        self._confidence = max(0.0, min(1.0, 1.0 - self._divergence / 2.0))
        self._warm = warm

        # ---------- 7. prognostics: trends -> RUL --------------------------
        self._wear_hist.append((self._t_real, self.wear))
        self._foul_hist.append((self._t_real, self.fouling))
        self.rul_h, self.rul_lo, self.rul_hi = self._prognose()

    # ------------------------------------------------------------------ #
    def _prognose(self):
        """Extrapolate the worst degradation trend to its failure threshold.
        Returns (RUL, lo, hi) in ENGINE-HOURS, or (None, None, None) when no
        degradation trend is present."""
        worst = None
        for hist, limit in ((self._wear_hist, 0.95), (self._foul_hist, 0.90)):
            if len(hist) < 40:
                continue
            t_now = hist[-1][0]
            pts = [(tt, vv) for (tt, vv) in hist if tt >= t_now - 30.0]
            if len(pts) < 20:
                continue
            tt = np.array([p[0] for p in pts])
            vv = np.array([p[1] for p in pts])
            coef = np.polyfit(tt, vv, 1)
            slope = float(coef[0])
            if float(tt.max() - tt.min()) < 15.0:
                continue
            v_now = float(vv[-1])
            margin = limit - v_now
            if slope <= 1e-5:
                if margin < 0 and (worst is None or worst[0] > 0.0):
                    worst = (0.0, 0.0, 0.0)
                continue
            rul_s = margin / slope
            rul_eh = max(0.0, rul_s) / 60.0                     # 60 engine-s per real s
            noise = float(np.std(vv - np.polyval(coef, tt)))
            spread = max(0.15, 2.0 * noise / max(slope, 1e-6) / 60.0)
            if worst is None or rul_eh < worst[0]:
                worst = (rul_eh, max(0.0, rul_eh - spread), rul_eh + spread)
        if worst is None:
            return None, None, None
        return worst

    # ------------------------------------------------------------------ #
    def summary(self):
        return {
            "wear_pct": round(self.wear * 100.0, 1),
            "fouling_pct": round(self.fouling * 100.0, 1),
            "oil_health_pct": round(self.oil_health * 100.0, 1),
            "power_est_pct": round(self.power_est * 100.0, 1),
            "rul_h": None if self.rul_h is None else round(self.rul_h, 2),
            "rul_lo": None if self.rul_lo is None else round(self.rul_lo, 2),
            "rul_hi": None if self.rul_hi is None else round(self.rul_hi, 2),
            "synced": bool(self._synced),
            "divergence": round(self._divergence, 3),
            "confidence": round(self._confidence, 2),
            "ml": self.ml.diagnosis_summary(),
        }
