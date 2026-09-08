"""Runtime ML engine monitor — online inference + online learning.

Loads the models trained by train_ml.py (ml_models.json) and runs them at
telemetry rate against the live noisy sensor stream.  It never sees ground
truth: features come from the gauges plus the throttle / carb-heat commands.

What it produces each update:
  - residuals: measured-vs-learned-normal behaviour, EMA-smoothed (10 s) —
    the learned analogue of the physics twin's model residuals
  - diagnosis: softmax probabilities over
    {healthy, carb_ice, oil_leak, stress, dead}
  - fail risk: P(engine failure within the next 60 s)

Online learning: the behaviour model is re-fit every ~5 s against a rolling
window of recent samples, anchored to the trained weights with a ridge
prior — it adapts to the individual engine instance without drifting away
from everything the offline training knew.  Samples are only collected
while the engine is actually running (windmilling would poison the model).
"""

import math
import os
from collections import deque

import numpy as np

from simulator.config import PISTON_ENGINE
from simulator.ai_advisor.ml_core import (
    CLASS_NAMES, PREDICT_HORIZON_S, EngineFeatureHistory,
    targets_from_features, RidgeRegressor, SoftmaxRegression, Standardizer,
    load_models,
)

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models.json")

RES_EMA_TAU = 10.0        # s — matches the training-time residual smoothing
ONLINE_WINDOW = 1800      # samples (~3 min) for the online re-fit
REFIT_EVERY = 100         # samples between online re-fits
ONLINE_PRIOR_LAM = 400.0  # strong prior anchor: adapt gently, never drift
DEF_LOGIT_TEMP = 2.5      # used if the model file carries no calibrated value
RISK_EMA_TAU = 4.0        # s — smooth the failure-risk signal
WARMUP_S = 45.0           # cold-start transients poison the residuals —
                          # same idea as the physics twin's warm-up gate
RES_CLIP = 1.5            # hard clamp on residuals: the linear behaviour
                          # model extrapolates wildly off-manifold, and an
                          # unclipped residual would swamp the classifier


class MLEngineMonitor:
    def __init__(self, models_path=None):
        self.available = False
        self.n_learned = 0
        self._t = 0.0
        self._feat = EngineFeatureHistory()
        self._pending = deque()            # (samples_left, prediction)
        self._res_ema = None               # smoothed residual vector
        self._risk_ema = None              # smoothed failure risk
        self._onl_X = deque(maxlen=ONLINE_WINDOW)
        self._onl_Y = deque(maxlen=ONLINE_WINDOW)
        self._since_refit = 0
        self._diag = {"available": False}

        data = load_models(models_path or MODELS_PATH)
        if not data:
            self._diag["reason"] = "ml_models.json not found — run train_ml.py"
            return
        try:
            self.horizon_n = int(round(data["predict_horizon_s"] / 0.1))
            self.target_std = Standardizer.from_dict(data["target_std"])
            self.behaviour = RidgeRegressor.from_weights(
                np.array(data["behaviour"]["W"]), data["behaviour"]["lam"])
            self.behaviour_prior = self.behaviour.W.copy()
            self.x2_std = Standardizer.from_dict(data["x2_std"])
            self.diagnosis = SoftmaxRegression.from_weights(np.array(data["diagnosis"]["W"]))
            self.hazard = SoftmaxRegression.from_weights(np.array(data["hazard"]["W"]))
            # one-vs-rest fault detectors: per-fault probabilities that don't
            # compete, so ice + oil leak can be reported together
            self.ovr_W = [np.array(w) for w in data.get("ovr", {}).get("W", [])]
            self.ovr_temps = data.get("ovr", {}).get("temps", [DEF_LOGIT_TEMP] * len(self.ovr_W))
            self.class_names = data.get("class_names", CLASS_NAMES)
            self.temp_dx = float(data.get("diagnosis", {}).get("temp", DEF_LOGIT_TEMP))
            self.temp_hz = float(data.get("hazard", {}).get("temp", DEF_LOGIT_TEMP))
            self.available = True
            self._res_ema = np.zeros(len(data["target_names"]))
        except (KeyError, ValueError, TypeError) as ex:
            self.available = False
            self._diag["reason"] = f"model file unreadable: {ex}"

    # ------------------------------------------------------------------ #
    def update(self, dt, sensors, throttle, carb_heat):
        """One telemetry sample. Returns the diagnosis dict for the twin."""
        if not self.available:
            return self._diag
        self._t += dt
        fv = self._feat.push(dt, sensors, throttle, carb_heat)

        # --- 1-step-ahead behaviour prediction + deferred residual ---
        y_now = targets_from_features(fv)
        # predictions are clamped to the physical target range: off-manifold
        # extrapolation (e.g. during an oil collapse) would otherwise produce
        # absurd residuals and poison every downstream feature
        pred_now = np.clip(self.behaviour.predict(fv.reshape(1, -1))[0], -0.2, 1.3)
        self._pending.append([self.horizon_n, pred_now, y_now])
        res_sample = None
        for item in self._pending:
            item[0] -= 1
        while self._pending and self._pending[0][0] <= 0:
            _, p, y_meas = self._pending.popleft()
            res_sample = np.clip(y_meas - p, -RES_CLIP, RES_CLIP)
        if res_sample is not None and self._t >= WARMUP_S:
            # residuals only enter the EMA once the engine has warmed: the
            # cold-start fill transients otherwise bias every feature downstream
            if self._res_ema is None:
                self._res_ema = res_sample.copy()
            else:
                k = 1.0 - math.exp(-dt / RES_EMA_TAU)
                self._res_ema = self._res_ema + (res_sample - self._res_ema) * k

        # --- classification + hazard on [features | smoothed residuals] ---
        # logits are tempered: a linear softmax on clean simulator data is
        # badly overconfident on real sensor noise, and a 1.00 reading tells
        # the pilot nothing a 0.82 wouldn't
        if self._t < WARMUP_S or self._res_ema is None:
            self._diag = {"available": True, "class": "warming", "conf": 0.0,
                          "probs": {}, "fail_risk_60s": 0.0,
                          "residuals": [0.0] * len(self._res_ema if self._res_ema is not None
                                                   else [0.0, 0.0, 0.0, 0.0]),
                          "learned_samples": self.n_learned}
            return self._diag
        x2 = self.x2_std.transform(np.concatenate([fv, self._res_ema]).reshape(1, -1))
        x2 = np.clip(x2, -4.0, 4.0)   # robust to rare off-manifold features
        # per-fault probabilities from the one-vs-rest detectors (fallback to
        # the softmax arg-max if the model file predates them)
        if self.ovr_W:
            probs = np.zeros(len(self.class_names))
            for i, W in enumerate(self.ovr_W[:len(self.class_names)]):
                probs[i] = self._tempered(W, x2, self.ovr_temps[i])[0][1]
            k = int(np.argmax(probs))
        else:
            probs = self._tempered(self.diagnosis.W, x2, self.temp_dx)[0]
            k = int(np.argmax(probs))
        risk_raw = float(self._tempered(self.hazard.W, x2, self.temp_hz)[0][1])
        if self._risk_ema is None:
            self._risk_ema = risk_raw
        else:
            self._risk_ema += (risk_raw - self._risk_ema) * min(1.0, dt / RISK_EMA_TAU)
        k = int(np.argmax(probs))
        self._diag = {
            "available": True,
            "class": self.class_names[k],
            "conf": round(float(probs[k]), 3),
            "probs": {self.class_names[i]: round(float(probs[i]), 3)
                      for i in range(len(self.class_names))},
            "fail_risk_60s": round(float(self._risk_ema), 3),
            "residuals": [round(float(v), 4) for v in self._res_ema],
            "learned_samples": self.n_learned,
            "ovr": True,
        }

        # --- online learning of the behaviour model (running engine only) ---
        idle = PISTON_ENGINE["idle_rpm"]
        if float(sensors.get("rpm", 0.0)) > idle * 0.8:
            self._onl_X.append(fv)
            self._onl_Y.append(y_now)
            self._since_refit += 1
            if self._since_refit >= REFIT_EVERY and len(self._onl_X) >= 200:
                X = np.array(self._onl_X)
                Y = self.target_std.transform(np.array(self._onl_Y))
                self.behaviour.fit(X, Y, w_prior=self.behaviour_prior,
                                   prior_lam=ONLINE_PRIOR_LAM)
                self._since_refit = 0
                self.n_learned += len(X)
        return self._diag

    # ------------------------------------------------------------------ #
    @staticmethod
    def _tempered(W, Xs, temp):
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])   # bias column
        Z = (Xb @ W) / max(temp, 0.5)
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def diagnosis_summary(self):
        d = dict(self._diag)
        d["res_ema_tau_s"] = RES_EMA_TAU
        return d
