"""Numpy ML core for the engine AI — no external ML dependencies.

Three learned models power the "AI with ML" layer:

  1. Behaviour model (multi-output ridge regression) — predicts the normal
     engine response (EGT, CHT, oil pressure, fuel flow) ~1 s ahead from the
     operating conditions.  Residuals of this model are the learned analogue
     of the hand-tuned physics residuals the twin used before.
  2. Diagnosis model (multinomial logistic regression) — classifies the
     engine into {healthy, carb-ice, oil-leak, stress-damage, dead} from the
     sensor features plus the behaviour residuals.
  3. Hazard model (logistic regression) — predicts the probability of an
     engine failure (seizure) within the next 60 s of flight.

The same feature builder (EngineFeatureHistory) is used at training time and
at run time, so the models always see exactly the same view of the world —
and only what a real sensor stream could see: noisy gauges + the throttle
and carb-heat commands.  Never ground truth.
"""

import math
from collections import deque

import numpy as np

from simulator.config import PISTON_ENGINE

# ------------------------------------------------------------------ #
# Feature contract (shared by training and runtime)
# ------------------------------------------------------------------ #
FEATURE_NAMES = [
    "thr", "carb", "rpm_frac", "rpm_cmd_frac", "rpm_err", "rpm_rough",
    "cht_n", "cht_tr", "egt_n", "egt_tr", "oil_n", "oil_tr", "ff_n", "ff_tr",
    "d10_egt", "d10_cht", "d10_oil", "d10_ff",
]
TARGET_NAMES = ["egt_n", "cht_n", "oil_n", "ff_n"]
# positions of the regression targets inside the feature vector
TGT_IDX = [8, 6, 10, 12]
CLASS_NAMES = ["healthy", "carb_ice", "oil_leak", "stress", "dead"]
PREDICT_HORIZON_S = 1.0          # behaviour model predicts this far ahead
FAIL_HORIZON_S = 60.0            # hazard model horizon

_FF_MAX_LPH = (PISTON_ENGINE["bsfc_kg_per_kwh"] * PISTON_ENGINE["max_power_kw"]
               / PISTON_ENGINE["fuel_density_kg_per_L"])


class EngineFeatureHistory:
    """Turns the raw noisy sensor stream into the ML feature vector.

    Maintains the small state the features need (rpm roughness window,
    short/long EMAs for trends).  `push` is called once per telemetry sample
    (10 Hz in flight, 10 Hz in the training generator).
    """

    def __init__(self):
        self._rpm_win = deque(maxlen=30)       # ~3 s of samples
        self._ema = {"cht": None, "egt": None, "oil": None, "ff": None}
        self._tgt_hist = deque(maxlen=101)     # ~10 s of target history
        self._t = 0.0

    def push(self, dt, sensors, throttle, carb_heat):
        self._t += dt
        rpm = max(0.0, float(sensors.get("rpm", 0.0)))
        cht = float(sensors.get("cht_C", 0.0))
        egt = float(sensors.get("egt_C", 0.0))
        oil = float(sensors.get("oil_psi", 0.0))
        ff = max(0.0, float(sensors.get("fuel_flow_Lph", 0.0)))

        self._rpm_win.append(rpm)
        if len(self._rpm_win) >= 5:
            w = np.array(self._rpm_win)
            rough = float(np.std(w)) / max(PISTON_ENGINE["max_rpm"] * 0.05, 1.0)
        else:
            rough = 0.0

        # trend = (value - slow EMA) scaled to per-minute units; the fast EMA
        # is what the trend is measured against so steps register quickly
        for key, v in (("cht", cht), ("egt", egt), ("oil", oil), ("ff", ff)):
            prev = self._ema[key]
            self._ema[key] = v if prev is None else prev + (v - prev) * min(1.0, dt / 6.0)

        idle, mx = PISTON_ENGINE["idle_rpm"], PISTON_ENGINE["max_rpm"]
        rpm_frac = max(0.0, min(1.15, (rpm - idle) / max(mx - idle, 1.0)))
        thr = max(0.0, min(1.0, float(throttle)))
        rpm_cmd_frac = thr ** 0.85
        cht_n = (cht - PISTON_ENGINE["cht_nominal_c"]) / max(
            PISTON_ENGINE["cht_max_c"] - PISTON_ENGINE["cht_nominal_c"], 1.0)
        egt_n = (egt - PISTON_ENGINE["egt_nominal_c"]) / max(
            PISTON_ENGINE["egt_max_c"] - PISTON_ENGINE["egt_nominal_c"], 1.0)
        oil_n = oil / max(PISTON_ENGINE["oil_pressure_nominal_psi"], 1.0)
        ff_n = ff / max(_FF_MAX_LPH, 1.0)

        tr = lambda key, v: 0.0 if self._ema[key] is None else (v - self._ema[key]) * 10.0
        base = np.array([
            thr, 1.0 if carb_heat else 0.0, rpm_frac, rpm_cmd_frac,
            rpm_frac - rpm_cmd_frac, min(3.0, rough),
            cht_n, tr("cht", cht), egt_n, tr("egt", egt),
            oil_n, tr("oil", oil), ff_n, tr("ff", ff),
        ], dtype=np.float64)

        # 10-second level drops — a slow drain that the short trend features
        # average away (an oil leak dumps oil_n by 0.3+ in 10 s; normal
        # operation barely moves)
        self._tgt_hist.append((egt_n, cht_n, oil_n, ff_n))
        if len(self._tgt_hist) >= 101:
            old = np.array(self._tgt_hist[0])
            d10 = (np.array(self._tgt_hist[-1]) - old) * 10.0
        else:
            d10 = np.zeros(4)
        return np.concatenate([base, d10])


def targets_from_features(feat):
    """The behaviour model's regression targets, taken from a feature vector."""
    return np.array([feat[i] for i in TGT_IDX], dtype=np.float64)


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #
class Standardizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X, eps=1e-6):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean, std)

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]))


def _add_bias(X):
    return np.hstack([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1))])


class RidgeRegressor:
    """Closed-form multi-output ridge regression with an optional prior.

    The prior-anchored fit is what makes online learning safe at run time:
    W = (XᵀX + L·I)⁻¹ (XᵀY + L·W_prior) — the model may adapt to the
    specific engine instance it is flying on, but only in directions the
    recent data actually supports.
    """

    def __init__(self, lam=1.0):
        self.lam = float(lam)
        self.W = None

    def fit(self, X, Y, w_prior=None, prior_lam=0.0):
        Xb = _add_bias(X)
        Y = np.asarray(Y, dtype=np.float64)
        d = Xb.shape[1]
        lam_vec = np.full(d, self.lam)
        lam_vec[-1] = 0.0                       # never shrink the bias
        A = Xb.T @ Xb + np.diag(lam_vec)
        b = Xb.T @ Y
        if w_prior is not None and prior_lam > 0.0:
            Pl = lam_vec.copy()
            Pl[-1] = 0.0
            b = b + Pl[:, None] * np.asarray(w_prior, dtype=np.float64)
            A = A + np.diag(Pl)
        self.W = np.linalg.solve(A + 1e-9 * np.eye(d), b)
        return self

    def predict(self, X):
        return _add_bias(X) @ self.W

    @classmethod
    def from_weights(cls, W, lam):
        m = cls(lam=lam)
        m.W = np.asarray(W, dtype=np.float64)
        return m


class SoftmaxRegression:
    """Multinomial logistic regression, full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=350, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, int(epochs), float(l2)
        self.W = None

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y_idx, n_classes, class_weight=None):
        Xb = _add_bias(X)
        n, d = Xb.shape
        Y = np.zeros((n, n_classes))
        Y[np.arange(n), np.asarray(y_idx, dtype=int)] = 1.0
        if class_weight is not None:
            S = np.array([class_weight.get(int(y), 1.0) for y in y_idx])
        else:
            S = np.ones(n)
        S_norm = S / S.sum() * n
        rng = np.random.default_rng(0)
        self.W = np.zeros((d, n_classes))
        reg = np.ones_like(self.W) * self.l2
        reg[-1, :] = 0.0                        # no L2 on the bias
        for _ in range(self.epochs):
            P = self._softmax(Xb @ self.W)
            G = (Xb.T @ ((P - Y) * S_norm[:, None])) / n + reg * self.W
            self.W -= self.lr * G
        return self

    def predict_proba(self, X):
        return self._softmax(_add_bias(X) @ self.W)

    @classmethod
    def from_weights(cls, W, lr=0.5, epochs=0, l2=1e-3):
        m = cls(lr=lr, epochs=epochs, l2=l2)
        m.W = np.asarray(W, dtype=np.float64)
        return m


# ------------------------------------------------------------------ #
# Model persistence
# ------------------------------------------------------------------ #
def save_models(path, payload):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def calibrate_temperature(W, Xs, y_idx, target_conf):
    """Pick the softmax temperature whose mean max-probability matches the
    model's measured accuracy — probabilities then read like honest odds
    instead of saturating at 1.00."""
    Xb = _add_bias(Xs)
    Z = Xb @ W
    for T in np.arange(1.0, 40.0, 0.5):
        Pt = SoftmaxRegression._softmax(Z / T)
        if float(Pt.max(axis=1).mean()) <= target_conf:
            return round(float(T), 2)
    return 40.0


def load_models(path):
    import json, os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
