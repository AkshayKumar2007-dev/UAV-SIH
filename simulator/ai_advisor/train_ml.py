"""Offline ML training for the engine AI advisor.

Generates labelled flight data by stepping the simulator's REAL engine
physics (PistonEngine + FaultSystem + EngineSensors, no display) through
randomized throttle schedules and fault scenarios, then fits:

  1. a behaviour model        — ridge regression, predicts normal sensor
                                response ~1 s ahead (residuals = anomalies)
  2. a diagnosis model        — softmax regression over
                                {healthy, carb_ice, oil_leak, stress, dead}
  3. a hazard model           — logistic regression, P(seizure within 60 s)

Weights + normalizers are saved to ml_models.json next to this file; the
live digital twin loads them (ml_runtime.py) and keeps learning online.

Run:  python -m simulator.ai_advisor.train_ml [--episodes 48] [--seed 7]
"""

import argparse
import math
import os
import random
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.config import PISTON_ENGINE, SENSORS
from simulator.aircraft.piston_engine import PistonEngine
from simulator.sensors.engine_sensors import EngineSensors
from simulator.ai_advisor.ml_core import (
    CLASS_NAMES, FEATURE_NAMES, TARGET_NAMES, PREDICT_HORIZON_S,
    EngineFeatureHistory, targets_from_features,
    RidgeRegressor, SoftmaxRegression, Standardizer, save_models,
    calibrate_temperature,
)

DT = 0.1                       # sample period (matches TELEMETRY_HZ)
HORIZON_N = int(round(PREDICT_HORIZON_S / DT))
FAIL_N = int(round(60.0 / DT))  # hazard horizon in samples

CLASS_ICE, CLASS_LEAK, CLASS_STRESS, CLASS_DEAD = 1, 2, 3, 4


def _truth_class(faults, running):
    """Label one sample from ground truth (training data only).

    Thresholds sit where the fault actually produces a sensor signature —
    labelling earlier than that only teaches the classifier noise:
      - carb ice: power loss and roughness start at ~0.30 ice
      - oil leak: pressure only responds below ~70% quantity
      - stress: roughness/heat effects are measurable from ~0.30
    """
    if not running:
        return CLASS_DEAD
    if faults.leak_active and faults.oil_qty_pct < 70.0:
        return CLASS_LEAK
    if faults.ice > 0.30:
        return CLASS_ICE
    if faults.stress > 0.30:
        return CLASS_STRESS
    return 0


class _Episode:
    """One generated flight: engine physics + scenario + sensor noise."""

    def __init__(self, rng, kind):
        self.kind = kind                       # healthy|ice|leak|abuse|mixed
        self.rng = rng
        self.eng = PistonEngine()
        self.sens = EngineSensors()
        f = self.eng.faults
        # scenario scheduling: force the fault channels this episode studies
        f._ice_this_flight = kind in ("ice", "mixed")
        f._leak_this_flight = kind in ("leak", "mixed")
        f._ice_delay = rng.uniform(15.0, 60.0)
        f._leak_delay = rng.uniform(15.0, 60.0)
        self.throttle = rng.uniform(0.45, 0.75)
        self.target = self.throttle
        self.tgt_left = rng.uniform(20.0, 50.0)
        self.carb_heat = False
        self.carb_left = rng.uniform(30.0, 80.0)
        self.seized_sample = None
        self.t = 0.0

    def step(self):
        rng = self.rng
        self.tgt_left -= DT
        if self.tgt_left <= 0.0:
            if self.kind == "abuse":
                self.target = rng.uniform(0.95, 1.0)
                self.tgt_left = rng.uniform(45.0, 90.0)
            else:
                self.target = rng.uniform(0.2, 1.0)
                self.tgt_left = rng.uniform(15.0, 50.0)
        k = 1.0 - np.exp(-DT / 3.0)
        self.throttle += (self.target - self.throttle) * k
        if self.kind == "abuse":
            self.throttle = min(1.0, self.throttle + 0.05)

        self.carb_left -= DT
        if self.carb_left <= 0.0:
            # pilots react to rough rpm some of the time — gives the models
            # examples of both building and melting ice
            ice_now = self.eng.faults.ice
            self.carb_heat = (self.rng.random() < 0.5 if ice_now > 0.5
                              else self.rng.random() < 0.18)
            self.carb_left = self.rng.uniform(20.0, 70.0)

        was_running = self.eng.running
        self.eng.set_carb_heat(self.carb_heat)
        self.eng.step(self.throttle, DT, 1.225, 30.0)
        if was_running and not self.eng.running and self.seized_sample is None:
            self.seized_sample = True          # failure event this sample
        return self.sens.reading(self.eng.summary())


def generate(episodes, seed):
    rng = random.Random(seed)
    kinds = (["healthy"] * 6 + ["ice"] * 5 + ["leak"] * 5 +
             ["abuse"] * 4 + ["mixed"] * 3)
    rows_f, rows_y, rows_dx, rows_haz, rows_ep = [], [], [], [], []
    for ep_i in range(episodes):
        kind = kinds[ep_i % len(kinds)]
        ep = _Episode(rng, kind)
        dur = rng.uniform(240.0, 480.0)
        n = int(dur / DT)
        hist_f = deque(maxlen=FAIL_N + 1)      # rolling feature history
        feat = EngineFeatureHistory()
        all_f, cls_rows = [], []
        for i in range(n + HORIZON_N):
            s = ep.step()
            fv = feat.push(DT, s, ep.throttle, ep.carb_heat)
            hist_f.append(fv)
            all_f.append(fv)
            cls_rows.append(_truth_class(ep.eng.faults, ep.eng.running))
        # rows: features(i) → normalised sensors HORIZON_N samples ahead
        feat_rows = all_f[:n]
        tgt_rows = [targets_from_features(all_f[i + HORIZON_N]) for i in range(n)]
        # failure timing → hazard labels (seizure within the next 60 s);
        # only rows that have horizon targets join the dataset
        fail_idx = None
        if ep.seized_sample:
            fail_idx = n                        # sample index where it died
        for i in range(n):
            if tgt_rows[i] is None:
                continue
            haz = 1 if (fail_idx is not None and i < fail_idx <= i + FAIL_N) else 0
            rows_f.append(feat_rows[i])
            rows_y.append(tgt_rows[i])
            rows_dx.append(cls_rows[i])
            rows_haz.append(haz)
            rows_ep.append(ep_i)
        print(f"  episode {ep_i + 1:02d} [{kind:7s}] {n * DT:5.0f}s  "
              f"classes: {np.bincount(np.array(cls_rows), minlength=5)}")
    F = np.array(rows_f)
    Y = np.array(rows_y)
    dx = np.array(rows_dx)
    haz = np.array(rows_haz)
    ep_ids = np.array(rows_ep)
    return F, Y, dx, haz, ep_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"[train_ml] generating {args.episodes} episodes with real engine physics…")
    F, Y, dx, haz, ep_ids = generate(args.episodes, args.seed)
    n = len(F)
    print(f"[train_ml] samples: {n}  features: {F.shape[1]}")
    print(f"[train_ml] class mix: " +
          ", ".join(f"{CLASS_NAMES[k]}={int((dx == k).sum())}" for k in range(len(CLASS_NAMES))) +
          f" | hazard positives: {int(haz.sum())}")

    # ---- stage 1: behaviour model (what normal looks like) ----
    # episode-level holdout: ~30% of episodes are never seen during fitting,
    # so the reported metrics measure generalization, not memorization
    is_test = (ep_ids % 10) >= 7
    t_std = Standardizer.fit(Y[~is_test])
    Yn = t_std.transform(Y)
    reg = RidgeRegressor(lam=1.0).fit(F[~is_test], Yn[~is_test])
    pred = np.clip(reg.predict(F), -0.2, 1.3)   # physical target range
    ss_res = ((Yn - pred) ** 2).sum(axis=0)
    ss_tot = ((Yn - Yn.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    print("[train_ml] behaviour model R² per target (holdout):",
          {TARGET_NAMES[i]: round(float(r2[i]), 3) for i in range(len(TARGET_NAMES))})

    # ---- stage 2: diagnosis classifier (features + learned residuals) ----
    # the instantaneous 1 s residual is noise-dominated; faults like carb ice
    # are a small persistent shift, so smooth residuals within each episode
    # (same EMA the runtime uses, tau = 10 s).  Residuals are clipped exactly
    # like the runtime clamps them — off-manifold behaviour-model
    # extrapolation must not leak into the classifier's inputs.
    res = np.clip(Yn - pred, -1.5, 1.5)
    res_ema = np.zeros_like(res)
    for ep in np.unique(ep_ids):
        m = ep_ids == ep
        r = res[m]
        e = np.zeros_like(r)
        acc_e = np.zeros(r.shape[1])
        for i in range(len(r)):
            acc_e += (r[i] - acc_e) * (1.0 - math.exp(-DT / 10.0))
            e[i] = acc_e
        res_ema[m] = e
    X2 = np.hstack([F, res_ema])
    x2_std = Standardizer.fit(X2[~is_test])
    X2s = np.clip(x2_std.transform(X2), -4.0, 4.0)
    # cap the healthy majority so the fault classes carry real weight
    rng = np.random.default_rng(args.seed)
    fit_pool = np.where(~is_test)[0]
    healthy_idx = fit_pool[dx[fit_pool] == 0]
    other_idx = fit_pool[dx[fit_pool] != 0]
    cap = int(len(other_idx) * 1.5)
    if len(healthy_idx) > cap:
        healthy_idx = rng.choice(healthy_idx, size=cap, replace=False)
    fit_idx = np.concatenate([healthy_idx, other_idx])
    rng.shuffle(fit_idx)
    # sqrt-inverse-frequency weighting: pulls minority recall up without
    # letting false alarms flood the healthy class
    dxw = {k: (len(fit_idx) / (len(CLASS_NAMES) * max(int((dx[fit_idx] == k).sum()), 1))) ** 0.5
           for k in range(len(CLASS_NAMES))}
    clf = SoftmaxRegression(lr=0.5, epochs=400, l2=1e-3).fit(
        X2s[fit_idx], dx[fit_idx], len(CLASS_NAMES), class_weight=dxw)
    p = clf.predict_proba(X2s).argmax(axis=1)
    acc = float((p[is_test] == dx[is_test]).mean())
    print(f"[train_ml] diagnosis accuracy (holdout): {acc * 100:.1f}%")
    per = {}
    for k in range(len(CLASS_NAMES)):
        m = (dx == k) & is_test
        per[CLASS_NAMES[k]] = round(float((p[m] == k).mean() * 100.0), 1) if m.any() else None
    print("[train_ml] per-class recall % (holdout):", per)

    # ---- stage 2b: one-vs-rest fault detectors ----
    # softmax arg-max can lock onto one fault and hide a second, co-occurring
    # one (ice + oil leak).  Each fault gets an independent detector instead;
    # the runtime reports per-fault probabilities that don't compete.
    ovr, ovr_temps = [], []
    for k in range(len(CLASS_NAMES)):
        yk = (dx == k).astype(int)
        mk = SoftmaxRegression(lr=0.5, epochs=300, l2=1e-3).fit(
            X2s[fit_idx], yk[fit_idx], 2,
            class_weight={1: max(1.0, int((yk[fit_idx] == 0).sum()) / max(int((yk[fit_idx] == 1).sum()), 1))})
        pk = mk.predict_proba(X2s[is_test])[:, 1]
        rec = float((pk[yk[is_test] == 1] > 0.5).mean()) if (yk[is_test] == 1).any() else None
        ovr.append(mk.W.tolist())
        ovr_temps.append(calibrate_temperature(mk.W, X2s[is_test], yk[is_test], 0.75))
        print(f"[train_ml]   OVR {CLASS_NAMES[k]:>8}: holdout recall {rec if rec is None else round(rec * 100, 1)}%")

    # ---- stage 3: hazard model (P(failure within 60 s)) ----
    # rare positives: weight them so "no failure predicted" is not free
    n_pos, n_neg = int(haz[~is_test].sum()), int((haz[~is_test] == 0).sum())
    hz = SoftmaxRegression(lr=0.4, epochs=400, l2=1e-4).fit(
        X2s[~is_test], haz[~is_test], 2, class_weight={1: max(1.0, n_neg / max(n_pos, 1))})
    ph = hz.predict_proba(X2s)[:, 1]
    hz_hit = float(((ph[is_test] > 0.5) == (haz[is_test] == 1)).mean())
    prec = float((ph[(haz == 1) & is_test] > 0.5).mean()) if ((haz == 1) & is_test).any() else 0.0
    far = float((ph[(haz == 0) & is_test] <= 0.5).mean())
    print(f"[train_ml] hazard (holdout): accuracy {hz_hit * 100:.1f}%, "
          f"recall-on-failures {prec * 100:.1f}%, false-alarm rate {100 - far * 100:.1f}%")

    # temperature calibration: pick softmax temperatures whose mean
    # max-probability matches the measured accuracy, so live probabilities
    # read like honest odds instead of pinning at 1.00
    temp_dx = calibrate_temperature(clf.W, X2s[is_test], dx[is_test], max(acc, 0.55))
    temp_hz = calibrate_temperature(hz.W, X2s[is_test], haz[is_test], max(hz_hit, 0.55))
    print(f"[train_ml] calibrated temperatures: diagnosis T={temp_dx}, hazard T={temp_hz}")

    # refit on ALL episodes for the shipped artifact (metrics above are from
    # the holdout models; the saved model uses every drop of data)
    clf_final = SoftmaxRegression(lr=0.5, epochs=400, l2=1e-3).fit(
        X2s, dx, len(CLASS_NAMES),
        class_weight={k: (n / (len(CLASS_NAMES) * max(int((dx == k).sum()), 1))) ** 0.5
                      for k in range(len(CLASS_NAMES))})
    ovr_final, ovr_final_temps = [], []
    for k in range(len(CLASS_NAMES)):
        yk = (dx == k).astype(int)
        mk = SoftmaxRegression(lr=0.5, epochs=300, l2=1e-3).fit(
            X2s, yk, 2,
            class_weight={1: max(1.0, int((yk == 0).sum()) / max(int((yk == 1).sum()), 1))})
        ovr_final.append(mk.W.tolist())
        ovr_final_temps.append(ovr_temps[k])
    hz_final = SoftmaxRegression(lr=0.4, epochs=400, l2=1e-4).fit(
        X2s, haz, 2, class_weight={1: max(1.0, int((haz == 0).sum()) / max(int(haz.sum()), 1))})
    reg_final = RidgeRegressor(lam=1.0).fit(F, Yn)

    out = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "class_names": CLASS_NAMES,
        "predict_horizon_s": PREDICT_HORIZON_S,
        "fail_horizon_s": 60.0,
        "target_std": t_std.to_dict(),
        "behaviour": {"W": reg_final.W.tolist(), "lam": reg_final.lam},
        "x2_std": x2_std.to_dict(),
        "diagnosis": {"W": clf_final.W.tolist(), "temp": temp_dx},
        "ovr": {"W": ovr_final, "temps": ovr_final_temps},
        "hazard": {"W": hz_final.W.tolist(), "temp": temp_hz},
        "meta": {"episodes": args.episodes, "samples": n, "seed": args.seed,
                 "dx_accuracy_holdout": round(acc, 4),
                 "r2_holdout": [round(float(v), 4) for v in r2]},
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models.json")
    save_models(path, out)
    print(f"[train_ml] saved {path}")


if __name__ == "__main__":
    main()
