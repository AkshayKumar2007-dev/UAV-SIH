"""Dryden continuous-turbulence model (MIL-F-8785C, low-altitude form).

The simulator previously represented turbulence as smoothed Gaussian noise,
which has no physical scale: the gust magnitude was a tuning constant and the
spectrum fell off at whatever rate the exponential smoother happened to give.
This module generates the three gust components by passing white noise through
the Dryden forming filters, so the output spectra match the standard
atmospheric-turbulence PSDs and each component's variance equals the intensity
the spec prescribes.

Scope note: this implements the LOW-ALTITUDE (h < 1000 ft AGL) form, where the
scale lengths grow with height. That covers the regime this simulator flies in.
Above 1000 ft the spec switches to constant scale lengths; that branch is not
implemented, so the low-altitude form is applied with height clamped instead.

Normalisation
-------------
The Dryden transfer functions are usually quoted with an arbitrary input
normalisation, so the classic forms do not produce sigma-squared variance when
driven by unit-PSD white noise. Rather than propagate that ambiguity, each
filter here is normalised directly from the physical requirement

    Var(output) == sigma^2

For a first-order filter 1/(s+a) driven by white noise of one-sided PSD q, the
stationary variance is q/(2a), so q = 2*a*sigma^2. For the second-order form
(s+c)/(s+b)^2 it is q*(b^2+c^2)/(4*b^3); with c = b/sqrt(3) that reduces to
q/(3b), so q = 3*b*sigma^2. `_selftest` checks both the realised variance and
the measured spectrum against these closed forms.

Second-order realisation
------------------------
(s+c)/(s+b)^2 is realised as the cascade

    dx1/dt = -b*x1 + w
    dx2/dt =  x1 - b*x2
    y      =  x1 + (c - b)*x2

so the system matrix A = [[-b,0],[1,-b]] has the repeated eigenvalue -b and
exp(A t) = exp(-b t) * (I + N t) with N = A + bI and N^2 = 0. That makes both
the discrete transition matrix and the process-noise covariance available in
closed form, so no matrix-exponential library is needed.
"""

import math

import numpy as np

_FT_PER_M = 1.0 / 0.3048


def dryden_parameters(agl_m, w20_mps):
    """Scale lengths (m) and intensities (m/s) for the low-altitude Dryden model.

    `w20_mps` is the mean wind speed at 20 ft in m/s. The spec gives the
    intensity as sigma_w = 0.1 * W20, a ratio, so m/s in gives m/s out.

    Returns (Lu_m, Lv_m, Lw_m, sigma_u, sigma_v, sigma_w).
    """
    h_ft = max(agl_m, 3.0) * _FT_PER_M
    height_term = 0.177 + 0.000823 * h_ft

    lw_m = h_ft * 0.3048
    lu_m = lv_m = (h_ft / (height_term ** 1.2)) * 0.3048

    sigma_w = 0.1 * w20_mps
    sigma_u = sigma_v = sigma_w / (height_term ** 0.4)

    return lu_m, lv_m, lw_m, sigma_u, sigma_v, sigma_w


def _second_order_discrete(b, sigma, dt):
    """Discrete (A_d, Q_d) for the cascade, normalised for output variance sigma^2.

    b is the pole (V/L) and the driving noise has one-sided PSD q = 3*b*sigma^2.
    Returns the 2x2 transition matrix and the 2x2 process-noise covariance.
    """
    e = math.exp(-b * dt)
    e2 = math.exp(-2.0 * b * dt)
    bdt = b * dt

    a_d = np.array([[e, 0.0],
                    [dt * e, e]])

    q = 3.0 * b * sigma * sigma
    j0 = (1.0 - e2) / (2.0 * b)
    j1 = (1.0 - e2 * (1.0 + 2.0 * bdt)) / (4.0 * b * b)
    j2 = (2.0 - e2 * (4.0 * bdt * bdt + 4.0 * bdt + 2.0)) / (8.0 * b * b * b)

    q_d = q * np.array([[j0, j1],
                        [j1, j2]])
    # symmetrise against round-off so Cholesky cannot see a non-PSD matrix
    q_d = 0.5 * (q_d + q_d.T)
    return a_d, q_d


class DrydenTurbulence:
    """Continuous Dryden turbulence, advanced one physics step at a time."""

    def __init__(self, w20_mps=7.0, seed=20240905, enabled=True):
        self.w20_mps = float(w20_mps)
        self.enabled = bool(enabled)
        self.rng = np.random.default_rng(seed)

        self._x_u = 0.0
        self._x_vw = [np.zeros(2), np.zeros(2)]   # v and w cascade states
        self._last = np.zeros(3)
        self._coeff_key = None
        self._coeffs = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._x_u = 0.0
        self._x_vw = [np.zeros(2), np.zeros(2)]
        self._last[:] = 0.0
        self._coeff_key = None

    def _coefficients(self, airspeed_mps, agl_m):
        """Cache filter coefficients, keyed on airspeed and height.

        They only change when airspeed or height moves materially, and the
        filter state is carried across a retune so the gust does not jump.
        """
        key = (round(airspeed_mps, 2), round(agl_m, 1))
        if key == self._coeff_key:
            return self._coeffs

        v = max(airspeed_mps, 1.0)     # a parked aircraft still sits in moving air
        lu, lv, lw, su, sv, sw = dryden_parameters(agl_m, self.w20_mps)

        self._coeffs = {
            "au": v / lu, "su": su,
            "lu": lu, "lv": lv, "lw": lw,
            "bv": v / lv, "cv": v / (math.sqrt(3.0) * lv), "sv": sv,
            "bw": v / lw, "cw": v / (math.sqrt(3.0) * lw), "sw": sw,
        }
        self._coeff_key = key
        return self._coeffs

    def step(self, dt, airspeed_mps, agl_m):
        """Advance the filters and return the gust vector in NED (m/s)."""
        if not self.enabled or dt <= 0.0:
            self._last[:] = 0.0
            return self._last

        c = self._coefficients(airspeed_mps, agl_m)

        a_u = c["au"]
        e_a = math.exp(-a_u * dt)
        g_u = c["su"] * math.sqrt(max(1.0 - math.exp(-2.0 * a_u * dt), 0.0))
        self._x_u = e_a * self._x_u + g_u * self.rng.standard_normal()

        for i, (b, cc, sigma) in enumerate(((c["bv"], c["cv"], c["sv"]),
                                            (c["bw"], c["cw"], c["sw"]))):
            a_d, q_d = _second_order_discrete(b, sigma, dt)
            try:
                l_chol = np.linalg.cholesky(q_d)
            except np.linalg.LinAlgError:
                # Q_d is PSD by construction; nudge the diagonal if round-off
                # ever pushes it marginally negative
                l_chol = np.linalg.cholesky(q_d + np.eye(2) * 1e-15)
            self._x_vw[i] = a_d @ self._x_vw[i] + l_chol @ self.rng.standard_normal(2)

        x_v, x_w = self._x_vw[0], self._x_vw[1]
        out_v = x_v[0] + (c["cv"] - c["bv"]) * x_v[1]
        out_w = x_w[0] + (c["cw"] - c["bw"]) * x_w[1]

        # frozen-field approximation: the gust components are treated as aligned
        # with the flight path rather than rotated into it, matching the
        # pre-existing wind-field behaviour that callers already account for.
        self._last[:] = (self._x_u, out_v, out_w)
        return self._last

    def intensity_mps(self, agl_m):
        _, _, _, su, sv, sw = dryden_parameters(agl_m, self.w20_mps)
        return float(su), float(sv), float(sw)

    def summary(self):
        return {
            "model": "dryden",
            "enabled": bool(self.enabled),
            "w20_mps": round(self.w20_mps, 2),
            "gust_n_mps": round(float(self._last[0]), 3),
            "gust_e_mps": round(float(self._last[1]), 3),
            "gust_d_mps": round(float(self._last[2]), 3),
        }


# --------------------------------------------------------------------------- #
# Closed-form one-sided PSDs (per rad/s), consistent with the filters above.
#   Var = integral over omega in [0, inf) of the one-sided PSD
# This is the standard Dryden convention, so these can be compared directly
# against a one-sided periodogram computed the same way.
# --------------------------------------------------------------------------- #

def dryden_psd_u(omega_radps, airspeed_mps, lu_m, sigma_u):
    a = max(airspeed_mps, 1e-6) / max(lu_m, 1e-9)
    return (2.0 * a / math.pi) * sigma_u * sigma_u / (omega_radps ** 2 + a ** 2)


def dryden_psd_vw(omega_radps, airspeed_mps, l_m, sigma):
    b = max(airspeed_mps, 1e-6) / max(l_m, 1e-9)
    cc = b / math.sqrt(3.0)
    q = 3.0 * b * sigma * sigma / math.pi
    return q * (omega_radps ** 2 + cc ** 2) / ((omega_radps ** 2 + b ** 2) ** 2)


def _selftest():
    """Check realised variance against sigma^2 and the spectrum against theory.

    The variance check uses a long record for statistics. The spectral check
    averages a small band of bins around each probe frequency, because a single
    periodogram bin has enormous variance and would fail on noise alone.
    """
    dt = 0.02
    v = 42.0
    agl = 250.0
    w20 = 9.0
    n = 400_000

    turb = DrydenTurbulence(w20_mps=w20, seed=7)
    out = np.empty((n, 3))
    for i in range(n):
        out[i] = turb.step(dt, v, agl)

    lu, lv, lw, su, sv, sw = dryden_parameters(agl, w20)
    sigmas = (su, sv, sw)
    labels = ("u (north)", "v (east)", "w (down)")

    print(f"V={v:.0f} m/s  AGL={agl:.0f} m  W20={w20:.0f} m/s  dt={dt}  N={n}")
    print(f"scale lengths: Lu={lu:.1f} Lv={lv:.1f} Lw={lw:.1f} m")
    print(f"\n{'axis':<12}{'sigma_theory':>14}{'sigma_meas':>12}{'ratio':>9}   target 1.00")
    ok = True
    for i, (label, sigma) in enumerate(zip(labels, sigmas)):
        measured = float(np.std(out[:, i]))
        ratio = measured / sigma
        good = abs(ratio - 1.0) < 0.08
        ok = ok and good
        print(f"{label:<12}{sigma:>14.4f}{measured:>12.4f}{ratio:>9.3f}"
              f"{'' if good else '   <-- OFF'}")

    # Segment-averaged (Welch) PSD. A single periodogram bin has ~100% standard
    # deviation, so no small band of raw bins can resolve the theoretical
    # spectrum; averaging hundreds of half-overlapping windowed segments brings
    # the scatter down to a few percent and makes the comparison meaningful.
    seg = 2048
    step = seg // 2
    n_seg = (n - seg) // step + 1
    win = np.hanning(seg)
    win_norm = float((win ** 2).sum())
    omega = np.fft.rfftfreq(seg, dt) * 2.0 * math.pi

    print(f"\n{'omega rad/s':>12}{'PSD theory':>14}{'PSD meas':>14}{'ratio':>9}"
          f"   (one-sided, per rad/s)")
    for axis, (label, l_m, sigma) in enumerate((("u", lu, su), ("v", lv, sv), ("w", lw, sw))):
        x = out[:, axis] - out[:, axis].mean()
        acc = np.zeros(len(omega))
        for s in range(n_seg):
            chunk = x[s * step:s * step + seg] * win
            acc += np.abs(np.fft.rfft(chunk)) ** 2
        # one-sided PSD per rad/s, averaged over segments
        psd = (2.0 * dt / (win_norm * n_seg)) * acc / (2.0 * math.pi)

        for target in (0.2, 0.5, 1.0, 2.0, 4.0):
            idx = int(np.argmin(np.abs(omega - target)))
            meas = float(psd[idx])
            if axis == 0:
                theory = dryden_psd_u(omega[idx], v, l_m, sigma)
            else:
                theory = dryden_psd_vw(omega[idx], v, l_m, sigma)
            r = meas / theory if theory > 0 else float("nan")
            good = abs(r - 1.0) < 0.25
            ok = ok and good
            print(f"{omega[idx]:>12.3f}{theory:>14.6f}{meas:>14.6f}{r:>9.3f}"
                  f"  {label}{'' if good else '   <-- OFF'}")

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
