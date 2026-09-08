import numpy as np


class PID:
    def __init__(self, kp, ki, kd, i_lim, out_lim, d_filter_tau=0.15):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_lim = float(i_lim)
        self.out_lim = float(out_lim)
        self.d_tau = float(d_filter_tau)
        self.integral = 0.0
        self._prev_err = None
        self._d_state = 0.0

    def reset(self):
        self.integral = 0.0
        self._prev_err = None
        self._d_state = 0.0

    def update(self, setpoint, measurement, dt):
        err = float(setpoint) - float(measurement)
        self.integral += err * dt
        self.integral = np.clip(self.integral, -self.i_lim, self.i_lim)
        if self._prev_err is None:
            deriv = 0.0
        else:
            deriv = (err - self._prev_err) / max(dt, 1e-6)
        alpha = 1.0 - np.exp(-dt / self.d_tau)
        self._d_state = (1.0 - alpha) * self._d_state + alpha * deriv
        self._prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * self._d_state
        out_clipped = np.clip(out, -self.out_lim, self.out_lim)
        if self.ki != 0.0 and out != out_clipped and ((self.integral > 0 and err > 0) or (self.integral < 0 and err < 0)):
            self.integral -= err * dt
        return float(out_clipped)
