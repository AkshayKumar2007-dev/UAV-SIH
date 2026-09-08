import numpy as np
from simulator.config import SENSORS, G


class IMU:
    def __init__(self):
        cfg = SENSORS["imu"]
        self.rate_hz = cfg["rate_hz"]
        self.dt = 1.0 / self.rate_hz
        self._timer = 0.0
        self._new_data = False
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._accel = np.zeros(3)
        self._gyro = np.zeros(3)
        self._temp_c = 25.0
        self.health = "OK"

    def step(self, dt, true_accel_body, true_rates_body):
        self._timer += dt
        self._new_data = False
        cfg = SENSORS["imu"]
        tau_bias = cfg["bias_tau_s"]
        sigma_a_b = cfg["accel_bias_stability_mps2"]
        sigma_g_b = cfg["gyro_bias_stability_radps"]
        self._accel_bias += (-self._accel_bias / tau_bias + np.random.randn(3) * sigma_a_b * np.sqrt(2.0 / tau_bias) * dt)
        self._gyro_bias += (-self._gyro_bias / tau_bias + np.random.randn(3) * sigma_g_b * np.sqrt(2.0 / tau_bias) * dt)

        if self._timer >= self.dt:
            self._timer -= self.dt
            self._accel = true_accel_body + self._accel_bias + np.random.randn(3) * cfg["accel_noise_std_mps2"]
            self._gyro = true_rates_body + self._gyro_bias + np.random.randn(3) * cfg["gyro_noise_std_radps"]
            self._new_data = True
            self.health = "OK"

    @property
    def new_data(self):
        return self._new_data

    def reading(self):
        return {
            "accel_mps2": self._accel.copy(),
            "gyro_radps": self._gyro.copy(),
            "temp_c": float(self._temp_c),
            "bias_accel_mps2": self._accel_bias.copy(),
            "bias_gyro_radps": self._gyro_bias.copy(),
            "health": self.health,
            "new_data": bool(self._new_data),
        }
