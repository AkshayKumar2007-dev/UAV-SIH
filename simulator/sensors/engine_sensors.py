import numpy as np
from simulator.config import SENSORS


class EngineSensors:
    def __init__(self):
        self.health = "OK"

    def reading(self, eng_true):
        cfg = SENSORS["engine"]
        rpm = float(eng_true["rpm"] + np.random.randn() * eng_true["rpm"] * (cfg["rpm_noise_std_pct"] / 100.0))
        ff = float(max(0.0, eng_true["fuel_flow_Lph"] + np.random.randn() * cfg["fuel_flow_noise_std_Lph"]))
        fq = float(max(0.0, eng_true["fuel_L"] + np.random.randn() * cfg["fuel_qty_noise_std_L"]))
        cht = float(eng_true["cht_C"] + np.random.randn() * cfg["cht_noise_std_c"])
        egt = float(eng_true["egt_C"] + np.random.randn() * cfg["egt_noise_std_c"])
        oil = float(max(0.0, eng_true["oil_psi"] + np.random.randn() * cfg["oil_p_noise_std_psi"]))
        return {
            "rpm": rpm,
            "fuel_flow_Lph": ff,
            "fuel_qty_L": fq,
            "cht_C": cht,
            "egt_C": egt,
            "oil_psi": oil,
            "health": self.health,
            "raw_flags": list(eng_true.get("health_flags", [])),
        }
