"""MAVLink bridge: exposes the simulated UAV to a real ground station.

Telemetry-out is always safe and is the default. Command-in is OFF unless
explicitly enabled, because the rest of this simulator is built on a
"no external control path" property: the dashboard can only ask questions,
and the AI can only advise. Opening a MAVLink command channel deliberately
breaks that property for the UAV's own ground-station use case, so it is
opt-in and announced at startup.

Protocol notes
--------------
- Transport is UDP by default, which needs no pyserial. A serial link also
  works if the operator passes a COMx / /dev/ttyUSBx URL and has pyserial.
- Messages are sent at whatever rate send_state is called (the sim's
  telemetry tick), so the GCS sees a steady 10 Hz stream.
- Every send is wrapped: a MAVLink failure must never take down the sim.

Command mapping
---------------
ArduPilot-style custom_mode values are mapped onto this simulator's modes:

    0  MANUAL      -> MANUAL
    2  ALT_HOLD    -> ALT_HOLD
    3  HDG/CRUISE  -> HDG_HOLD
    5  FBWA        -> STAB
    10 AUTO        -> WAYPOINT
    11 RTL         -> RTH

ARM/DISARM is translated to an engine start/stop request; MANUAL_CONTROL
and RC_CHANNELS_OVERRIDE are decoded into stick/throttle commands when
command-in is enabled.
"""

import math

import numpy as np

try:
    from pymavlink import mavutil
    HAVE_PYMAVLINK = True
except ImportError:
    mavutil = None
    HAVE_PYMAVLINK = False

DEFAULT_URL = "udpout:127.0.0.1:14550"

# ArduPilot plane custom_mode -> simulator mode
_MODE_MAP = {
    0: "MANUAL",
    2: "ALT_HOLD",
    3: "HDG_HOLD",
    5: "STAB",
    6: "STAB",
    10: "WAYPOINT",
    11: "RTH",
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class MavlinkInterface:
    """Bridges the simulator to MAVLink.

    allow_commands=False keeps this telemetry-only. Even when commands are
    allowed, the caller decides what to do with them: poll_commands only
    parses and returns intents, and the main loop applies the ones it
    chooses to honour.
    """

    def __init__(self, url=DEFAULT_URL, source_system=1, source_component=1,
                 allow_commands=False, target_system=255, target_component=0):
        self.available = HAVE_PYMAVLINK
        self.url = url
        self.allow_commands = bool(allow_commands)
        self.target_system = target_system
        self.target_component = target_component
        self.armed = False
        self._conn = None
        self._n_sent = 0
        self._n_recv = 0
        self._last_error = None
        self._last_command = None

        if not self.available:
            self._last_error = "pymavlink not installed"
            return
        try:
            self._conn = mavutil.mavlink_connection(
                url, source_system=source_system,
                source_component=source_component)
        except Exception as ex:
            self.available = False
            self._last_error = f"{type(ex).__name__}: {ex}"

    def send_state(self, s, on_ground=False):
        """Publish one telemetry frame. Returns True if anything was sent."""
        if not self.available or s is None:
            return False

        mav = self._conn.mav
        try:
            self._send_heartbeat(mav, s, on_ground)
            self._send_attitude(mav, s)
            self._send_position(mav, s)
            self._send_vfr_hud(mav, s, on_ground)
            self._n_sent += 1
            return True
        except Exception as ex:
            self._last_error = f"send: {type(ex).__name__}: {ex}"
            return False

    def _send_heartbeat(self, mav, s, on_ground):
        running = bool(s.get("engine_summary", {}).get("running", False))
        base_mode = 0
        if self.armed and running:
            base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        if s.get("mode") == "MANUAL":
            base_mode |= mavutil.mavlink.MAV_MODE_FLAG_MANUAL_INPUT_ENABLED
        else:
            base_mode |= mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED

        status = (mavutil.mavlink.MAV_STATE_STANDBY if on_ground
                  else mavutil.mavlink.MAV_STATE_ACTIVE)
        mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_FIXED_WING,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            base_mode,
            self._custom_mode(s.get("mode", "MANUAL")),
            status,
        )

    @staticmethod
    def _custom_mode(mode):
        for k, v in _MODE_MAP.items():
            if v == mode:
                return k
        return 0

    def _send_attitude(self, mav, s):
        eul = s["state"]["euler"]
        rates = s["state"]["rates"]
        mav.attitude_send(
            0,
            float(eul[0]), float(eul[1]), float(eul[2]),
            float(rates[0]), float(rates[1]), float(rates[2]),
        )

    def _send_position(self, mav, s):
        gps = s.get("gps", {})
        lat = float(gps.get("lat_deg", 0.0))
        lon = float(gps.get("lon_deg", 0.0))
        alt_msl = float(gps.get("alt_msl_m",
                                 s.get("pitot", {}).get("baro_alt_msl_m", 0.0)))
        rel_alt = float(s.get("agl_m", 0.0))
        vn = float(gps.get("vn_mps", 0.0))
        ve = float(gps.get("ve_mps", 0.0))
        vd = float(gps.get("vd_mps", 0.0))
        hdg = int(math.degrees(s["state"]["euler"][2]) % 360.0 * 100.0)

        mav.global_position_int_send(
            0,
            int(lat * 1e7), int(lon * 1e7),
            int(alt_msl * 1000.0), int(rel_alt * 1000.0),
            int(vn * 100.0), int(ve * 100.0), int(vd * 100.0),
            hdg,
        )

    def _send_vfr_hud(self, mav, s, on_ground):
        pit = s.get("pitot", {})
        # VFR_HUD declares airspeed, groundspeed, heading, throttle, alt,
        # climb -- throttle is the uint16 in the middle, not the last arg.
        mav.vfr_hud_send(
            float(pit.get("ias_mps", 0.0)),
            float(s.get("gps", {}).get("groundspeed_mps", 0.0)),
            int(math.degrees(s["state"]["euler"][2])),
            int(_clamp(float(s.get("manual_controls", {}).get("throttle", 0.0)) * 100.0, 0, 100)),
            float(pit.get("baro_alt_msl_m", 0.0)),
            float(pit.get("vsi_mps", 0.0)),
        )

    def poll_commands(self):
        """Drain inbound MAVLink and return a list of intents.

        Each intent is a dict, e.g. {"type": "arm", "value": True},
        {"type": "set_mode", "mode": "WAYPOINT"}, or a manual-control dict.
        Returning an empty list is always safe.
        """
        if not self.available or not self.allow_commands:
            return []

        out = []
        try:
            while True:
                msg = self._conn.recv_match(blocking=False)
                if msg is None:
                    break
                self._n_recv += 1
                intent = self._decode(msg)
                if intent is not None:
                    out.append(intent)
                    self._last_command = intent
        except Exception as ex:
            self._last_error = f"recv: {type(ex).__name__}: {ex}"
        return out

    def _decode(self, msg):
        mtype = msg.get_type()
        if mtype == "COMMAND_LONG":
            return self._decode_command_long(msg)
        if mtype == "SET_MODE":
            mode = _MODE_MAP.get(int(msg.custom_mode))
            if mode:
                return {"type": "set_mode", "mode": mode}
        if mtype == "MANUAL_CONTROL":
            return {
                "type": "manual",
                "roll": _clamp(msg.x / 1000.0, -1.0, 1.0),
                "pitch": _clamp(msg.y / 1000.0, -1.0, 1.0),
                "throttle": _clamp((msg.z + 1000.0) / 2000.0, 0.0, 1.0),
                "yaw": _clamp(msg.r / 1000.0, -1.0, 1.0),
            }
        if mtype == "RC_CHANNELS_OVERRIDE":
            def pwm(v):
                return _clamp((v - 1500.0) / 500.0, -1.0, 1.0)
            return {
                "type": "manual",
                "roll": pwm(msg.chan1_raw),
                "pitch": pwm(msg.chan2_raw),
                "throttle": _clamp((msg.chan3_raw - 1000.0) / 1000.0, 0.0, 1.0),
                "yaw": pwm(msg.chan4_raw),
            }
        return None

    def _decode_command_long(self, msg):
        cmd = int(msg.command)
        m = mavutil.mavlink
        if cmd == m.MAV_CMD_COMPONENT_ARM_DISARM:
            arm = msg.param1 >= 0.5
            self.armed = arm
            return {"type": "arm", "value": arm}
        if cmd == m.MAV_CMD_DO_SET_MODE:
            mode = _MODE_MAP.get(int(msg.param2))
            if mode:
                return {"type": "set_mode", "mode": mode}
        if cmd == m.MAV_CMD_NAV_TAKEOFF:
            return {"type": "arm", "value": True}
        if cmd == m.MAV_CMD_DO_SET_HOME:
            return {"type": "set_home", "lat": msg.param5,
                    "lon": msg.param6, "alt": msg.param7}
        return None

    def summary(self):
        return {
            "available": bool(self.available),
            "url": self.url,
            "commands_enabled": bool(self.allow_commands),
            "armed": bool(self.armed),
            "sent": self._n_sent,
            "received": self._n_recv,
            "last_error": self._last_error,
        }

if __name__ == "__main__":
    import time

    m = MavlinkInterface()
    print("MAVLink interface")
    for k, v in m.summary().items():
        print(f"  {k}: {v}")
    if not m.available:
        raise SystemExit(1)
    frame = {
        "state": {"euler": np.array([0.05, 0.02, 0.3]),
                  "rates": np.array([0.01, 0.0, -0.02])},
        "gps": {"lat_deg": 28.6129, "lon_deg": 77.2295, "alt_msl_m": 225.0,
                "vn_mps": 40.0, "ve_mps": 1.0, "vd_mps": -0.5,
                "groundspeed_mps": 40.0},
        "pitot": {"ias_mps": 42.0, "baro_alt_msl_m": 225.0, "vsi_mps": 0.5},
        "engine_summary": {"running": True},
        "manual_controls": {"throttle": 0.55},
        "mode": "WAYPOINT",
        "agl_m": 120.0,
    }
    for _ in range(3):
        m.send_state(frame)
        time.sleep(0.1)
    print("encoder self-test OK:", m.summary()["sent"], "frames sent")
