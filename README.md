# UAV Flight Simulator

A 6-DOF fixed-wing UAV flight simulator with a real-time 3D telemetry dashboard and an
engine-health ML advisor.

## Overview

The simulator integrates rigid-body flight dynamics at a fixed 50 Hz physics step against an
aerodynamic model, a turbocharged piston engine, an ISA atmosphere with wind, and sampled
terrain. State is published over a WebSocket to a browser dashboard, and an ML-based health
advisor monitors engine telemetry against a digital twin.

The reference platform is MALE-class (Bayraktar TB2 / Rotax-914 heavy-fuel class): roughly
750 kg MTOW, 14 m span, ~86 kW turbocharged piston, 240 L fuel.

## Features

- **6-DOF rigid-body dynamics** — position/velocity in NED, Euler attitude, body rates, plus
  propeller gyroscopic and p-factor moments.
- **Aerodynamics** — lift, drag and pitching moment from angle of attack and sideslip, with
  control surfaces (aileron, elevator, rudder), flaps, gear drag, stall modelling and
  terrain-relative ground contact (wheel rolling resistance and brakes).
- **Piston engine** — RPM- and density-dependent thrust with fuel burn, thermal state, wear,
  icing and leak modelling. Engine aging runs on a configurable time scale so a full mission
  arc is visible in a short demo; flight dynamics stay real time.
- **Sensors** — GPS, IMU, pitot-static, and engine sensors, with injectable fault models.
- **Flight control** — PID inner loops, autopilot modes (stabilise, altitude hold, waypoint),
  and guidance.
- **AI advisor** — engine digital twin plus ML health monitoring (`ml_core`, `ml_runtime`,
  `train_ml`, `monitor`). Advisory only: it surfaces warnings and suggestions and never
  commands the aircraft.
- **Telemetry** — WebSocket stream and a 3D browser dashboard with HUD.
- **UI** — Pygame window with chase camera and manual input handling.

## Layout

```
simulator/
  config.py            Tunable constants (rates, home, airframe, aero, engine, crash limits)
  main.py              Entry point: physics loop, sensors, autopilot, telemetry, renderer
  ai_advisor/          Engine digital twin and ML health advisor
    ml_core.py           Model definitions and feature handling
    ml_runtime.py        Inference at runtime
    train_ml.py          Offline training
    ml_models.json       Trained model parameters
    monitor.py           HealthMonitor / PilotAssistant
    twin.py              EngineDigitalTwin
  aircraft/
    rigid_body.py        6-DOF integrator
    aerodynamics.py      Force and moment model
    piston_engine.py     Engine and thrust model
    faults.py            Failure injection
  environment/
    atmosphere.py        ISA atmosphere and wind field
    world.py             Terrain, no-fly zones, waypoint missions
  flight_controller/
    pid.py               PID primitives
    autopilot.py         Autopilot modes
    guidance.py          Guidance laws
  sensors/
    gps.py, imu.py, pitot.py, engine_sensors.py
  telemetry/
    server.py            WebSocket + HTTP server
    dashboard.html       3D dashboard client
  ui/
    renderer.py, chase_view.py, input_handler.py
```

## Requirements

- Python 3.10+
- `numpy`, `websockets`, and either `pygame-ce` (Python 3.13+) or `pygame`

```bash
pip install -r requirements.txt
```

## Running

```bash
cd UAV
python -m simulator.main
```

Then open the dashboard at http://127.0.0.1:8766 (WebSocket telemetry on
`ws://127.0.0.1:8765`). The Pygame window accepts manual control input and mirrors the
simulated state; the dashboard renders it in 3D.

## Tests

Run from the `UAV` directory:

```bash
python -m simulator._smoketest
python -m simulator._stabtest
python -m simulator._twintest
python -m simulator._aitest
python -m simulator._faulttest
python -m simulator._crashtest
python -m simulator._trimtest
```

| Test | Covers |
|------|--------|
| `_smoketest` | Startup, physics stepping, clean shutdown |
| `_stabtest` | Autopilot stabilise-mode convergence |
| `_twintest` | Engine digital twin response |
| `_aitest` | Advisor output shape and advisory-only safety |
| `_faulttest` | Injected sensor and engine faults |
| `_crashtest` | Ground impact detection and automatic reset |
| `_trimtest` | Manual-control level-cruise trim |

`_debug_main.py` is a development entry point that runs the loop headless for inspection.

## Safety note

The AI advisor is read-only with respect to flight controls. Corrective authority stays with
the autopilot and the human operator.
</｜｜DSML｜｜ parameter>