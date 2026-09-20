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

Optional MAVLink bridge (requires `pymavlink`):

```bash
python -m simulator.main --mavlink udpout:127.0.0.1:14550
```

This streams HEARTBEAT / ATTITUDE / GLOBAL_POSITION_INT / VFR_HUD at 10 Hz so a ground
station (e.g. Mission Planner or QGroundControl) can watch the simulated aircraft. UDP
needs no extra setup; a serial port URL like `/dev/ttyUSB0` or `COM7` works too.

By default the link is **telemetry-only**: the ground station can watch but cannot command
the aircraft, matching the simulator's advisory-only design. Pass `--mavlink-commands` to
also accept arm/disarm, mode changes and manual control from the ground station — this is
announced in the on-screen advisories when enabled.

Then open the dashboard at http://127.0.0.1:8766 (WebSocket telemetry on
`ws://127.0.0.1:8765`). The Pygame window accepts manual control input and mirrors the
simulated state; the dashboard renders it in 3D.

## AI copilot (LLM)

The chat assistant and the "AI suggestions" panel run through an LLM when one is available,
and silently fall back to the offline rule engine otherwise — the simulator never *needs*
any external service. The LLM is text-only by design: it answers questions and drafts
suggestion cards; it has no connection to the aircraft controls.

**Defaults to a local Ollama server.** With Ollama running and a model pulled, nothing more
is needed:

```bash
ollama pull qwen2.5:3b        # or any model you like
cd UAV
python -m simulator.main       # copilot auto-detects http://127.0.0.1:11434/v1
```

Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, vLLM, LM Studio, …) via env vars
or CLI flags:

```bash
# env (read at import)
export CYBERSPARKS_LLM_BASE="https://api.openai.com/v1"
export CYBERSPARKS_LLM_MODEL="gpt-4o-mini"
export CYBERSPARKS_LLM_KEY="sk-..."
python -m simulator.main

# or per-run flags (override env)
python -m simulator.main --llm-url http://127.0.0.1:1234/v1 --llm-model llama-3.2-3b
python -m simulator.main --no-llm   # force the offline rule engine
```

The dashboard headers show a badge — green `LLM · <model>` when the model is answering,
amber `offline rules` when the fallback is active. Chat replies stay on the same WebSocket
and the LLM call never blocks the 10 Hz telemetry stream; suggestion cards are re-drafted
by the model at most every 15 s on a background thread.

## AI agents (MCP server)

`simulator/mcp_server.py` bridges the running simulator to any AI agent that speaks the
Model Context Protocol (Claude Desktop, `claude code`, Cursor, or a custom MCP client):

```bash
python -m simulator.main                # terminal 1 — the simulator
python -m simulator.mcp_server          # terminal 2 — stdio transport (for MCP clients)
python -m simulator.mcp_server --http 8767   # or HTTP JSON-RPC for network agents
```

The bridge is a thin client of the simulator's existing WebSocket protocol — no simulator
core changes are needed. Tools exposed:

| Tool | What it does |
|------|--------------|
| `get_status` | Mode, sim time, crash state, overall health, AI backend (LLM vs rules) |
| `get_telemetry` | Latest live snapshot: flight, engine, GPS, wind, AI assessment, twin/ML |
| `ask_copilot` | Chat with the AI flight assistant (LLM, falling back to rules); `verbose` returns provenance |
| `get_ai_suggestions` | Read the live "AI suggestions" panel: cards, active alerts, overall health |
| `get_chat_log` | Read the shared chat transcript (dashboard + MCP turns), newest last |
| `inject_fault` | carb_ice / oil_leak / stress / seizure / clear_all (dashboard power) |
| `reset_sim` | Reset to the runway (dashboard power) |
| `list_scenarios` | List bundled YAML/JSON scenario files |

The chat and suggestions panels are published over MCP. `ask_copilot` and the dashboard
chatbox share one transcript: a question asked from an MCP agent appears in the dashboard
chat (tagged `[MCP]`) and vice versa, and `get_chat_log` returns the merged history. When a
bridge is attached, the dashboard's chat and suggestions headers show an `MCP · published`
badge. Read-only surface: `get_ai_suggestions` and `get_chat_log` can only observe.

To point Claude Desktop at it, add to `claude_desktop_config.json` (adjust `cwd`):

```json
{
  "mcpServers": {
    "cybersparks-uav-sim": {
      "command": "python",
      "args": ["-m", "simulator.mcp_server"],
      "cwd": "/full/path/to/UAV"
    }
  }
}
```

The trust model is inherited unchanged: the copilot can only answer and suggest (no
flight-control path), and fault injection / reset are exactly the dashboard's powers.

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
python -m simulator._llmtest
python -m simulator._mcptest
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
| `_llmtest` | LLM copilot: rules fallback, suggestion parsing, fake OpenAI endpoint |
| `_mcptest` | MCP server: protocol unit tests + end-to-end against a spawned sim |
| `mavlink_interface` self-test | MAVLink frame encoding without a listener |

`_debug_main.py` is a development entry point that runs the loop headless for inspection.

`mavlink_interface.py` is the MAVLink bridge (telemetry-out by default, opt-in command-in).
Its `__main__` block is a self-test that encodes a few frames without needing a listener.

## Safety note

The AI advisor is read-only with respect to flight controls. Corrective authority stays with
the autopilot and the human operator.