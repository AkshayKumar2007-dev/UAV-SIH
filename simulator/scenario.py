"""Declarative scenario files for the UAV simulator.

A scenario is a small YAML (or JSON) document that overrides parts of
`simulator.config`. Scenarios exist so a test flight -- a heavy crosswind, a
degraded engine, a relocated airfield -- can be reproduced from a file instead
of editing constants, and so several can be run without the process restarting.

Overrides are applied IN PLACE to the config dictionaries. Every module does
`from simulator.config import WIND` (etc.), which binds the dict object rather
than a copy, so mutating the dict here is visible everywhere without a reload.
Rebinding the module-level name would not be, which is why the section
dictionaries are never reassigned -- only their contents are merged.

Top-level schema
----------------
The top-level keys are the config sections, plus three reserved keys:

    name / description   metadata, echoed back in the apply manifest
    engine_time_scale    float; multiplies the engine-aging clock
    scenery              {regenerate: bool, seed: int} -- rebuild trees and
                         buildings after any `world` edit, using the current
                         airport layout for the keep-out zones

Any other top-level key is rejected, as is an unknown key inside a section:
a typo'd override that silently did nothing would be worse than an error.
"""

import json
import os

import numpy as np

import simulator.config as config

try:
    import yaml
except ImportError:      # JSON scenarios still work without PyYAML
    yaml = None


class ScenarioError(Exception):
    """A malformed scenario file, or an override that targets nothing."""


# Scenario section -> attribute name on the config module.
_SECTIONS = {
    "home": "HOME",
    "initial_conditions": "INITIAL_CONDITIONS",
    "airframe": "AIRFRAME",
    "aero": "AERO",
    "engine": "PISTON_ENGINE",
    "wind": "WIND",
    "world": "WORLD",
    "sensors": "SENSORS",
    "pid": "PID",
    "faults": "FAULTS",
    "crash": "CRASH",
    "guidance": "GUIDANCE",
    "ui": "UI",
}

_RESERVED = {"name", "description", "engine_time_scale", "scenery"}

# INITIAL_CONDITIONS holds numpy vectors; a scenario supplies plain lists.
_VECTOR_KEYS = ("pos_ned_m", "vel_body_mps", "euler_rad", "rates_radps")

_DEFAULT_SCENERY_SEED = 20240905

SCENARIO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scenarios")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def deep_update(dst, src, path, changes):
    """Merge `src` into dict `dst` in place.

    Nested dicts merge recursively; scalars, lists and tuples replace. A key
    that is a dict on one side and not the other is an error rather than a
    silent overwrite -- that mismatch almost always means the scenario and the
    config have drifted apart, and quietly clobbering a whole subtree would
    hide it.
    """
    for key, value in src.items():
        here = f"{path}.{key}"
        if key not in dst:
            raise ScenarioError(f"{here}: not a known config key")
        current = dst[key]
        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise ScenarioError(
                    f"{here}: expected a mapping of sub-keys, got "
                    f"{type(value).__name__}")
            deep_update(current, value, here, changes)
        else:
            dst[key] = value
            changes.append(f"{here} = {_fmt(value)}")
    return dst


def apply(spec, config_module=None):
    """Apply a scenario mapping to the config dictionaries in place.

    Returns a manifest: {name, description, changes}. Raises ScenarioError on
    an unknown key or a type mismatch. The config module is injectable so the
    loader can be tested against a throwaway copy.
    """
    if not isinstance(spec, dict):
        raise ScenarioError("scenario root must be a mapping")

    cfg = config_module or config

    unknown = sorted(set(spec) - set(_SECTIONS) - _RESERVED)
    if unknown:
        valid = ", ".join(sorted(set(_SECTIONS) | _RESERVED))
        raise ScenarioError(
            f"unknown scenario key(s): {', '.join(unknown)} (valid: {valid})")

    changes = []
    for section, cfg_name in _SECTIONS.items():
        if section not in spec:
            continue
        body = spec[section]
        if not isinstance(body, dict):
            raise ScenarioError(f"{section}: expected a mapping")
        target = getattr(cfg, cfg_name)
        deep_update(target, body, cfg_name, changes)
        if section == "initial_conditions":
            for key in _VECTOR_KEYS:
                if key in target:
                    target[key] = np.asarray(target[key], dtype=float)

    if "engine_time_scale" in spec:
        scale = spec["engine_time_scale"]
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise ScenarioError("engine_time_scale must be a number")
        if scale <= 0.0:
            raise ScenarioError("engine_time_scale must be positive")
        cfg.ENGINE_TIME_SCALE = float(scale)
        changes.append(f"ENGINE_TIME_SCALE = {_fmt(float(scale))}")

    scenery = spec.get("scenery")
    if scenery is not None:
        if not isinstance(scenery, dict):
            raise ScenarioError("scenery: expected a mapping")
        if scenery.get("regenerate", True):
            stats = cfg.build_scenery(int(scenery.get("seed", _DEFAULT_SCENERY_SEED)))
            changes.append(
                f"scenery regenerated ({stats['trees']} trees, "
                f"{stats['buildings']} buildings)")

    return {
        "name": str(spec.get("name", "unnamed")),
        "description": str(spec.get("description", "")),
        "changes": changes,
    }


def load(path):
    """Read a scenario file. Format is chosen by extension (.yaml/.yml/.json)."""
    if not os.path.isfile(path):
        raise ScenarioError(f"no such scenario file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        data = json.loads(text)
    elif ext in (".yaml", ".yml"):
        if yaml is None:
            raise ScenarioError(
                f"{path}: PyYAML is required to read .yaml scenarios")
        data = yaml.safe_load(text)
    else:
        raise ScenarioError(
            f"{path}: unsupported extension {ext!r} (use .yaml, .yml or .json)")

    if data is None:
        raise ScenarioError(f"{path}: scenario file is empty")
    return data


def apply_file(path, config_module=None):
    """Load and apply a scenario file; returns the apply manifest."""
    return apply(load(path), config_module=config_module)


def list_bundled(directory=None):
    """Bundled scenarios as sorted (name, path) pairs."""
    directory = directory or SCENARIO_DIR
    if not os.path.isdir(directory):
        return []
    found = []
    for entry in sorted(os.listdir(directory)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in (".yaml", ".yml", ".json"):
            found.append((os.path.splitext(entry)[0], os.path.join(directory, entry)))
    return found


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("bundled scenarios:")
        for name, path in list_bundled():
            print(f"  {name:<28} {path}")
        raise SystemExit(0)

    try:
        manifest = apply_file(sys.argv[1])
    except ScenarioError as exc:
        print(f"scenario error: {exc}")
        raise SystemExit(2)

    print(f"applied '{manifest['name']}' from {sys.argv[1]}")
    if manifest["description"]:
        print(f"  {manifest['description']}")
    for line in manifest["changes"]:
        print(f"  - {line}")
    print(f"{len(manifest['changes'])} override(s) applied")
