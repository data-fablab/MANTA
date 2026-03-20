"""Centralized YAML configuration loader.

Loads all pipeline parameters from a single YAML file and constructs
the corresponding dataclass instances (MissionCondition, FeasibilityConfig,
CGConfig, ControlConfig).
"""

from pathlib import Path
from typing import Any

import yaml

from .aero.mission import MissionCondition, FeasibilityConfig
from .systems.cg import CGConfig
from .aero.avl_runner import ControlSurface, ControlConfig
from .propulsion.edf_model import (
    EDF_70MM, EDF_90MM,
    BATTERY_3S_800, BATTERY_3S_1300, BATTERY_3S_1800, BATTERY_4S_1000,
)

# Lookup tables for presets
_EDF_PRESETS = {"EDF_70MM": EDF_70MM, "EDF_90MM": EDF_90MM}
_BATTERY_PRESETS = {
    "BATTERY_3S_800": BATTERY_3S_800,
    "BATTERY_3S_1300": BATTERY_3S_1300,
    "BATTERY_3S_1800": BATTERY_3S_1800,
    "BATTERY_4S_1000": BATTERY_4S_1000,
}

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load raw YAML config as dict."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def mission_from_config(cfg: dict) -> MissionCondition:
    """Build MissionCondition from config dict."""
    m = cfg.get("mission", {})
    return MissionCondition(
        velocity=m.get("velocity_ms", 20.0),
        altitude=m.get("altitude_m", 100.0),
        mtow=m.get("mtow_kg", 1.5),
        avionics_mass=m.get("avionics_mass_kg", 0.10),
        edf=_EDF_PRESETS.get(m.get("edf", "EDF_70MM"), EDF_70MM),
        battery=_BATTERY_PRESETS.get(m.get("battery", "BATTERY_3S_1800"), BATTERY_3S_1800),
    )


def feasibility_from_config(cfg: dict) -> FeasibilityConfig:
    """Build FeasibilityConfig from config dict."""
    f = cfg.get("feasibility", {})
    w = f.get("weights", {})
    kwargs = {}
    # Map flat keys
    for key in ["sm_min", "sm_max", "cm_max", "cl_max", "cl_min", "ar_min",
                "volume_min", "cn_beta_min", "td_min", "endurance_min",
                "drag_margin", "vs_max", "cl_max_clean", "alpha_trim_max",
                "elevon_deflection_max", "cl_beta_cn_beta_max",
                "manufacturability_min"]:
        if key in f:
            kwargs[key] = f[key]
    # Map weight keys
    for key, val in w.items():
        kwargs[key] = val
    return FeasibilityConfig(**kwargs)


def cg_from_config(cfg: dict) -> CGConfig:
    """Build CGConfig from config dict."""
    c = cfg.get("cg", {})
    return CGConfig(
        battery_xc=c.get("battery_xc", 0.22),
        edf_xc=c.get("edf_xc", 0.40),
        avionics_xc=c.get("avionics_xc", 0.28),
        servo_mass=c.get("servo_mass_kg", 0.040),
        servo_xc=c.get("servo_xc", 0.70),
    )


def controls_from_config(cfg: dict) -> ControlConfig:
    """Build ControlConfig from config dict."""
    ctrl = cfg.get("controls", {})
    surfaces = []
    for name, spec in ctrl.items():
        surfaces.append(ControlSurface(
            name=name,
            cf_ratio=spec.get("cf_ratio", 0.25),
            eta_inner=spec.get("eta_inner", 0.0),
            eta_outer=spec.get("eta_outer", 0.50),
            sgn_dup=spec.get("sgn_dup", 1.0),
            gain=spec.get("gain", 1.0),
        ))
    return ControlConfig(surfaces=surfaces)


def load_all(path: str | Path | None = None) -> dict[str, Any]:
    """Load config and build all dataclass instances.

    Returns dict with keys: 'raw', 'mission', 'feasibility', 'cg', 'controls'.
    """
    cfg = load_config(path)
    return {
        "raw": cfg,
        "mission": mission_from_config(cfg),
        "feasibility": feasibility_from_config(cfg),
        "cg": cg_from_config(cfg),
        "controls": controls_from_config(cfg),
    }
