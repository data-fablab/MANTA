"""Centralized YAML configuration loader.

Loads all pipeline parameters from a single YAML file and constructs
the corresponding dataclass instances.
"""

from dataclasses import dataclass, field
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


# ── New config dataclasses ──────────────────────────────────────────────────

@dataclass
class AeroModelConfig:
    """Aerodynamic model tuning parameters."""
    re_clip_min: float = 50_000
    re_clip_max: float = 500_000
    alpha_clip_min: float = -5.0
    alpha_clip_max: float = 15.0
    cd_clip_min: float = 0.003
    cd_clip_max: float = 0.10
    interference_wing: float = 1.05
    interference_body: float = 1.10
    wetted_area_ratio: float = 2.05
    k_3d: float = 2.5
    k_3d_tc_threshold: float = 0.12
    alpha_trim_clip_min: float = -2.0
    alpha_trim_clip_max: float = 12.0


@dataclass
class DuctGeometryConfig:
    """Duct geometry placement and sizing."""
    intake_x_frac: float = 0.04
    fan_x_frac: float = 0.40
    exhaust_x_frac: float = 0.93
    intake_width_factor: float = 1.8
    intake_depth_factor: float = 0.5
    intake_ramp_factor: float = 3.0
    nozzle_area_ratio: float = 0.80
    duct_wall_mm: float = 1.0
    housing_length_factor: float = 1.5
    material_density: float = 1500.0
    structural_adder: float = 1.30


@dataclass
class DuctAeroConfig:
    """Duct aerodynamic performance parameters."""
    pr_intake: float = 0.95
    k_sduct: float = 0.15
    pr_nozzle: float = 0.98
    cd_intake_base: float = 0.05
    static_capture_factor: float = 1.2


@dataclass
class StructureConfig:
    """Structural mass model parameters."""
    skin_density_kg_m2: float = 0.6
    structural_adder: float = 1.20
    body_tc_correction: float = 0.8
    gyration_radius_factor: float = 0.3


@dataclass
class ManufConfig:
    """Manufacturability scoring weights and thresholds."""
    weights: dict[str, float] = field(default_factory=lambda: {
        "twist_smoothness": 0.15,
        "dihedral_smoothness": 0.10,
        "dihedral_simplicity": 0.10,
        "tip_robustness": 0.15,
        "taper_simplicity": 0.10,
        "blend_smoothness": 0.10,
        "sweep_continuity": 0.05,
        "twist_simplicity": 0.10,
        "equipment_space": 0.10,
        "mold_size": 0.05,
    })
    thresholds: dict[str, list[float]] = field(default_factory=lambda: {
        "twist_gradient": [5.0, 20.0],
        "dihedral_gradient": [30.0, 100.0],
        "dihedral_breaks": [0.0, 3.0],
        "tip_thickness_mm": [8.0, 3.0],
        "taper_ratio": [0.25, 0.08],
        "chord_ratio": [1.3, 2.0],
        "sweep_delta": [3.0, 12.0],
        "twist_range": [3.0, 8.0],
        "volume": [0.004, 0.002],
        "half_span": [0.8, 1.0],
    })


# ── Config loaders ──────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> dict:
    """Load raw YAML config as dict."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def mission_from_config(cfg: dict) -> MissionCondition:
    """Build MissionCondition from config dict."""
    m = cfg.get("mission", {})
    return MissionCondition(
        velocity=m.get("velocity_ms", 25.0),
        altitude=m.get("altitude_m", 100.0),
        mtow=m.get("mtow_kg", 2.5),
        avionics_mass=m.get("avionics_mass_kg", 0.179),
        auxiliary_mass=m.get("auxiliary_mass_kg", 0.080),
        payload_mass=m.get("payload_mass_kg", 0.200),
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


def aero_model_from_config(cfg: dict) -> AeroModelConfig:
    """Build AeroModelConfig from config dict."""
    a = cfg.get("aero_model", {})
    kwargs = {}
    re_clip = a.get("re_clip", [50_000, 500_000])
    kwargs["re_clip_min"] = re_clip[0]
    kwargs["re_clip_max"] = re_clip[1]
    alpha_clip = a.get("alpha_clip", [-5.0, 15.0])
    kwargs["alpha_clip_min"] = alpha_clip[0]
    kwargs["alpha_clip_max"] = alpha_clip[1]
    cd_clip = a.get("cd_clip", [0.003, 0.10])
    kwargs["cd_clip_min"] = cd_clip[0]
    kwargs["cd_clip_max"] = cd_clip[1]
    trim_clip = a.get("alpha_trim_clip", [-2.0, 12.0])
    kwargs["alpha_trim_clip_min"] = trim_clip[0]
    kwargs["alpha_trim_clip_max"] = trim_clip[1]
    for key in ["interference_wing", "interference_body", "wetted_area_ratio",
                "k_3d", "k_3d_tc_threshold"]:
        if key in a:
            kwargs[key] = a[key]
    return AeroModelConfig(**kwargs)


def duct_from_config(cfg: dict) -> DuctGeometryConfig:
    """Build DuctGeometryConfig from config dict."""
    d = cfg.get("duct", {})
    kwargs = {}
    for key in DuctGeometryConfig.__dataclass_fields__:
        if key in d:
            kwargs[key] = d[key]
    return DuctGeometryConfig(**kwargs)


def duct_aero_from_config(cfg: dict) -> DuctAeroConfig:
    """Build DuctAeroConfig from config dict."""
    d = cfg.get("duct_aero", {})
    kwargs = {}
    for key in DuctAeroConfig.__dataclass_fields__:
        if key in d:
            kwargs[key] = d[key]
    return DuctAeroConfig(**kwargs)


def structure_from_config(cfg: dict) -> StructureConfig:
    """Build StructureConfig from config dict."""
    s = cfg.get("structure", {})
    kwargs = {}
    for key in StructureConfig.__dataclass_fields__:
        if key in s:
            kwargs[key] = s[key]
    return StructureConfig(**kwargs)


def manuf_from_config(cfg: dict) -> ManufConfig:
    """Build ManufConfig from config dict."""
    m = cfg.get("manufacturability", {})
    kwargs = {}
    if "weights" in m:
        kwargs["weights"] = m["weights"]
    if "thresholds" in m:
        kwargs["thresholds"] = m["thresholds"]
    return ManufConfig(**kwargs)


def load_all(path: str | Path | None = None) -> dict[str, Any]:
    """Load config and build all dataclass instances.

    Returns dict with keys: 'raw', 'mission', 'feasibility', 'cg', 'controls',
    'aero_model', 'duct', 'duct_aero', 'structure', 'manufacturability',
    'avl_command'.
    """
    cfg = load_config(path)
    return {
        "raw": cfg,
        "mission": mission_from_config(cfg),
        "feasibility": feasibility_from_config(cfg),
        "cg": cg_from_config(cfg),
        "controls": controls_from_config(cfg),
        "aero_model": aero_model_from_config(cfg),
        "duct": duct_from_config(cfg),
        "duct_aero": duct_aero_from_config(cfg),
        "structure": structure_from_config(cfg),
        "manufacturability": manuf_from_config(cfg),
        "avl_command": cfg.get("avl", {}).get("command", "avl"),
    }
