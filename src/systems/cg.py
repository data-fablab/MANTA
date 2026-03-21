"""Aircraft center-of-gravity computation from component placement.

Computes CG from structural mass distribution + discrete components
(battery, EDF, avionics). Replaces the hardcoded x_ref = 0.30 * body_chord.
"""

import numpy as np
from dataclasses import dataclass, field
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    compute_wing_area, compute_mac, compute_aspect_ratio,
    estimate_structural_mass, OUTER_WING_STATIONS,
)


@dataclass
class ComponentPlacement:
    """A discrete component with mass and position in body frame."""
    name: str
    mass: float       # [kg]
    x_cg: float       # [m] from body LE (positive aft)
    y_cg: float = 0.0 # [m] lateral (0 for symmetric)
    z_cg: float = 0.0 # [m] vertical (positive up)


@dataclass
class CGConfig:
    """Configuration for component placement.

    Positions are expressed as fractions of body_root_chord (x/c_body),
    so they scale naturally with the design.
    """
    # Battery position: fraction of body_root_chord from LE
    battery_xc: float = 0.22       # forward for CG control (typical flying wing)

    # EDF motor: at fan face position
    edf_xc: float = 0.40           # fan face location in body

    # Avionics (receiver, FC, GPS, ESC, wiring): forward bay
    avionics_xc: float = 0.28

    # Servo mass per pair [kg] (2 servos for elevons + 2 for ailerons)
    servo_mass: float = 0.040      # total for all servos (~10g each × 4)
    servo_xc: float = 0.70         # near trailing edge (at hinge line)


DEFAULT_CG_CONFIG = CGConfig()


def compute_cg(
    params: BWBParams,
    battery_mass: float,
    edf_mass: float,
    avionics_mass: float,
    config: CGConfig | None = None,
    structure_config=None,
) -> dict:
    """Compute aircraft CG from structural mass distribution + components.

    Parameters
    ----------
    params : BWBParams
        Aircraft geometry parameters.
    battery_mass, edf_mass, avionics_mass : float
        Component masses [kg] from MissionCondition.
    config : CGConfig, optional
        Component placement configuration.

    Returns
    -------
    dict with x_cg, y_cg, z_cg [m], total_mass [kg], components list,
    and x_cg_frac (x_cg / body_root_chord for quick SM check).
    """
    from ..config import StructureConfig
    config = config or DEFAULT_CG_CONFIG
    if structure_config is None:
        structure_config = StructureConfig()
    body_chord = params.body_root_chord
    bw = params.body_halfwidth

    # ── Structural mass split: body shell vs wing shell + spar ──
    # Compute mass fractions from wetted area ratio (same logic as
    # estimate_structural_mass, but split into body/wing contributions)
    from ..parameterization.bwb_aircraft import build_kulfan_airfoil_at_station

    root_kaf = build_kulfan_airfoil_at_station(params, 0.0)
    tip_kaf = build_kulfan_airfoil_at_station(params, 1.0)
    avg_tc_wing = (root_kaf.max_thickness() + tip_kaf.max_thickness()) / 2
    avg_tc_body = params.body_tc_root * structure_config.body_tc_correction

    wing_wetted = 2.05 * (params.wing_root_chord + params.tip_chord) * params.outer_half_span
    body_wetted = 2.05 * (body_chord + params.wing_root_chord) * bw

    density_kg_m2 = structure_config.skin_density_kg_m2
    shell_wing = density_kg_m2 * wing_wetted * (1 + avg_tc_wing)
    shell_body = density_kg_m2 * body_wetted * (1 + avg_tc_body)

    total_struct = estimate_structural_mass(params)
    shell_total = shell_wing + shell_body
    # Spar mass and adder go to wing
    adder = structure_config.structural_adder
    spar_and_rib = total_struct - shell_total * adder / adder  # remove adder for split
    # Approximate: body gets shell_body, wing gets shell_wing + spar
    body_struct = shell_body * adder
    wing_struct = total_struct - body_struct

    components = []

    # ── Body structure CG ──
    # Body shell centroid: ~40% of body root chord (thickest region)
    components.append(ComponentPlacement(
        "body_structure", body_struct,
        x_cg=body_chord * 0.40,
    ))

    # ── Wing structure CG ──
    # Wing structure centroid at MAC aerodynamic center location
    # MAC station is at ~36% of outer semi-span for typical taper ratios
    mac = compute_mac(params)
    lam = params.taper_ratio
    # Spanwise position of MAC (from body edge)
    y_mac = bw + params.outer_half_span * (1 + 2 * lam) / (3 * (1 + lam))
    # LE x-position at MAC station
    eta_mac = (y_mac - bw) / params.outer_half_span if params.outer_half_span > 0 else 0.0
    sweep_rad = np.radians(params.le_sweep_deg)
    body_sweep_rad = np.radians(params.body_sweep_deg)
    x_le_blend = bw * np.tan(body_sweep_rad)
    x_le_mac = x_le_blend + eta_mac * params.outer_half_span * np.tan(sweep_rad)
    # Wing structure CG at 40% MAC behind LE at MAC station
    components.append(ComponentPlacement(
        "wing_structure", wing_struct,
        x_cg=x_le_mac + 0.40 * mac,
        y_cg=y_mac,
    ))

    # ── Battery ──
    components.append(ComponentPlacement(
        "battery", battery_mass,
        x_cg=body_chord * config.battery_xc,
    ))

    # ── EDF motor ──
    components.append(ComponentPlacement(
        "edf_motor", edf_mass,
        x_cg=body_chord * config.edf_xc,
    ))

    # ── Duct structure ──
    try:
        from ..propulsion.duct_geometry import (
            compute_duct_placement, compute_duct_structure_mass,
        )
        from ..propulsion.edf_model import EDF_70MM
        placement = compute_duct_placement(params, EDF_70MM)
        duct_mass = compute_duct_structure_mass(placement)
        components.append(ComponentPlacement(
            "duct_structure", duct_mass,
            x_cg=body_chord * placement.fan_x_frac,  # CG near fan face
        ))
    except Exception:
        pass  # no propulsion module or duct computation failed

    # ── Avionics (detailed BOM or placeholder) ──
    try:
        from .avionics import AvionicsSpec
        avionics_spec = AvionicsSpec.default_bwb()
        for ac in avionics_spec.components:
            components.append(ComponentPlacement(
                ac.name, ac.mass_g / 1000.0,
                x_cg=body_chord * ac.position_xc,
            ))
    except ImportError:
        # Fallback to placeholder
        components.append(ComponentPlacement(
            "avionics", avionics_mass,
            x_cg=body_chord * config.avionics_xc,
        ))
        if config.servo_mass > 0:
            components.append(ComponentPlacement(
                "servos", config.servo_mass,
                x_cg=body_chord * config.servo_xc,
            ))

    # ── Weighted CG computation ──
    total_mass = sum(c.mass for c in components)
    if total_mass <= 0:
        return {"x_cg": body_chord * 0.30, "y_cg": 0.0, "z_cg": 0.0,
                "total_mass": 0.0, "components": components, "x_cg_frac": 0.30}

    x_cg = sum(c.mass * c.x_cg for c in components) / total_mass
    y_cg = sum(c.mass * c.y_cg for c in components) / total_mass
    z_cg = sum(c.mass * c.z_cg for c in components) / total_mass

    return {
        "x_cg": x_cg,
        "y_cg": y_cg,
        "z_cg": z_cg,
        "total_mass": total_mass,
        "x_cg_frac": x_cg / body_chord if body_chord > 0 else 0.0,
        "components": components,
    }


def compute_mass_budget(
    params: BWBParams,
    battery_mass: float,
    edf_mass: float,
    avionics_mass: float,
    mtow: float = 1.5,
    config: CGConfig | None = None,
    structure_config=None,
) -> dict:
    """Detailed mass budget with margins and reserves.

    Returns hierarchical dict with category subtotals, margins, and
    comparison to MTOW.
    """
    from ..config import StructureConfig
    config = config or DEFAULT_CG_CONFIG
    if structure_config is None:
        structure_config = StructureConfig()
    cg_result = compute_cg(params, battery_mass, edf_mass, avionics_mass, config, structure_config)

    # Split structural mass
    total_struct = estimate_structural_mass(params, mtow=mtow)

    from ..parameterization.bwb_aircraft import build_kulfan_airfoil_at_station
    root_kaf = build_kulfan_airfoil_at_station(params, 0.0)
    tip_kaf = build_kulfan_airfoil_at_station(params, 1.0)
    avg_tc_wing = (root_kaf.max_thickness() + tip_kaf.max_thickness()) / 2
    avg_tc_body = params.body_tc_root * structure_config.body_tc_correction
    body_chord = params.body_root_chord

    wing_wetted = 2.05 * (params.wing_root_chord + params.tip_chord) * params.outer_half_span
    body_wetted = 2.05 * (body_chord + params.wing_root_chord) * params.body_halfwidth

    density_kg_m2 = structure_config.skin_density_kg_m2
    shell_wing = density_kg_m2 * wing_wetted * (1 + avg_tc_wing)
    shell_body = density_kg_m2 * body_wetted * (1 + avg_tc_body)
    adder = structure_config.structural_adder
    body_struct = shell_body * adder
    wing_struct = total_struct - body_struct

    # Itemized budget
    items = {
        "airframe": {
            "body_shell": {"mass_kg": shell_body, "margin": adder - 1.0},
            "wing_shell": {"mass_kg": shell_wing, "margin": adder - 1.0},
            "spar_ribs_adhesive": {"mass_kg": wing_struct - shell_wing * adder, "margin": 0.10},
            "subtotal": total_struct,
        },
        "propulsion": {
            "edf_motor": {"mass_kg": edf_mass, "margin": 0.0},
            "battery": {"mass_kg": battery_mass, "margin": 0.0},
            "subtotal": edf_mass + battery_mass,
        },
        "systems": {
            "avionics": {"mass_kg": avionics_mass, "margin": 0.10},
            "servos": {"mass_kg": config.servo_mass, "margin": 0.05},
            "wiring_connectors": {"mass_kg": 0.020, "margin": 0.15},
            "subtotal": avionics_mass + config.servo_mass + 0.020,
        },
    }

    # Totals
    dry_mass = sum(cat["subtotal"] for cat in items.values())
    reserve = mtow - dry_mass
    margin_pct = reserve / mtow * 100 if mtow > 0 else 0

    items["summary"] = {
        "dry_mass_kg": dry_mass,
        "mtow_kg": mtow,
        "reserve_kg": reserve,
        "margin_pct": margin_pct,
    }

    return items
