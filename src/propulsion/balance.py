"""Propulsion balance — centralized thrust, endurance, intake drag, duct mass.

All propulsion-derived quantities computed in one place.
Supports both scalar (evaluator) and vectorized (surrogate reconstruct) modes.
"""

import numpy as np
from .edf_model import (
    EDFSpec, BatterySpec, EDF_70MM, BATTERY_3S_1800,
    thrust_at_speed, endurance as _compute_endurance_full,
    intake_drag as _intake_drag_raw,
)


def compute_propulsion_balance(drag_force, velocity: float,
                               edf: EDFSpec, battery: BatterySpec) -> dict:
    """Full propulsion balance from drag force.

    Works with scalar or numpy array `drag_force`.

    Returns dict with: T_available, T_over_D, P_elec, endurance_s,
    endurance_min, range_km.
    """
    t_avail = thrust_at_speed(edf, velocity)

    # Vectorized path
    if isinstance(drag_force, np.ndarray):
        t_available = np.full_like(drag_force, t_avail)
        t_over_d = np.where(drag_force > 0.01, t_available / drag_force, np.inf)

        p_thrust = drag_force * velocity
        eta_total = edf.eta_fan * edf.eta_motor
        p_elec = np.where(eta_total > 0, p_thrust / eta_total, np.inf)

        usable_frac = 0.80
        eta_discharge = battery.discharge_efficiency(usable_frac)
        e_battery = battery.energy_j
        endurance_s = np.where(
            p_elec > 0,
            e_battery * usable_frac * eta_discharge / p_elec,
            np.inf,
        )
        endurance_min = endurance_s / 60.0
        range_km = endurance_s * velocity / 1000.0

        return {
            "T_available": t_available,
            "T_over_D": t_over_d,
            "P_elec": p_elec,
            "endurance_s": endurance_s,
            "endurance_min": endurance_min,
            "range_km": range_km,
        }

    # Scalar path — delegate to existing edf_model.endurance()
    prop = _compute_endurance_full(drag_force, velocity, edf, battery)
    prop["T_available"] = t_avail
    prop["T_over_D"] = t_avail / drag_force if drag_force > 0.01 else float("inf")
    return prop


def compute_intake_cd(edf: EDFSpec, s_ref: float) -> float:
    """Intake drag coefficient."""
    return _intake_drag_raw(edf, s_ref)


def compute_duct_mass(params, edf: EDFSpec, config=None) -> float:
    """Duct structure mass [kg] from params + EDF spec."""
    from .duct_geometry import compute_duct_placement, compute_duct_structure_mass
    if config is None:
        from ..config import duct_from_config, load_config
        config = duct_from_config(load_config())
    placement = compute_duct_placement(params, edf, config=config)
    return compute_duct_structure_mass(placement, config=config)


def compute_bump_drag(params, edf: EDFSpec, velocity: float,
                      kinematic_viscosity: float, s_ref: float,
                      config=None) -> tuple[float, dict]:
    """Bump fairing CD0 from duct protrusion above body OML.

    Estimates bump dimensions from clearance data (how much duct
    protrudes above body upper surface).

    Returns (cd0_bump, bump_info_dict).
    """
    from .duct_geometry import compute_duct_placement, validate_duct_clearance
    from ..aero.drag import compute_bump_cd0
    if config is None:
        from ..config import duct_from_config, load_config
        config = duct_from_config(load_config())
    placement = compute_duct_placement(params, edf, config=config)
    _, clr_results = validate_duct_clearance(placement, params, min_clearance_mm=0.0)

    # Bump height = max protrusion above body upper surface
    max_protrusion = 0.0
    bump_start_x = placement.exhaust_x
    bump_end_x = placement.intake_x
    for r in clr_results:
        if r.clearance_top_mm < 0:
            protrusion = -r.clearance_top_mm / 1000.0
            max_protrusion = max(max_protrusion, protrusion)
            bump_start_x = min(bump_start_x, r.x_mm / 1000.0)
            bump_end_x = max(bump_end_x, r.x_mm / 1000.0)

    bump_info = {
        "max_height_mm": max_protrusion * 1000,
        "length_mm": (bump_end_x - bump_start_x) * 1000,
        "width_mm": placement.duct_od * 1000,
    }

    if max_protrusion < 0.001:
        return 0.0, bump_info

    bump_length = bump_end_x - bump_start_x
    bump_width = placement.duct_od
    cd0_bump = compute_bump_cd0(
        max_protrusion, bump_length, bump_width,
        velocity, kinematic_viscosity, s_ref,
    )
    return cd0_bump, bump_info


def compute_duct_clearance(params, edf: EDFSpec, config=None):
    """Duct-body clearance validation.

    Returns (duct_ok, min_clearance_mm, placement).
    """
    from .duct_geometry import (
        compute_duct_placement, validate_duct_clearance,
        min_effective_clearance_mm,
    )
    if config is None:
        from ..config import duct_from_config, load_config
        config = duct_from_config(load_config())
    placement = compute_duct_placement(params, edf, config=config)
    duct_ok, clr = validate_duct_clearance(placement, params, min_clearance_mm=0.0)
    min_clr = min_effective_clearance_mm(clr)
    return duct_ok, min_clr, placement
