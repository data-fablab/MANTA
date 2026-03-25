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
                               edf: EDFSpec, battery: BatterySpec,
                               pr_total: float = 1.0,
                               velocity_ratio: float = 1.5,
                               alpha_deg: float = 0.0,
                               x_thrust: float = 0.0,
                               z_thrust: float = 0.0,
                               x_cg: float = 0.0,
                               z_cg: float = 0.0,
                               q_S_c: float = None) -> dict:
    """Full propulsion balance from drag force.

    Works with scalar or numpy array `drag_force`.

    Parameters
    ----------
    pr_total : float
        Duct total pressure recovery (0-1). Default 1.0 = no installation loss.
    velocity_ratio : float
        V_exit / V_freestream (~1.5 for EDF 70mm at 25 m/s).
        Amplifies PR loss: ΔT/T = (1 - PR) × velocity_ratio (Mattingly).
    alpha_deg : float
        Cruise angle of attack [deg]. After nozzle twist compensation,
        thrust is parallel to fuselage axis; the only misalignment with
        the velocity vector is alpha_cruise.
    x_thrust, z_thrust : float
        Exhaust application point [m] in body coordinates.
    x_cg, z_cg : float
        Centre of gravity [m] in body coordinates.
    q_S_c : float or None
        Dynamic pressure × S_ref × MAC [N·m] for CM non-dimensionalisation.

    Returns dict with: T_available, T_over_D, P_elec, endurance_s,
    endurance_min, range_km, T_axial, T_normal, M_thrust, CM_thrust.
    """
    t_uninstalled = thrust_at_speed(edf, velocity)
    installation_loss = (1.0 - pr_total) * velocity_ratio
    t_avail = t_uninstalled * max(1.0 - installation_loss, 0.0)

    alpha_rad = np.radians(alpha_deg)
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)

    # Vectorized path
    if isinstance(drag_force, np.ndarray):
        t_available = np.full_like(drag_force, t_avail)
        t_axial = t_available * cos_a
        t_normal = t_available * sin_a
        t_over_d = np.where(drag_force > 0.01, t_axial / drag_force, np.inf)

        # Pitching moment from thrust offset relative to CG
        m_thrust = (t_available * sin_a * (x_thrust - x_cg)
                    - t_available * cos_a * (z_thrust - z_cg))
        cm_thrust = m_thrust / q_S_c if q_S_c is not None and q_S_c > 0 else np.zeros_like(drag_force)

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
            "T_axial": t_axial,
            "T_normal": t_normal,
            "T_over_D": t_over_d,
            "M_thrust": m_thrust,
            "CM_thrust": cm_thrust,
            "P_elec": p_elec,
            "endurance_s": endurance_s,
            "endurance_min": endurance_min,
            "range_km": range_km,
        }

    # Scalar path — delegate to existing edf_model.endurance()
    prop = _compute_endurance_full(drag_force, velocity, edf, battery)
    prop["T_available"] = t_avail
    t_axial = t_avail * cos_a
    t_normal = t_avail * sin_a
    prop["T_axial"] = t_axial
    prop["T_normal"] = t_normal
    prop["T_over_D"] = t_axial / drag_force if drag_force > 0.01 else float("inf")

    m_thrust = (t_avail * sin_a * (x_thrust - x_cg)
                - t_avail * cos_a * (z_thrust - z_cg))
    prop["M_thrust"] = m_thrust
    prop["CM_thrust"] = m_thrust / q_S_c if q_S_c is not None and q_S_c > 0 else 0.0
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
