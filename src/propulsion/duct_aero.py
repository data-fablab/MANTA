"""Aerodynamic sizing and performance of the propulsion duct system.

Computes pressure losses, mass flow continuity, intake capture area,
and nozzle area ratio for the internal duct system. All calculations
assume incompressible flow (M < 0.1, appropriate for V < 35 m/s).

References:
- Seddon & Goldsmith, "Intake Aerodynamics", 2nd ed. (S-duct losses)
- NACA RM-A7I30: Submerged intake design (NACA scoop geometry)
- Actuator disk theory for fan face velocity estimation
"""

import numpy as np
from dataclasses import dataclass
from .edf_model import EDFSpec, thrust_at_speed
from .duct_geometry import DuctPlacement
from ..parameterization.design_variables import BWBParams


# Standard atmosphere at 100m altitude
RHO_AIR = 1.215       # [kg/m³]
MU_AIR = 1.81e-5      # [Pa·s] dynamic viscosity


@dataclass
class DuctAeroResult:
    """Aerodynamic performance of the complete duct system."""

    # Mass flow
    mass_flow: float           # [kg/s]
    fan_face_velocity: float   # [m/s]

    # Areas
    intake_capture_area: float  # [m²] required capture area
    fan_area: float             # [m²] fan face area
    exhaust_exit_area: float    # [m²] nozzle exit area
    nozzle_area_ratio: float    # A_exit / A_fan

    # Pressure recovery
    pr_intake: float            # intake pressure recovery (0-1)
    pr_sduct: float             # S-duct pressure recovery (0-1)
    pr_nozzle: float            # nozzle pressure recovery (0-1)
    pr_total: float             # total system pressure recovery

    # Velocities
    exit_velocity: float        # [m/s] nozzle exit velocity
    velocity_ratio: float       # V_exit / V_inf

    # Drag
    cd_intake: float            # intake drag coefficient increment

    # Geometric ratios
    sduct_offset_ratio: float   # offset / length for S-duct (< 0.5 desired)

    # Warnings
    warnings: list[str]


def compute_duct_aero(placement: DuctPlacement, edf: EDFSpec,
                      params: BWBParams, velocity: float = 20.0,
                      rho: float = RHO_AIR) -> DuctAeroResult:
    """Compute complete duct aerodynamic performance.

    Parameters
    ----------
    placement : DuctPlacement
    edf : EDFSpec
    params : BWBParams
    velocity : float
        Freestream velocity [m/s].
    rho : float
        Air density [kg/m³].

    Returns
    -------
    DuctAeroResult
    """
    warnings = []

    # ── Fan face conditions ──
    fan_area = np.pi / 4 * edf.fan_diameter ** 2

    # Fan face velocity from actuator disk theory
    # V_fan ≈ V_inf + T / (2 * rho * A_fan * V_eff)
    # Simplified: T = rho * A_fan * V_fan * (V_exit - V_inf)
    # For small UAV EDF, V_fan ≈ 1.3-1.8 × V_inf
    thrust = thrust_at_speed(edf, velocity)
    if velocity > 0.5:
        # Actuator disk: V_fan = V_inf + delta_V/2
        delta_v = thrust / (rho * fan_area * velocity) if fan_area > 0 else 0
        fan_face_velocity = velocity + delta_v / 2
    else:
        fan_face_velocity = np.sqrt(2 * thrust / (rho * fan_area)) if fan_area > 0 else 0

    # Mass flow
    mass_flow = rho * fan_area * fan_face_velocity

    # ── Intake capture area (mass flow continuity) ──
    if velocity > 0.5:
        intake_capture_area = mass_flow / (rho * velocity)
    else:
        intake_capture_area = fan_area * 1.2  # static: larger than fan

    actual_intake_area = placement.intake_width * placement.intake_depth
    if actual_intake_area < intake_capture_area * 0.9:
        warnings.append(
            f"Intake area {actual_intake_area*1e4:.1f} cm² < required "
            f"{intake_capture_area*1e4:.1f} cm² (mass flow deficit)"
        )

    # ── Intake pressure recovery ──
    # NACA flush intake: PR ≈ 0.92-0.97 depending on BL thickness
    # Conservative estimate for thick BL on BWB upper surface
    pr_intake = 0.95

    # ── S-duct pressure recovery ──
    # Seddon & Goldsmith correlation: dp/q = K * (offset/L)^2
    # K ≈ 0.10-0.20 for well-designed S-ducts
    sduct_offset = abs(placement.intake_z - placement.fan_z)
    sduct_length = abs(placement.fan_x - (placement.intake_x + placement.intake_length))
    sduct_length = max(sduct_length, 0.01)  # avoid division by zero

    offset_ratio = sduct_offset / sduct_length

    if offset_ratio > 0.5:
        warnings.append(
            f"S-duct offset/length = {offset_ratio:.2f} > 0.5 "
            "(high risk of flow separation)"
        )

    k_sduct = 0.15  # moderate design quality
    dp_q_sduct = k_sduct * offset_ratio ** 2
    pr_sduct = 1.0 - dp_q_sduct

    # ── Nozzle performance ──
    exhaust_exit_area = placement.exhaust_width * placement.exhaust_height
    nozzle_area_ratio = exhaust_exit_area / fan_area if fan_area > 0 else 1.0

    # Nozzle pressure loss (friction + expansion/contraction)
    # For a short convergent nozzle: PR ≈ 0.97-0.99
    pr_nozzle = 0.98

    # Exit velocity from mass flow continuity
    if exhaust_exit_area > 0:
        exit_velocity = mass_flow / (rho * exhaust_exit_area)
    else:
        exit_velocity = fan_face_velocity

    velocity_ratio = exit_velocity / velocity if velocity > 0.5 else 0.0

    # ── Total pressure recovery ──
    pr_total = pr_intake * pr_sduct * pr_nozzle

    if pr_total < 0.90:
        warnings.append(
            f"Total pressure recovery {pr_total:.3f} < 0.90 "
            "(excessive duct losses, consider redesign)"
        )

    # ── Intake drag coefficient ──
    s_ref = float(np.pi / 4 * edf.intake_diameter ** 2)
    wing_area = ((params.body_root_chord + params.wing_root_chord) * params.body_halfwidth
                 + (params.wing_root_chord + params.tip_chord) * params.outer_half_span)
    cd_intake = 0.05 * s_ref / wing_area if wing_area > 0 else 0.0

    return DuctAeroResult(
        mass_flow=mass_flow,
        fan_face_velocity=fan_face_velocity,
        intake_capture_area=intake_capture_area,
        fan_area=fan_area,
        exhaust_exit_area=exhaust_exit_area,
        nozzle_area_ratio=nozzle_area_ratio,
        pr_intake=pr_intake,
        pr_sduct=pr_sduct,
        pr_nozzle=pr_nozzle,
        pr_total=pr_total,
        exit_velocity=exit_velocity,
        velocity_ratio=velocity_ratio,
        cd_intake=cd_intake,
        sduct_offset_ratio=offset_ratio,
        warnings=warnings,
    )


def print_duct_aero_summary(result: DuctAeroResult):
    """Print a formatted summary of duct aerodynamic performance."""
    print("=" * 60)
    print("  DUCT AERODYNAMIC PERFORMANCE")
    print("=" * 60)
    print(f"  Mass flow rate:       {result.mass_flow:.4f} kg/s")
    print(f"  Fan face velocity:    {result.fan_face_velocity:.1f} m/s")
    print()
    print(f"  Intake capture area:  {result.intake_capture_area * 1e4:.1f} cm²")
    print(f"  Fan face area:        {result.fan_area * 1e4:.1f} cm²")
    print(f"  Exhaust exit area:    {result.exhaust_exit_area * 1e4:.1f} cm²")
    print(f"  Nozzle area ratio:    {result.nozzle_area_ratio:.3f}")
    print()
    print(f"  Pressure recovery:")
    print(f"    Intake (NACA):      {result.pr_intake:.3f}")
    print(f"    S-duct:             {result.pr_sduct:.3f}")
    print(f"    Nozzle:             {result.pr_nozzle:.3f}")
    print(f"    TOTAL:              {result.pr_total:.3f}")
    print()
    print(f"  Exit velocity:        {result.exit_velocity:.1f} m/s")
    print(f"  Velocity ratio:       {result.velocity_ratio:.2f}")
    print(f"  S-duct offset/L:      {result.sduct_offset_ratio:.3f}")
    print(f"  CD intake increment:  {result.cd_intake:.5f}")
    print()
    if result.warnings:
        print("  WARNINGS:")
        for w in result.warnings:
            print(f"    - {w}")
    else:
        print("  No warnings — duct design is within acceptable limits.")
    print("=" * 60)
