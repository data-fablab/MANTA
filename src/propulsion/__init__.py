from .edf_model import EDFSpec, EDF_70MM, EDF_90MM
from .edf_model import thrust_at_speed, power_required, endurance, intake_drag

from .duct_geometry import (
    DuctPlacement,
    ClearanceResult,
    size_and_place_duct,
    compute_duct_placement,  # backward-compatible alias
    compute_duct_centerline,
    duct_cross_section,
    validate_duct_clearance,
    min_effective_clearance_mm,
    compute_duct_structure_mass,
    build_intake_geometry,
    build_sduct_geometry,
    build_edf_housing,
    build_exhaust_geometry,
    build_propulsion_mesh,
)

from .duct_aero import DuctAeroResult, compute_duct_aero, print_duct_aero_summary

from .balance import (
    compute_propulsion_balance, compute_intake_cd,
    compute_duct_mass, compute_bump_drag, compute_duct_clearance,
)

__all__ = [
    "EDFSpec", "EDF_70MM", "EDF_90MM",
    "thrust_at_speed", "power_required", "endurance", "intake_drag",
    "DuctPlacement", "ClearanceResult",
    "size_and_place_duct", "compute_duct_placement",
    "compute_duct_centerline", "duct_cross_section",
    "validate_duct_clearance", "min_effective_clearance_mm",
    "compute_duct_structure_mass",
    "build_intake_geometry", "build_sduct_geometry",
    "build_edf_housing", "build_exhaust_geometry",
    "build_propulsion_mesh",
    "DuctAeroResult", "compute_duct_aero", "print_duct_aero_summary",
    "compute_propulsion_balance", "compute_intake_cd",
    "compute_duct_mass", "compute_bump_drag", "compute_duct_clearance",
]
