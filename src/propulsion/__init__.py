from .edf_model import EDFSpec, EDF_70MM, EDF_90MM
from .edf_model import thrust_at_speed, power_required, endurance, duct_fits_in_body, intake_drag

from .duct_geometry import (
    DuctPlacement,
    ClearanceResult,
    compute_duct_placement,
    compute_duct_centerline,
    duct_cross_section,
    validate_duct_clearance,
    build_intake_geometry,
    build_sduct_geometry,
    build_edf_housing,
    build_exhaust_geometry,
    build_exhaust_fairing,
    build_propulsion_mesh,
    get_duct_profiles,
    get_duct_centerline_csv,
)

from .duct_aero import DuctAeroResult, compute_duct_aero, print_duct_aero_summary

__all__ = [
    # EDF model
    "EDFSpec", "EDF_70MM", "EDF_90MM",
    "thrust_at_speed", "power_required", "endurance",
    "duct_fits_in_body", "intake_drag",
    # Duct geometry
    "DuctPlacement", "ClearanceResult",
    "compute_duct_placement", "compute_duct_centerline",
    "duct_cross_section", "validate_duct_clearance",
    "build_intake_geometry", "build_sduct_geometry",
    "build_edf_housing", "build_exhaust_geometry", "build_exhaust_fairing",
    "build_propulsion_mesh",
    "get_duct_profiles", "get_duct_centerline_csv",
    # Duct aero
    "DuctAeroResult", "compute_duct_aero", "print_duct_aero_summary",
]
