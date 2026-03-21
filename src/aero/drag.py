"""Drag models for BWB aircraft.

Separates wing drag (NeuralFoil strip theory) from body drag (3D form factor).
This is a key improvement over v1 where NeuralFoil was used for thick body
sections (t/c > 15%) where 2D strip theory is inaccurate.
"""

import numpy as np
import aerosandbox as asb
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import outer_wing_twists
from ..parameterization.bwb_aircraft import (
    build_kulfan_airfoil_at_station,
    OUTER_WING_STATIONS, N_OUTER_SEGMENTS,
)


def oswald_efficiency(ar: float, sweep_deg: float, taper: float) -> float:
    """Estimate Oswald efficiency factor for a flying wing."""
    e_theo = 1.0 / (1.05 + 0.007 * np.pi * ar)
    e_sweep = 1.0 - 0.0004 * sweep_deg
    e_taper = 1.0 - 0.2 * abs(taper - 0.30)
    return float(np.clip(e_theo * e_sweep * e_taper, 0.5, 0.95))


def compute_wing_cd0(
    params: BWBParams, alpha_eq: float, velocity: float, kinematic_viscosity: float,
    config=None,
) -> float:
    """Viscous drag CD0 of the outer wing via NeuralFoil strip integration.

    Only applies to the outer wing where t/c < 15% and NeuralFoil is reliable.
    """
    from ..config import AeroModelConfig
    if config is None:
        config = AeroModelConfig()
    p = params
    outer_span = p.outer_half_span
    root_chord = p.wing_root_chord
    tip_chord = p.tip_chord
    twists = outer_wing_twists(p)

    s_ref = _compute_outer_wing_area(p)
    cd_total = 0.0

    for i in range(N_OUTER_SEGMENTS):
        frac_lo = OUTER_WING_STATIONS[i]
        frac_hi = OUTER_WING_STATIONS[i + 1]
        frac_mid = (frac_lo + frac_hi) / 2

        chord_local = root_chord + frac_mid * (tip_chord - root_chord)
        twist_local = (twists[i] + twists[i + 1]) / 2

        re_local = velocity * chord_local / kinematic_viscosity
        re_local = np.clip(re_local, config.re_clip_min, config.re_clip_max)
        alpha_local = np.clip(alpha_eq + twist_local, config.alpha_clip_min, config.alpha_clip_max)

        try:
            kaf = build_kulfan_airfoil_at_station(p, frac_mid)
            aero_2d = kaf.get_aero_from_neuralfoil(
                alpha=alpha_local, Re=re_local, mach=0.0, model_size="large",
            )
            cd_section = float(np.clip(aero_2d["CD"], config.cd_clip_min, config.cd_clip_max))
        except Exception:
            # Flat-plate fallback
            cf = 1.328 / np.sqrt(max(re_local, 1e3))
            cd_section = cf * 2.05

        dy = outer_span * (frac_hi - frac_lo)
        cd_total += cd_section * chord_local * dy

    # Both half-wings, normalized by total S_ref
    s_ref_total = _compute_total_area(p)
    cd0_wing = 2 * cd_total / s_ref_total * config.interference_wing
    return cd0_wing


def compute_body_cd0(
    params: BWBParams, alpha_eq: float, velocity: float,
    kinematic_viscosity: float, s_ref: float,
    cd_intake: float = 0.0, config=None,
) -> float:
    """Center-body form drag with 3D correction.

    For thick sections (t/c > 12%), NeuralFoil 2D is unreliable.
    Uses analytical skin friction + Raymer form factor + 3D correction.
    """
    from ..config import AeroModelConfig
    if config is None:
        config = AeroModelConfig()
    p = params
    bw = p.body_halfwidth

    # Average body properties
    avg_chord = (p.body_root_chord + p.wing_root_chord) / 2
    avg_tc = (p.body_tc_root + 0.105) / 2  # center → blend

    # Reynolds number on body
    re_body = velocity * avg_chord / kinematic_viscosity

    # Skin friction (laminar-turbulent blend)
    if re_body > 5e5:
        cf = 0.455 / (np.log10(re_body) ** 2.58)
    elif re_body > 1e4:
        re_tr = min(2e5, re_body * 0.5)
        cf_lam = 1.328 / np.sqrt(re_body)
        cf_turb = 0.455 / (np.log10(re_body) ** 2.58)
        frac_lam = re_tr / re_body
        cf = frac_lam * cf_lam + (1 - frac_lam) * cf_turb
    else:
        cf = 1.328 / np.sqrt(max(re_body, 1e3))

    # Form factor (Raymer): accounts for pressure drag from thickness
    ff = 1.0 + 2.0 * avg_tc + 60 * avg_tc**4

    # 3D form-drag correction for thick bodies
    K_3d = config.k_3d
    if avg_tc > config.k_3d_tc_threshold:
        ff_3d = 1.0 + K_3d * (avg_tc - config.k_3d_tc_threshold)**2
    else:
        ff_3d = 1.0

    # Wetted area ratio
    s_wet_body = config.wetted_area_ratio * (p.body_root_chord + p.wing_root_chord) * bw
    s_wet_ratio = 2 * s_wet_body / s_ref  # both halves

    # Body CD0
    cd0_body = cf * ff * ff_3d * s_wet_ratio

    # Interference factor
    cd0_body *= config.interference_body

    # Add intake drag
    cd0_body += cd_intake

    return cd0_body


def estimate_parasite_drag(params: BWBParams, velocity: float,
                           kinematic_viscosity: float) -> float:
    """Quick analytical CD0 estimate (fallback). Full aircraft."""
    s_ref = _compute_total_area(params)
    mac = (2/3) * params.wing_root_chord * (1 + params.taper_ratio + params.taper_ratio**2) / (1 + params.taper_ratio)
    re_mac = velocity * mac / kinematic_viscosity

    if re_mac > 5e5:
        cf = 0.455 / (np.log10(re_mac) ** 2.58)
    else:
        cf = 1.328 / np.sqrt(max(re_mac, 1e3))

    avg_tc = 0.10
    ff = 1.0 + 2.0 * avg_tc + 60 * avg_tc**4
    cd0 = cf * ff * 2.05 * 1.10
    return cd0


def _compute_outer_wing_area(p: BWBParams) -> float:
    return (p.wing_root_chord + p.tip_chord) * p.outer_half_span


def _compute_total_area(p: BWBParams) -> float:
    body_area = (p.body_root_chord + p.wing_root_chord) * p.body_halfwidth
    outer_area = (p.wing_root_chord + p.tip_chord) * p.outer_half_span
    return body_area + outer_area
