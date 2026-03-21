"""BWB aircraft geometry — 2-wing AeroSandbox model.

The aircraft consists of:
1. Center-body (asb.Wing, symmetric=True): thick BWB fuselage, 3 stations
2. Outer wing (asb.Wing, symmetric=True): tapered Kulfan CST wing, 6 stations

The two wings meet at the blend station (y = body_halfwidth) with C0 continuity:
- Chord match: body tip chord = wing root chord
- Airfoil match: body tip airfoil = wing root Kulfan airfoil
- LE position match: continuous leading edge
"""

import numpy as np
import aerosandbox as asb
from .design_variables import BWBParams

# Template Kulfan weights (from v1 default reflex airfoil)
_TEMPLATE_UPPER = np.array([
    0.1553, 0.1686, 0.2833, 0.1191, 0.2572, 0.1890, 0.2404, 0.2363,
])
_TEMPLATE_LOWER = np.array([
    -0.1482, -0.0983, -0.0218, -0.1063, -0.0346, -0.0355, -0.0198, -0.0132,
])
_FREE_INDICES = [2, 6, 7]  # optimizer-controlled Kulfan indices

# Outer wing stations (fraction of outer half-span)
# Aligned with dihedral breaks + control surface boundaries:
#   0.00  elevon inboard start (blend)
#   0.25  elevon split (dihedral break)
#   0.50  elevon end
#   0.55  aileron start
#   0.75  mid-outboard
#   0.90  aileron end
#   1.00  tip
OUTER_WING_STATIONS = [0.0, 0.25, 0.50, 0.55, 0.75, 0.90, 1.0]
N_OUTER_SEGMENTS = len(OUTER_WING_STATIONS) - 1

def outer_wing_twists(p: 'BWBParams') -> list[float]:
    """Build the 7-element twist list for OUTER_WING_STATIONS.

    The twist at eta=0.55 is linearly interpolated between twist_2
    (eta=0.50) and twist_3 (eta=0.75).
    """
    twist_055 = p.twist_2 + (0.55 - 0.50) / (0.75 - 0.50) * (p.twist_3 - p.twist_2)
    return [0.0, p.twist_1, p.twist_2, twist_055, p.twist_3, p.twist_4, p.twist_tip]


def outer_wing_dihedrals(p: 'BWBParams') -> list[float]:
    """Build the 6-element dihedral list for OUTER_WING_STATIONS segments.

    The segment eta=0.50->0.55 inherits dihedral_2 (same dihedral region).
    """
    return [p.dihedral_0, p.dihedral_1, p.dihedral_2, p.dihedral_2,
            p.dihedral_3, p.dihedral_tip]


# Body stations (fraction of body half-width)
BODY_STATIONS = [0.0, 0.50, 1.0]
N_BODY_SEGMENTS = len(BODY_STATIONS) - 1


# ═══════════════════════════════════════════════════════════════════════════
# Kulfan airfoil construction (for outer wing)
# ═══════════════════════════════════════════════════════════════════════════

def _build_kulfan_weights(u2, u6, u7, l2, l6, l7):
    """Build full 8-weight arrays from 3 free weights per surface."""
    upper = _TEMPLATE_UPPER.copy()
    lower = _TEMPLATE_LOWER.copy()
    upper[_FREE_INDICES[0]] = u2
    upper[_FREE_INDICES[1]] = u6
    upper[_FREE_INDICES[2]] = u7
    lower[_FREE_INDICES[0]] = l2
    lower[_FREE_INDICES[1]] = l6
    lower[_FREE_INDICES[2]] = l7
    return upper, lower


def build_kulfan_airfoil(u2, u6, u7, l2, l6, l7, name="kulfan") -> asb.KulfanAirfoil:
    """Build KulfanAirfoil from reduced design variables."""
    upper, lower = _build_kulfan_weights(u2, u6, u7, l2, l6, l7)
    return asb.KulfanAirfoil(
        name=name,
        upper_weights=upper,
        lower_weights=lower,
        leading_edge_weight=0.16,
        TE_thickness=0.0,
    )


def build_kulfan_airfoil_at_station(
    params: BWBParams, frac: float, name: str = "s",
) -> asb.KulfanAirfoil:
    """Build KulfanAirfoil at a spanwise station (frac in [0, 1] of outer wing).

    Root (frac=0) uses kulfan_root_* weights directly.
    Tip (frac=1) applies delta-offsets to approximate different tip airfoil.
    """
    p = params

    # Tip weights: root + delta-offsets applied to mid-chord and TE weights
    # delta_tc affects upper weights (thicker or thinner)
    # delta_camber affects the difference between upper and lower
    dt = p.kulfan_tip_delta_tc
    dc = p.kulfan_tip_delta_camber

    tip_u2 = p.kulfan_root_u2 + dt + dc
    tip_u6 = p.kulfan_root_u6 + dt * 0.5
    tip_u7 = p.kulfan_root_u7 + dt * 0.3
    tip_l2 = p.kulfan_root_l2 - dt + dc
    tip_l6 = p.kulfan_root_l6 - dt * 0.5
    tip_l7 = p.kulfan_root_l7 - dt * 0.3

    # Interpolate root → tip
    u2 = p.kulfan_root_u2 + frac * (tip_u2 - p.kulfan_root_u2)
    u6 = p.kulfan_root_u6 + frac * (tip_u6 - p.kulfan_root_u6)
    u7 = p.kulfan_root_u7 + frac * (tip_u7 - p.kulfan_root_u7)
    l2 = p.kulfan_root_l2 + frac * (tip_l2 - p.kulfan_root_l2)
    l6 = p.kulfan_root_l6 + frac * (tip_l6 - p.kulfan_root_l6)
    l7 = p.kulfan_root_l7 + frac * (tip_l7 - p.kulfan_root_l7)

    return build_kulfan_airfoil(u2, u6, u7, l2, l6, l7, name=name)


def flatten_airfoil_aft(coords_2d: np.ndarray, x_hinge: float = 0.75) -> np.ndarray:
    """Replace the aft section of an airfoil with straight lines.

    For x/c >= x_hinge, the upper and lower surfaces are replaced by
    straight lines from the Kulfan point at x_hinge to the TE point.
    This creates a flat control surface zone with a physical kink at
    the hinge line.

    Parameters
    ----------
    coords_2d : (N, 2) array in Selig format (TE_upper → LE → TE_lower).
    x_hinge : float
        Hinge x/c position (default 0.75 for 25% chord flap).

    Returns
    -------
    (N, 2) array with linearized aft section. Forward of x_hinge is unchanged.
    """
    result = coords_2d.copy()
    mid = np.argmin(result[:, 0])  # LE index

    # Upper surface (Selig: indices 0..mid, TE→LE)
    xs_upper = result[:mid + 1, 0]  # TE→LE direction (decreasing x)
    te_z_upper = result[0, 1]       # TE upper z/c
    # Interpolate hinge z from the original Kulfan curve
    # Need LE→TE direction for np.interp (increasing x)
    hinge_z_upper = np.interp(x_hinge, xs_upper[::-1], result[:mid + 1, 1][::-1])
    for i in range(mid + 1):
        xc = result[i, 0]
        if xc >= x_hinge:
            t = (xc - x_hinge) / (1.0 - x_hinge) if (1.0 - x_hinge) > 1e-10 else 1.0
            result[i, 1] = hinge_z_upper + t * (te_z_upper - hinge_z_upper)

    # Lower surface (Selig: indices mid..end, LE→TE)
    te_z_lower = result[-1, 1]      # TE lower z/c
    hinge_z_lower = np.interp(x_hinge, result[mid:, 0], result[mid:, 1])
    for i in range(mid, len(result)):
        xc = result[i, 0]
        if xc >= x_hinge:
            t = (xc - x_hinge) / (1.0 - x_hinge) if (1.0 - x_hinge) > 1e-10 else 1.0
            result[i, 1] = hinge_z_lower + t * (te_z_lower - hinge_z_lower)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Body airfoil construction (thick, non-Kulfan)
# ═══════════════════════════════════════════════════════════════════════════

def build_body_airfoil(
    tc: float, camber: float, reflex: float, le_droop: float,
    name: str = "body", n_pts: int = 80,
) -> asb.Airfoil:
    """Build a thick body airfoil from direct geometric parameters.

    Uses NACA 4-digit thickness distribution + parametric camber line.
    Suitable for t/c up to 25% where Kulfan CST becomes unreliable.

    Parameters
    ----------
    tc : float      — max thickness-to-chord (0.12 to 0.25)
    camber : float  — max camber (0.01 to 0.06)
    reflex : float  — TE reflex: upward deflection of aft 30% camber line
    le_droop : float — LE radius weight (controls nose rounding)
    """
    x = _cosine_spacing(n_pts)

    # NACA 4-digit thickness distribution, scaled to desired t/c
    # y_t = (t/0.2) * [0.2969√x - 0.1260x - 0.3516x² + 0.2843x³ - 0.1015x⁴]
    yt = (tc / 0.20) * (
        0.2969 * np.sqrt(np.maximum(x, 0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # LE droop: modify the first term coefficient for more/less LE radius
    # Default NACA coefficient is 0.2969; le_droop scales it
    le_factor = le_droop / 0.16  # normalized to default
    yt_le_correction = (le_factor - 1.0) * 0.2969 * np.sqrt(np.maximum(x, 0)) * (tc / 0.20)
    yt += yt_le_correction * np.exp(-8 * x)  # only affects LE region

    # Camber line: parabolic with reflex
    # Base camber: max at ~40% chord
    yc = 4 * camber * x * (1 - x)

    # Reflex: deflect aft 30% upward (pitch-up moment)
    reflex_start = 0.70
    mask = x > reflex_start
    reflex_x = (x[mask] - reflex_start) / (1.0 - reflex_start)
    yc[mask] += reflex * reflex_x**2

    # Combine: upper = camber + thickness, lower = camber - thickness
    xu = x
    yu = yc + yt
    xl = x
    yl = yc - yt

    # Smooth TE closure: blend to y=0 over last 5% chord
    # Avoids the sharp corner caused by reflex + forced closure
    te_blend_start = 0.95
    te_mask = x > te_blend_start
    te_frac = (x[te_mask] - te_blend_start) / (1.0 - te_blend_start)  # 0→1
    blend = (1.0 - te_frac ** 2)  # smooth ramp to 0
    yu[te_mask] *= blend
    yl[te_mask] *= blend

    # Assemble in Selig format (upper TE→LE then lower LE→TE)
    coords_upper = np.column_stack([xu[::-1], yu[::-1]])
    coords_lower = np.column_stack([xl[1:], yl[1:]])  # skip duplicate LE
    coordinates = np.vstack([coords_upper, coords_lower])

    return asb.Airfoil(name=name, coordinates=coordinates)


def _cosine_spacing(n: int) -> np.ndarray:
    """Cosine-spaced x/c from 0 to 1 (denser at LE and TE)."""
    beta = np.linspace(0, np.pi, n)
    return 0.5 * (1 - np.cos(beta))


# ═══════════════════════════════════════════════════════════════════════════
# Aircraft assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_airplane(params: BWBParams, x_cg: float | None = None) -> asb.Airplane:
    """Build AeroSandbox Airplane with center-body + outer wing.

    The center-body is modeled as a separate asb.Wing (symmetric=True)
    with 3 stations at y = [0, body_hw/2, body_hw].

    The outer wing is modeled as a separate asb.Wing (symmetric=True)
    with 6 stations from y = body_hw to y = half_span.

    Parameters
    ----------
    params : BWBParams
        Aircraft geometry parameters.
    x_cg : float, optional
        CG x-position [m] from body LE. If None, uses fallback (0.30 × body chord).
    """
    body_wing = _build_center_body(params)
    outer_wing = _build_outer_wing(params)

    # CG reference: computed from component placement or fallback
    if x_cg is not None:
        x_ref = x_cg
    else:
        # Fallback: approximate CG at 30% body root chord
        x_ref = params.body_root_chord * 0.30

    # Explicit reference quantities for consistent AVL normalization.
    # s_ref = total projected wing area, c_ref = mean geometric chord,
    # b_ref = full span. This ensures CM_alpha / CL_alpha gives SM
    # in units of c_ref, matching the CG correction in reconstruct.py.
    s = compute_wing_area(params)
    span = 2 * params.half_span
    c = s / span  # mean geometric chord
    airplane = asb.Airplane(
        name="nEUROn BWB v2",
        xyz_ref=[x_ref, 0, 0],
        s_ref=s,
        c_ref=c,
        b_ref=span,
        wings=[body_wing, outer_wing],
    )
    return airplane


def _build_center_body(params: BWBParams) -> asb.Wing:
    """Build center-body as a thick, swept, low-AR wing."""
    p = params
    body_chord = p.body_root_chord
    blend_chord = p.wing_root_chord  # C0 continuity

    body_sweep = np.radians(p.body_sweep_deg)
    bw = p.body_halfwidth

    # Airfoil at centerline: thick body airfoil
    af_center = build_body_airfoil(
        tc=p.body_tc_root, camber=p.body_camber,
        reflex=p.body_reflex, le_droop=p.body_le_droop,
        name="body_center",
    )

    # Airfoil at mid-body: interpolated
    mid_tc = p.body_tc_root * 0.7 + 0.3 * 0.105  # blend toward wing
    mid_camber = p.body_camber * 0.7 + 0.3 * 0.024
    af_mid = build_body_airfoil(
        tc=mid_tc, camber=mid_camber,
        reflex=p.body_reflex * 0.5, le_droop=p.body_le_droop * 0.8,
        name="body_mid",
    )

    # Airfoil at blend: must match outer wing root Kulfan
    kaf_blend = build_kulfan_airfoil_at_station(p, 0.0, name="body_blend")
    af_blend = kaf_blend.to_airfoil()

    # Positions
    # Station 0 (centerline): y=0
    # Station 1 (mid): y=body_hw/2
    # Station 2 (blend): y=body_hw
    mid_y = bw * 0.5

    # LE offsets from sweep
    dx_mid = mid_y * np.tan(body_sweep)
    dx_blend = bw * np.tan(body_sweep)

    # Chord distribution: linear from body_chord to blend_chord
    chord_mid = body_chord + 0.5 * (blend_chord - body_chord)

    # Twist: body_twist at center, interpolated to 0 at blend (outer wing root twist = 0)
    twist_mid = p.body_twist * 0.5

    xsecs = [
        asb.WingXSec(
            xyz_le=[0.0, 0.0, 0.0],
            chord=body_chord,
            twist=p.body_twist,
            airfoil=af_center,
        ),
        asb.WingXSec(
            xyz_le=[dx_mid, mid_y, 0.0],
            chord=chord_mid,
            twist=twist_mid,
            airfoil=af_mid,
        ),
        asb.WingXSec(
            xyz_le=[dx_blend, bw, 0.0],
            chord=blend_chord,
            twist=0.0,  # match outer wing root twist
            airfoil=af_blend,
        ),
    ]

    return asb.Wing(name="Center Body", symmetric=True, xsecs=xsecs)


def _build_outer_wing(params: BWBParams) -> asb.Wing:
    """Build outer wing from blend station to tip."""
    p = params
    bw = p.body_halfwidth
    outer_span = p.outer_half_span
    root_chord = p.wing_root_chord
    tip_chord = p.tip_chord

    # Body blend LE position (must match center-body tip LE)
    body_sweep = np.radians(p.body_sweep_deg)
    x_blend = bw * np.tan(body_sweep)
    y_blend = bw

    # Chords at each station (linear taper)
    chords = [root_chord + frac * (tip_chord - root_chord) for frac in OUTER_WING_STATIONS]

    twists = outer_wing_twists(p)
    dihedrals = outer_wing_dihedrals(p)

    # Compute 3D positions from blend station outward
    wing_sweep_rad = np.radians(p.le_sweep_deg)
    positions = [[x_blend, y_blend, 0.0]]

    for i in range(N_OUTER_SEGMENTS):
        prev = positions[-1]
        dy_seg = outer_span * (OUTER_WING_STATIONS[i + 1] - OUTER_WING_STATIONS[i])
        dih_rad = np.radians(dihedrals[i])
        dx = dy_seg * np.tan(wing_sweep_rad)
        dy = dy_seg * np.cos(dih_rad)
        dz = dy_seg * np.sin(dih_rad)
        positions.append([prev[0] + dx, prev[1] + dy, prev[2] + dz])

    # Build wing cross-sections
    xsecs = []
    for i, (frac, chord, twist, xyz) in enumerate(
        zip(OUTER_WING_STATIONS, chords, twists, positions)
    ):
        kaf = build_kulfan_airfoil_at_station(p, frac, name=f"wing_s{i}")
        af = kaf.to_airfoil()

        xsecs.append(asb.WingXSec(
            xyz_le=xyz,
            chord=chord,
            twist=twist,
            airfoil=af,
        ))

    return asb.Wing(name="Outer Wing", symmetric=True, xsecs=xsecs)


# ═══════════════════════════════════════════════════════════════════════════
# Geometric computations
# ═══════════════════════════════════════════════════════════════════════════

def compute_wing_area(params: BWBParams) -> float:
    """Total projected reference wing area [m²] (body + outer wing)."""
    p = params
    # Body area: trapezoidal (body_root_chord → wing_root_chord) × body_halfwidth × 2
    body_area = (p.body_root_chord + p.wing_root_chord) * p.body_halfwidth

    # Outer wing area: trapezoidal (wing_root_chord → tip_chord) × outer_half_span × 2
    outer_area = (p.wing_root_chord + p.tip_chord) * p.outer_half_span

    return body_area + outer_area  # already for full span (symmetric)


def compute_aspect_ratio(params: BWBParams) -> float:
    """Wing aspect ratio (full span)."""
    span = 2 * params.half_span
    area = compute_wing_area(params)
    return span**2 / area if area > 0 else 0.0


def compute_mac(params: BWBParams) -> float:
    """Mean aerodynamic chord of the outer wing."""
    lam = params.taper_ratio
    return (2 / 3) * params.wing_root_chord * (1 + lam + lam**2) / (1 + lam)


def estimate_structural_mass(
    params: BWBParams,
    density_kg_m2: float = 0.6,
    n_load: float = 2.5,
    fos: float = 1.5,
    sigma_material: float = 200e6,
    rho_material: float = 1500.0,
    mtow: float = 1.5,
    structure_config=None,
) -> float:
    """Estimate structural mass [kg] for composite BWB.

    Parameters
    ----------
    density_kg_m2 : skin areal density [kg/m²]
    n_load : limit load factor (2.5 for UAV)
    fos : factor of safety (1.5 standard)
    sigma_material : ultimate stress [Pa] (200 MPa for composite)
    rho_material : material density [kg/m³] (1500 for composite)
    mtow : max takeoff weight [kg] (for bending moment)
    """
    from ..config import StructureConfig
    if structure_config is None:
        structure_config = StructureConfig()
    if density_kg_m2 == 0.6:
        density_kg_m2 = structure_config.skin_density_kg_m2

    # Avg thickness from root Kulfan
    root_kaf = build_kulfan_airfoil_at_station(params, 0.0)
    tip_kaf = build_kulfan_airfoil_at_station(params, 1.0)
    avg_tc_wing = (root_kaf.max_thickness() + tip_kaf.max_thickness()) / 2

    # Body contribution: thicker → more material
    avg_tc_body = params.body_tc_root * structure_config.body_tc_correction

    # Shell mass: wetted area × density × (1 + t/c correction)
    wing_wetted = 2.05 * (params.wing_root_chord + params.tip_chord) * params.outer_half_span
    body_wetted = 2.05 * (params.body_root_chord + params.wing_root_chord) * params.body_halfwidth

    shell_mass = density_kg_m2 * (
        wing_wetted * (1 + avg_tc_wing) +
        body_wetted * (1 + avg_tc_body)
    )

    # Spar mass: beam bending model
    # Triangular load approximation → root bending moment
    from ..constants import G
    weight = mtow * G
    half_span = params.half_span
    m_root = 0.5 * n_load * fos * weight * half_span / 3.0

    # Spar height from wing thickness at MAC
    mac = compute_mac(params)
    h_spar = max(avg_tc_wing * mac, 0.005)

    # Required spar cap area (upper + lower caps)
    a_cap = m_root / (sigma_material * h_spar)
    spar_mass = 2.0 * a_cap * half_span * rho_material

    return (shell_mass + spar_mass) * structure_config.structural_adder


def compute_internal_volume(params: BWBParams, packing_factor: float = 0.55) -> float:
    """Usable internal volume [m³] in the BWB center body.

    Integrates the body cross-section from centerline to body_halfwidth.
    Much more accurate than v1's elliptical approximation because we use
    the actual body airfoil thickness distribution.
    """
    p = params
    bw = p.body_halfwidth
    n_steps = 20

    total_volume = 0.0
    for i in range(n_steps):
        frac = (i + 0.5) / n_steps
        y = bw * frac
        dy = bw / n_steps

        # Local chord: linear interpolation body_root → wing_root
        chord_local = p.body_root_chord + frac * (p.wing_root_chord - p.body_root_chord)

        # Local t/c: interpolate from body center toward blend
        tc_local = p.body_tc_root + frac * (0.105 - p.body_tc_root)  # 0.105 = default wing root t/c

        # Elliptical cross-section: height = tc * chord, width approximated
        height = tc_local * chord_local
        area_cross = np.pi / 4 * chord_local * height * 0.60  # 60% fill factor

        total_volume += area_cross * dy

    return 2 * total_volume * packing_factor  # both halves
