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

# Body Kulfan templates (thick reflex airfoil, ~18% t/c baseline)
# Thicker than wing template to accommodate 12-25% t/c range.
_BODY_TEMPLATE_UPPER = np.array([
    0.2640, 0.2880, 0.4560, 0.2160, 0.3840, 0.3000, 0.2400, 0.2160,
])
_BODY_TEMPLATE_LOWER = np.array([
    -0.2520, -0.1800, -0.0720, -0.1800, -0.0840, -0.0720, -0.0360, -0.0120,
])
_BODY_FREE_INDICES = [2, 6, 7]  # same indices as wing

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


# ═══════════════════════════════════════════════════════════════════════════
# Body Kulfan airfoil construction (thick center-body)
# ═══════════════════════════════════════════════════════════════════════════

def _build_body_kulfan_weights(u2, u6, u7, l2, l6, l7):
    """Build full 8-weight arrays from 3 free body weights per surface."""
    upper = _BODY_TEMPLATE_UPPER.copy()
    lower = _BODY_TEMPLATE_LOWER.copy()
    upper[_BODY_FREE_INDICES[0]] = u2
    upper[_BODY_FREE_INDICES[1]] = u6
    upper[_BODY_FREE_INDICES[2]] = u7
    lower[_BODY_FREE_INDICES[0]] = l2
    lower[_BODY_FREE_INDICES[1]] = l6
    lower[_BODY_FREE_INDICES[2]] = l7
    return upper, lower


def build_body_kulfan_airfoil(u2, u6, u7, l2, l6, l7,
                               name="body_kulfan") -> asb.KulfanAirfoil:
    """Build thick body KulfanAirfoil from reduced design variables."""
    upper, lower = _build_body_kulfan_weights(u2, u6, u7, l2, l6, l7)
    return asb.KulfanAirfoil(
        name=name,
        upper_weights=upper,
        lower_weights=lower,
        leading_edge_weight=0.25,  # larger than wing (0.16) for thicker LE
        TE_thickness=0.0,
    )


def body_tc_from_kulfan(u2, u6, u7, l2, l6, l7) -> float:
    """Extract effective t/c from body Kulfan weights.

    Used by downstream code (drag, CG, features) as drop-in
    replacement for the old params.body_tc_root field.
    """
    kaf = build_body_kulfan_airfoil(u2, u6, u7, l2, l6, l7)
    return float(kaf.max_thickness())


def body_height_at_xc(params: 'BWBParams', x_frac: float) -> float:
    """Body airfoil thickness at arbitrary x/c, as a chord fraction.

    Evaluates the centerline body Kulfan CST at a single chordwise station.
    """
    kaf = build_body_kulfan_airfoil(
        params.body_kulfan_u2, params.body_kulfan_u6, params.body_kulfan_u7,
        params.body_kulfan_l2, params.body_kulfan_l6, params.body_kulfan_l7,
    )
    coords = kaf.to_airfoil(n_coordinates_per_side=80).coordinates
    x_pts = coords[:, 0]
    z_pts = coords[:, 1]
    n = len(x_pts) // 2
    # Upper surface: first n+1 points (from TE to LE to TE), reversed
    x_up = x_pts[:n + 1][::-1]
    z_up = z_pts[:n + 1][::-1]
    x_lo = x_pts[n:]
    z_lo = z_pts[n:]
    z_upper = float(np.interp(x_frac, x_up, z_up))
    z_lower = float(np.interp(x_frac, x_lo, z_lo))
    return z_upper - z_lower


def vectorized_body_tc_at_xc(
    body_u2: np.ndarray, body_u6: np.ndarray, body_u7: np.ndarray,
    body_l2: np.ndarray, body_l6: np.ndarray, body_l7: np.ndarray,
    x_frac: float,
) -> np.ndarray:
    """Evaluate body airfoil t/c at arbitrary x/c, vectorized over N designs.

    Uses the Kulfan CST class function analytically:
        z(x) = x^0.5 * (1-x) * Σ(w_i * C(n,i) * x^i * (1-x)^(n-i))
    where C(n,i) are binomial coefficients and n=7 (order 8 Bernstein).
    """
    n_order = 7  # degree = len(weights) - 1
    # Binomial coefficients C(7, i) for i=0..7
    binom_coeffs = np.array([1, 7, 21, 35, 35, 21, 7, 1], dtype=float)

    # Bernstein basis at x_frac (same for all designs)
    x = x_frac
    basis = binom_coeffs * (x ** np.arange(8)) * ((1 - x) ** np.arange(n_order, -1, -1))

    # Class function: x^0.5 * (1-x)
    class_fn = x ** 0.5 * (1 - x)

    # Build full 8-weight arrays for N designs
    # Template values are broadcast, free indices overwritten
    N = len(body_u2)
    upper_w = np.tile(_BODY_TEMPLATE_UPPER, (N, 1))  # (N, 8)
    lower_w = np.tile(_BODY_TEMPLATE_LOWER, (N, 1))
    upper_w[:, _BODY_FREE_INDICES[0]] = body_u2
    upper_w[:, _BODY_FREE_INDICES[1]] = body_u6
    upper_w[:, _BODY_FREE_INDICES[2]] = body_u7
    lower_w[:, _BODY_FREE_INDICES[0]] = body_l2
    lower_w[:, _BODY_FREE_INDICES[1]] = body_l6
    lower_w[:, _BODY_FREE_INDICES[2]] = body_l7

    # z_upper = class_fn * dot(upper_w, basis), z_lower = class_fn * dot(lower_w, basis)
    z_upper = class_fn * (upper_w @ basis)  # (N,)
    z_lower = class_fn * (lower_w @ basis)  # (N,)

    return z_upper - z_lower  # t/c at x_frac


def build_body_kulfan_at_station(params: 'BWBParams', frac: float,
                                  name: str = "body_s") -> asb.KulfanAirfoil:
    """Build body Kulfan airfoil at a spanwise station within the center-body.

    frac=0.0 → centerline (pure body Kulfan)
    frac=1.0 → blend station (pure wing root Kulfan, for C0 continuity)

    Interpolates both the free weights and the template weights between
    body and wing templates.
    """
    p = params
    # Body center weights
    bu2, bu6, bu7 = p.body_kulfan_u2, p.body_kulfan_u6, p.body_kulfan_u7
    bl2, bl6, bl7 = p.body_kulfan_l2, p.body_kulfan_l6, p.body_kulfan_l7
    # Wing root weights
    wu2, wu6, wu7 = p.kulfan_root_u2, p.kulfan_root_u6, p.kulfan_root_u7
    wl2, wl6, wl7 = p.kulfan_root_l2, p.kulfan_root_l6, p.kulfan_root_l7

    # Interpolate free weights
    u2 = bu2 * (1 - frac) + wu2 * frac
    u6 = bu6 * (1 - frac) + wu6 * frac
    u7 = bu7 * (1 - frac) + wu7 * frac
    l2 = bl2 * (1 - frac) + wl2 * frac
    l6 = bl6 * (1 - frac) + wl6 * frac
    l7 = bl7 * (1 - frac) + wl7 * frac

    # Interpolate full template arrays
    upper = _BODY_TEMPLATE_UPPER * (1 - frac) + _TEMPLATE_UPPER * frac
    lower = _BODY_TEMPLATE_LOWER * (1 - frac) + _TEMPLATE_LOWER * frac
    upper[_BODY_FREE_INDICES[0]] = u2
    upper[_BODY_FREE_INDICES[1]] = u6
    upper[_BODY_FREE_INDICES[2]] = u7
    lower[_BODY_FREE_INDICES[0]] = l2
    lower[_BODY_FREE_INDICES[1]] = l6
    lower[_BODY_FREE_INDICES[2]] = l7

    le_weight = 0.25 * (1 - frac) + 0.16 * frac

    return asb.KulfanAirfoil(
        name=name,
        upper_weights=upper,
        lower_weights=lower,
        leading_edge_weight=le_weight,
        TE_thickness=0.0,
    )


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


def apply_duct_bump(coords_2d: np.ndarray, placement, body_chord: float,
                    body_halfwidth: float, station_y: float) -> np.ndarray:
    """Raise upper surface where the duct protrudes above body OML.

    Uses DuctSpine for consistent Bezier centerline (not linear interp).
    Applies a smooth cosine fairing on the upper surface between
    intake_x_frac and exhaust_x_frac.  The bump height tapers to zero
    at the body halfwidth (duct is centered at y=0).

    Parameters
    ----------
    coords_2d : (N, 2) Selig-format airfoil coordinates (x/c, z/c).
    placement : DuctPlacement with duct geometry.
    body_chord : body root chord [m].
    body_halfwidth : [m].
    station_y : spanwise y-position of this station [m].

    Returns
    -------
    (N, 2) array with upper surface raised where duct protrudes.
    """
    if placement is None or station_y >= body_halfwidth:
        return coords_2d

    from ..propulsion.duct_geometry import DuctSpine
    spine = DuctSpine(placement)

    result = coords_2d.copy()
    mid = np.argmin(result[:, 0])  # LE index

    # Spanwise taper: bump full at y=0, zero at body_halfwidth
    y_taper = max(0.0, 1.0 - station_y / body_halfwidth)

    # Upper surface: indices 0..mid in Selig (TE→LE), reverse for LE→TE
    x_up = result[:mid + 1, 0][::-1]
    z_up = result[:mid + 1, 1][::-1]

    x_start = placement.intake_x_frac
    x_end = placement.exhaust_x_frac

    for j in range(len(x_up)):
        xc = x_up[j]
        if x_start <= xc <= x_end:
            # Duct centerline z/c from spine (follows Bezier, not linear)
            t = spine.t_at_x(xc * body_chord)
            pos = spine.position(t)
            duct_center_z = pos[2] / body_chord
            duct_r = spine.radius(t, radius_mode="structural")
            duct_top_zc = duct_center_z + duct_r / body_chord

            protrusion = duct_top_zc - z_up[j]
            if protrusion > 0:
                frac = (xc - x_start) / max(x_end - x_start, 1e-6)
                envelope = 0.5 * (1 - np.cos(2 * np.pi * frac))
                z_up[j] += protrusion * envelope * y_taper

    # Write back (reverse to Selig TE→LE)
    result[:mid + 1, 0] = x_up[::-1]
    result[:mid + 1, 1] = z_up[::-1]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Aircraft assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_airplane(params: BWBParams, x_cg: float | None = None,
                   duct_placement=None) -> asb.Airplane:
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
    body_wing = _build_center_body(params, duct_placement=duct_placement)
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
        name="MANTA BWB",
        xyz_ref=[x_ref, 0, 0],
        s_ref=s,
        c_ref=c,
        b_ref=span,
        wings=[body_wing, outer_wing],
    )
    return airplane


def _build_center_body(params: BWBParams, duct_placement=None) -> asb.Wing:
    """Build center-body as a thick, swept, low-AR wing."""
    p = params
    body_chord = p.body_root_chord
    blend_chord = p.wing_root_chord  # C0 continuity

    body_sweep = np.radians(p.body_sweep_deg)
    bw = p.body_halfwidth

    # Airfoil at centerline: thick body Kulfan
    af_center = build_body_kulfan_at_station(p, 0.0, name="body_center").to_airfoil()

    # Airfoil at mid-body: interpolated body→wing Kulfan (frac=0.5)
    af_mid = build_body_kulfan_at_station(p, 0.5, name="body_mid").to_airfoil()

    # Airfoil at blend: must match outer wing root Kulfan (C0 continuity)
    kaf_blend = build_kulfan_airfoil_at_station(p, 0.0, name="body_blend")
    af_blend = kaf_blend.to_airfoil()

    # Apply duct bump to center and mid stations (blend is at wing root, no duct)
    if duct_placement is not None:
        bc = p.body_root_chord
        mid_y = bw * 0.5
        coords_c = apply_duct_bump(af_center.coordinates, duct_placement, bc, bw, 0.0)
        af_center = asb.Airfoil(name="body_center_bump", coordinates=coords_c)
        coords_m = apply_duct_bump(af_mid.coordinates, duct_placement, bc, bw, mid_y)
        af_mid = asb.Airfoil(name="body_mid_bump", coordinates=coords_m)

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
