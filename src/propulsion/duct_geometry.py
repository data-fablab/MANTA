"""Parametric 3D geometry for the propulsion duct system.

Generates the complete internal flow path:
    Dorsal NACA intake → S-duct → EDF housing → Convergent exhaust nozzle

All geometry is derived from BWBParams + EDFSpec — no additional design
variables. The duct placement is computed automatically to fit within the
BWB center body envelope.

Coordinate system (same as bwb_aircraft.py):
    X — chordwise (LE = 0, TE = chord)
    Y — spanwise  (centerline = 0)
    Z — vertical  (positive up)

All dimensions in meters.
"""

import numpy as np
from dataclasses import dataclass
from ..parameterization.design_variables import BWBParams
from .edf_model import EDFSpec, EDF_70MM


# ═══════════════════════════════════════════════════════════════════════════
# Duct placement parameters (derived, not design variables)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DuctPlacement:
    """Derived propulsion placement parameters.

    All positions are in meters, fractions are relative to body_root_chord.
    These are NOT design variables — they are computed from BWBParams + EDFSpec.
    """

    # Intake (dorsal NACA flush scoop on upper surface)
    intake_x_frac: float          # intake center x/c on body chord
    intake_x: float               # [m] intake center x position
    intake_z: float               # [m] intake center z (on upper OML surface)
    intake_width: float           # [m] NACA scoop width (spanwise)
    intake_length: float          # [m] NACA scoop ramp length (chordwise)
    intake_depth: float           # [m] NACA scoop max ramp depth

    # Fan face
    fan_x_frac: float             # fan face center x/c
    fan_x: float                  # [m] fan face center x position
    fan_z: float                  # [m] fan face center z (near camber line)
    fan_diameter: float           # [m] fan diameter
    duct_od: float                # [m] duct outer diameter (fan + shroud)

    # Exhaust nozzle exit
    exhaust_x_frac: float         # exhaust exit x/c
    exhaust_x: float              # [m] exhaust exit x position
    exhaust_z: float              # [m] exhaust exit z position
    exhaust_width: float          # [m] exit slot width
    exhaust_height: float         # [m] exit slot height

    # Duct dimensions
    duct_wall_thickness: float    # [m] wall thickness
    housing_length: float         # [m] EDF housing length (cylindrical section)
    body_chord: float             # [m] body root chord (reference)


def compute_duct_placement(params: BWBParams, edf: EDFSpec) -> DuctPlacement:
    """Derive duct placement from aircraft geometry and EDF specs.

    Automatically positions the intake, fan, and exhaust to fit within
    the BWB center body while maintaining aerodynamic constraints.
    """
    body_chord = params.body_root_chord
    body_tc = params.body_tc_root

    # ── Intake placement ──
    # Dorsal lip intake very close to the leading edge (nEUROn-style).
    # Shallow ramp (25mm depth, 50mm length) to maximise S-duct length
    # and keep offset/L < 0.45 for good pressure recovery.
    intake_x_frac = 0.04
    intake_x = intake_x_frac * body_chord

    # Upper surface z at intake location (from body airfoil)
    z_upper, z_camber, _ = _body_surface_z(intake_x_frac, body_tc,
                                            params.body_camber, params.body_reflex,
                                            params.body_le_droop)
    intake_z = z_upper * body_chord

    # Intake dimensions: wide D-shaped dorsal lip (nEUROn style).
    # Wider and shallower than a classic NACA scoop — the D-shape
    # transitions smoothly from the flat upper surface into the S-duct.
    # Width = 1.8 × fan diameter, depth ≈ 0.5 × fan_dia, length = 3 × depth.
    intake_width = edf.fan_diameter * 1.8     # wider for D-shape
    intake_depth = edf.fan_diameter * 0.5     # ~35mm for 70mm EDF
    intake_length = intake_depth * 3.0        # longer ramp for smooth transition

    # ── Fan face placement ──
    # Fan at 40% chord — thick part of the body, centered on the camber
    # line so the duct stays entirely within the body envelope.
    fan_x_frac = 0.40
    fan_x = fan_x_frac * body_chord

    z_upper_fan, z_camber_fan, _ = _body_surface_z(fan_x_frac, body_tc,
                                                     params.body_camber,
                                                     params.body_reflex,
                                                     params.body_le_droop)
    # Fan centered on camber line (symmetric clearance top/bottom)
    fan_z = z_camber_fan * body_chord

    # ── Exhaust placement ──
    # Flat slot near trailing edge (x/c=0.93) where 27mm of thickness
    # is available for the convergent nozzle exit.
    exhaust_x_frac = 0.93
    exhaust_x = exhaust_x_frac * body_chord

    z_upper_ex, z_camber_ex, _ = _body_surface_z(exhaust_x_frac, body_tc,
                                                   params.body_camber,
                                                   params.body_reflex,
                                                   params.body_le_droop)
    exhaust_z = z_camber_ex * body_chord

    # Exhaust slot: flatten to wide, low-profile slot for stealth
    fan_area = np.pi / 4 * edf.fan_diameter ** 2
    nozzle_area_ratio = 0.80  # convergent nozzle
    exit_area = fan_area * nozzle_area_ratio
    # Aspect ratio ~3:1 for flat exhaust
    exhaust_width = np.sqrt(exit_area * 3.0)
    exhaust_height = exit_area / exhaust_width

    # ── Duct dimensions ──
    duct_wall = 0.001  # 1mm wall (carbon prepreg or 3D print LW-PLA)
    housing_length = edf.fan_diameter * 1.5  # motor + fan assembly length

    return DuctPlacement(
        intake_x_frac=intake_x_frac,
        intake_x=intake_x,
        intake_z=intake_z,
        intake_width=intake_width,
        intake_length=intake_length,
        intake_depth=intake_depth,
        fan_x_frac=fan_x_frac,
        fan_x=fan_x,
        fan_z=fan_z,
        fan_diameter=edf.fan_diameter,
        duct_od=edf.duct_outer_diameter,
        exhaust_x_frac=exhaust_x_frac,
        exhaust_x=exhaust_x,
        exhaust_z=exhaust_z,
        exhaust_width=exhaust_width,
        exhaust_height=exhaust_height,
        duct_wall_thickness=duct_wall,
        housing_length=housing_length,
        body_chord=body_chord,
    )


def compute_duct_structure_mass(
    placement: DuctPlacement,
    material_density: float = 1500.0,
) -> float:
    """Estimate mass of the duct wall structure (carbon composite).

    Computes the wall volume from the duct cross-section envelope along
    the full centerline (intake→exhaust), multiplied by composite density.
    Adds 30% for flanges, mounting pylons, firewall, and fasteners.

    Parameters
    ----------
    placement : DuctPlacement
    material_density : float
        Composite density [kg/m³]. Default 1500 (carbon-epoxy).

    Returns
    -------
    float : duct structure mass [kg] (wall + 30% mounting hardware).
    """
    centerline = compute_duct_centerline(placement, n_pts=40)
    wall_t = placement.duct_wall_thickness  # [m]

    x_start = centerline[0, 0]
    x_end = centerline[-1, 0]
    x_span = max(x_end - x_start, 1e-9)

    wall_volume = 0.0
    for i in range(len(centerline) - 1):
        cx = (centerline[i, 0] + centerline[i + 1, 0]) / 2
        t = np.clip((cx - x_start) / x_span, 0.0, 1.0)

        # Cross-section at this station
        cs = duct_cross_section(t, placement, n_pts=32)
        # Perimeter ≈ sum of point-to-point distances
        cs_closed = np.vstack([cs, cs[0]])
        perimeter = float(np.sum(np.sqrt(np.diff(cs_closed[:, 0])**2 +
                                          np.diff(cs_closed[:, 1])**2)))

        # Segment length along centerline
        dx = float(np.linalg.norm(centerline[i + 1] - centerline[i]))

        wall_volume += perimeter * wall_t * dx

    wall_mass = wall_volume * material_density
    return wall_mass * 1.30  # +30% for flanges, pylons, firewall


# ═══════════════════════════════════════════════════════════════════════════
# Body surface evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _body_surface_z(x_frac: float, tc: float, camber: float,
                    reflex: float, le_droop: float) -> tuple[float, float, float]:
    """Evaluate body airfoil surfaces and camber z/c at x/c.

    Returns (z_upper/c, z_camber/c, z_lower/c) — normalized by chord.
    Uses the same NACA 4-digit + parametric camber as bwb_aircraft.py.
    """
    x = x_frac

    # NACA 4-digit thickness distribution
    yt = (tc / 0.20) * (
        0.2969 * np.sqrt(max(x, 0))
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    # LE droop correction
    le_factor = le_droop / 0.16
    yt_le_corr = (le_factor - 1.0) * 0.2969 * np.sqrt(max(x, 0)) * (tc / 0.20)
    yt += yt_le_corr * np.exp(-8 * x)

    # Camber line
    yc = 4 * camber * x * (1 - x)

    # Reflex
    reflex_start = 0.70
    if x > reflex_start:
        reflex_x = (x - reflex_start) / (1.0 - reflex_start)
        yc += reflex * reflex_x ** 2

    z_upper = yc + yt
    z_camber = yc
    z_lower = yc - yt
    return z_upper, z_camber, z_lower


# ═══════════════════════════════════════════════════════════════════════════
# Clearance validation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClearanceResult:
    """Per-station duct-to-OML clearance result."""

    x_frac: float              # station x/c
    x_mm: float                # station x in mm
    centerline_z_mm: float     # duct centerline z in mm
    duct_half_h_mm: float      # duct half-height at this station (mm)
    body_z_upper_mm: float     # OML upper surface z (mm)
    body_z_lower_mm: float     # OML lower surface z (mm)
    clearance_top_mm: float    # gap: OML upper - duct top (positive = OK)
    clearance_bot_mm: float    # gap: duct bottom - OML lower (positive = OK)
    is_ok: bool                # True if both clearances >= min_clearance


def validate_duct_clearance(
    placement: DuctPlacement,
    params: BWBParams,
    min_clearance_mm: float = 5.0,
    n_stations: int = 50,
) -> tuple[bool, list['ClearanceResult']]:
    """Check that the duct fits within the body envelope at all stations.

    Evaluates clearance between the duct outer boundary and the body
    OML (upper and lower surfaces) at ``n_stations`` points along the
    duct centerline.

    The intake transition region (t < 0.15) is treated specially: the
    intake is a flush scoop cut into the upper surface, so top clearance
    is measured from the OML to the duct bottom rather than the duct top.

    Returns (all_ok, results) where all_ok is True when every station
    has at least ``min_clearance_mm`` of margin on both sides.
    """
    centerline = compute_duct_centerline(placement, n_pts=n_stations)
    body_chord = placement.body_chord

    # X range of the duct (intake entry → exhaust exit)
    x_start = centerline[0, 0]
    x_end = centerline[-1, 0]
    x_span = max(x_end - x_start, 1e-9)

    results: list[ClearanceResult] = []
    all_ok = True

    for i in range(len(centerline)):
        cx, _, cz = centerline[i]

        # Parametric t along duct (0=intake, 1=exhaust)
        t = np.clip((cx - x_start) / x_span, 0.0, 1.0)

        # Duct cross-section half-height at this station
        cs_2d = duct_cross_section(t, placement, n_pts=32)
        half_h = float(np.max(cs_2d[:, 1]) - np.min(cs_2d[:, 1])) / 2.0

        # Body envelope at this x/c
        x_frac = np.clip(cx / body_chord, 0.01, 0.99)
        z_up_c, _, z_lo_c = _body_surface_z(
            x_frac, params.body_tc_root, params.body_camber,
            params.body_reflex, params.body_le_droop,
        )
        z_up = z_up_c * body_chord   # meters
        z_lo = z_lo_c * body_chord   # meters

        # Clearances in mm
        # The intake (t < 0.15) is a flush scoop cut into the upper surface:
        # the duct top IS the OML, only bottom clearance matters.
        # The exhaust (t > 0.85) is a slot cut through the trailing edge:
        # both top and bottom are openings, clearance is measured from
        # the duct centerline to the OML only.
        if t < 0.15:
            cl_top = (z_up - cz) * 1000.0
            cl_bot = ((cz - half_h) - z_lo) * 1000.0
        elif t > 0.85:
            cl_top = (z_up - cz) * 1000.0
            cl_bot = (cz - z_lo) * 1000.0
        else:
            cl_top = (z_up - (cz + half_h)) * 1000.0
            cl_bot = ((cz - half_h) - z_lo) * 1000.0

        ok = cl_top >= min_clearance_mm and cl_bot >= min_clearance_mm
        if not ok:
            all_ok = False

        results.append(ClearanceResult(
            x_frac=round(x_frac, 4),
            x_mm=round(cx * 1000, 1),
            centerline_z_mm=round(cz * 1000, 1),
            duct_half_h_mm=round(half_h * 1000, 1),
            body_z_upper_mm=round(z_up * 1000, 1),
            body_z_lower_mm=round(z_lo * 1000, 1),
            clearance_top_mm=round(cl_top, 1),
            clearance_bot_mm=round(cl_bot, 1),
            is_ok=ok,
        ))

    return all_ok, results


# ═══════════════════════════════════════════════════════════════════════════
# Duct centerline — two cubic Bezier segments
# ═══════════════════════════════════════════════════════════════════════════

def compute_duct_centerline(placement: DuctPlacement,
                            n_pts: int = 60) -> np.ndarray:
    """Compute the full duct centerline from intake to exhaust.

    The centerline is composed of two cubic Bezier segments:
    1. Intake → Fan face (S-duct, curves downward)
    2. Fan face → Exhaust exit (nozzle, curves to TE)

    Returns (n_pts, 3) array [x, y=0, z] in meters.
    """
    p = placement

    # ── Segment 1: Intake throat → Fan (S-duct) ──
    # P0: throat bottom (end of NACA ramp, submerged below OML)
    # Per NACA submerged inlet design: the S-duct starts at the
    # deepest point of the intake ramp, NOT at the upper surface.
    p0 = np.array([p.intake_x + p.intake_length, 0.0,
                   p.intake_z - p.intake_depth])
    # P3: fan face center
    p3 = np.array([p.fan_x, 0.0, p.fan_z])

    # Control points: tangent at throat is nearly horizontal (shallow
    # ramp exit, ~5-7° per NACA guidelines), tangent at fan is horizontal.
    dx_13 = p3[0] - p0[0]
    p1 = p0 + np.array([dx_13 * 0.4, 0.0, 0.0])     # horizontal departure
    p2 = p3 + np.array([-dx_13 * 0.35, 0.0, 0.0])    # horizontal approach

    n1 = n_pts // 2
    seg1 = _cubic_bezier(p0, p1, p2, p3, n1)

    # ── Segment 2: Fan → Exhaust (nozzle) ──
    q0 = p3  # continuity at fan face
    q3 = np.array([p.exhaust_x, 0.0, p.exhaust_z])

    dx_q = q3[0] - q0[0]
    q1 = q0 + np.array([dx_q * 0.35, 0.0, 0.0])      # horizontal departure
    q2 = q3 + np.array([-dx_q * 0.3, 0.0, 0.0])       # approach along chord

    n2 = n_pts - n1
    seg2 = _cubic_bezier(q0, q1, q2, q3, n2 + 1)

    # Merge (drop duplicate at junction)
    centerline = np.vstack([seg1, seg2[1:]])
    return centerline


def _cubic_bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    """Evaluate cubic Bezier curve at n uniform parameter values."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    curve = ((1 - t) ** 3 * p0
             + 3 * (1 - t) ** 2 * t * p1
             + 3 * (1 - t) * t ** 2 * p2
             + t ** 3 * p3)
    return curve


# ═══════════════════════════════════════════════════════════════════════════
# Cross-section generation — superellipse family
# ═══════════════════════════════════════════════════════════════════════════

def duct_cross_section(t: float, placement: DuctPlacement,
                       n_pts: int = 32) -> np.ndarray:
    """Generate a duct cross-section at parametric station t.

    t = 0.0 → intake (D-shape: flat top, semicircular bottom — nEUROn style)
    t = 0.5 → fan face (circular)
    t = 1.0 → exhaust exit (elliptical, flat)

    For t < 0.25, sections are D-shaped (flat top flush with OML surface,
    rounded bottom). This blends smoothly to the circular fan face at t=0.5.

    Returns (n_pts, 2) closed curve in local cross-section frame
    (horizontal, vertical), centered at origin.
    """
    p = placement

    # Key station parameters: (half_width, half_height, superellipse_exponent)
    stations = {
        0.0: (p.intake_width / 2, p.intake_depth / 2, 2.5),       # D-shape (wide, shallow)
        0.3: (p.duct_od / 2 * 1.1, p.duct_od / 2 * 0.9, 2.5),    # transition
        0.5: (p.fan_diameter / 2, p.fan_diameter / 2, 2.0),        # circular
        0.7: (p.fan_diameter / 2 * 1.05, p.fan_diameter / 2 * 0.9, 2.2),  # transition
        1.0: (p.exhaust_width / 2, p.exhaust_height / 2, 2.5),    # elliptical
    }

    # Interpolate parameters at t
    ts = sorted(stations.keys())
    a_vals = [stations[s][0] for s in ts]
    b_vals = [stations[s][1] for s in ts]
    n_vals = [stations[s][2] for s in ts]

    a = np.interp(t, ts, a_vals)
    b = np.interp(t, ts, b_vals)
    n_exp = np.interp(t, ts, n_vals)

    # D-shape blend: for t < 0.25, flatten the top of the section
    # (the upper half becomes a straight line = flush with OML surface).
    # d_blend goes from 1.0 (full D-shape) at t=0 to 0.0 (symmetric) at t=0.25.
    d_blend = max(0.0, 1.0 - t / 0.25)

    if d_blend > 0.01:
        return _dshape(a, b, n_exp, n_pts, d_blend)
    else:
        return _superellipse(a, b, n_exp, n_pts)


def _dshape(a: float, b: float, n: float, n_pts: int,
            blend: float = 1.0) -> np.ndarray:
    """Generate a D-shaped cross-section (flat top, rounded bottom).

    blend=1.0: fully D-shaped (top is flat line at y=0)
    blend=0.0: fully symmetric (same as superellipse)

    The top half is flattened: the upper y-coordinates are compressed
    toward zero, creating a D-shape that smoothly transitions to a
    symmetric ellipse as blend→0.

    Returns (n_pts, 2) array, counterclockwise, starting at (a, 0).
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Base superellipse
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / n)
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / n)

    # Flatten the top (y > 0): compress upper half toward zero
    upper = y > 0
    y[upper] *= (1.0 - blend)  # blend=1 → flat, blend=0 → unchanged

    return np.column_stack([x, y])


def _superellipse(a: float, b: float, n: float, n_pts: int) -> np.ndarray:
    """Generate a closed superellipse |x/a|^n + |y/b|^n = 1.

    Returns (n_pts, 2) array, counterclockwise, starting at (a, 0).
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Signed superellipse parametrization
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / n)
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / n)

    return np.column_stack([x, y])


# ═══════════════════════════════════════════════════════════════════════════
# Frenet frame along centerline
# ═══════════════════════════════════════════════════════════════════════════

def _compute_frenet_frames(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Frenet-like frames (T, N, B) along the duct centerline.

    Since the centerline lies in the XZ plane (y=0), we use:
    - T = tangent vector (along centerline)
    - B = Y-axis (spanwise, out of the XZ plane)
    - N = B × T (in XZ plane, perpendicular to tangent)

    Returns three (n, 3) arrays: tangents, normals, binormals.
    """
    n = len(centerline)

    # Tangent via central differences
    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]

    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    tangents = tangents / norms

    # Binormal = Y-axis (constant, since centerline is in XZ plane)
    binormals = np.tile([0.0, 1.0, 0.0], (n, 1))

    # Normal = B × T
    normals = np.cross(binormals, tangents)
    n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norms = np.maximum(n_norms, 1e-12)
    normals = normals / n_norms

    return tangents, normals, binormals


def _place_cross_section(center: np.ndarray, normal: np.ndarray,
                         binormal: np.ndarray, cs_2d: np.ndarray) -> np.ndarray:
    """Transform a 2D cross-section into 3D at a centerline point.

    cs_2d: (n_pts, 2) — [horizontal, vertical] in local frame.
    horizontal → binormal direction (spanwise)
    vertical → normal direction (in XZ plane)

    Returns (n_pts, 3) world coordinates.
    """
    pts_3d = (center[np.newaxis, :]
              + cs_2d[:, 0:1] * binormal[np.newaxis, :]
              + cs_2d[:, 1:2] * normal[np.newaxis, :])
    return pts_3d


# ═══════════════════════════════════════════════════════════════════════════
# Component geometry builders
# ═══════════════════════════════════════════════════════════════════════════

def build_intake_geometry(placement: DuctPlacement, params: BWBParams,
                          n_profile: int = 32, n_axial: int = 10) -> list[np.ndarray]:
    """Build the NACA flush intake scoop geometry.

    The NACA submerged intake ramp follows a cubic depth profile:
        depth(x) = d_max * (x / L)^3
    where L is the ramp length and d_max is the maximum ramp depth.

    Returns list of (n_profile, 3) cross-section arrays, from ramp
    leading edge to the duct entry, suitable for loft triangulation.
    """
    p = placement
    body_chord = p.body_chord

    sections = []
    for i in range(n_axial):
        frac = i / max(n_axial - 1, 1)

        # Axial position along the ramp
        x_local = p.intake_x + frac * p.intake_length

        # Ramp depth (cubic profile)
        depth = p.intake_depth * frac ** 3

        # Width tapers from 0 at lip to full width
        width = p.intake_width * max(frac ** 0.5, 0.15)

        # Upper surface z at this x position
        x_frac = x_local / body_chord
        x_frac = np.clip(x_frac, 0.01, 0.99)
        z_upper, _, _ = _body_surface_z(x_frac, params.body_tc_root,
                                      params.body_camber, params.body_reflex,
                                      params.body_le_droop)
        z_top = z_upper * body_chord

        # Cross-section: rectangular scoop cut into the upper surface
        # Top edge = OML surface, bottom edge = ramp floor
        half_w = width / 2
        z_bottom = z_top - depth

        # Build rectangular cross-section (n_profile points)
        cs = _rectangular_section(half_w, z_top, z_bottom, n_profile)

        # Place in 3D (y=0 symmetry plane, centered spanwise)
        pts_3d = np.column_stack([
            np.full(n_profile, x_local),
            cs[:, 0],  # y (spanwise)
            cs[:, 1],  # z (vertical)
        ])
        sections.append(pts_3d)

    return sections


def _rectangular_section(half_w: float, z_top: float, z_bottom: float,
                         n_pts: int) -> np.ndarray:
    """Build a rectangular cross-section as a closed curve.

    Returns (n_pts, 2) array with [y, z] coordinates.
    The rectangle has corners at (±half_w, z_top) and (±half_w, z_bottom).
    """
    # Distribute points around the rectangle perimeter
    # Right wall → bottom → left wall → top
    n_side = max(n_pts // 4, 2)
    n_top_bot = max(n_pts // 4, 2)

    # Adjust to get exactly n_pts
    n_right = n_side
    n_bottom = n_top_bot
    n_left = n_side
    n_top = n_pts - n_right - n_bottom - n_left

    points = []

    # Right wall (top to bottom)
    for i in range(n_right):
        f = i / max(n_right, 1)
        points.append([half_w, z_top + f * (z_bottom - z_top)])

    # Bottom (right to left)
    for i in range(n_bottom):
        f = i / max(n_bottom, 1)
        points.append([half_w + f * (-2 * half_w), z_bottom])

    # Left wall (bottom to top)
    for i in range(n_left):
        f = i / max(n_left, 1)
        points.append([-half_w, z_bottom + f * (z_top - z_bottom)])

    # Top (left to right)
    for i in range(n_top):
        f = i / max(n_top, 1)
        points.append([-half_w + f * (2 * half_w), z_top])

    return np.array(points[:n_pts])


def build_sduct_geometry(placement: DuctPlacement, params: BWBParams,
                         edf: EDFSpec, n_stations: int = 20,
                         n_profile: int = 32) -> list[np.ndarray]:
    """Build the S-duct cross-sections from intake exit to fan face.

    Each section is oriented perpendicular to the centerline tangent
    using the Frenet frame. Cross-sections transition smoothly from
    the rectangular intake shape to the circular fan face.

    Returns list of (n_profile, 3) arrays in world coordinates.
    """
    centerline = compute_duct_centerline(placement)
    tangents, normals, binormals = _compute_frenet_frames(centerline)

    # The S-duct covers the first half of the centerline (t: 0 → 0.5)
    n_cl = len(centerline)
    n_sduct = n_cl // 2  # first half

    # Sample n_stations evenly from the S-duct portion
    indices = np.linspace(0, n_sduct - 1, n_stations, dtype=int)

    sections = []
    for idx in indices:
        t_param = idx / max(n_cl - 1, 1)  # parametric position [0, ~0.5]
        t_cs = t_param * 2.0  # map to [0, 1] for cross-section (0=intake, 0.5=fan)
        t_cs = min(t_cs, 0.5)

        cs_2d = duct_cross_section(t_cs, placement, n_profile)
        pts_3d = _place_cross_section(centerline[idx], normals[idx],
                                       binormals[idx], cs_2d)
        sections.append(pts_3d)

    return sections


def build_edf_housing(placement: DuctPlacement, edf: EDFSpec,
                      n_profile: int = 32, n_axial: int = 6) -> list[np.ndarray]:
    """Build the cylindrical EDF housing section.

    A short cylindrical duct centered on the fan face, accommodating
    the motor + fan assembly.

    Returns list of (n_profile, 3) cross-section arrays.
    """
    p = placement
    half_len = p.housing_length / 2

    # Centerline at fan face
    center = np.array([p.fan_x, 0.0, p.fan_z])

    # Housing axis direction: approximate from centerline tangent at fan
    # For simplicity, use the local tangent direction
    cl = compute_duct_centerline(placement)
    mid_idx = len(cl) // 2
    tangent = cl[min(mid_idx + 1, len(cl) - 1)] - cl[max(mid_idx - 1, 0)]
    tangent = tangent / np.linalg.norm(tangent)

    binormal = np.array([0.0, 1.0, 0.0])
    normal = np.cross(binormal, tangent)
    normal = normal / np.linalg.norm(normal)

    # Circular cross-section at fan diameter
    cs_2d = duct_cross_section(0.5, placement, n_profile)  # t=0.5 = circular

    sections = []
    for i in range(n_axial):
        frac = i / max(n_axial - 1, 1)
        offset = -half_len + frac * p.housing_length
        c = center + offset * tangent
        pts_3d = _place_cross_section(c, normal, binormal, cs_2d)
        sections.append(pts_3d)

    return sections


def build_exhaust_geometry(placement: DuctPlacement, params: BWBParams,
                           edf: EDFSpec, n_stations: int = 12,
                           n_profile: int = 32) -> list[np.ndarray]:
    """Build the convergent exhaust nozzle from fan exit to trailing edge.

    Cross-sections transition from circular (fan exit) to a wide,
    flat elliptical slot at the trailing edge.

    Returns list of (n_profile, 3) arrays in world coordinates.
    """
    centerline = compute_duct_centerline(placement)
    tangents, normals, binormals = _compute_frenet_frames(centerline)

    n_cl = len(centerline)
    n_nozzle_start = n_cl // 2  # second half of centerline

    # Sample n_stations from the nozzle portion
    indices = np.linspace(n_nozzle_start, n_cl - 1, n_stations, dtype=int)

    sections = []
    for idx in indices:
        t_param = idx / max(n_cl - 1, 1)  # [~0.5, 1.0]
        t_cs = t_param  # maps directly to cross-section parametrization

        cs_2d = duct_cross_section(t_cs, placement, n_profile)
        pts_3d = _place_cross_section(centerline[idx], normals[idx],
                                       binormals[idx], cs_2d)
        sections.append(pts_3d)

    return sections


def build_exhaust_fairing(placement: DuctPlacement, params: BWBParams,
                          edf: EDFSpec, skin_mm: float = 3.0,
                          n_stations: int = 20,
                          n_profile: int = 48) -> list[np.ndarray]:
    """Build the duct shell (coque) — the outer wall of the full duct.

    Creates cross-sections that follow the duct shape + skin thickness
    along the entire duct length (intake throat → exhaust exit).
    When fused with the OML, only the portions that protrude beyond
    the body envelope create a local bump (at the thin TE region).

    Parameters
    ----------
    skin_mm : float
        Skin/clearance thickness added around the duct (mm).
    n_stations : int
        Number of axial stations for the fairing loft.
    n_profile : int
        Points per cross-section.

    Returns
    -------
    list of (n_profile, 3) arrays in world coordinates (meters).
    """
    p = placement
    bc = p.body_chord
    skin = skin_mm / 1000.0  # mm → m

    centerline = compute_duct_centerline(p)

    # Shell covers the FULL duct: intake lip → throat → S-duct → exhaust
    # The centerline starts at the throat; we prepend the intake ramp.
    cl_x = centerline[:, 0]
    cl_z = centerline[:, 2]

    # Intake ramp: from intake_x to throat (centerline[0])
    n_ramp = max(n_stations // 4, 3)
    ramp_x = np.linspace(p.intake_x, cl_x[0], n_ramp, endpoint=False)

    # Main duct: from throat to exhaust (along centerline)
    n_main = n_stations - n_ramp
    main_x = np.linspace(cl_x[0], cl_x[-1], n_main)

    sections = []

    # ── Intake ramp sections ──
    for x in ramp_x:
        frac = (x - p.intake_x) / max(p.intake_length, 1e-9)
        frac = np.clip(frac, 0.0, 1.0)

        # Ramp depth follows cubic profile
        depth = p.intake_depth * frac ** 3
        # z at ramp center (between upper surface and ramp floor)
        ramp_z = p.intake_z - depth / 2.0

        # Width tapers from narrow at lip to full at throat
        half_w = (p.intake_width / 2 + skin) * max(frac ** 0.5, 0.15)
        half_h = (depth / 2.0 + skin) if depth > 1e-6 else skin

        # Build elliptical cross-section
        theta = np.linspace(0, 2 * np.pi, n_profile, endpoint=False)
        pts_3d = np.column_stack([
            np.full(n_profile, x),
            half_w * np.cos(theta),
            ramp_z + half_h * np.sin(theta),
        ])
        sections.append(pts_3d)

    # ── Main duct sections (throat → exhaust) ──
    for x in main_x:
        cz = float(np.interp(x, cl_x, cl_z))

        x_span = max(cl_x[-1] - cl_x[0], 1e-9)
        t = np.clip((x - cl_x[0]) / x_span, 0.0, 1.0)

        cs_2d = duct_cross_section(t, p, n_pts=n_profile)

        # Expand by skin thickness
        cs_scaled = cs_2d.copy()
        cs_scaled[:, 0] *= 1.0 + skin / max(np.abs(cs_2d[:, 0]).max(), 1e-6)
        cs_scaled[:, 1] *= 1.0 + skin / max(np.abs(cs_2d[:, 1]).max(), 1e-6)

        pts_3d = np.column_stack([
            np.full(n_profile, x),
            cs_scaled[:, 0],
            cz + cs_scaled[:, 1],
        ])
        sections.append(pts_3d)

    return sections


# ═══════════════════════════════════════════════════════════════════════════
# Full propulsion mesh assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_propulsion_mesh(params: BWBParams, edf: EDFSpec,
                          n_profile: int = 32) -> list:
    """Build the complete propulsion duct as triangulated mesh.

    Assembles intake, S-duct, EDF housing, and exhaust nozzle into
    a single triangle list compatible with the STL export pipeline.

    Returns list of [p0, p1, p2] triangle vertex arrays (same format
    as _loft_sections in export.py).
    """
    placement = compute_duct_placement(params, edf)
    triangles = []

    # Build all component cross-sections
    intake_secs = build_intake_geometry(placement, params,
                                         n_profile=n_profile, n_axial=8)
    sduct_secs = build_sduct_geometry(placement, params, edf,
                                       n_stations=16, n_profile=n_profile)
    housing_secs = build_edf_housing(placement, edf,
                                      n_profile=n_profile, n_axial=4)
    exhaust_secs = build_exhaust_geometry(placement, params, edf,
                                           n_stations=10, n_profile=n_profile)

    # Loft each component
    for component_secs in [intake_secs, sduct_secs, housing_secs, exhaust_secs]:
        for i in range(len(component_secs) - 1):
            _loft_duct_sections(component_secs[i], component_secs[i + 1], triangles)

    # Cap intake entrance and exhaust exit
    if intake_secs:
        _cap_duct_section(intake_secs[0], triangles, flip=True)
    if exhaust_secs:
        _cap_duct_section(exhaust_secs[-1], triangles, flip=False)

    return triangles


def build_propulsion_mesh_mirrored(params: BWBParams, edf: EDFSpec,
                                    n_profile: int = 32) -> list:
    """Build propulsion duct mesh for both left and right sides.

    The duct sits on the centerline (y=0), so only a single duct is
    generated (no mirror needed for a single-engine configuration).
    Returns the same triangle list as build_propulsion_mesh.
    """
    return build_propulsion_mesh(params, edf, n_profile)


# ═══════════════════════════════════════════════════════════════════════════
# Mesh helpers
# ═══════════════════════════════════════════════════════════════════════════

def _loft_duct_sections(sec_a: np.ndarray, sec_b: np.ndarray, triangles: list):
    """Create quad strip (2 triangles each) between two duct sections."""
    n = min(len(sec_a), len(sec_b))
    for j in range(n - 1):
        p0 = sec_a[j]
        p1 = sec_a[j + 1]
        p2 = sec_b[j + 1]
        p3 = sec_b[j]
        triangles.append([p0, p1, p2])
        triangles.append([p0, p2, p3])
    # Close the loop (last point → first point)
    triangles.append([sec_a[-1], sec_a[0], sec_b[0]])
    triangles.append([sec_a[-1], sec_b[0], sec_b[-1]])


def _cap_duct_section(section: np.ndarray, triangles: list, flip: bool = False):
    """Close a duct section with a fan of triangles from the centroid."""
    centroid = section.mean(axis=0)
    n = len(section)
    for j in range(n):
        j_next = (j + 1) % n
        if flip:
            triangles.append([centroid, section[j_next], section[j]])
        else:
            triangles.append([centroid, section[j], section[j_next]])


# ═══════════════════════════════════════════════════════════════════════════
# Duct profile export helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_duct_profiles(placement: DuctPlacement, n_profile: int = 32) -> dict:
    """Get key duct cross-section profiles for CAD export.

    Returns dict mapping station names to (n_profile, 2) arrays
    of cross-section coordinates (in local frame, meters).
    """
    profiles = {}
    station_names = {
        0.0: "intake",
        0.25: "sduct_25pct",
        0.5: "fan_face",
        0.75: "nozzle_75pct",
        1.0: "exhaust_exit",
    }

    for t, name in station_names.items():
        cs = duct_cross_section(t, placement, n_profile)
        profiles[name] = cs

    return profiles


def get_duct_centerline_csv(placement: DuctPlacement,
                            n_pts: int = 50) -> np.ndarray:
    """Get duct centerline points for CSV export.

    Returns (n_pts, 3) array [x, y, z] in meters.
    """
    return compute_duct_centerline(placement, n_pts)
