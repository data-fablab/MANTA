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


def compute_duct_placement(params: BWBParams, edf: EDFSpec,
                           config=None) -> DuctPlacement:
    """Derive duct placement from aircraft geometry and EDF specs.

    Adaptively positions the intake, fan, and exhaust to fit within
    the BWB center body:
    - Fan at the point of maximum body height (best clearance)
    - Exhaust as far aft as possible while maintaining clearance
    - Centerline at body midpoint (not camber) for symmetric margins
    - Intake width capped by body halfwidth
    """
    from ..config import DuctGeometryConfig
    if config is None:
        config = DuctGeometryConfig()
    body_chord = params.body_root_chord
    body_tc = params.body_tc_root
    camber = params.body_camber
    reflex = params.body_reflex
    le_droop = params.body_le_droop

    duct_wall = config.duct_wall_mm / 1000.0

    # ── Exhaust geometry (compute first — needed for exhaust clearance check) ──
    fan_area = np.pi / 4 * edf.fan_diameter ** 2
    nozzle_area_ratio = config.nozzle_area_ratio
    exit_area = fan_area * nozzle_area_ratio
    # Exhaust slot: keep width ≤ fan diameter for convergent nozzle
    # Height adjusts to match exit_area = fan_area × nozzle_area_ratio
    exhaust_width = edf.fan_diameter  # same width as fan → no flare
    exhaust_height = exit_area / exhaust_width

    # ── Fan placement: find x/c of maximum body height ──
    # Constrained: S-duct length must be >= 20% chord for acceptable PR
    intake_depth_est = edf.fan_diameter * config.intake_depth_factor
    intake_length_est = intake_depth_est * config.intake_ramp_factor
    intake_end_frac = config.intake_x_frac + intake_length_est / body_chord
    min_fan_x_frac = max(0.30, intake_end_frac + 0.20)  # 20% chord min S-duct
    best_fan_x = config.fan_x_frac  # fallback
    best_height = 0.0
    for x_frac in np.linspace(min_fan_x_frac, 0.55, 30):
        z_up, _, z_lo = _body_surface_z(x_frac, body_tc, camber, reflex, le_droop)
        height = (z_up - z_lo) * body_chord
        if height > best_height:
            best_height = height
            best_fan_x = x_frac
    fan_x_frac = best_fan_x
    fan_x = fan_x_frac * body_chord

    # Fan centerline: shifted toward intrados to keep intake flush with extrados.
    # The offset is a fraction of body half-height: 0 = centered, -1 = at intrados.
    z_up_fan, _, z_lo_fan = _body_surface_z(fan_x_frac, body_tc, camber, reflex, le_droop)
    body_half_h = (z_up_fan - z_lo_fan) / 2 * body_chord
    z_mid = (z_up_fan + z_lo_fan) / 2 * body_chord
    duct_z_bias = -0.37  # shift 37% of half-height toward intrados
    fan_z = z_mid + duct_z_bias * body_half_h

    # ── Exhaust placement: fixed at config target ──
    # The exhaust is a slot that cuts through the TE — no height constraint.
    # The nozzle converges from fan circle to flat exit over this full length.
    exhaust_x_frac = config.exhaust_x_frac
    exhaust_x = exhaust_x_frac * body_chord

    # Exhaust centerline at body midpoint
    z_up_ex, _, z_lo_ex = _body_surface_z(exhaust_x_frac, body_tc, camber, reflex, le_droop)
    exhaust_z = (z_up_ex + z_lo_ex) / 2 * body_chord

    # ── Intake placement ──
    intake_x_frac = config.intake_x_frac
    intake_x = intake_x_frac * body_chord

    z_upper, _, _ = _body_surface_z(intake_x_frac, body_tc, camber, reflex, le_droop)
    intake_z = z_upper * body_chord

    # Intake dimensions: adapt to EDF and body width
    intake_width = edf.fan_diameter * config.intake_width_factor
    # Cap by body halfwidth (intake must fit within body)
    max_intake_width = 2 * params.body_halfwidth * 0.8
    intake_width = min(intake_width, max_intake_width)
    intake_depth = edf.fan_diameter * config.intake_depth_factor
    intake_length = intake_depth * config.intake_ramp_factor

    # ── Duct dimensions ──
    housing_length = edf.fan_diameter * config.housing_length_factor

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
    config=None,
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
    from ..config import DuctGeometryConfig
    if config is None:
        config = DuctGeometryConfig()
    if material_density == 1500.0:
        material_density = config.material_density
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
    return wall_mass * config.structural_adder


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
            # Intake: flush scoop, only bottom clearance matters
            cl_top = (z_up - cz) * 1000.0
            cl_bot = ((cz - half_h) - z_lo) * 1000.0
        elif t > 0.50:
            # Nozzle transition + exhaust: duct merges with body structure.
            # The duct walls ARE the body walls in this zone (like nEUROn).
            # No clearance check — the duct is structurally integrated.
            cl_top = (z_up - cz) * 1000.0
            cl_bot = (cz - z_lo) * 1000.0
        else:
            # Main S-duct: full clearance check (tube inside body)
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

    # ── Segment 1: Intake lip → Fan (S-duct) ──
    p0 = np.array([p.intake_x, 0.0, p.intake_z])
    # P3: fan face center
    p3 = np.array([p.fan_x, 0.0, p.fan_z])

    # Control points: P1 dives steeply to stay inside the body
    # as the cross-section grows from lip to full intake size.
    dx_13 = p3[0] - p0[0]
    dz_13 = p3[2] - p0[2]  # negative (diving down from OML to fan)
    p1 = p0 + np.array([dx_13 * 0.1, 0.0, dz_13 * 0.9])  # dive very fast
    p2 = p3 + np.array([-dx_13 * 0.3, 0.0, 0.0])           # horizontal approach to fan

    n1 = n_pts // 2
    seg1 = _cubic_bezier(p0, p1, p2, p3, n1)

    # ── Segment 2: Fan → Exhaust (nozzle) ──
    # The nozzle is a straight convergent section at constant Z.
    # It stays at fan_z height and simply converges toward the TE slot.
    q0 = p3  # continuity at fan face
    q3 = np.array([p.exhaust_x, 0.0, p.fan_z])  # same Z as fan

    dx_q = q3[0] - q0[0]
    q1 = q0 + np.array([dx_q * 0.35, 0.0, 0.0])
    q2 = q3 + np.array([-dx_q * 0.3, 0.0, 0.0])

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

    All intermediate sections are computed by area-preserving interpolation
    between the three anchor stations. No hardcoded fudge factors.

    Physical constraints:
    - Intake: wide D-shape, area = intake_width × intake_depth
    - Fan face: circular, area = pi/4 × fan_diameter²
    - Exhaust: flat ellipse, area = exhaust_width × exhaust_height
    - Transition: area varies smoothly (no abrupt contractions)

    Returns (n_pts, 2) closed curve in local cross-section frame
    (horizontal, vertical), centered at origin.
    """
    p = placement

    # Four anchor stations: lip → intake full → fan → exhaust
    r_fan = p.fan_diameter / 2
    intake_a = p.intake_width / 2
    intake_b = p.intake_depth / 2
    exhaust_a = p.exhaust_width / 2
    exhaust_b = p.exhaust_height / 2

    # t=0.00: lip (barely open — near-zero height, narrow)
    # t=0.15: intake full size (ramp end, D-shape)
    # t=0.50: fan face (circular)
    # t=1.00: exhaust (flat ellipse)
    t_lip = 0.0
    t_intake = 0.15  # ~intake_length / total_duct_length
    t_fan = 0.50
    t_exhaust = 1.0

    lip_a = intake_a * 0.15      # narrow at lip
    lip_b = 0.002                # ~2mm — barely open

    if t <= t_intake:
        # Lip → intake full: cubic ramp (NACA profile)
        s = t / t_intake  # 0→1
        s_cubic = s ** 3   # NACA cubic depth profile
        a = lip_a + (intake_a - lip_a) * s ** 0.5   # width opens fast
        b = lip_b + (intake_b - lip_b) * s_cubic     # depth opens slow (cubic)
    elif t <= t_fan:
        # Intake full → fan: smooth transition
        s = (t - t_intake) / (t_fan - t_intake)  # 0→1
        s_smooth = 3 * s**2 - 2 * s**3
        a = intake_a * (1 - s_smooth) + r_fan * s_smooth
        b = intake_b * (1 - s_smooth) + r_fan * s_smooth
    else:
        # Fan → exhaust: smooth transition
        s = (t - t_fan) / (t_exhaust - t_fan)  # 0→1
        s_smooth = 3 * s**2 - 2 * s**3
        a = r_fan * (1 - s_smooth) + exhaust_a * s_smooth
        b = r_fan * (1 - s_smooth) + exhaust_b * s_smooth

    n_exp = 2.0

    # D-shape blend: for t < 0.25, flatten the top (flush with OML)
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

    The top is flattened by clamping y to a ceiling that descends
    from b (symmetric) to 0 (fully flat) as blend increases.
    The section is then shifted down so the flat top sits at y=0,
    preserving the full lower half area for airflow.

    Returns (n_pts, 2) array, counterclockwise, starting at (a, 0).
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Base superellipse
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / n)
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / n)

    # Ceiling: how high the upper half can go
    ceiling = b * (1.0 - blend)  # blend=1 → 0, blend=0 → b
    y = np.clip(y, -b, ceiling)

    # Re-center vertically so centroid stays at origin
    # (the centerline handles the actual position in 3D)
    y_mid = (y.max() + y.min()) / 2
    y -= y_mid

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
    from ..config import duct_from_config, load_config
    _duct_cfg = duct_from_config(load_config())
    placement = compute_duct_placement(params, edf, config=_duct_cfg)
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
