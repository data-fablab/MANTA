"""Control surface geometry materialization.

Computes 3D hinge lines, outlines, and dimensional data for elevons
and ailerons on the outer wing.  Uses the same coordinate system and
airfoil construction pipeline as the visualization/export module.
"""

from dataclasses import dataclass
import numpy as np

from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    build_kulfan_airfoil_at_station,
    OUTER_WING_STATIONS,
    N_OUTER_SEGMENTS,
)
from ..aero.avl_runner import ControlConfig, ControlSurface


# ═══════════════════════════════════════════════════════════════════════════
# Wing LE position helper (factorized from export.py _build_wing_sections)
# ═══════════════════════════════════════════════════════════════════════════

def compute_wing_le_positions(
    params: BWBParams,
    eta_stations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D leading-edge positions and local chords on the outer wing.

    Parameters
    ----------
    params : BWBParams
    eta_stations : (N,) array of eta fractions [0, 1] on the outer wing.
        If None, uses the 6 defining stations from OUTER_WING_STATIONS.

    Returns
    -------
    le_xyz : (N, 3) array of LE positions in meters (X, Y, Z)
    chords : (N,) array of local chord lengths in meters
    twists : (N,) array of local twist angles in degrees
    """
    p = params
    bw = p.body_halfwidth
    outer_span = p.outer_half_span
    body_sweep_rad = np.radians(p.body_sweep_deg)
    wing_sweep_rad = np.radians(p.le_sweep_deg)

    # Defining stations: LE positions and properties (7 stations, 6 segments)
    def_etas = np.array(OUTER_WING_STATIONS)
    twist_055 = p.twist_2 + (0.55 - 0.50) / (0.75 - 0.50) * (p.twist_3 - p.twist_2)
    def_twists = np.array([0.0, p.twist_1, p.twist_2, twist_055, p.twist_3, p.twist_4, p.twist_tip])
    def_dihedrals = np.array([p.dihedral_0, p.dihedral_1, p.dihedral_2, p.dihedral_2, p.dihedral_3, p.dihedral_tip])

    # Chords (linear taper)
    def_chords = np.array([
        p.wing_root_chord + eta * (p.tip_chord - p.wing_root_chord)
        for eta in def_etas
    ])

    # 3D positions at defining stations
    x_blend = bw * np.tan(body_sweep_rad)
    def_pos = np.zeros((len(def_etas), 3))
    def_pos[0] = [x_blend, bw, 0.0]
    for i in range(N_OUTER_SEGMENTS):
        dy_seg = outer_span * (def_etas[i + 1] - def_etas[i])
        dih_rad = np.radians(def_dihedrals[i])
        dx = dy_seg * np.tan(wing_sweep_rad)
        dy = dy_seg * np.cos(dih_rad)
        dz = dy_seg * np.sin(dih_rad)
        def_pos[i + 1] = def_pos[i] + [dx, dy, dz]

    if eta_stations is None:
        return def_pos, def_chords, def_twists

    # Interpolate for arbitrary eta stations
    eta_stations = np.asarray(eta_stations)
    le_xyz = np.column_stack([
        np.interp(eta_stations, def_etas, def_pos[:, 0]),
        np.interp(eta_stations, def_etas, def_pos[:, 1]),
        np.interp(eta_stations, def_etas, def_pos[:, 2]),
    ])
    chords = np.interp(eta_stations, def_etas, def_chords)
    twists = np.interp(eta_stations, def_etas, def_twists)

    return le_xyz, chords, twists


# ═══════════════════════════════════════════════════════════════════════════
# Control surface geometry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ControlSurfaceGeometry:
    """Materialized 3D geometry of a single control surface.

    All arrays use the aircraft coordinate system (X aft, Y right, Z up)
    in meters.
    """
    name: str
    hinge_line_upper: np.ndarray   # (N, 3) upper-surface points at x/c = xhinge
    hinge_line_lower: np.ndarray   # (N, 3) lower-surface points at x/c = xhinge
    te_line_upper: np.ndarray      # (N, 3) upper trailing edge points
    te_line_lower: np.ndarray      # (N, 3) lower trailing edge points
    outline: np.ndarray            # (M, 3) closed contour (for visualization)
    eta_stations: np.ndarray       # (N,) sampled eta fractions
    chord_at_stations: np.ndarray  # (N,) local chord in meters
    flap_chord: np.ndarray         # (N,) flap chord in meters
    te_thickness: np.ndarray       # (N,) airfoil thickness at hinge line in meters
    span: float                    # control surface span in meters
    area: float                    # control surface area in m²
    cf_ratio: float                # chord fraction (0.25)
    eta_inner: float
    eta_outer: float
    sgn_dup: float


def _find_surface_point(coords_2d: np.ndarray, x_target: float,
                        surface: str = "upper") -> np.ndarray:
    """Find the (x, z) point on upper or lower surface at a given x/c.

    coords_2d follows Selig convention: TE_upper → LE → TE_lower.
    Upper surface = first half (decreasing x), lower = second half (increasing x).
    """
    n = len(coords_2d)
    mid = np.argmin(coords_2d[:, 0])  # LE index

    if surface == "upper":
        xs = coords_2d[:mid + 1, 0][::-1]  # LE→TE direction (increasing x)
        zs = coords_2d[:mid + 1, 1][::-1]
    else:
        xs = coords_2d[mid:, 0]  # LE→TE direction (increasing x)
        zs = coords_2d[mid:, 1]

    z_interp = np.interp(x_target, xs, zs)
    return np.array([x_target, z_interp])


def compute_control_surface_geometry(
    params: BWBParams,
    controls: ControlConfig | None = None,
    n_spanwise: int = 20,
    n_profile: int = 100,
) -> list[ControlSurfaceGeometry]:
    """Compute 3D geometry of all control surfaces on the outer wing.

    Parameters
    ----------
    params : BWBParams
        Aircraft design parameters.
    controls : ControlConfig or None
        Control surface layout. Defaults to ControlConfig.default_bwb().
    n_spanwise : int
        Number of spanwise stations per control surface.
    n_profile : int
        Airfoil points per side for interpolation accuracy.

    Returns
    -------
    List of ControlSurfaceGeometry, one per control surface (right wing only).
    Mirror across Y=0 for left wing.
    """
    if controls is None:
        controls = ControlConfig.default_bwb()

    results = []

    for surf in controls.surfaces:
        etas = np.linspace(surf.eta_inner, surf.eta_outer, n_spanwise)
        le_xyz, chords, twists = compute_wing_le_positions(params, etas)

        xhinge = surf.xhinge  # e.g. 0.75 for 25% chord flap

        hinge_upper_pts = []
        hinge_lower_pts = []
        te_upper_pts = []
        te_lower_pts = []
        flap_chords = []
        te_thicknesses = []

        for i, eta in enumerate(etas):
            chord = chords[i]
            twist_rad = np.radians(twists[i])
            le = le_xyz[i]

            # Generate Kulfan airfoil at this station
            kaf = build_kulfan_airfoil_at_station(params, eta)
            coords_2d = kaf.to_airfoil(n_coordinates_per_side=n_profile).coordinates

            # Find hinge and TE points on upper and lower surfaces
            hinge_up_2d = _find_surface_point(coords_2d, xhinge, "upper")
            hinge_lo_2d = _find_surface_point(coords_2d, xhinge, "lower")
            te_up_2d = _find_surface_point(coords_2d, 1.0, "upper")
            te_lo_2d = _find_surface_point(coords_2d, 1.0, "lower")

            # Scale by chord
            def to_3d(pt_2d):
                x_local = pt_2d[0] * chord
                z_local = pt_2d[1] * chord
                # Apply twist
                x_rot = x_local * np.cos(twist_rad) + z_local * np.sin(twist_rad)
                z_rot = -x_local * np.sin(twist_rad) + z_local * np.cos(twist_rad)
                # Translate to 3D
                return np.array([x_rot + le[0], le[1], z_rot + le[2]])

            hinge_upper_pts.append(to_3d(hinge_up_2d))
            hinge_lower_pts.append(to_3d(hinge_lo_2d))
            te_upper_pts.append(to_3d(te_up_2d))
            te_lower_pts.append(to_3d(te_lo_2d))

            flap_chords.append(chord * surf.cf_ratio)
            te_thicknesses.append(chord * abs(hinge_up_2d[1] - hinge_lo_2d[1]))

        hinge_upper = np.array(hinge_upper_pts)
        hinge_lower = np.array(hinge_lower_pts)
        te_upper = np.array(te_upper_pts)
        te_lower = np.array(te_lower_pts)
        flap_chord_arr = np.array(flap_chords)
        te_thick_arr = np.array(te_thicknesses)

        # Build closed outline (upper hinge → upper TE → lower TE rev → lower hinge rev)
        outline = np.vstack([
            hinge_upper,
            te_upper[-1:],       # outer TE corner
            te_lower[-1:],
            hinge_lower[::-1],
            te_lower[:1],        # inner TE corner
            te_upper[:1],
            hinge_upper[:1],     # close
        ])

        # Span and area
        dy = np.diff(le_xyz[:, 1])  # spanwise increments (Y)
        span = np.sum(np.sqrt(np.diff(le_xyz[:, 0])**2 +
                               np.diff(le_xyz[:, 1])**2 +
                               np.diff(le_xyz[:, 2])**2))
        avg_flap_chords = 0.5 * (flap_chord_arr[:-1] + flap_chord_arr[1:])
        area = np.sum(avg_flap_chords * np.abs(dy))

        results.append(ControlSurfaceGeometry(
            name=surf.name,
            hinge_line_upper=hinge_upper,
            hinge_line_lower=hinge_lower,
            te_line_upper=te_upper,
            te_line_lower=te_lower,
            outline=outline,
            eta_stations=etas,
            chord_at_stations=chords,
            flap_chord=flap_chord_arr,
            te_thickness=te_thick_arr,
            span=span,
            area=area,
            cf_ratio=surf.cf_ratio,
            eta_inner=surf.eta_inner,
            eta_outer=surf.eta_outer,
            sgn_dup=surf.sgn_dup,
        ))

    return results


def classify_wing_segments(
    controls: ControlConfig | None = None,
) -> list[tuple[int, int, str]]:
    """Classify each outer wing segment as 'ruled' or 'spline'.

    Segments covered by a control surface are 'ruled' (flat panel).
    Others are 'spline' (C2 BSpline loft).

    Returns list of (idx_start, idx_end, loft_type) tuples,
    where indices refer to OUTER_WING_STATIONS.
    """
    if controls is None:
        controls = ControlConfig.default_bwb()

    etas = np.array(OUTER_WING_STATIONS)
    n_seg = len(etas) - 1
    seg_types = ['spline'] * n_seg

    for surf in controls.surfaces:
        for i in range(n_seg):
            eta_lo, eta_hi = etas[i], etas[i + 1]
            mid = (eta_lo + eta_hi) / 2
            if surf.eta_inner <= mid <= surf.eta_outer:
                seg_types[i] = 'ruled'

    # Group consecutive segments of the same type
    result = []
    i = 0
    while i < n_seg:
        j = i + 1
        while j < n_seg and seg_types[j] == seg_types[i]:
            j += 1
        result.append((i, j, seg_types[i]))
        i = j

    return result


def print_control_surface_summary(geometries: list[ControlSurfaceGeometry]) -> None:
    """Print a manufacturing-oriented summary of control surface dimensions."""
    for g in geometries:
        sym = "symmetric (pitch)" if g.sgn_dup > 0 else "antisymmetric (roll)"
        print(f"\n{'='*60}")
        print(f"  {g.name.upper()} — {sym}")
        print(f"{'='*60}")
        print(f"  Span range:    eta = {g.eta_inner:.2f} -> {g.eta_outer:.2f}")
        print(f"  Span:          {g.span*1000:.0f} mm")
        print(f"  Area:          {g.area*1e4:.0f} cm2")
        print(f"  Chord frac:    {g.cf_ratio:.0%}")
        print()
        print(f"  {'Station eta':>12s} {'Chord':>8s} {'Flap':>8s} {'TE thick':>9s} {'Hinge pin':>10s}")
        print(f"  {'-'*50}")
        # Show 5-6 representative stations
        n = len(g.eta_stations)
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        for idx in indices:
            eta = g.eta_stations[idx]
            chord_mm = g.chord_at_stations[idx] * 1000
            flap_mm = g.flap_chord[idx] * 1000
            te_mm = g.te_thickness[idx] * 1000
            # Hinge pin recommendation
            if te_mm >= 5.0:
                pin = "5-6mm"
            elif te_mm >= 3.5:
                pin = "4mm"
            elif te_mm >= 2.5:
                pin = "3mm"
            else:
                pin = "MARGINAL"
            print(f"  {eta:12.2f} {chord_mm:7.0f}mm {flap_mm:7.0f}mm {te_mm:8.1f}mm {pin:>10s}")
