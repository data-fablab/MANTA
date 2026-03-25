"""Export BWB aircraft geometry to STL mesh.

Produces triangulated STL files suitable for visualization and 3D printing.
"""

import numpy as np
from stl import mesh as stl_mesh
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    build_kulfan_airfoil_at_station,
    build_body_kulfan_at_station,
    OUTER_WING_STATIONS,
    N_OUTER_SEGMENTS,
    BODY_STATIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# STL export — body + outer wing
# ═══════════════════════════════════════════════════════════════════════════

def export_aircraft_stl(params: BWBParams, path: str, n_profile: int = 30,
                        n_interp: int = 8,
                        include_propulsion: bool = False,
                        edf: 'EDFSpec | None' = None):
    """Export complete BWB aircraft (body + outer wing) as a single STL file.

    Uses cubic spline interpolation in the spanwise direction to produce
    smooth surfaces between the defining sections (spline loft).

    Parameters
    ----------
    params : BWBParams
    path : str
        Output .stl file path.
    n_profile : int
        Number of points per airfoil section (higher = smoother).
    n_interp : int
        Number of interpolated sections between each pair of defining
        stations. Higher = smoother surface (8 is a good default).
    include_propulsion : bool
        If True, add propulsion duct geometry (intake, S-duct, nozzle)
        to the STL mesh. Requires edf to be provided.
    edf : EDFSpec or None
        EDF specification for propulsion geometry. Required if
        include_propulsion is True. Defaults to EDF_70MM.

    Returns
    -------
    int
        Number of triangles written.
    """
    p = params

    # Compute duct placement for body bump (consistent with AVL model)
    _duct_placement = None
    if include_propulsion and edf is not None:
        from ..propulsion.duct_geometry import compute_duct_placement
        try:
            _duct_placement = compute_duct_placement(p, edf)
        except Exception:
            pass

    # ── Build all sections: body (3) + wing (6) ──
    body_sections, _ = _build_body_sections(p, n_profile,
                                             duct_placement=_duct_placement)
    wing_sections = _build_wing_sections(p, n_profile)

    # Merge: body[0..2] + wing[1..5]  (wing[0] == body[-1] at blend)
    all_sections = body_sections + wing_sections[1:]

    # Build a unified parametric coordinate along half-span
    bw = p.body_halfwidth
    body_t = [frac * bw for frac in BODY_STATIONS]
    wing_t = [bw + frac * p.outer_half_span for frac in OUTER_WING_STATIONS]
    all_t = body_t + wing_t[1:]

    # ── Single continuous spline loft with symmetric root BC ──
    dense_right = _spline_loft_sections(all_sections, all_t, n_interp,
                                        symmetric_root=True)

    # Reverse point order so cross-product normals point outward (+Y)
    dense_right = [sec[::-1] for sec in dense_right]

    # ── Mirror for left side ──
    dense_left = [_mirror_section(sec) for sec in dense_right]

    # ── Loft triangles ──
    triangles = []

    # Right half
    for i in range(len(dense_right) - 1):
        _loft_sections(dense_right[i], dense_right[i + 1], triangles)

    # Left half (reversed order for correct normals)
    for i in range(len(dense_left) - 1):
        _loft_sections(dense_left[i + 1], dense_left[i], triangles)

    # Tip caps
    _cap_section(dense_right[-1], triangles, flip=False)
    _cap_section(dense_left[-1], triangles, flip=True)

    # No root cap: left and right lofts share the y=0 section,
    # closing the surface at the symmetry plane.

    # Propulsion duct geometry
    if include_propulsion:
        from ..propulsion.duct_geometry import build_propulsion_mesh
        from ..propulsion.edf_model import EDF_70MM
        prop_edf = edf if edf is not None else EDF_70MM
        prop_tris = build_propulsion_mesh(params, prop_edf, n_profile=32)
        triangles.extend(prop_tris)

    # Create STL mesh
    stl_obj = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for j, tri in enumerate(triangles):
        stl_obj.vectors[j] = tri

    stl_obj.save(path)
    return len(triangles)


# ═══════════════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_body_sections(params: BWBParams, n_profile: int,
                         n_body_stations: int = 3,
                         duct_placement=None) -> tuple[list[np.ndarray], list[float]]:
    """Build 3D airfoil sections for the center body.

    Parameters
    ----------
    n_profile : int
        Points per side (upper/lower). Each section will have 2*n_profile-1
        points with cosine spacing (dense at LE and TE).
    n_body_stations : int
        Number of spanwise stations.
    """
    p = params
    bw = p.body_halfwidth
    body_chord_root = p.body_root_chord
    body_chord_blend = p.wing_root_chord
    body_sweep_rad = np.radians(p.body_sweep_deg)

    body_fracs = np.linspace(0, 1, n_body_stations)

    sections = []
    for frac in body_fracs:
        chord = body_chord_root + frac * (body_chord_blend - body_chord_root)
        twist_deg = p.body_twist * (1.0 - frac)
        y = frac * bw
        le_x = y * np.tan(body_sweep_rad)

        if frac >= 1.0 - 1e-6:
            kaf = build_kulfan_airfoil_at_station(p, 0.0, name="body_blend")
        else:
            kaf = build_body_kulfan_at_station(p, frac, name=f"body_{frac:.0%}")
        coords_2d = kaf.to_airfoil(n_coordinates_per_side=n_profile).coordinates

        # Apply duct bump if available (match AVL model)
        if duct_placement is not None and frac < 1.0 - 1e-6:
            from ..parameterization.bwb_aircraft import apply_duct_bump
            coords_2d = apply_duct_bump(
                coords_2d, duct_placement, p.body_root_chord,
                p.body_halfwidth, frac * bw,
            )

        x_local = coords_2d[:, 0] * chord
        z_local = coords_2d[:, 1] * chord

        twist_rad = np.radians(twist_deg)
        x_rot = x_local * np.cos(twist_rad) + z_local * np.sin(twist_rad)
        z_rot = -x_local * np.sin(twist_rad) + z_local * np.cos(twist_rad)

        pts_3d = np.column_stack([
            x_rot + le_x,
            np.full(len(x_rot), y),
            z_rot,
        ])
        sections.append(pts_3d)

    return sections, list(body_fracs)


def _build_section_at_eta(params: BWBParams, eta: float, n_profile: int,
                          le: np.ndarray, chord: float, twist_deg: float,
                          flatten_aft: float | None = None) -> np.ndarray:
    """Build a single 3D airfoil section at arbitrary eta.

    If flatten_aft is set (e.g. 0.75), the aft portion of the airfoil
    is replaced with straight lines — creating a flat control surface zone.
    """
    kaf = build_kulfan_airfoil_at_station(params, eta)
    coords_2d = kaf.to_airfoil(n_coordinates_per_side=n_profile).coordinates
    if flatten_aft is not None:
        from ..parameterization.bwb_aircraft import flatten_airfoil_aft
        coords_2d = flatten_airfoil_aft(coords_2d, x_hinge=flatten_aft)
    x_local = coords_2d[:, 0] * chord
    z_local = coords_2d[:, 1] * chord
    twist_rad = np.radians(twist_deg)
    x_rot = x_local * np.cos(twist_rad) + z_local * np.sin(twist_rad)
    z_rot = -x_local * np.sin(twist_rad) + z_local * np.cos(twist_rad)
    return np.column_stack([x_rot + le[0], np.full(len(x_rot), le[1]), z_rot + le[2]])


def _build_wing_sections(params: BWBParams, n_profile: int) -> list[np.ndarray]:
    """Build wing sections with flat TE in control surface zones."""
    from ..geometry.control_surfaces import classify_wing_segments, compute_wing_le_positions
    from ..aero.avl_runner import ControlConfig

    seg_groups = classify_wing_segments()
    x_hinge = ControlConfig.default_bwb().surfaces[0].xhinge  # 0.75

    def in_control_zone(eta):
        for idx_s, idx_e, lt in seg_groups:
            if lt == 'ruled':
                lo = OUTER_WING_STATIONS[idx_s]
                hi = OUTER_WING_STATIONS[idx_e]
                if lo - 1e-6 <= eta <= hi + 1e-6:
                    return True
        return False

    le_xyz, chords, twists = compute_wing_le_positions(params)

    sections = []
    for i, eta in enumerate(OUTER_WING_STATIONS):
        flatten = x_hinge if in_control_zone(eta) else None
        sec = _build_section_at_eta(params, eta, n_profile,
                                     le_xyz[i], chords[i], twists[i],
                                     flatten_aft=flatten)
        sections.append(sec)

    return sections


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mirror_section(section: np.ndarray) -> np.ndarray:
    """Mirror a 3D section across the XZ plane (negate Y)."""
    mirrored = section.copy()
    mirrored[:, 1] = -mirrored[:, 1]
    return mirrored


def _spline_loft_sections(sections: list[np.ndarray], stations: list[float],
                          n_interp: int = 8,
                          symmetric_root: bool = False) -> list[np.ndarray]:
    """PCHIP interpolation in the spanwise direction.

    Uses Piecewise Cubic Hermite Interpolating Polynomials (PCHIP):
    smooth (C1 continuous) but monotone per interval — no overshoot.

    At the symmetry plane (root), X and Z derivatives are forced to zero
    so left/right halves join without a ridge.

    Parameters
    ----------
    sections : list of (n_profile, 3) arrays
        The defining 3D airfoil sections.
    stations : list of float
        Spanwise parametric positions for each section (monotonic).
    n_interp : int
        Interpolated sections between each pair of defining stations.
    symmetric_root : bool
        If True, enforce zero derivative at root for X and Z axes.

    Returns
    -------
    list of (n_profile, 3) arrays
        Dense list of sections.
    """
    from scipy.interpolate import PchipInterpolator

    K = len(sections)
    n_pts = sections[0].shape[0]
    t = np.array(stations[:K], dtype=float)
    stack = np.array(sections)  # (K, n_pts, 3)

    # Dense parametric positions
    t_dense = []
    for i in range(K - 1):
        seg = np.linspace(t[i], t[i + 1], n_interp + 2)
        if i > 0:
            seg = seg[1:]
        t_dense.extend(seg.tolist())
    t_dense = np.array(t_dense)

    dense_sections = np.zeros((len(t_dense), n_pts, 3))
    for j in range(n_pts):
        for axis in range(3):
            vals = stack[:, j, axis]

            if symmetric_root and axis != 1:
                t_ext = np.concatenate([[-t[1]], t])
                v_ext = np.concatenate([[vals[1]], vals])
                pch = PchipInterpolator(t_ext, v_ext)
            else:
                pch = PchipInterpolator(t, vals)

            dense_sections[:, j, axis] = pch(t_dense)

    return [dense_sections[i] for i in range(len(t_dense))]


def _loft_sections(sec_a: np.ndarray, sec_b: np.ndarray, triangles: list):
    """Create quad strip (as 2 triangles each) between two sections."""
    n = len(sec_a)
    for j in range(n - 1):
        p0 = sec_a[j]
        p1 = sec_a[j + 1]
        p2 = sec_b[j + 1]
        p3 = sec_b[j]
        triangles.append([p0, p1, p2])
        triangles.append([p0, p2, p3])


def _cap_section(section: np.ndarray, triangles: list, flip: bool = False):
    """Close an airfoil section with a fan of triangles from the centroid."""
    centroid = section.mean(axis=0)
    n = len(section)
    for j in range(n - 1):
        if flip:
            triangles.append([centroid, section[j + 1], section[j]])
        else:
            triangles.append([centroid, section[j], section[j + 1]])
