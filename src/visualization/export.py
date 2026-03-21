"""Export BWB aircraft geometry to STL, STEP, and CAD-ready formats.

Supports:
- STL mesh (triangulated, for visualization)
- STEP solid (NURBS surfaces via CadQuery, for CAD/manufacturing)
- CAD profiles (.dat) + LE/TE splines for manual loft
"""

import json
import numpy as np
from pathlib import Path
from stl import mesh as stl_mesh
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    build_kulfan_airfoil_at_station,
    build_body_airfoil,
    OUTER_WING_STATIONS,
    N_OUTER_SEGMENTS,
    BODY_STATIONS,
    compute_wing_area,
    compute_aspect_ratio,
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

    # ── Build all sections: body (3) + wing (6) ──
    # The last body section and first wing section share the same position
    # (blend station), so we merge them into one continuous set of 8 sections.
    body_sections = _build_body_sections(p, n_profile)
    wing_sections = _build_wing_sections(p, n_profile)

    # Merge: body[0..2] + wing[1..5]  (wing[0] == body[-1] at blend)
    all_sections = body_sections + wing_sections[1:]

    # Build a unified parametric coordinate along half-span:
    # body stations map to [0, body_halfwidth], wing to [body_halfwidth, half_span]
    bw = p.body_halfwidth
    hs = p.half_span
    body_t = [frac * bw for frac in BODY_STATIONS]  # [0, 0.5*bw, bw]
    wing_t = [bw + frac * p.outer_half_span for frac in OUTER_WING_STATIONS]  # [bw, ..., hs]
    all_t = body_t + wing_t[1:]  # merged, no duplicate at blend

    # ── Single continuous spline loft with symmetric root BC ──
    dense_right = _spline_loft_sections(all_sections, all_t, n_interp,
                                        symmetric_root=True)

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

    # Root cap (centerline, y=0)
    _cap_section(dense_right[0], triangles, flip=True)

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


def _build_body_sections(params: BWBParams, n_profile: int) -> list[np.ndarray]:
    """Build 3D airfoil sections for the center body (3 stations).

    Parameters
    ----------
    n_profile : int
        Points per side (upper/lower). Each section will have 2*n_profile-1
        points with cosine spacing (dense at LE and TE).
    """
    p = params
    bw = p.body_halfwidth
    body_chord_root = p.body_root_chord
    body_chord_blend = p.wing_root_chord
    body_sweep_rad = np.radians(p.body_sweep_deg)

    # Chords at each body station (linear interpolation)
    body_chords = [body_chord_root + frac * (body_chord_blend - body_chord_root)
                   for frac in BODY_STATIONS]

    # Twist: body_twist at center, 0 at blend
    body_twists = [p.body_twist * (1.0 - frac) for frac in BODY_STATIONS]

    # y and LE x positions
    body_y = [frac * bw for frac in BODY_STATIONS]
    body_le_x = [y * np.tan(body_sweep_rad) for y in body_y]

    sections = []
    for i, frac in enumerate(BODY_STATIONS):
        # Airfoil selection — generate with native cosine spacing
        if frac == 0.0:
            # Centerline: thick body airfoil
            af = build_body_airfoil(
                tc=p.body_tc_root, camber=p.body_camber,
                reflex=p.body_reflex, le_droop=p.body_le_droop,
                name="body_center", n_pts=n_profile,
            )
            coords_2d = af.coordinates
        elif frac == 1.0:
            # Blend station: Kulfan airfoil (match outer wing root)
            kaf = build_kulfan_airfoil_at_station(p, 0.0, name="body_blend")
            coords_2d = kaf.to_airfoil(n_coordinates_per_side=n_profile).coordinates
        else:
            # Mid-body: blended body airfoil
            mid_tc = p.body_tc_root * (1 - frac) + frac * 0.105
            mid_camber = p.body_camber * (1 - frac) + frac * 0.024
            af = build_body_airfoil(
                tc=mid_tc, camber=mid_camber,
                reflex=p.body_reflex * (1 - frac),
                le_droop=p.body_le_droop * (1 - 0.2 * frac),
                name=f"body_mid_{frac:.0%}", n_pts=n_profile,
            )
            coords_2d = af.coordinates

        # Scale by chord
        chord = body_chords[i]
        x_local = coords_2d[:, 0] * chord
        z_local = coords_2d[:, 1] * chord

        # Apply twist
        twist_rad = np.radians(body_twists[i])
        x_rot = x_local * np.cos(twist_rad) + z_local * np.sin(twist_rad)
        z_rot = -x_local * np.sin(twist_rad) + z_local * np.cos(twist_rad)

        # Translate to 3D position
        pts_3d = np.column_stack([
            x_rot + body_le_x[i],
            np.full(len(x_rot), body_y[i]),
            z_rot,
        ])
        sections.append(pts_3d)

    return sections


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
    """Build wing sections with flat TE in control surface zones.

    Control surface zones get flattened airfoils (linear TE aft of hinge).
    Other zones keep full Kulfan profiles. Uses only the 7 defining stations
    — no intermediate sections (they destabilize the C2 loft).
    """
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
# CAD profiles export — body + wing stations
# ═══════════════════════════════════════════════════════════════════════════

def export_cad_profiles(params: BWBParams, output_dir: str,
                        n_profile: int = 40, n_spline: int = 50,
                        include_propulsion: bool = False,
                        edf: 'EDFSpec | None' = None):
    """Export flat airfoil profiles + LE/TE splines for CAD loft.

    Creates:
      output_dir/
        profiles/
          body_00_center.dat
          body_01_mid.dat
          body_02_blend.dat
          wing_00_root.dat
          wing_01_25pct.dat
          ...
          wing_05_tip.dat
        spline_le.csv
        spline_te.csv
        spline_le_left.csv
        spline_te_left.csv
        assembly.json

    Returns
    -------
    dict
        Assembly data (same content as assembly.json).
    """
    p = params
    bw = p.body_halfwidth
    body_chord_root = p.body_root_chord
    body_chord_blend = p.wing_root_chord
    body_sweep_rad = np.radians(p.body_sweep_deg)
    wing_sweep_rad = np.radians(p.le_sweep_deg)
    outer_span = p.outer_half_span
    tip_chord = p.tip_chord

    out = Path(output_dir)
    prof_dir = out / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    section_data = []
    all_le_positions = []
    all_te_positions = []

    # ── Body stations ──
    body_labels = ["center", "mid", "blend"]
    body_chords = [body_chord_root + frac * (body_chord_blend - body_chord_root)
                   for frac in BODY_STATIONS]
    body_twists = [p.body_twist * (1.0 - frac) for frac in BODY_STATIONS]
    body_y = [frac * bw for frac in BODY_STATIONS]
    body_le_x = [y * np.tan(body_sweep_rad) for y in body_y]

    for i, frac in enumerate(BODY_STATIONS):
        # Build airfoil
        if frac == 0.0:
            af = build_body_airfoil(
                tc=p.body_tc_root, camber=p.body_camber,
                reflex=p.body_reflex, le_droop=p.body_le_droop,
                name="body_center",
            )
            coords = af.coordinates
            tc_val = p.body_tc_root
        elif frac == 1.0:
            kaf = build_kulfan_airfoil_at_station(p, 0.0, name="body_blend")
            coords = kaf.to_airfoil().coordinates
            tc_val = float(kaf.max_thickness())
        else:
            mid_tc = p.body_tc_root * (1 - frac) + frac * 0.105
            mid_camber = p.body_camber * (1 - frac) + frac * 0.024
            af = build_body_airfoil(
                tc=mid_tc, camber=mid_camber,
                reflex=p.body_reflex * (1 - frac),
                le_droop=p.body_le_droop * (1 - 0.2 * frac),
                name=f"body_mid",
            )
            coords = af.coordinates
            tc_val = mid_tc

        coords = _resample_airfoil(coords, n_profile)

        chord = body_chords[i]
        twist = body_twists[i]
        le_x = body_le_x[i]
        le_y = body_y[i]
        le_z = 0.0

        # TE position (accounting for twist)
        twist_rad = np.radians(twist)
        te_x = le_x + chord * np.cos(twist_rad)
        te_z = le_z - chord * np.sin(twist_rad)

        all_le_positions.append([le_x, le_y, le_z])
        all_te_positions.append([te_x, le_y, te_z])

        # Save .dat file
        dat_name = f"body_{i:02d}_{body_labels[i]}.dat"
        dat_path = prof_dir / dat_name
        with open(dat_path, "w") as f:
            f.write(f"nEUROn_v2 body {body_labels[i]} "
                    f"t/c={tc_val:.1%} chord={chord*1000:.1f}mm\n")
            for x, y in coords:
                f.write(f"  {x:.6f}  {y:.6f}\n")

        section_data.append({
            "index": i,
            "region": "body",
            "label": body_labels[i],
            "station_frac": BODY_STATIONS[i],
            "dat_file": dat_name,
            "chord_m": chord,
            "thickness_ratio": tc_val,
            "twist_deg": twist,
            "le_x_m": le_x,
            "le_y_m": le_y,
            "le_z_m": le_z,
            "te_x_m": te_x,
            "te_y_m": le_y,
            "te_z_m": te_z,
        })

    # ── Outer wing stations ──
    wing_labels = ["root", "25pct", "50pct", "55pct", "75pct", "90pct", "tip"]
    wing_chords = [p.wing_root_chord + frac * (tip_chord - p.wing_root_chord)
                   for frac in OUTER_WING_STATIONS]
    from ..parameterization.bwb_aircraft import outer_wing_twists, outer_wing_dihedrals
    wing_twists = outer_wing_twists(p)
    wing_dihedrals = outer_wing_dihedrals(p)

    # LE position at blend
    x_blend = bw * np.tan(body_sweep_rad)
    wing_positions = [[x_blend, bw, 0.0]]
    for seg_i in range(N_OUTER_SEGMENTS):
        prev = wing_positions[-1]
        dy_seg = outer_span * (OUTER_WING_STATIONS[seg_i + 1] - OUTER_WING_STATIONS[seg_i])
        dih_rad = np.radians(wing_dihedrals[seg_i])
        dx = dy_seg * np.tan(wing_sweep_rad)
        dy = dy_seg * np.cos(dih_rad)
        dz = dy_seg * np.sin(dih_rad)
        wing_positions.append([prev[0] + dx, prev[1] + dy, prev[2] + dz])

    for i, frac in enumerate(OUTER_WING_STATIONS):
        kaf = build_kulfan_airfoil_at_station(p, frac, name=f"wing_{wing_labels[i]}")
        af = kaf.to_airfoil()
        coords = af.coordinates
        coords = _resample_airfoil(coords, n_profile)
        tc_val = float(kaf.max_thickness())

        chord = wing_chords[i]
        twist = wing_twists[i]
        le = wing_positions[i]

        # TE position
        twist_rad = np.radians(twist)
        te_x = le[0] + chord * np.cos(twist_rad)
        te_z = le[2] - chord * np.sin(twist_rad)

        all_le_positions.append(le)
        all_te_positions.append([te_x, le[1], te_z])

        # Save .dat file
        dat_name = f"wing_{i:02d}_{wing_labels[i]}.dat"
        dat_path = prof_dir / dat_name
        with open(dat_path, "w") as f:
            f.write(f"nEUROn_v2 wing {wing_labels[i]} "
                    f"t/c={tc_val:.1%} chord={chord*1000:.1f}mm\n")
            for x, y in coords:
                f.write(f"  {x:.6f}  {y:.6f}\n")

        section_data.append({
            "index": len(BODY_STATIONS) + i,
            "region": "wing",
            "label": wing_labels[i],
            "station_frac": OUTER_WING_STATIONS[i],
            "dat_file": dat_name,
            "chord_m": chord,
            "thickness_ratio": tc_val,
            "twist_deg": twist,
            "le_x_m": le[0],
            "le_y_m": le[1],
            "le_z_m": le[2],
            "te_x_m": te_x,
            "te_y_m": le[1],
            "te_z_m": te_z,
        })

    # ── Spline exports ──
    le_arr = np.array(all_le_positions)
    te_arr = np.array(all_te_positions)

    le_spline = _cubic_interpolate_3d(le_arr, n_spline)
    te_spline = _cubic_interpolate_3d(te_arr, n_spline)

    _save_spline_csv(out / "spline_le.csv", le_spline, "Leading Edge Right")
    _save_spline_csv(out / "spline_te.csv", te_spline, "Trailing Edge Right")

    # Mirror splines for left side
    le_spline_left = le_spline.copy()
    le_spline_left[:, 1] = -le_spline_left[:, 1]
    te_spline_left = te_spline.copy()
    te_spline_left[:, 1] = -te_spline_left[:, 1]
    _save_spline_csv(out / "spline_le_left.csv", le_spline_left[::-1], "Leading Edge Left")
    _save_spline_csv(out / "spline_te_left.csv", te_spline_left[::-1], "Trailing Edge Left")

    # ── Propulsion duct profiles ──
    propulsion_data = None
    if include_propulsion:
        from ..propulsion.duct_geometry import (
            compute_duct_placement, get_duct_profiles, get_duct_centerline_csv,
        )
        from ..propulsion.edf_model import EDF_70MM

        prop_edf = edf if edf is not None else EDF_70MM
        placement = compute_duct_placement(p, prop_edf)
        duct_profiles = get_duct_profiles(placement, n_profile=40)

        # Save duct cross-section profiles
        duct_profile_files = []
        for i, (name, cs) in enumerate(duct_profiles.items()):
            dat_name = f"duct_{i:02d}_{name}.dat"
            dat_path = prof_dir / dat_name
            with open(dat_path, "w") as f:
                f.write(f"nEUROn_v2 duct {name} (meters, local frame)\n")
                for h, v in cs:
                    f.write(f"  {h:.6f}  {v:.6f}\n")
            duct_profile_files.append(dat_name)

        # Save duct centerline
        cl = get_duct_centerline_csv(placement, n_pts=50)
        _save_spline_csv(out / "duct_centerline.csv", cl, "Duct Centerline")

        propulsion_data = {
            "edf_name": prop_edf.name,
            "fan_diameter_m": prop_edf.fan_diameter,
            "duct_od_m": prop_edf.duct_outer_diameter,
            "intake_x_frac": placement.intake_x_frac,
            "fan_x_frac": placement.fan_x_frac,
            "exhaust_x_frac": placement.exhaust_x_frac,
            "intake_x_m": placement.intake_x,
            "intake_z_m": placement.intake_z,
            "fan_x_m": placement.fan_x,
            "fan_z_m": placement.fan_z,
            "exhaust_x_m": placement.exhaust_x,
            "exhaust_z_m": placement.exhaust_z,
            "exhaust_width_m": placement.exhaust_width,
            "exhaust_height_m": placement.exhaust_height,
            "duct_profile_files": duct_profile_files,
            "duct_centerline_file": "duct_centerline.csv",
        }

    # ── Assembly JSON ──
    assembly = {
        "source": "nEUROn_v2 BWB optimizer",
        "half_span_m": p.half_span,
        "body_halfwidth_m": p.body_halfwidth,
        "body_root_chord_m": p.body_root_chord,
        "wing_root_chord_m": p.wing_root_chord,
        "taper_ratio": p.taper_ratio,
        "le_sweep_deg": p.le_sweep_deg,
        "body_sweep_deg": p.body_sweep_deg,
        "aspect_ratio": float(compute_aspect_ratio(p)),
        "wing_area_m2": float(compute_wing_area(p)),
        "n_body_stations": len(BODY_STATIONS),
        "n_wing_stations": len(OUTER_WING_STATIONS),
        "sections": section_data,
        "spline_files": {
            "le_right": "spline_le.csv",
            "te_right": "spline_te.csv",
            "le_left": "spline_le_left.csv",
            "te_left": "spline_te_left.csv",
        },
        "cad_instructions": (
            "1. Import each .dat profile as a 2D wire\n"
            "2. Scale each profile by its chord_m\n"
            "3. Apply twist rotation around LE point\n"
            "4. Position at (le_x_m, le_y_m, le_z_m)\n"
            "5. Import spline_le.csv as a 3D BSpline guide\n"
            "6. Loft body profiles (body_00..body_02) along body LE spline\n"
            "7. Loft wing profiles (wing_00..wing_05) along wing LE spline\n"
            "8. body_02_blend and wing_00_root share the same airfoil (C0 continuity)\n"
            "9. Mirror for the left side\n"
            "10. For propulsion: import duct profiles and loft along duct_centerline.csv\n"
            "11. Boolean cut intake opening from body upper surface"
        ),
    }
    if propulsion_data is not None:
        assembly["propulsion"] = propulsion_data

    with open(out / "assembly.json", "w") as f:
        json.dump(assembly, f, indent=2)

    return assembly


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resample_airfoil(coords: np.ndarray, n: int) -> np.ndarray:
    """Resample airfoil to exactly n points with uniform arc-length spacing."""
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    target_s = np.linspace(0, total, n)
    x_new = np.interp(target_s, cumlen, coords[:, 0])
    y_new = np.interp(target_s, cumlen, coords[:, 1])
    return np.column_stack([x_new, y_new])


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
    This gives smooth surfaces while preserving winglet dihedral breaks.

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
                # Force zero derivative at root for X, Z (mirror symmetry).
                # PCHIP doesn't support custom BCs, so we prepend a mirror
                # point at t=-t[1] with the same value as t[1] to create
                # a symmetric pattern that naturally gives zero slope at t=0.
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


def _cubic_interpolate_3d(points: np.ndarray, n_out: int) -> np.ndarray:
    """PCHIP interpolation through 3D points — smooth, no overshoot.

    Uses monotone piecewise cubic Hermite interpolation (C1 smooth)
    which preserves dihedral breaks without oscillation.
    """
    from scipy.interpolate import PchipInterpolator

    # Remove duplicate consecutive points
    mask = np.ones(len(points), dtype=bool)
    for i in range(1, len(points)):
        if np.allclose(points[i], points[i - 1], atol=1e-10):
            mask[i] = False
    points = points[mask]

    diffs = np.diff(points, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(seg_len)])
    t_norm = t / t[-1]

    t_out = np.linspace(0, 1, n_out)

    # Mirror first point for symmetric root (zero slope at centerline)
    t_ext = np.concatenate([[-t_norm[1]], t_norm])

    result = np.zeros((n_out, 3))
    for axis in range(3):
        vals = points[:, axis]
        if axis == 1:  # Y: antisymmetric (mirror flips sign)
            v_ext = np.concatenate([[-vals[1]], vals])
        else:  # X, Z: symmetric (mirror keeps value)
            v_ext = np.concatenate([[vals[1]], vals])
        pch = PchipInterpolator(t_ext, v_ext)
        result[:, axis] = pch(t_out)

    return result


def _save_spline_csv(path, points: np.ndarray, name: str):
    """Save spline points as CSV (X, Y, Z in meters)."""
    with open(path, "w") as f:
        f.write(f"# {name} spline — {len(points)} points (meters)\n")
        f.write("X,Y,Z\n")
        for x, y, z in points:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# STEP export — watertight solid via BRepOffsetAPI_ThruSections loft
# ═══════════════════════════════════════════════════════════════════════════

def _make_open_wire(pts_mm: np.ndarray, tol_mm: float = 1e-3):
    """Convert an open polyline (n, 3) in mm to an OCC BSpline Wire.

    Unlike _make_periodic_wire, this creates a non-periodic curve
    (for hinge lines, control surface boundaries, etc.).
    """
    from OCP.TColgp import TColgp_HArray1OfPnt
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    n = len(pts_mm)
    harray = TColgp_HArray1OfPnt(1, n)
    for i in range(n):
        harray.SetValue(i + 1, gp_Pnt(float(pts_mm[i, 0]),
                                        float(pts_mm[i, 1]),
                                        float(pts_mm[i, 2])))

    interp = GeomAPI_Interpolate(harray, False, tol_mm)  # periodic=False
    interp.Perform()
    if not interp.IsDone():
        raise RuntimeError("GeomAPI_Interpolate failed for open BSpline wire")

    curve = interp.Curve()
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return wire


def _make_periodic_wire(pts_3d: np.ndarray, tol_mm: float = 1e-3):
    """Convert a closed (n_profile, 3) section in meters to an OCC periodic BSpline Wire.

    The input array follows Selig convention (TE→LE→TE, first ≈ last).
    We drop the duplicate last point and use periodic BSpline interpolation
    so the curve is exactly C2-closed at the trailing edge.

    Parameters
    ----------
    pts_3d : (n, 3) array in meters
    tol_mm : float
        Interpolation tolerance in mm (default 1 micron).

    Returns
    -------
    TopoDS_Wire
    """
    from OCP.TColgp import TColgp_HArray1OfPnt
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    # Drop duplicate closing point
    pts_mm = pts_3d[:-1] * 1000.0  # meters → mm
    n = len(pts_mm)

    harray = TColgp_HArray1OfPnt(1, n)
    for i in range(n):
        harray.SetValue(i + 1, gp_Pnt(float(pts_mm[i, 0]),
                                        float(pts_mm[i, 1]),
                                        float(pts_mm[i, 2])))

    interp = GeomAPI_Interpolate(harray, True, tol_mm)  # periodic=True
    interp.Perform()
    if not interp.IsDone():
        raise RuntimeError("GeomAPI_Interpolate failed for periodic BSpline wire")

    curve = interp.Curve()
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return wire


# ═══════════════════════════════════════════════════════════════════════════
# Densified loft — uniform intermediate sections for flat control surfaces
# ═══════════════════════════════════════════════════════════════════════════

def _densify_sections(
    params: BWBParams,
    all_sections: list[np.ndarray],
    n_profile: int = 100,
    n_interp: int = 4,
) -> list[np.ndarray]:
    """Add intermediate sections uniformly across the entire span.

    Inserts *n_interp* sections between each pair of defining stations
    to produce a uniformly dense set.  This balances the C2 loft
    parameterisation and prevents distortion from uneven spacing.

    - **Body segments** (merged 0-1, 1-2): linear interpolation of
      3D coordinates (body airfoils transition smoothly).
    - **Wing control zones**: true Kulfan CST profiles with
      ``flatten_airfoil_aft`` (correct aerodynamic shape + flat TE).
    - **Wing non-control zones**: true Kulfan CST profiles without
      flattening (correct aerodynamic shape, smooth TE).

    Parameters
    ----------
    params : BWBParams
        Aircraft design parameters.
    all_sections : list of (n_profile, 3) arrays
        The 9 merged defining sections (body[0:3] + wing[1:7]).
    n_profile : int
        Points per airfoil side for intermediate profiles.
    n_interp : int
        Number of intermediate sections per segment pair. Default 4.

    Returns
    -------
    list of (n_profile, 3) arrays
        Densified section list (defining + intermediates).
    """
    from ..geometry.control_surfaces import classify_wing_segments, compute_wing_le_positions
    from ..aero.avl_runner import ControlConfig

    wing_groups = classify_wing_segments()
    x_hinge = ControlConfig.default_bwb().surfaces[0].xhinge  # 0.75
    BODY_OFFSET = 2  # wing[j] → merged index 2+j
    N_BODY = len(BODY_STATIONS)  # 3

    # Map each (merged_start, merged_end) pair to its type
    # Body pairs: linear interpolation
    # Wing ruled pairs: Kulfan + flatten
    # Wing spline pairs: Kulfan (no flatten)
    pair_info: dict[tuple[int, int], tuple[str, float, float]] = {}

    # Body segments
    for k in range(N_BODY - 1):
        pair_info[(k, k + 1)] = ('body', BODY_STATIONS[k], BODY_STATIONS[k + 1])

    # Wing segments
    for ws, we, ltype in wing_groups:
        for k in range(ws, we):
            eta_lo = OUTER_WING_STATIONS[k]
            eta_hi = OUTER_WING_STATIONS[k + 1]
            seg_type = 'ruled' if ltype == 'ruled' else 'spline'
            pair_info[(BODY_OFFSET + k, BODY_OFFSET + k + 1)] = (seg_type, eta_lo, eta_hi)

    densified = [all_sections[0]]
    for i in range(len(all_sections) - 1):
        info = pair_info.get((i, i + 1))
        if info is not None:
            seg_type, param_lo, param_hi = info
            for k in range(1, n_interp + 1):
                t = k / (n_interp + 1)
                if seg_type == 'body':
                    # Linear interpolation of 3D coordinates
                    densified.append(
                        (1.0 - t) * all_sections[i] + t * all_sections[i + 1]
                    )
                else:
                    # True Kulfan profile at interpolated eta
                    eta = param_lo + t * (param_hi - param_lo)
                    le_xyz, chords, twists = compute_wing_le_positions(
                        params, np.array([eta]),
                    )
                    flatten = x_hinge if seg_type == 'ruled' else None
                    sec = _build_section_at_eta(
                        params, eta, n_profile,
                        le_xyz[0], chords[0], twists[0],
                        flatten_aft=flatten,
                    )
                    densified.append(sec)
        densified.append(all_sections[i + 1])

    return densified


def _build_oml_solid(
    params: BWBParams, n_profile: int = 100,
) -> tuple['TopoDS_Shape', list, list]:
    """Build the fused OML solid (right + left halves).

    Returns
    -------
    oml_solid : TopoDS_Shape
        Full aircraft OML solid (both halves fused).
    all_sections : list of (n_profile, 3) arrays
        The 9 defining sections before densification.
    dense_sections : list of (n_profile, 3) arrays
        All sections after densification (defining + intermediates).
    """
    import warnings
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.Approx import Approx_ChordLength
    from OCP.gp import gp_Pnt, gp_Trsf, gp_Ax2, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    p = params

    # Build defining sections
    body_sections = _build_body_sections(p, n_profile)
    wing_sections = _build_wing_sections(p, n_profile)
    all_sections = body_sections + wing_sections[1:]  # merge at blend

    # Densify with intermediate sections (flat TE in control zones)
    dense_sections = _densify_sections(p, all_sections, n_profile)
    wires = [_make_periodic_wire(sec) for sec in dense_sections]

    # C2 loft
    loft = BRepOffsetAPI_ThruSections(True, False)
    loft.SetMaxDegree(6)
    loft.SetContinuity(GeomAbs_C2)
    loft.SetParType(Approx_ChordLength)
    loft.CheckCompatibility(True)
    for wire in wires:
        loft.AddWire(wire)
    loft.Build()
    if not loft.IsDone():
        raise RuntimeError("BRepOffsetAPI_ThruSections loft failed")
    right_solid = loft.Shape()

    # Mirror across XZ plane and fuse both halves
    # Note: BRepAlgoAPI_Fuse is used directly here (not CellsBuilder)
    # because CellsBuilder can return incomplete results for mirror fuses.
    trsf = gp_Trsf()
    trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
    left_solid = BRepBuilderAPI_Transform(right_solid, trsf, True).Shape()

    fuse = BRepAlgoAPI_Fuse(right_solid, left_solid)
    fuse.Build()
    if not fuse.IsDone():
        warnings.warn("Direct fuse failed, falling back to sewing")
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
        from OCP.TopoDS import TopoDS

        sew = BRepBuilderAPI_Sewing(0.01)
        sew.Add(right_solid)
        sew.Add(left_solid)
        sew.Perform()
        sewn = sew.SewedShape()
        oml_solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sewn)).Solid()
    else:
        oml_solid = fuse.Shape()

    return oml_solid, all_sections, dense_sections


def export_aircraft_step(
    params: BWBParams,
    path: str,
    n_profile: int = 100,
    include_propulsion: bool = False,
    edf: 'EDFSpec | None' = None,
) -> dict:
    """Export BWB aircraft as a watertight STEP solid.

    Builds the OML via densified C2 ThruSections loft, mirrors and fuses
    both halves.  Optionally integrates propulsion (duct, shell, boolean
    cuts, hinge wires, IGES control surfaces).

    Parameters
    ----------
    params : BWBParams
    path : str
        Output .step file path.
    n_profile : int
        Points per airfoil section. Default 100.
    include_propulsion : bool
        If True, add propulsion duct geometry with boolean integration.
    edf : EDFSpec or None
        EDF specification. Defaults to EDF_70MM.

    Returns
    -------
    dict
        Validation metrics.
    """
    if include_propulsion:
        return _export_with_propulsion(params, path, n_profile, edf)
    return _export_oml_only(params, path, n_profile)


def _export_oml_only(params: BWBParams, path: str,
                     n_profile: int = 100) -> dict:
    """Export OML-only STEP solid with validation."""
    import os
    import warnings
    import cadquery as cq
    from OCP.gp import gp_Pnt
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    full_solid, all_sections, dense_sections = _build_oml_solid(params, n_profile)

    # Validate topology
    analyzer = BRepCheck_Analyzer(full_solid)
    is_valid = analyzer.IsValid()

    # Volume and surface area
    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(full_solid, vol_props)
    volume_mm3 = abs(vol_props.Mass())

    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(full_solid, area_props)
    surface_area_mm2 = area_props.Mass()

    # Max deviation at control points
    max_dev = 0.0
    try:
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex

        for sec in all_sections:
            for i in range(0, len(sec), max(1, len(sec) // 12)):
                pt_mm = sec[i] * 1000.0
                vertex = BRepBuilderAPI_MakeVertex(
                    gp_Pnt(float(pt_mm[0]), float(pt_mm[1]), float(pt_mm[2]))
                ).Vertex()
                dist_calc = BRepExtrema_DistShapeShape(vertex, full_solid)
                if dist_calc.IsDone() and dist_calc.NbSolution() > 0:
                    max_dev = max(max_dev, dist_calc.Value())
    except Exception:
        max_dev = -1.0

    # Export
    result_shape = cq.Workplane("XY").add(cq.Shape(full_solid))
    cq.exporters.export(result_shape, path, cq.exporters.ExportTypes.STEP)

    file_size_kb = os.path.getsize(path) / 1024
    if file_size_kb < 426:
        warnings.warn(
            f"STEP file size {file_size_kb:.0f} KB < 426 KB reference minimum. "
            "Quality may be degraded."
        )

    return {
        "path": path,
        "file_size_kb": round(file_size_kb, 1),
        "volume_mm3": round(volume_mm3, 1),
        "surface_area_mm2": round(surface_area_mm2, 1),
        "is_valid": is_valid,
        "max_deviation_mm": round(max_dev, 4),
        "n_sections_defining": len(all_sections),
        "n_sections_dense": len(dense_sections),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP v3 — integrated propulsion (OML + duct + boolean cuts)
# ═══════════════════════════════════════════════════════════════════════════

def _make_closed_wire(pts_3d_mm: np.ndarray, tol_mm: float = 1e-3):
    """Build a closed BSpline wire from (n, 3) points in mm.

    Unlike _make_periodic_wire which expects Selig airfoil convention,
    this function works with any closed loop of points (duct cross-sections).
    """
    from OCP.TColgp import TColgp_HArray1OfPnt
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    n = len(pts_3d_mm)
    harray = TColgp_HArray1OfPnt(1, n)
    for i in range(n):
        harray.SetValue(i + 1, gp_Pnt(float(pts_3d_mm[i, 0]),
                                        float(pts_3d_mm[i, 1]),
                                        float(pts_3d_mm[i, 2])))

    interp = GeomAPI_Interpolate(harray, True, tol_mm)  # periodic=True
    interp.Perform()
    if not interp.IsDone():
        raise RuntimeError("GeomAPI_Interpolate failed for closed duct wire")

    curve = interp.Curve()
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return wire


def _loft_duct_solid(sections_3d: list[np.ndarray]) -> 'TopoDS_Shape':
    """Loft a list of (n, 3) cross-section arrays (meters) into a solid.

    Uses BRepOffsetAPI_ThruSections for smooth interpolation.
    Returns an OCC solid shape.
    """
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.Approx import Approx_ChordLength

    loft = BRepOffsetAPI_ThruSections(True, False)  # solid=True, ruled=False
    loft.SetMaxDegree(6)
    loft.SetContinuity(GeomAbs_C2)
    loft.SetParType(Approx_ChordLength)
    loft.CheckCompatibility(True)

    for sec in sections_3d:
        pts_mm = sec * 1000.0  # meters → mm
        wire = _make_closed_wire(pts_mm)
        loft.AddWire(wire)

    loft.Build()
    if not loft.IsDone():
        raise RuntimeError("Duct loft (ThruSections) failed")
    return loft.Shape()


def _export_with_propulsion(params: BWBParams, path: str,
                            n_profile: int = 100,
                            edf: 'EDFSpec | None' = None) -> dict:
    """Export BWB aircraft with integrated propulsion as STEP compound."""
    import os
    import warnings
    import cadquery as cq
    from OCP.gp import gp_Pnt, gp_Trsf, gp_Ax2, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder

    from .step_booleans import robust_fuse_and_cut

    from ..propulsion.duct_geometry import (
        compute_duct_placement, validate_duct_clearance,
        build_sduct_geometry, build_edf_housing, build_exhaust_geometry,
        build_exhaust_fairing, duct_cross_section,
    )
    from ..propulsion.edf_model import EDF_70MM

    prop_edf = edf if edf is not None else EDF_70MM
    p = params

    # ==================================================================
    # Step 1: OML solid
    # ==================================================================
    print("[prop] 1/6  Building OML solid...")
    oml_solid, all_sections, dense_sections = _build_oml_solid(p, n_profile)

    oml_path = path.replace('.step', '_oml_only.step')
    cq.exporters.export(
        cq.Workplane("XY").add(cq.Shape(oml_solid)),
        oml_path, cq.exporters.ExportTypes.STEP,
    )
    oml_kb = os.path.getsize(oml_path) / 1024

    vol_oml = GProp_GProps()
    BRepGProp.VolumeProperties_s(oml_solid, vol_oml)
    oml_volume = abs(vol_oml.Mass())
    print("[prop]      OML OK  (%.0f KB, %.1f cm3)" % (oml_kb, oml_volume / 1e6))

    # ==================================================================
    # Step 2: Duct placement
    # ==================================================================
    print("[prop] 2/6  Computing duct placement...")
    placement = compute_duct_placement(p, prop_edf)

    # ==================================================================
    # Step 3: Internal duct solid (at actual position, no extensions)
    # ==================================================================
    print("[prop] 3/6  Building internal duct solid...")
    n_duct_profile = 48

    import numpy as _np

    sduct_secs = build_sduct_geometry(placement, p, prop_edf,
                                       n_stations=12, n_profile=n_duct_profile)
    housing_secs = build_edf_housing(placement, prop_edf,
                                       n_profile=n_duct_profile, n_axial=4)
    exhaust_secs = build_exhaust_geometry(placement, p, prop_edf,
                                            n_stations=8, n_profile=n_duct_profile)

    all_duct_secs = sduct_secs + housing_secs[1:] + exhaust_secs[1:]

    duct_solid = None
    duct_path = None
    duct_kb = 0
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec

        # 1. Main duct loft (no extensions)
        main_duct = _loft_duct_solid(all_duct_secs)

        # 2. Intake extension: extrude first section 60mm upstream (-X)
        first_wire = _make_closed_wire(all_duct_secs[0] * 1000.0)
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        first_face = BRepBuilderAPI_MakeFace(first_wire, True).Face()
        intake_prism = BRepPrimAPI_MakePrism(
            first_face, gp_Vec(-200.0, 0.0, 0.0)  # 200mm in -X (mm)
        ).Shape()

        # 3. Exhaust extension: extrude last section 100mm downstream (+X)
        last_wire = _make_closed_wire(all_duct_secs[-1] * 1000.0)
        last_face = BRepBuilderAPI_MakeFace(last_wire, True).Face()
        exhaust_prism = BRepPrimAPI_MakePrism(
            last_face, gp_Vec(100.0, 0.0, 0.0)  # 100mm in +X (mm)
        ).Shape()

        # 4. Fuse main + intake_ext + exhaust_ext
        fuse1 = BRepAlgoAPI_Fuse(main_duct, intake_prism)
        fuse1.Build()
        fuse2 = BRepAlgoAPI_Fuse(fuse1.Shape(), exhaust_prism)
        fuse2.Build()
        duct_solid = fuse2.Shape()
        duct_path = path.replace('.step', '_duct_only.step')
        cq.exporters.export(
            cq.Workplane("XY").add(cq.Shape(duct_solid)),
            duct_path, cq.exporters.ExportTypes.STEP,
        )
        duct_kb = os.path.getsize(duct_path) / 1024
        print("[prop]      Duct OK (%d sections, %.0f KB)" % (len(all_duct_secs), duct_kb))
    except Exception as e:
        warnings.warn("Duct loft failed: %s" % e)

    # ==================================================================
    # Step 4: Clearance validation
    # ==================================================================
    print("[prop] 4/6  Validating duct clearance...")
    clr_ok, clr_results = validate_duct_clearance(placement, p)
    if not clr_ok:
        violations = [r for r in clr_results if not r.is_ok]
        for v in violations[:5]:
            warnings.warn(
                "Clearance violation at x/c=%.3f: top=%.1fmm bot=%.1fmm"
                % (v.x_frac, v.clearance_top_mm, v.clearance_bot_mm)
            )
    else:
        min_top = min(r.clearance_top_mm for r in clr_results)
        min_bot = min(r.clearance_bot_mm for r in clr_results)
        print("[prop]      Clearance OK (min top=%.1fmm, min bot=%.1fmm)" % (min_top, min_bot))

    # ==================================================================
    # Step 5: Build duct shell (coque) = duct sections + skin
    # ==================================================================
    # The shell uses the EXACT same 3D sections as the duct solid,
    # but each section is dilated outward from its centroid by skin_mm.
    # This guarantees the shell perfectly wraps the duct.
    print("[prop] 5/6  Building duct shell (coque)...")
    shell_solid = None
    skin_mm = 3.0

    try:
        import numpy as _np
        shell_secs = []
        for sec in all_duct_secs:
            centroid = sec.mean(axis=0)
            # Expand each point outward from centroid
            directions = sec - centroid
            norms = _np.linalg.norm(directions, axis=1, keepdims=True)
            norms = _np.maximum(norms, 1e-9)
            unit_dirs = directions / norms
            expanded = sec + unit_dirs * (skin_mm / 1000.0)  # mm → m
            shell_secs.append(expanded)

        shell_solid = _loft_duct_solid(shell_secs)
        vol_s = GProp_GProps()
        BRepGProp.VolumeProperties_s(shell_solid, vol_s)
        shell_kb = 0
        shell_path = path.replace('.step', '_shell_only.step')
        cq.exporters.export(
            cq.Workplane("XY").add(cq.Shape(shell_solid)),
            shell_path, cq.exporters.ExportTypes.STEP,
        )
        shell_kb = os.path.getsize(shell_path) / 1024
        print("[prop]      Shell OK (vol=%.1f cm3, %.0f KB)" % (
            abs(vol_s.Mass()) / 1e6, shell_kb))
    except Exception as e:
        warnings.warn("Shell build failed: %s" % e)

    # ==================================================================
    # Step 6: Boolean integration — Robust pipeline
    # ==================================================================
    # Uses step_booleans.py: heal → unify → CellsBuilder → validate
    # with cascading fallbacks (CellsBuilder → Standard → Splitter).
    print("[prop] 6/6  Boolean: robust Fuse(OML, Shell) + Cut(Duct)...")
    result_solid = oml_solid
    bool_ok = False
    bool_method = "none"

    if shell_solid is not None and duct_solid is not None:
        try:
            integrated, status = robust_fuse_and_cut(
                oml_solid, shell_solid, duct_solid,
                label="v3",
            )
            if integrated is not None:
                result_solid = integrated
                bool_ok = True
                bool_method = status
                print("[prop]      Boolean OK: %s" % status)
            else:
                warnings.warn("Boolean pipeline failed: %s — exporting multi-body fallback" % status)
        except Exception as e:
            warnings.warn("Boolean failed: %s — exporting multi-body fallback" % e)

    # Fallback: multi-body compound if booleans failed
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    if bool_ok:
        builder.Add(compound, result_solid)
    else:
        builder.Add(compound, oml_solid)
        if shell_solid is not None:
            builder.Add(compound, shell_solid)
        if duct_solid is not None:
            builder.Add(compound, duct_solid)

    # ── Add control surface hinge wires to compound ──
    n_hinge_wires = 0
    try:
        from ..geometry.control_surfaces import compute_control_surface_geometry
        from ..aero.avl_runner import ControlConfig

        geoms = compute_control_surface_geometry(p, ControlConfig.default_bwb(), n_spanwise=20)

        for g in geoms:
            # Build hinge line wire (upper surface) — right side
            hinge_pts_mm = g.hinge_line_upper * 1000.0  # m → mm
            if len(hinge_pts_mm) >= 2:
                wire_r = _make_open_wire(hinge_pts_mm)
                builder.Add(compound, wire_r)
                # Mirror for left side (negate Y)
                trsf_y = gp_Trsf()
                trsf_y.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
                wire_l = BRepBuilderAPI_Transform(wire_r, trsf_y, True).Shape()
                builder.Add(compound, wire_l)
                n_hinge_wires += 2
        print("[prop]      Added %d hinge wires (%d control surfaces × 2 sides)" %
              (n_hinge_wires, len(geoms)))
    except Exception as e:
        warnings.warn("Could not add hinge wires: %s" % e)

    analyzer = BRepCheck_Analyzer(compound)
    is_valid = analyzer.IsValid()

    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(compound, vol_props)
    volume_mm3 = abs(vol_props.Mass())

    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(compound, area_props)
    surface_area_mm2 = area_props.Mass()

    # Use OCC STEP writer directly (CadQuery's exporter drops wires)
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.Interface import Interface_Static

    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    writer.Transfer(compound, STEPControl_AsIs)
    writer.Write(path)
    file_size_kb = os.path.getsize(path) / 1024

    if bool_ok:
        compound_mode = "integrated solid (%s)" % bool_method
    else:
        compound_mode = "multi-body fallback: OML"
        if shell_solid is not None:
            compound_mode += " + shell"
        if duct_solid is not None:
            compound_mode += " + duct"

    # ── Export OML-conformant control surface faces as IGES ──
    hinge_path = path.replace('.step', '_control_surfaces.iges').replace('.STEP', '_control_surfaces.iges')
    hinge_kb = 0.0
    if n_hinge_wires > 0:
        from OCP.IGESControl import IGESControl_Writer
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.GeomAbs import GeomAbs_C2
        iges_writer = IGESControl_Writer("MM", 0)

        for idx, g in enumerate(geoms):
            n_span = len(g.eta_stations)

            # Simple 2-row ruled surface (hinge → TE) per panel.
            # Since the OML profiles are flattened in control zones,
            # these ruled faces match the OML quasi-perfectly.
            pts_grid = TColgp_Array2OfPnt(1, 2, 1, n_span)
            for j in range(n_span):
                h = g.hinge_line_upper[j] * 1000.0
                t = g.te_line_upper[j] * 1000.0
                pts_grid.SetValue(1, j + 1, gp_Pnt(float(h[0]), float(h[1]), float(h[2])))
                pts_grid.SetValue(2, j + 1, gp_Pnt(float(t[0]), float(t[1]), float(t[2])))

            surf = GeomAPI_PointsToBSplineSurface(pts_grid, 3, 8, GeomAbs_C2, 0.1)
            if surf.IsDone():
                face = BRepBuilderAPI_MakeFace(surf.Surface(), 1e-3).Face()
                iges_writer.AddShape(face)
                # Mirror for left side
                trsf_y = gp_Trsf()
                trsf_y.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
                face_l = BRepBuilderAPI_Transform(face, trsf_y, True).Shape()
                iges_writer.AddShape(face_l)

        iges_writer.ComputeModel()
        iges_writer.Write(hinge_path)
        hinge_kb = os.path.getsize(hinge_path) / 1024
        print("[prop]      Control surfaces: %.0f KB -> %s" % (hinge_kb, hinge_path))

    result = {
        "path": path,
        "file_size_kb": round(file_size_kb, 1),
        "volume_mm3": round(volume_mm3, 1),
        "surface_area_mm2": round(surface_area_mm2, 1),
        "is_valid": is_valid,
        "has_duct": duct_solid is not None,
        "has_shell": shell_solid is not None,
        "clearance_ok": clr_ok,
        "compound_mode": compound_mode,
        "oml_path": oml_path,
        "oml_size_kb": round(oml_kb, 1),
        "duct_path": duct_path,
        "duct_size_kb": round(duct_kb, 1),
        "hinge_path": hinge_path,
        "hinge_size_kb": round(hinge_kb, 1),
        "n_sections_defining": len(all_sections),
        "n_sections_dense": len(dense_sections),
        "n_duct_sections": len(all_duct_secs),
        "n_hinge_wires": n_hinge_wires,
    }

    print("[prop] Done: %.0f KB, valid=%s, boolean=%s, vol=%.1f cm3, hinge_wires=%d" % (
        file_size_kb, is_valid, bool_ok, volume_mm3 / 1e6, n_hinge_wires))

    return result
