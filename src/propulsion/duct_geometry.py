"""Parametric duct geometry for dorsal EDF propulsion.

Physics-based sizing and placement of the internal flow path:
    Dorsal NACA intake → S-duct → EDF housing → Convergent nozzle

Sizing sequence (first-principles):
1. Fan area from EDF spec
2. Intake area from capture ratio (A_intake = A_fan × capture_ratio)
3. Intake depth from width/depth ratio + ramp angle
4. Intake placement: scan body for sufficient height margin
5. Fan placement: maximize clearance, not body height
6. S-duct: route with offset/length < 0.25 (Seddon)
7. Nozzle: converge to exit area ratio
8. Exhaust: TE slot

References:
- NACA RM-A7I30: flush intake geometry
- Seddon & Goldsmith, "Intake Aerodynamics": S-duct correlations
- NASA TN-D-4014: ducted fan performance

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
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DuctPlacement:
    """Derived duct placement — computed from BWBParams + EDFSpec + config."""

    # Intake (dorsal NACA flush scoop)
    intake_x_frac: float
    intake_x: float               # [m]
    intake_z: float               # [m] on upper OML
    intake_width: float           # [m]
    intake_depth: float           # [m] max ramp depth
    intake_length: float          # [m] ramp length

    # Fan face
    fan_x_frac: float
    fan_x: float                  # [m]
    fan_z: float                  # [m]
    fan_diameter: float           # [m]
    duct_od: float                # [m] outer diameter (fan + shroud)

    # Exhaust
    exhaust_x_frac: float
    exhaust_x: float              # [m]
    exhaust_z: float              # [m]
    exhaust_width: float          # [m] exit slot width
    exhaust_height: float         # [m] exit slot height

    # Duct dimensions
    duct_wall_thickness: float    # [m]
    housing_length: float         # [m] EDF cylindrical section
    body_chord: float             # [m] reference
    intake_fallback: bool = False # True if no suitable intake location found


@dataclass
class ClearanceResult:
    """Clearance check at one axial station."""
    x_frac: float
    x_mm: float
    centerline_z_mm: float
    duct_half_h_mm: float
    body_z_upper_mm: float
    body_z_lower_mm: float
    clearance_top_mm: float
    clearance_bot_mm: float
    is_ok: bool


# ═══════════════════════════════════════════════════════════════════════════
# Body surface helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_body_surface_cache(params: BWBParams, n_pts: int = 200) -> dict:
    """Pre-compute body upper/lower surface arrays for fast lookup."""
    from ..parameterization.bwb_aircraft import build_body_kulfan_at_station
    kaf = build_body_kulfan_at_station(params, 0.0)
    coords = kaf.to_airfoil(n_coordinates_per_side=n_pts).coordinates
    mid = len(coords) // 2
    # Upper surface: TE→LE in Selig order, reverse to get LE→TE
    x_upper = coords[:mid + 1, 0][::-1]
    z_upper = coords[:mid + 1, 1][::-1]
    # Lower surface: LE→TE
    x_lower = coords[mid:, 0]
    z_lower = coords[mid:, 1]
    return {"x_upper": x_upper, "z_upper": z_upper,
            "x_lower": x_lower, "z_lower": z_lower}


def _body_surface_z(x_frac: float, params: BWBParams,
                    _cache=None) -> tuple[float, float, float]:
    """Body upper/camber/lower z/c at chordwise fraction x_frac.

    Returns (z_upper/c, z_camber/c, z_lower/c).
    """
    if _cache is None:
        _cache = _build_body_surface_cache(params)
    z_up = float(np.interp(x_frac, _cache["x_upper"], _cache["z_upper"]))
    z_lo = float(np.interp(x_frac, _cache["x_lower"], _cache["z_lower"]))
    z_cam = (z_up + z_lo) / 2
    return z_up, z_cam, z_lo


# ═══════════════════════════════════════════════════════════════════════════
# Sizing & placement (physics-based)
# ═══════════════════════════════════════════════════════════════════════════

def size_and_place_duct(params: BWBParams, edf: EDFSpec,
                        config=None) -> DuctPlacement:
    """Size and place the duct system from first principles.

    Sizing sequence:
    1. Fan area & intake area from EDF spec + capture ratio
    2. Intake depth/width from NACA guidelines (ramp angle, W/D ratio)
    3. Intake placement: scan body for sufficient height margin
    4. Fan placement: maximize clearance margin (not body height)
    5. S-duct: verify offset/length ratio
    6. Nozzle: compute exit area
    7. Exhaust: TE slot
    """
    from ..config import DuctGeometryConfig
    if config is None:
        config = DuctGeometryConfig()

    body_chord = params.body_root_chord
    duct_wall = config.duct_wall_mm / 1000.0
    _cache = _build_body_surface_cache(params)

    # ── Step 1: Fan & intake areas ──
    fan_area = np.pi / 4 * edf.fan_diameter ** 2
    intake_area = fan_area * config.intake_capture_ratio

    # ── Step 2: Intake geometry from NACA guidelines ──
    # Width/depth ratio → depth = sqrt(intake_area / W_D_ratio)
    w_d = config.intake_width_depth_ratio
    intake_depth = np.sqrt(intake_area / w_d)
    intake_width = intake_depth * w_d
    # Cap width by 80% of body width
    max_w = 2 * params.body_halfwidth * 0.8
    if intake_width > max_w:
        intake_width = max_w
        intake_depth = intake_area / intake_width
    # Ramp length from ramp angle, capped at 20% of chord
    # (NACA 5-7° is for full-scale; small UAVs need steeper ramps)
    ramp_angle_rad = np.radians(config.intake_ramp_angle_deg)
    intake_length = intake_depth / np.tan(ramp_angle_rad)
    max_intake_length = 0.20 * body_chord
    if intake_length > max_intake_length:
        intake_length = max_intake_length

    # ── Step 3: Intake placement — find x/c where body is deep enough ──
    min_body_margin = config.intake_min_body_margin_mm / 1000.0
    required_height = intake_depth + min_body_margin
    # Scan from 5% to 25% chord
    intake_x_frac = 0.10  # default
    intake_fallback = True
    for xf in np.linspace(0.05, 0.25, 40):
        z_up, _, z_lo = _body_surface_z(xf, params, _cache)
        body_h = (z_up - z_lo) * body_chord
        if body_h >= required_height:
            intake_x_frac = xf
            intake_fallback = False
            break
    intake_x = intake_x_frac * body_chord
    z_up_intake, _, _ = _body_surface_z(intake_x_frac, params, _cache)
    intake_z = z_up_intake * body_chord

    # ── Step 4: Fan placement — maximize clearance ──
    # Scan from end-of-intake+S-duct to 55% chord
    intake_end_frac = intake_x_frac + intake_length / body_chord
    duct_radius = edf.duct_outer_diameter / 2 + duct_wall
    min_clearance = config.fan_clearance_min_mm / 1000.0

    # Minimum S-duct length from offset/length constraint
    # We'll check this after finding the fan position
    min_fan_x_frac = intake_end_frac + 0.10  # at least 10% chord S-duct

    best_fan_x_frac = 0.40  # fallback
    best_clearance = -np.inf
    best_fan_z = 0.0
    for xf in np.linspace(max(min_fan_x_frac, 0.25), 0.55, 40):
        z_up, _, z_lo = _body_surface_z(xf, params, _cache)
        # Place duct center so bottom clears intrados by min_clearance.
        # If duct doesn't fit, it protrudes above (dorsal bump) — that's OK.
        fan_z_candidate = z_lo * body_chord + duct_radius + min_clearance
        clr_bot = min_clearance  # guaranteed by construction
        clr_top = z_up * body_chord - (fan_z_candidate + duct_radius)
        # Optimize: maximize bottom clearance (stay away from intrados)
        # while minimizing bump (top protrusion)
        bump = max(0, -clr_top)
        score = clr_bot - bump * 0.5 - bump ** 2 * 2.0
        if score > best_clearance:
            best_clearance = score
            best_fan_x_frac = xf
            best_fan_z = fan_z_candidate

    fan_x_frac = best_fan_x_frac
    fan_x = fan_x_frac * body_chord

    # Verify S-duct offset/length
    # S-duct starts at bottom of intake ramp, not the surface
    intake_ramp_bottom_z = intake_z - intake_depth
    sduct_offset = abs(intake_ramp_bottom_z - best_fan_z)
    sduct_length = fan_x - (intake_x + intake_length)
    if sduct_length > 0.01:
        offset_ratio = sduct_offset / sduct_length
        if offset_ratio > config.max_offset_length_ratio:
            # Push fan aft to satisfy constraint
            required_length = sduct_offset / config.max_offset_length_ratio
            fan_x = intake_x + intake_length + required_length
            fan_x_frac = fan_x / body_chord
            # Recompute fan z at new position
            z_up, _, z_lo = _body_surface_z(fan_x_frac, params, _cache)
            best_fan_z = z_lo * body_chord + duct_radius + min_clearance

    fan_z = best_fan_z

    # ── Step 5: Exhaust ──
    exhaust_x_frac = config.exhaust_x_frac
    exhaust_x = exhaust_x_frac * body_chord
    exit_area = fan_area * config.nozzle_area_ratio
    exhaust_width = edf.fan_diameter
    exhaust_height = exit_area / exhaust_width
    z_up_ex, _, z_lo_ex = _body_surface_z(exhaust_x_frac, params, _cache)
    exhaust_z = (z_up_ex + z_lo_ex) / 2 * body_chord

    # ── Step 6: Housing ──
    housing_length = edf.fan_diameter * config.housing_length_factor

    return DuctPlacement(
        intake_x_frac=intake_x_frac, intake_x=intake_x, intake_z=intake_z,
        intake_width=intake_width, intake_depth=intake_depth,
        intake_length=intake_length,
        fan_x_frac=fan_x_frac, fan_x=fan_x, fan_z=fan_z,
        fan_diameter=edf.fan_diameter, duct_od=edf.duct_outer_diameter,
        exhaust_x_frac=exhaust_x_frac, exhaust_x=exhaust_x,
        exhaust_z=exhaust_z,
        exhaust_width=exhaust_width, exhaust_height=exhaust_height,
        duct_wall_thickness=duct_wall, housing_length=housing_length,
        body_chord=body_chord, intake_fallback=intake_fallback,
    )


# Backward-compatible alias
compute_duct_placement = size_and_place_duct


# ═══════════════════════════════════════════════════════════════════════════
# DuctSpine — single source of truth for duct geometry
# ═══════════════════════════════════════════════════════════════════════════

class DuctSpine:
    """Parametric duct spine — centerline + cross-sections from arc-length t.

    Encapsulates the complete duct geometry as a single object.  Every
    consumer (3-D builders, 2-D plot, clearance, bump) queries this spine
    instead of independently reconstructing coordinates.

    Parameters
    ----------
    placement : DuctPlacement
        Sized duct from ``size_and_place_duct``.

    The spine is parameterised by *arc-length fraction* ``t ∈ [0, 1]``:

    ========= =========== ============ =========== ===================
    t = 0     t_ramp_end  t_housing_s  t_housing_e t = 1
    INTAKE    S-DUCT      HOUSING      NOZZLE
    (rect)    (rect→circ) (circle)     (circ→ell)
    ========= =========== ============ =========== ===================
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, placement: DuctPlacement, body_twist_deg: float = 0.0,
                 n_table: int = 500):
        p = self._p = placement
        self._body_twist_rad = np.radians(body_twist_deg)

        # ── Build the three geometric curves ──────────────────────────
        # seg0: intake ramp — quadratic z = intake_z - depth * frac²
        ramp_end_x = p.intake_x + p.intake_length
        ramp_end_z = p.intake_z - p.intake_depth

        n0 = max(n_table // 5, 20)
        frac0 = np.linspace(0, 1, n0)
        seg0 = np.column_stack([
            p.intake_x + frac0 * p.intake_length,
            np.zeros(n0),
            p.intake_z - p.intake_depth * frac0 ** 2,
        ])

        # seg1: S-duct cubic Bezier  (ramp bottom → fan face)
        p0 = np.array([ramp_end_x, 0.0, ramp_end_z])
        p3 = np.array([p.fan_x, 0.0, p.fan_z])
        dx = p3[0] - p0[0]
        ramp_slope = -2 * p.intake_depth / max(p.intake_length, 1e-6)
        ramp_dir = np.array([1.0, 0.0, ramp_slope])
        ramp_dir /= np.linalg.norm(ramp_dir)
        self._bez1 = (p0, p0 + ramp_dir * dx * 0.4,
                       p3 + np.array([-dx * 0.4, 0.0, 0.0]), p3)

        n1 = max(n_table * 2 // 5, 40)
        seg1 = _cubic_bezier(*self._bez1, n1)

        # seg2: nozzle cubic Bezier  (fan face → exhaust)
        q0 = p3.copy()
        q3 = np.array([p.exhaust_x, 0.0, p.exhaust_z])
        dx2 = q3[0] - q0[0]
        self._bez2 = (q0, q0 + np.array([dx2 * 0.3, 0.0, 0.0]),
                       q3 + np.array([-dx2 * 0.3, 0.0, 0.0]), q3)

        n2 = n_table - n0 - n1
        seg2 = _cubic_bezier(*self._bez2, max(n2, 20))

        # Concatenate (removing duplicate junction points)
        self._pts = np.vstack([seg0, seg1[1:], seg2[1:]])
        n_total = len(self._pts)

        # ── Arc-length table ──────────────────────────────────────────
        diffs = np.diff(self._pts, axis=0)
        ds = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(ds)])
        total_len = cum[-1]
        self._s_total = total_len
        self._cum_s = cum                       # (n_total,)
        self._t_table = cum / total_len         # normalised to [0, 1]

        # ── Segment boundaries in t-space ─────────────────────────────
        # seg0 has n0 points → last index of seg0 = n0 - 1
        self._idx_ramp_end = n0 - 1
        # seg1[1:] adds n1-1 points → last index = n0-1 + n1-1
        self._idx_fan_face = n0 - 1 + n1 - 1

        self._t_ramp_end = self._t_table[self._idx_ramp_end]
        self._t_fan_face = self._t_table[self._idx_fan_face]

        # Housing boundaries in x-space → map to t
        housing_x_start = p.fan_x - p.housing_length / 2
        housing_x_end = p.fan_x + p.housing_length / 2
        # Clamp: housing can't start before ramp end or extend past exhaust
        housing_x_start = max(housing_x_start, ramp_end_x)
        housing_x_end = min(housing_x_end, p.exhaust_x)

        self._t_housing_start = float(np.interp(
            housing_x_start, self._pts[:, 0], self._t_table))
        self._t_housing_end = float(np.interp(
            housing_x_end, self._pts[:, 0], self._t_table))

        # Clamp: S-duct must have non-zero length
        if self._t_housing_start <= self._t_ramp_end + 1e-6:
            self._t_housing_start = self._t_ramp_end + 1e-6
        if self._t_housing_end <= self._t_housing_start + 1e-6:
            self._t_housing_end = self._t_housing_start + 1e-6

        # ── Radii ─────────────────────────────────────────────────────
        self._r_structural = p.duct_od / 2 + p.duct_wall_thickness
        self._r_flow = p.fan_diameter / 2

    # ------------------------------------------------------------------
    # Segment boundary properties
    # ------------------------------------------------------------------
    @property
    def t_ramp_end(self) -> float:
        """End of intake ramp / start of S-duct."""
        return self._t_ramp_end

    @property
    def t_fan_face(self) -> float:
        """Fan face — inside the housing, NOT a builder boundary."""
        return self._t_fan_face

    @property
    def t_housing_start(self) -> float:
        """Start of EDF cylindrical housing."""
        return self._t_housing_start

    @property
    def t_housing_end(self) -> float:
        """End of EDF housing / start of nozzle."""
        return self._t_housing_end

    @property
    def placement(self) -> DuctPlacement:
        return self._p

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------
    def position(self, t: float) -> np.ndarray:
        """(x, y=0, z) at arc-length fraction *t* ∈ [0, 1]."""
        t = float(np.clip(t, 0, 1))
        x = np.interp(t, self._t_table, self._pts[:, 0])
        z = np.interp(t, self._t_table, self._pts[:, 2])
        return np.array([x, 0.0, z])

    def positions(self, t_arr: np.ndarray) -> np.ndarray:
        """Vectorised: (N, 3) positions at an array of *t* values."""
        t_arr = np.clip(np.asarray(t_arr, dtype=float), 0, 1)
        x = np.interp(t_arr, self._t_table, self._pts[:, 0])
        z = np.interp(t_arr, self._t_table, self._pts[:, 2])
        return np.column_stack([x, np.zeros_like(x), z])

    def t_at_x(self, x_meters: float) -> float:
        """Inverse lookup: x [m] → arc-length fraction *t*."""
        return float(np.interp(x_meters, self._pts[:, 0], self._t_table))

    def segment_at(self, t: float) -> str:
        """Name of the segment at *t*: intake / sduct / housing / nozzle."""
        if t <= self._t_ramp_end:
            return "intake"
        elif t <= self._t_housing_start:
            return "sduct"
        elif t <= self._t_housing_end:
            return "housing"
        else:
            return "nozzle"

    # ------------------------------------------------------------------
    # Radius (half-height for clearance / envelope)
    # ------------------------------------------------------------------
    def radius(self, t: float, radius_mode: str = "structural") -> float:
        """Effective duct half-height at *t*.

        Returns the vertical half-extent of the cross-section, consistent
        with what ``cross_section(t)`` produces:

          intake  → rect half-height (intake_depth × frac / 2)
          sduct   → blend rect half-height → circle radius
          housing → circle radius
          nozzle  → cosine blend circle → exhaust_height / 2

        Parameters
        ----------
        radius_mode : 'structural' or 'flow'
        """
        r = self._r_structural if radius_mode == "structural" else self._r_flow

        if t <= 0:
            return 0.0
        if t < self._t_ramp_end:
            # Intake: rectangular section, half-height grows with frac
            frac = t / self._t_ramp_end
            return self._p.intake_depth * frac / 2
        if t <= self._t_housing_start:
            # S-duct: blend from rect half-height to circle radius
            span = self._t_housing_start - self._t_ramp_end
            blend = (t - self._t_ramp_end) / span if span > 1e-9 else 1.0
            rect_hh = self._p.intake_depth / 2
            return rect_hh * (1 - blend) + r * blend
        if t <= self._t_housing_end:
            return r
        # Nozzle taper: cosine blend to exhaust half-height
        s = (t - self._t_housing_end) / max(1.0 - self._t_housing_end, 1e-9)
        blend = 0.5 * (1 - np.cos(np.pi * s))
        return r * (1 - blend) + (self._p.exhaust_height / 2) * blend

    # ------------------------------------------------------------------
    # Cross-sections
    # ------------------------------------------------------------------
    def cross_section(self, t: float, n_pts: int = 48,
                      radius_mode: str = "flow") -> np.ndarray:
        """2-D cross-section (n_pts, 2) in local (horiz, vert) frame.

        Shape varies with segment:
          intake  → rectangular, opening up
          sduct   → superellipse blend rect → circle
          housing → circle
          nozzle  → circle → exhaust ellipse
        """
        p = self._p
        r = self._r_structural if radius_mode == "structural" else self._r_flow

        if t <= self._t_ramp_end:
            # Intake: rectangle opening from zero to full
            frac = t / self._t_ramp_end if self._t_ramp_end > 1e-9 else 1.0
            frac = max(frac, 0.01)
            # Use blend_rect_circle at blend=0 for C0 continuity with sduct
            w = p.intake_width / 2 * frac
            h = p.intake_depth * frac
            return _blend_rect_circle(w, h, r, 0.0, n_pts)

        if t <= self._t_housing_start:
            # S-duct: rect → circle blend
            span = self._t_housing_start - self._t_ramp_end
            blend = (t - self._t_ramp_end) / span if span > 1e-9 else 1.0
            return _blend_rect_circle(
                p.intake_width / 2, p.intake_depth, r, blend, n_pts)

        if t <= self._t_housing_end:
            # Housing: pure circle
            theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        # Nozzle: circle → area-preserving flat ellipse
        # Height blends toward exhaust target; width adjusts to conserve area.
        span = 1.0 - self._t_housing_end
        blend = (t - self._t_housing_end) / span if span > 1e-9 else 1.0
        b = p.exhaust_height / 2 * blend + r * (1 - blend)
        fan_area = np.pi * r ** 2
        exit_area = p.exhaust_width * p.exhaust_height
        target_area = fan_area * (1 - blend) + exit_area * blend
        a = target_area / (np.pi * max(b, 1e-6))
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        return np.column_stack([a * np.cos(theta), b * np.sin(theta)])

    def section_3d(self, t: float, n_pts: int = 48,
                   radius_mode: str = "flow") -> np.ndarray:
        """3-D section (n_pts, 3): cross-section placed at position(t).

        Applies body twist rotation (same as STL body export) so the duct
        mesh is consistent with the body mesh.
        """
        pos = self.position(t)
        cs = self.cross_section(t, n_pts, radius_mode)
        # Place section in un-twisted body coordinates
        x_local = pos[0]
        z_local = cs[:, 1] + pos[2]
        # Apply body twist around LE (x=0, z=0) — matches export.py
        tw = self._body_twist_rad
        x_rot = x_local * np.cos(tw) + z_local * np.sin(tw)
        z_rot = -x_local * np.sin(tw) + z_local * np.cos(tw)
        return np.column_stack([x_rot, cs[:, 0], z_rot])


# ═══════════════════════════════════════════════════════════════════════════
# Structure mass
# ═══════════════════════════════════════════════════════════════════════════

def compute_duct_structure_mass(placement: DuctPlacement,
                                material_density: float = 1500.0,
                                config=None) -> float:
    """Duct structure mass [kg] from wall volume + structural adder."""
    from ..config import DuctGeometryConfig
    if config is None:
        config = DuctGeometryConfig()
    density = config.material_density
    wall_t = config.duct_wall_mm / 1000.0
    adder = config.structural_adder

    # Approximate duct as cylinder from intake to exhaust
    total_length = placement.exhaust_x - placement.intake_x
    avg_perimeter = np.pi * placement.duct_od  # conservative (circular)
    wall_volume = avg_perimeter * wall_t * total_length
    return wall_volume * density * adder


# ═══════════════════════════════════════════════════════════════════════════
# Clearance validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_duct_clearance(placement: DuctPlacement, params: BWBParams,
                            min_clearance_mm: float = 5.0,
                            n_stations: int = 50) -> tuple[bool, list[ClearanceResult]]:
    """Check duct-to-body clearance at axial stations along the duct.

    Uses DuctSpine for consistent centerline and radius queries.
    Returns (all_ok, list[ClearanceResult]).
    """
    spine = DuctSpine(placement)
    _cache = _build_body_surface_cache(params)
    bc = placement.body_chord

    t_arr = np.linspace(0, 1, n_stations)
    results = []
    all_ok = True
    for t in t_arr:
        pos = spine.position(t)
        x_abs = pos[0]
        z_center = pos[2]
        x_frac = x_abs / bc

        if x_frac < 0.01 or x_frac > 0.99:
            continue

        z_up, _, z_lo = _body_surface_z(x_frac, params, _cache)
        duct_hh = spine.radius(t, radius_mode="structural")

        clr_top = (z_up * bc - (z_center + duct_hh)) * 1000
        clr_bot = ((z_center - duct_hh) - z_lo * bc) * 1000

        is_ok = clr_top >= min_clearance_mm and clr_bot >= min_clearance_mm
        if not is_ok:
            all_ok = False

        results.append(ClearanceResult(
            x_frac=x_frac, x_mm=x_abs * 1000,
            centerline_z_mm=z_center * 1000,
            duct_half_h_mm=duct_hh * 1000,
            body_z_upper_mm=z_up * bc * 1000,
            body_z_lower_mm=z_lo * bc * 1000,
            clearance_top_mm=clr_top, clearance_bot_mm=clr_bot,
            is_ok=is_ok,
        ))

    return all_ok, results


def min_effective_clearance_mm(results: list[ClearanceResult]) -> float:
    """Minimum clearance across all stations."""
    if not results:
        return 0.0
    return min(min(r.clearance_top_mm, r.clearance_bot_mm) for r in results)


# ═══════════════════════════════════════════════════════════════════════════
# Duct centerline
# ═══════════════════════════════════════════════════════════════════════════

def compute_duct_centerline(placement: DuctPlacement,
                            n_pts: int = 60) -> np.ndarray:
    """Duct centerline as (n_pts, 3) array [x, y=0, z] in meters.

    Delegates to DuctSpine for arc-length-uniform sampling.
    """
    spine = DuctSpine(placement)
    return spine.positions(np.linspace(0, 1, n_pts))


def _cubic_bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    """Evaluate cubic Bezier curve at n uniform parameter points."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-sections
# ═══════════════════════════════════════════════════════════════════════════

def duct_cross_section(t: float, placement: DuctPlacement,
                       n_pts: int = 48) -> np.ndarray:
    """2D cross-section at arc-length fraction t ∈ [0, 1].

    Delegates to DuctSpine for segment-aware cross-sections.
    Returns (n_pts, 2) array in local (horizontal, vertical) frame.
    """
    spine = DuctSpine(placement)
    return spine.cross_section(t, n_pts, radius_mode="flow")


def _rect_section(half_w: float, height: float, n_pts: int) -> np.ndarray:
    """Rectangular cross-section with rounded corners."""
    n_side = n_pts // 4
    top = np.column_stack([np.linspace(-half_w, half_w, n_side),
                           np.full(n_side, height / 2)])
    right = np.column_stack([np.full(n_side, half_w),
                             np.linspace(height / 2, -height / 2, n_side)])
    bottom = np.column_stack([np.linspace(half_w, -half_w, n_side),
                              np.full(n_side, -height / 2)])
    left = np.column_stack([np.full(n_side, -half_w),
                            np.linspace(-height / 2, height / 2, n_side)])
    return np.vstack([top, right[1:], bottom[1:], left[1:]])


def _blend_rect_circle(half_w, depth, radius, blend, n_pts):
    """Blend between rectangular and circular cross-section."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    # Circle
    cx = radius * np.cos(theta)
    cy = radius * np.sin(theta)
    # Rectangle (superellipse with high exponent)
    rx = half_w * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.3
    ry = (depth / 2) * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.3
    x = rx * (1 - blend) + cx * blend
    y = ry * (1 - blend) + cy * blend
    return np.column_stack([x, y])


# ═══════════════════════════════════════════════════════════════════════════
# Component geometry builders (for visualization & STL)
# ═══════════════════════════════════════════════════════════════════════════

def build_intake_geometry(placement: DuctPlacement, params: BWBParams,
                          n_profile: int = 32, n_axial: int = 8) -> list[np.ndarray]:
    """Build intake cross-sections — segment [0, t_ramp_end]."""
    spine = DuctSpine(placement)
    t_arr = np.linspace(0, spine.t_ramp_end, n_axial)
    return [spine.section_3d(t, n_profile) for t in t_arr]


def build_sduct_geometry(placement: DuctPlacement, params: BWBParams,
                         edf: EDFSpec,
                         n_stations: int = 12,
                         n_profile: int = 32) -> list[np.ndarray]:
    """Build S-duct cross-sections — segment [t_ramp_end, t_housing_start]."""
    spine = DuctSpine(placement)
    t_arr = np.linspace(spine.t_ramp_end, spine.t_housing_start, n_stations)
    return [spine.section_3d(t, n_profile) for t in t_arr]


def build_edf_housing(placement: DuctPlacement, edf: EDFSpec,
                      n_profile: int = 32, n_axial: int = 4) -> list[np.ndarray]:
    """Build EDF housing sections — segment [t_housing_start, t_housing_end]."""
    spine = DuctSpine(placement)
    t_arr = np.linspace(spine.t_housing_start, spine.t_housing_end, n_axial)
    return [spine.section_3d(t, n_profile) for t in t_arr]


def build_exhaust_geometry(placement: DuctPlacement, params: BWBParams,
                           edf: EDFSpec,
                           n_stations: int = 8,
                           n_profile: int = 32) -> list[np.ndarray]:
    """Build nozzle sections — segment [t_housing_end, 1.0]."""
    spine = DuctSpine(placement)
    t_arr = np.linspace(spine.t_housing_end, 1.0, n_stations)
    return [spine.section_3d(t, n_profile) for t in t_arr]


# ═══════════════════════════════════════════════════════════════════════════
# STL mesh builder
# ═══════════════════════════════════════════════════════════════════════════

def build_propulsion_mesh(params: BWBParams, edf: EDFSpec,
                          n_profile: int = 32) -> list:
    """Build complete duct as watertight triangle list for STL export.

    Uses continuous lofting across all 4 segments — no inter-component gaps.
    """
    from ..config import duct_from_config, load_config
    _cfg = duct_from_config(load_config())
    placement = size_and_place_duct(params, edf, config=_cfg)
    spine = DuctSpine(placement, body_twist_deg=params.body_twist)

    # Single continuous t-array with shared boundary points (no duplicates)
    t_intake = np.linspace(0, spine.t_ramp_end, 8)
    t_sduct = np.linspace(spine.t_ramp_end, spine.t_housing_start, 16)[1:]
    t_housing = np.linspace(spine.t_housing_start, spine.t_housing_end, 4)[1:]
    t_exhaust = np.linspace(spine.t_housing_end, 1.0, 10)[1:]
    t_all = np.concatenate([t_intake, t_sduct, t_housing, t_exhaust])

    sections = [spine.section_3d(t, n_profile) for t in t_all]

    triangles = []
    for i in range(len(sections) - 1):
        _loft_sections(sections[i], sections[i + 1], triangles)
    # Cap ends
    if sections:
        _cap_section(sections[0], triangles, flip=True)
        _cap_section(sections[-1], triangles, flip=False)

    return triangles


def _loft_sections(sec_a: np.ndarray, sec_b: np.ndarray, triangles: list):
    """Quad strip between two sections."""
    n = min(len(sec_a), len(sec_b))
    for j in range(n - 1):
        triangles.append([sec_a[j], sec_a[j + 1], sec_b[j + 1]])
        triangles.append([sec_a[j], sec_b[j + 1], sec_b[j]])
    # Close loop
    triangles.append([sec_a[-1], sec_a[0], sec_b[0]])
    triangles.append([sec_a[-1], sec_b[0], sec_b[-1]])


def _cap_section(section: np.ndarray, triangles: list, flip: bool = False):
    """Fan triangles from centroid."""
    centroid = section.mean(axis=0)
    n = len(section)
    for j in range(n):
        j_next = (j + 1) % n
        if flip:
            triangles.append([centroid, section[j_next], section[j]])
        else:
            triangles.append([centroid, section[j], section[j_next]])
