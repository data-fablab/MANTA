"""30-variable BWB flying wing parametric model.

Design variables cover:
- Outer wing planform (4): half_span, root_chord, taper_ratio, LE sweep
- Center-body BWB (8): chord ratio, width, t/c, camber, reflex, twist, sweep, LE droop
- Outer wing twist (5): at 5 spanwise stations
- Outer wing dihedral (5): per segment including winglet
- Kulfan CST airfoil (8): 6 root weights + 2 tip delta-offsets
"""

import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class BWBParams:
    """30-variable BWB flying wing with proper center-body and outer wing."""

    # ── Planform (4) ──
    half_span: float = 0.75             # [m] total half wingspan
    wing_root_chord: float = 0.42       # [m] outer wing root chord (= body tip chord)
    taper_ratio: float = 0.15           # tip/root chord ratio
    le_sweep_deg: float = 26.0          # [deg] outer wing LE sweep

    # ── Center-Body BWB (8) ──
    body_chord_ratio: float = 1.4       # body_root_chord / wing_root_chord
    body_halfwidth: float = 0.12        # [m] center body half-width
    body_tc_root: float = 0.18          # t/c at centerline
    body_camber: float = 0.03           # camber at centerline
    body_reflex: float = 0.02           # TE reflex (controls CM)
    body_twist: float = 2.0             # [deg] centerline incidence
    body_sweep_delta: float = 5.0       # [deg] extra LE sweep over wing sweep
    body_le_droop: float = 0.16         # LE radius weight (nose rounding)

    # ── Outer Wing Twist (5) ── root twist = 0 (continuity from body tip)
    twist_1: float = -0.5               # [deg] at 25% outer span
    twist_2: float = -1.2               # [deg] at 50% outer span
    twist_3: float = -2.0               # [deg] at 75% outer span
    twist_4: float = -2.8               # [deg] at 90% outer span
    twist_tip: float = -3.5             # [deg] at tip

    # ── Outer Wing Dihedral (5) ──
    dihedral_0: float = 18.0            # [deg] root to 25% outer span (gull)
    dihedral_1: float = 4.0             # [deg] 25% to 50%
    dihedral_2: float = 0.0             # [deg] 50% to 75%
    dihedral_3: float = -1.5            # [deg] 75% to 90%
    dihedral_tip: float = 5.0           # [deg] 90% to tip (winglet)

    # ── Kulfan CST Airfoil (8) ── root + tip delta-offsets
    kulfan_root_u2: float = 0.3359      # root upper weight[2] (mid-chord)
    kulfan_root_u6: float = 0.2861      # root upper weight[6] (TE inner)
    kulfan_root_u7: float = 0.2819      # root upper weight[7] (TE reflex)
    kulfan_root_l2: float = -0.0129     # root lower weight[2]
    kulfan_root_l6: float = -0.0112     # root lower weight[6]
    kulfan_root_l7: float = -0.0030     # root lower weight[7]
    kulfan_tip_delta_tc: float = 0.0    # tip t/c offset from root
    kulfan_tip_delta_camber: float = 0.0  # tip camber offset from root

    # ── Derived geometry ──
    @property
    def body_root_chord(self) -> float:
        """Center body chord at y=0 [m]."""
        return self.wing_root_chord * self.body_chord_ratio

    @property
    def body_sweep_deg(self) -> float:
        """Center body LE sweep [deg]."""
        return self.le_sweep_deg + self.body_sweep_delta

    @property
    def outer_half_span(self) -> float:
        """Outer wing half-span (from body edge to tip) [m]."""
        return self.half_span - self.body_halfwidth

    @property
    def tip_chord(self) -> float:
        """Outer wing tip chord [m]."""
        return self.wing_root_chord * self.taper_ratio


# ── Bounds ───────────────────────────────────────────────────────────────

BOUNDS = {
    # Planform
    "half_span":            (0.55, 1.00),
    "wing_root_chord":      (0.25, 0.55),
    "taper_ratio":          (0.08, 0.40),
    "le_sweep_deg":         (18.0, 35.0),
    # Center-body
    "body_chord_ratio":     (1.1, 2.0),
    "body_halfwidth":       (0.08, 0.25),
    "body_tc_root":         (0.12, 0.25),
    "body_camber":          (0.01, 0.06),
    "body_reflex":          (-0.02, 0.08),
    "body_twist":           (-2.0, 6.0),
    "body_sweep_delta":     (0.0, 15.0),
    "body_le_droop":        (0.10, 0.30),
    # Twist
    "twist_1":              (-3.0, 1.0),
    "twist_2":              (-5.0, 0.0),
    "twist_3":              (-7.0, 0.0),
    "twist_4":              (-7.0, 0.0),
    "twist_tip":            (-8.0, -0.5),
    # Dihedral
    "dihedral_0":           (5.0, 30.0),
    "dihedral_1":           (-5.0, 15.0),
    "dihedral_2":           (-8.0, 5.0),
    "dihedral_3":           (-8.0, 5.0),
    "dihedral_tip":         (0.0, 90.0),
    # Kulfan CST
    "kulfan_root_u2":       (0.10, 0.55),
    "kulfan_root_u6":       (0.08, 0.48),
    "kulfan_root_u7":       (0.02, 0.48),
    "kulfan_root_l2":       (-0.22, 0.26),
    "kulfan_root_l6":       (-0.22, 0.25),
    "kulfan_root_l7":       (-0.28, 0.25),
    "kulfan_tip_delta_tc":  (-0.04, 0.02),
    "kulfan_tip_delta_camber": (-0.03, 0.02),
}

VAR_NAMES = list(BOUNDS.keys())
N_VARS = len(VAR_NAMES)

assert N_VARS == 30, f"Expected 30 variables, got {N_VARS}"


# ── Conversion utilities ─────────────────────────────────────────────────

def params_from_vector(x: np.ndarray) -> BWBParams:
    """Create BWBParams from a flat design vector (30,)."""
    return BWBParams(**{name: float(x[i]) for i, name in enumerate(VAR_NAMES)})


def params_to_vector(p: BWBParams) -> np.ndarray:
    """Convert BWBParams to a flat design vector (30,)."""
    d = asdict(p)
    return np.array([d[name] for name in VAR_NAMES])


def get_bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (lower_bounds, upper_bounds) as numpy arrays."""
    lb = np.array([BOUNDS[n][0] for n in VAR_NAMES])
    ub = np.array([BOUNDS[n][1] for n in VAR_NAMES])
    return lb, ub


def get_scipy_bounds() -> list[tuple[float, float]]:
    """Return bounds as list of (lo, hi) tuples for scipy."""
    return [BOUNDS[n] for n in VAR_NAMES]
