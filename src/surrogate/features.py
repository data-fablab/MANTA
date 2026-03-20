"""Feature augmentation for 30-variable BWB design space.

Takes (N, 30) raw design vectors and appends ~22 derived geometric features,
producing (N, 52) augmented feature matrices for the surrogate MLP.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Design variable indices (matching VAR_NAMES in design_variables.py)
# ---------------------------------------------------------------------------
_IDX_HALF_SPAN = 0
_IDX_WING_ROOT_CHORD = 1
_IDX_TAPER_RATIO = 2
_IDX_LE_SWEEP_DEG = 3

_IDX_BODY_CHORD_RATIO = 4
_IDX_BODY_HALFWIDTH = 5
_IDX_BODY_TC_ROOT = 6
_IDX_BODY_CAMBER = 7
_IDX_BODY_REFLEX = 8
_IDX_BODY_TWIST = 9
_IDX_BODY_SWEEP_DELTA = 10
_IDX_BODY_LE_DROOP = 11

_IDX_TWIST = slice(12, 17)       # twist_1 .. twist_tip
_IDX_TWIST_TIP = 16

_IDX_DIHEDRAL = slice(17, 22)    # dihedral_0 .. dihedral_tip

_IDX_KU_ROOT_U2 = 22
_IDX_KU_ROOT_U6 = 23
_IDX_KU_ROOT_U7 = 24
_IDX_KU_ROOT_L2 = 25
_IDX_KU_ROOT_L6 = 26
_IDX_KU_ROOT_L7 = 27
_IDX_KU_TIP_DELTA_TC = 28
_IDX_KU_TIP_DELTA_CAMBER = 29

# Outer wing dihedral segment fractions (sum = 1.0)
_DIHEDRAL_SEGMENTS = np.array([0.25, 0.25, 0.25, 0.15, 0.10])


def augment_features(X: np.ndarray) -> np.ndarray:
    """Append 19 derived geometric features to raw 30-var input -> (N, 49).

    Removed 3 features that were near-perfect duplicates (|r|>0.99):
    - taper_elliptic_dev  ≡ |taper_ratio - 0.4|  (r=-1.0 with taper_ratio)
    - total_twist         ≡ twist_tip             (r=+1.0 with twist_tip)
    - blend_tc_mismatch   ≡ |body_tc - 0.105|     (r=+1.0 with body_tc_root)

    Parameters
    ----------
    X : np.ndarray, shape (N, 30)
        Raw design vectors.

    Returns
    -------
    np.ndarray, shape (N, 49)
        Augmented feature matrix.
    """
    # --- Extract raw variables ---
    hs = X[:, _IDX_HALF_SPAN]
    rc = X[:, _IDX_WING_ROOT_CHORD]
    tr = X[:, _IDX_TAPER_RATIO]
    sweep_deg = X[:, _IDX_LE_SWEEP_DEG]

    bcr = X[:, _IDX_BODY_CHORD_RATIO]
    bw = X[:, _IDX_BODY_HALFWIDTH]
    btc = X[:, _IDX_BODY_TC_ROOT]
    bsd = X[:, _IDX_BODY_SWEEP_DELTA]

    twists = X[:, _IDX_TWIST]
    twist_tip = X[:, _IDX_TWIST_TIP]
    dihedrals = X[:, _IDX_DIHEDRAL]

    ku_root_u2 = X[:, _IDX_KU_ROOT_U2]
    ku_root_u7 = X[:, _IDX_KU_ROOT_U7]
    ku_root_l2 = X[:, _IDX_KU_ROOT_L2]
    ku_root_l7 = X[:, _IDX_KU_ROOT_L7]

    # --- Derived geometry ---
    tip_chord = rc * tr
    body_root_chord = rc * bcr
    outer_span = hs - bw

    # Wing area: body trapezoid + outer wing trapezoid
    wing_area = (body_root_chord + rc) * bw + (rc + tip_chord) * outer_span

    # Aspect ratio: (2*half_span)^2 / wing_area
    aspect_ratio = (2 * hs) ** 2 / np.maximum(wing_area, 1e-6)

    # Mean aerodynamic chord (outer wing)
    mac = (2 / 3) * rc * (1 + tr + tr ** 2) / (1 + tr)

    # Twist features (total_twist removed: identical to twist_tip which is already in X)
    twist_gradient = twist_tip / np.maximum(outer_span, 1e-6)
    mean_twist = twists.mean(axis=1)

    # Effective span accounting for dihedral
    dih_rad = np.radians(dihedrals)
    seg_lengths = outer_span[:, None] * _DIHEDRAL_SEGMENTS
    effective_span = 2 * (seg_lengths * np.cos(dih_rad)).sum(axis=1)

    # Quarter-chord sweep
    sweep_rad = np.radians(sweep_deg)
    qc_sweep = np.degrees(np.arctan(
        np.tan(sweep_rad) + (tr - 1) * rc / (4 * np.maximum(outer_span, 1e-6))
    ))

    # Airfoil-derived features
    reflex_root = ku_root_u7 + ku_root_l7
    camber_root = ku_root_u2 + ku_root_l2

    # Aerodynamic center offset (from wing root LE)
    ac_offset = 0.25 * mac + (outer_span / 3) * (1 + 2 * tr) / (1 + tr) * np.tan(sweep_rad)

    # Body-derived features
    # (taper_elliptic_dev removed: r=-1.0 with taper_ratio)
    # (blend_tc_mismatch removed: r=+1.0 with body_tc_root)
    body_volume = body_root_chord * btc * bw * (np.pi / 4) * 0.6
    body_frontal_area = body_root_chord * btc
    body_wetted_area = 2.05 * (body_root_chord + rc) * bw
    body_aspect_ratio = 2 * bw / np.maximum((body_root_chord + rc) / 2, 1e-6)

    # Body total sweep
    body_sweep_total = sweep_deg + bsd

    # Span-body ratio
    span_body_ratio = bw / np.maximum(hs, 1e-6)

    return np.column_stack([
        X,
        wing_area,              # 30
        aspect_ratio,           # 31
        mac,                    # 32
        tip_chord,              # 33
        body_root_chord,        # 34
        outer_span,             # 35
        twist_gradient,         # 36
        mean_twist,             # 37
        effective_span,         # 38
        qc_sweep,               # 39
        reflex_root,            # 40
        camber_root,            # 41
        ac_offset,              # 42
        body_volume,            # 43
        body_frontal_area,      # 44
        body_wetted_area,       # 45
        body_aspect_ratio,      # 46
        body_sweep_total,       # 47
        span_body_ratio,        # 48
    ])
