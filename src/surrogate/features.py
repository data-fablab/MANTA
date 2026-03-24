"""Feature augmentation for 32-variable BWB design space.

Takes (N, 32) raw design vectors and appends 20 derived geometric features,
producing (N, 52) augmented feature matrices for the surrogate MLP.
"""

import numpy as np

from ..parameterization.bwb_aircraft import vectorized_body_tc_at_xc

# ---------------------------------------------------------------------------
# Design variable indices (matching VAR_NAMES in design_variables.py)
# ---------------------------------------------------------------------------
_IDX_HALF_SPAN = 0
_IDX_WING_ROOT_CHORD = 1
_IDX_TAPER_RATIO = 2
_IDX_LE_SWEEP_DEG = 3

_IDX_BODY_CHORD_RATIO = 4
_IDX_BODY_HALFWIDTH = 5
_IDX_BODY_KU_U2 = 6
_IDX_BODY_KU_U6 = 7
_IDX_BODY_KU_U7 = 8
_IDX_BODY_KU_L2 = 9
_IDX_BODY_KU_L6 = 10
_IDX_BODY_KU_L7 = 11
_IDX_BODY_TWIST = 12
_IDX_BODY_SWEEP_DELTA = 13

_IDX_TWIST = slice(14, 19)       # twist_1 .. twist_tip
_IDX_TWIST_TIP = 18

_IDX_DIHEDRAL = slice(19, 24)    # dihedral_0 .. dihedral_tip

_IDX_KU_ROOT_U2 = 24
_IDX_KU_ROOT_U6 = 25
_IDX_KU_ROOT_U7 = 26
_IDX_KU_ROOT_L2 = 27
_IDX_KU_ROOT_L6 = 28
_IDX_KU_ROOT_L7 = 29
_IDX_KU_TIP_DELTA_TC = 30
_IDX_KU_TIP_DELTA_CAMBER = 31

# Outer wing dihedral segment fractions (sum = 1.0)
_DIHEDRAL_SEGMENTS = np.array([0.25, 0.25, 0.25, 0.15, 0.10])

_EDF_DUCT_OD = 0.078  # [m] EDF 70mm duct outer diameter (for duct ratio feature)


def augment_features(X: np.ndarray) -> np.ndarray:
    """Append 20 derived geometric features to raw 32-var input -> (N, 52).

    Parameters
    ----------
    X : np.ndarray, shape (N, 32)
        Raw design vectors.

    Returns
    -------
    np.ndarray, shape (N, 52)
        Augmented feature matrix.
    """
    # --- Extract raw variables ---
    hs = X[:, _IDX_HALF_SPAN]
    rc = X[:, _IDX_WING_ROOT_CHORD]
    tr = X[:, _IDX_TAPER_RATIO]
    sweep_deg = X[:, _IDX_LE_SWEEP_DEG]

    bcr = X[:, _IDX_BODY_CHORD_RATIO]
    bw = X[:, _IDX_BODY_HALFWIDTH]
    bsd = X[:, _IDX_BODY_SWEEP_DELTA]

    # Body t/c at max-thickness station (x/c=0.30) from CST evaluation
    btc = np.clip(vectorized_body_tc_at_xc(
        X[:, _IDX_BODY_KU_U2], X[:, _IDX_BODY_KU_U6], X[:, _IDX_BODY_KU_U7],
        X[:, _IDX_BODY_KU_L2], X[:, _IDX_BODY_KU_L6], X[:, _IDX_BODY_KU_L7],
        x_frac=0.30,
    ), 0.10, 0.30)

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

    # Twist features
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
    body_volume = body_root_chord * btc * bw * (np.pi / 4) * 0.6
    body_frontal_area = body_root_chord * btc
    body_wetted_area = 2.05 * (body_root_chord + rc) * bw
    body_aspect_ratio = 2 * bw / np.maximum((body_root_chord + rc) / 2, 1e-6)

    # Body total sweep
    body_sweep_total = sweep_deg + bsd

    # Span-body ratio
    span_body_ratio = bw / np.maximum(hs, 1e-6)

    # Duct-body coupling: body height at fan station (x/c=0.40) vs duct OD
    btc_fan = vectorized_body_tc_at_xc(
        X[:, _IDX_BODY_KU_U2], X[:, _IDX_BODY_KU_U6], X[:, _IDX_BODY_KU_U7],
        X[:, _IDX_BODY_KU_L2], X[:, _IDX_BODY_KU_L6], X[:, _IDX_BODY_KU_L7],
        x_frac=0.40,
    )
    body_height_fan_ratio = btc_fan * body_root_chord / _EDF_DUCT_OD

    return np.column_stack([
        X,
        wing_area,              # 32
        aspect_ratio,           # 33
        mac,                    # 34
        tip_chord,              # 35
        body_root_chord,        # 36
        outer_span,             # 37
        twist_gradient,         # 38
        mean_twist,             # 39
        effective_span,         # 40
        qc_sweep,               # 41
        reflex_root,            # 42
        camber_root,            # 43
        ac_offset,              # 44
        body_volume,            # 45
        body_frontal_area,      # 46
        body_wetted_area,       # 47
        body_aspect_ratio,      # 48
        body_sweep_total,       # 49
        span_body_ratio,        # 50
        body_height_fan_ratio,  # 51
    ])
