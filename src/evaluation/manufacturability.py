"""Manufacturability metrics for BWB flying wing designs.

Quantifies geometric properties that correlate with fabrication difficulty:
- Curvature gradients (twist, dihedral transitions)
- Minimum thickness
- Mold complexity (dihedral sign changes, taper severity)
- Body-wing blend smoothness

All metrics are computable from BWBParams alone (no simulation needed).
"""

import numpy as np
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    OUTER_WING_STATIONS,
    compute_internal_volume,
    outer_wing_twists,
    outer_wing_dihedrals,
)


def compute_manufacturability(params: BWBParams, config=None) -> dict:
    """Compute manufacturability metrics for a BWB design.

    Returns a dict with individual metrics and a composite score [0-1]
    where 1 = easy to manufacture, 0 = very difficult.
    """
    p = params
    metrics = {}

    # ── 1. Twist distribution analysis ────────────────────────────────────
    # Aggressive washout complicates composite layup and mold design.
    twists = outer_wing_twists(p)
    stations = OUTER_WING_STATIONS  # [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    outer_span = p.outer_half_span  # meters

    twist_gradients = []  # °/m between adjacent stations
    for i in range(len(twists) - 1):
        dy = (stations[i + 1] - stations[i]) * outer_span
        if dy > 0:
            twist_gradients.append(abs(twists[i + 1] - twists[i]) / dy)

    metrics["twist_gradient_max"] = max(twist_gradients) if twist_gradients else 0.0
    metrics["twist_gradient_mean"] = np.mean(twist_gradients) if twist_gradients else 0.0
    metrics["twist_range"] = abs(max(twists) - min(twists))

    # ── 2. Dihedral distribution analysis ─────────────────────────────────
    # Dihedral sign changes require complex jigs / multi-part molds.
    dihedrals = outer_wing_dihedrals(p)

    dihedral_gradients = []
    for i in range(len(dihedrals) - 1):
        dy = (stations[i + 1] - stations[i]) * outer_span
        if dy > 0:
            dihedral_gradients.append(abs(dihedrals[i + 1] - dihedrals[i]) / dy)

    metrics["dihedral_gradient_max"] = max(dihedral_gradients) if dihedral_gradients else 0.0
    metrics["dihedral_range"] = abs(max(dihedrals) - min(dihedrals))

    # Count dihedral sign changes (each requires a mold split)
    n_breaks = 0
    for i in range(len(dihedrals) - 1):
        if dihedrals[i] * dihedrals[i + 1] < 0:
            n_breaks += 1
    metrics["n_dihedral_breaks"] = n_breaks

    # ── 3. Minimum thickness ──────────────────────────────────────────────
    # Very thin tips are fragile and hard to layup.
    tip_chord = p.tip_chord  # meters
    # Approximate tip t/c from root + delta
    root_tc = 0.12  # NACA base for Kulfan at root
    tip_tc = root_tc + p.kulfan_tip_delta_tc
    tip_thickness_mm = tip_chord * max(tip_tc, 0.06) * 1000

    # Body minimum thickness at centerline
    body_thickness_mm = p.body_root_chord * p.body_tc_root * 1000

    metrics["thickness_tip_mm"] = tip_thickness_mm
    metrics["thickness_body_mm"] = body_thickness_mm
    metrics["thickness_min_mm"] = min(tip_thickness_mm, body_thickness_mm)

    # ── 4. Taper severity ─────────────────────────────────────────────────
    # High taper (low ratio) → sharp tip, difficult layup, structural weakness.
    metrics["taper_ratio"] = p.taper_ratio
    metrics["taper_severity"] = 1.0 - p.taper_ratio  # 0 = rect, 1 = point

    # ── 5. Body-wing blend smoothness ─────────────────────────────────────
    # Large chord ratio = abrupt blend at body/wing junction.
    metrics["body_chord_ratio"] = p.body_chord_ratio
    # Sweep discontinuity at blend
    metrics["sweep_delta"] = abs(p.body_sweep_delta)

    # ── 6. Surface complexity ─────────────────────────────────────────────
    # LE droop creates compound curvature at the nose.
    metrics["le_droop"] = p.body_le_droop
    # Camber + reflex interaction
    metrics["camber_reflex_range"] = abs(p.body_camber) + abs(p.body_reflex)

    # ── 7. Internal volume (equipment accessibility) ──────────────────────
    try:
        vol = compute_internal_volume(p)
    except Exception:
        vol = 0.0
    metrics["internal_volume_m3"] = vol

    # ── 8. Span / aspect ratio (structural sizing driver) ─────────────────
    metrics["half_span"] = p.half_span

    # ══════════════════════════════════════════════════════════════════════
    # Composite score [0, 1]: 1 = easy to manufacture
    # ══════════════════════════════════════════════════════════════════════
    from ..config import ManufConfig
    if config is None:
        config = ManufConfig()
    w = config.weights
    t = config.thresholds

    scores = []

    s_twist = _ramp(metrics["twist_gradient_max"], good=t["twist_gradient"][0], bad=t["twist_gradient"][1])
    scores.append(("twist_smoothness", s_twist, w["twist_smoothness"]))

    s_dihedral = _ramp(metrics["dihedral_gradient_max"], good=t["dihedral_gradient"][0], bad=t["dihedral_gradient"][1])
    scores.append(("dihedral_smoothness", s_dihedral, w["dihedral_smoothness"]))

    s_breaks = _ramp(metrics["n_dihedral_breaks"], good=t["dihedral_breaks"][0], bad=t["dihedral_breaks"][1])
    scores.append(("dihedral_simplicity", s_breaks, w["dihedral_simplicity"]))

    s_thickness = _ramp_inv(metrics["thickness_tip_mm"], good=t["tip_thickness_mm"][0], bad=t["tip_thickness_mm"][1])
    scores.append(("tip_robustness", s_thickness, w["tip_robustness"]))

    s_taper = _ramp_inv(metrics["taper_ratio"], good=t["taper_ratio"][0], bad=t["taper_ratio"][1])
    scores.append(("taper_simplicity", s_taper, w["taper_simplicity"]))

    s_blend = _ramp(metrics["body_chord_ratio"], good=t["chord_ratio"][0], bad=t["chord_ratio"][1])
    scores.append(("blend_smoothness", s_blend, w["blend_smoothness"]))

    s_sweep = _ramp(metrics["sweep_delta"], good=t["sweep_delta"][0], bad=t["sweep_delta"][1])
    scores.append(("sweep_continuity", s_sweep, w["sweep_continuity"]))

    s_twist_range = _ramp(metrics["twist_range"], good=t["twist_range"][0], bad=t["twist_range"][1])
    scores.append(("twist_simplicity", s_twist_range, w["twist_simplicity"]))

    s_vol = _ramp_inv(metrics["internal_volume_m3"], good=t["volume"][0], bad=t["volume"][1])
    scores.append(("equipment_space", s_vol, w["equipment_space"]))

    s_span = _ramp(metrics["half_span"], good=t["half_span"][0], bad=t["half_span"][1])
    scores.append(("mold_size", s_span, w["mold_size"]))

    # Store sub-scores
    total_weight = sum(w for _, _, w in scores)
    composite = sum(s * w for _, s, w in scores) / total_weight

    for name, s, w in scores:
        metrics[f"sub_{name}"] = round(s, 4)

    metrics["manufacturability_score"] = round(composite, 4)

    return metrics


def _ramp(value: float, good: float, bad: float) -> float:
    """Score 1.0 when value <= good, 0.0 when value >= bad, linear between."""
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return (bad - value) / (bad - good)


def _ramp_inv(value: float, good: float, bad: float) -> float:
    """Score 1.0 when value >= good, 0.0 when value <= bad, linear between."""
    if value >= good:
        return 1.0
    if value <= bad:
        return 0.0
    return (value - bad) / (good - bad)
