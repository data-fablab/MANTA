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
)


def compute_manufacturability(params: BWBParams) -> dict:
    """Compute manufacturability metrics for a BWB design.

    Returns a dict with individual metrics and a composite score [0-1]
    where 1 = easy to manufacture, 0 = very difficult.
    """
    p = params
    metrics = {}

    # ── 1. Twist distribution analysis ────────────────────────────────────
    # Aggressive washout complicates composite layup and mold design.
    twists = [0.0, p.twist_1, p.twist_2, p.twist_3, p.twist_4, p.twist_tip]
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
    dihedrals = [p.dihedral_0, p.dihedral_1, p.dihedral_2,
                 p.dihedral_3, p.dihedral_tip]

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
    # Each sub-score maps a metric to [0, 1] via linear ramps.

    scores = []

    # Twist gradient: < 5 °/m is easy, > 20 °/m is very hard
    s_twist = _ramp(metrics["twist_gradient_max"], good=5.0, bad=20.0)
    scores.append(("twist_smoothness", s_twist, 0.15))

    # Dihedral gradient: < 30 °/m is easy, > 100 °/m is hard
    s_dihedral = _ramp(metrics["dihedral_gradient_max"], good=30.0, bad=100.0)
    scores.append(("dihedral_smoothness", s_dihedral, 0.10))

    # Dihedral breaks: 0 is ideal, 3+ is bad
    s_breaks = _ramp(metrics["n_dihedral_breaks"], good=0.0, bad=3.0)
    scores.append(("dihedral_simplicity", s_breaks, 0.10))

    # Tip thickness: > 8 mm is easy, < 3 mm is hard
    s_thickness = _ramp_inv(metrics["thickness_tip_mm"], good=8.0, bad=3.0)
    scores.append(("tip_robustness", s_thickness, 0.15))

    # Taper: ratio > 0.25 is easy, < 0.08 is hard
    s_taper = _ramp_inv(metrics["taper_ratio"], good=0.25, bad=0.08)
    scores.append(("taper_simplicity", s_taper, 0.10))

    # Chord ratio: < 1.3 is easy, > 2.0 is hard
    s_blend = _ramp(metrics["body_chord_ratio"], good=1.3, bad=2.0)
    scores.append(("blend_smoothness", s_blend, 0.10))

    # Sweep delta: < 3° is easy, > 12° is hard
    s_sweep = _ramp(metrics["sweep_delta"], good=3.0, bad=12.0)
    scores.append(("sweep_continuity", s_sweep, 0.05))

    # Twist range: < 3° is easy, > 8° is hard
    s_twist_range = _ramp(metrics["twist_range"], good=3.0, bad=8.0)
    scores.append(("twist_simplicity", s_twist_range, 0.10))

    # Volume: > 0.004 m³ is easy, < 0.002 m³ is hard
    s_vol = _ramp_inv(metrics["internal_volume_m3"], good=0.004, bad=0.002)
    scores.append(("equipment_space", s_vol, 0.10))

    # Span: < 0.8m is easy, > 1.0m is hard (transport, mold size)
    s_span = _ramp(metrics["half_span"], good=0.8, bad=1.0)
    scores.append(("mold_size", s_span, 0.05))

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
