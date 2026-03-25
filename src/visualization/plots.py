"""Visualization utilities for the BWB flying wing optimization (v2).

Extends v1 plots with center-body + outer wing views and propulsion balance.
"""

import numpy as np
import matplotlib
if matplotlib.get_backend().lower() in ("agg", ""):
    try:
        # Don't override if running inside Jupyter (inline backend)
        get_ipython()  # raises NameError outside IPython/Jupyter
    except NameError:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .style import COLORS, FONT_SIZES, clean_spines, save_fig
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    build_kulfan_airfoil_at_station,
    build_body_kulfan_at_station,
    OUTER_WING_STATIONS,
    N_OUTER_SEGMENTS,
    BODY_STATIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Planform — top view + front view with center body and outer wing
# ═══════════════════════════════════════════════════════════════════════════

def plot_planform(params: BWBParams, ax=None, save_path: str | None = None):
    """Plot BWB planform (top view) and front view with gull-wing dihedral.

    Shows the center body (y=0 to body_halfwidth) and outer wing
    (body_halfwidth to half_span) in different colours on both views.
    """
    show = ax is None
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig = ax.figure
        axes = [ax, ax]

    p = params

    # ── Center-body geometry ──
    body_chord_root = p.body_root_chord
    body_chord_blend = p.wing_root_chord  # C0 continuity
    body_sweep_rad = np.radians(p.body_sweep_deg)
    bw = p.body_halfwidth

    # Body station positions (y and LE x)
    body_y = [frac * bw for frac in BODY_STATIONS]
    body_le_x = [y * np.tan(body_sweep_rad) for y in body_y]
    body_chords = [body_chord_root + frac * (body_chord_blend - body_chord_root)
                   for frac in BODY_STATIONS]

    # ── Outer-wing geometry ──
    tip_chord = p.tip_chord
    wing_sweep_rad = np.radians(p.le_sweep_deg)
    outer_span = p.outer_half_span

    # LE position at blend must match body tip
    x_blend = bw * np.tan(body_sweep_rad)

    wing_chords = [p.wing_root_chord + frac * (tip_chord - p.wing_root_chord)
                   for frac in OUTER_WING_STATIONS]
    from ..parameterization.bwb_aircraft import outer_wing_twists, outer_wing_dihedrals
    wing_twists = outer_wing_twists(p)
    wing_dihedrals = outer_wing_dihedrals(p)

    # 3-D positions along outer wing
    wing_positions = [[x_blend, bw, 0.0]]
    for i in range(N_OUTER_SEGMENTS):
        prev = wing_positions[-1]
        dy_seg = outer_span * (OUTER_WING_STATIONS[i + 1] - OUTER_WING_STATIONS[i])
        dih_rad = np.radians(wing_dihedrals[i])
        dx = dy_seg * np.tan(wing_sweep_rad)
        dy = dy_seg * np.cos(dih_rad)
        dz = dy_seg * np.sin(dih_rad)
        wing_positions.append([prev[0] + dx, prev[1] + dy, prev[2] + dz])

    # ────────────────────────── TOP VIEW ──────────────────────────
    ax_top = axes[0]
    body_color = COLORS["body"]
    wing_color = COLORS["wing"]

    for sign in [1, -1]:
        # Center body
        le_y_b = [y * sign for y in body_y]
        le_x_b = body_le_x
        te_x_b = [lx + c for lx, c in zip(body_le_x, body_chords)]

        ax_top.plot(le_y_b, le_x_b, color=body_color, linewidth=2)
        ax_top.plot(le_y_b, te_x_b, color=body_color, linewidth=2)
        ax_top.fill_betweenx(le_y_b, le_x_b, te_x_b,
                             alpha=0.12, color=body_color,
                             label="Center body" if sign == 1 else None)

        # Outer wing
        le_y_w = [pos[1] * sign for pos in wing_positions]
        le_x_w = [pos[0] for pos in wing_positions]
        te_x_w = [pos[0] + c for pos, c in zip(wing_positions, wing_chords)]

        ax_top.plot(le_y_w, le_x_w, color=wing_color, linewidth=2)
        ax_top.plot(le_y_w, te_x_w, color=wing_color, linewidth=2)
        ax_top.fill_betweenx(le_y_w, le_x_w, te_x_w,
                             alpha=0.12, color=wing_color,
                             label="Outer wing" if sign == 1 else None)

        # Tip closure
        ax_top.plot([le_y_w[-1], le_y_w[-1]],
                    [le_x_w[-1], te_x_w[-1]],
                    color=wing_color, linewidth=2)

    # Root closure
    ax_top.plot([0, 0], [0, body_chord_root], color=body_color, linewidth=2)

    # Section lines (body)
    for y, lx, c in zip(body_y, body_le_x, body_chords):
        for sign in [1, -1]:
            ax_top.plot([y * sign, y * sign], [lx, lx + c],
                        "k--", linewidth=0.4, alpha=0.3)

    # Section lines (wing)
    for pos, c in zip(wing_positions, wing_chords):
        for sign in [1, -1]:
            ax_top.plot([pos[1] * sign, pos[1] * sign],
                        [pos[0], pos[0] + c],
                        "k--", linewidth=0.4, alpha=0.3)

    # CG ref
    ax_top.plot(0, body_chord_root * 0.30, "ro", markersize=6, label="CG ref")

    from ..parameterization.bwb_aircraft import compute_wing_area, compute_aspect_ratio
    area = compute_wing_area(p)
    ar = compute_aspect_ratio(p)

    ax_top.set_xlabel("Span [m]")
    ax_top.set_ylabel("Chord [m]")
    ax_top.set_title(f"Top View — span={2*p.half_span:.2f}m, "
                     f"S={area:.3f}m\u00b2, AR={ar:.1f}")
    ax_top.set_aspect("equal")
    ax_top.legend(fontsize=FONT_SIZES["legend"])
    ax_top.grid(True, alpha=0.2)
    ax_top.invert_yaxis()

    # ────────────────────────── FRONT VIEW ──────────────────────────
    ax_front = axes[1]

    for sign in [1, -1]:
        # Center body is flat (z=0) from y=0 to y=body_halfwidth
        body_y_front = [y * sign for y in body_y]
        body_z_front = [0.0] * len(body_y)
        ax_front.plot(body_y_front, body_z_front, color=body_color,
                      linewidth=2.5,
                      label="Center body" if sign == 1 else None)

        # Outer wing with dihedral
        wing_y_front = [pos[1] * sign for pos in wing_positions]
        wing_z_front = [pos[2] for pos in wing_positions]
        ax_front.plot(wing_y_front, wing_z_front, color=wing_color,
                      linewidth=2.5,
                      label="Outer wing" if sign == 1 else None)

    ax_front.plot(0, 0, "ro", markersize=6)
    ax_front.set_xlabel("Span [m]")
    ax_front.set_ylabel("Height [m]")
    ax_front.set_title(f"Front View — dihedral [{p.dihedral_0:.0f}\u00b0, "
                       f"{p.dihedral_1:.0f}\u00b0, {p.dihedral_2:.0f}\u00b0, "
                       f"{p.dihedral_3:.0f}\u00b0, {p.dihedral_tip:.0f}\u00b0]")
    ax_front.set_aspect("equal")
    ax_front.legend(fontsize=FONT_SIZES["legend"])
    ax_front.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        if save_path:
            save_fig(fig, save_path)
        plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Airfoil profiles — body center, body blend, wing root, wing tip
# ═══════════════════════════════════════════════════════════════════════════

def plot_airfoils(params: BWBParams, save_path: str | None = None):
    """Plot 4 key airfoil cross-sections: body_center, body_blend, wing_root, wing_tip."""
    p = params
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # 1. Body center airfoil (Kulfan CST)
    kaf_center = build_body_kulfan_at_station(p, 0.0, name="body_center")
    af_center = kaf_center.to_airfoil()
    coords = af_center.coordinates
    ax = axes[0, 0]
    ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.1, color="darkorange")
    ax.set_ylabel("y/c")
    ax.set_title(f"Body Center — t/c={p.body_tc_root:.1%}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. Body blend (= wing root Kulfan at frac=0)
    kaf_blend = build_kulfan_airfoil_at_station(p, 0.0, name="body_blend")
    af_blend = kaf_blend.to_airfoil()
    coords = af_blend.coordinates
    tc_blend = kaf_blend.max_thickness()
    ax = axes[0, 1]
    ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.1, color="darkorange")
    ax.set_ylabel("y/c")
    ax.set_title(f"Body Blend / Wing Root — t/c={tc_blend:.1%}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 3. Wing root (same as blend but shown for clarity at frac=0.0)
    kaf_root = build_kulfan_airfoil_at_station(p, 0.0, name="wing_root")
    af_root = kaf_root.to_airfoil()
    coords = af_root.coordinates
    tc_root = kaf_root.max_thickness()
    ax = axes[1, 0]
    ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.1, color="steelblue")
    ax.set_ylabel("y/c")
    ax.set_xlabel("x/c")
    ax.set_title(f"Wing Root — t/c={tc_root:.1%}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 4. Wing tip (frac=1.0)
    kaf_tip = build_kulfan_airfoil_at_station(p, 1.0, name="wing_tip")
    af_tip = kaf_tip.to_airfoil()
    coords = af_tip.coordinates
    tc_tip = kaf_tip.max_thickness()
    ax = axes[1, 1]
    ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=1.5)
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.1, color="steelblue")
    ax.set_ylabel("y/c")
    ax.set_xlabel("x/c")
    ax.set_title(f"Wing Tip — t/c={tc_tip:.1%}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CAD-style multi-view drawing
# ═══════════════════════════════════════════════════════════════════════════

def plot_cad_multiview(params: BWBParams, placement=None,
                       controls=None, cg_data=None,
                       save_path: str | None = None) -> plt.Figure:
    """CAD-style 3-view drawing with duct integration.

    Layout (third-angle projection):
      Top-left:  Top view (XY) — planform + control surfaces
      Top-right: Side view (XZ) — body profile + duct + bump
      Bot-left:  Front view (YZ) — gull wing dihedral
      Bot-right: Airfoil sections overlay

    Parameters
    ----------
    params : BWBParams
    placement : DuctPlacement, optional
    controls : list of control surface geometries, optional
    cg_data : dict with 'x_cg', 'x_cg_frac', optional
    save_path : str, optional
    """
    from matplotlib.patches import Circle
    from matplotlib.gridspec import GridSpec
    from ..parameterization.bwb_aircraft import (
        compute_wing_area, compute_aspect_ratio, compute_mac,
        build_body_kulfan_at_station, build_kulfan_airfoil_at_station,
        apply_duct_bump, outer_wing_twists, outer_wing_dihedrals,
    )
    from ..propulsion.duct_geometry import (
        compute_duct_centerline, validate_duct_clearance,
    )

    p = params
    bc = p.body_root_chord
    bw = p.body_halfwidth
    s_ref = compute_wing_area(p)
    ar = compute_aspect_ratio(p)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.15,
                  height_ratios=[1.2, 1], width_ratios=[1, 1.3])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_front = fig.add_subplot(gs[1, 0])
    ax_af = fig.add_subplot(gs[1, 1])

    body_color = COLORS["body"]
    wing_color = COLORS["wing"]

    # ═══════════════════════════════════════════════════════════════════
    # TOP-LEFT: Top View (XY) — planform + control surfaces
    # ═══════════════════════════════════════════════════════════════════
    body_sweep_rad = np.radians(p.body_sweep_deg)
    wing_sweep_rad = np.radians(p.le_sweep_deg)
    outer_span = p.outer_half_span

    body_y = [frac * bw for frac in BODY_STATIONS]
    body_le_x = [y * np.tan(body_sweep_rad) for y in body_y]
    body_chords = [bc + frac * (p.wing_root_chord - bc) for frac in BODY_STATIONS]

    x_blend = bw * np.tan(body_sweep_rad)
    wing_dihedrals_deg = outer_wing_dihedrals(p)
    wing_chords = [p.wing_root_chord + frac * (p.tip_chord - p.wing_root_chord)
                   for frac in OUTER_WING_STATIONS]
    wing_pos = [[x_blend, bw, 0.0]]
    for i in range(N_OUTER_SEGMENTS):
        prev = wing_pos[-1]
        dy_seg = outer_span * (OUTER_WING_STATIONS[i+1] - OUTER_WING_STATIONS[i])
        dih_rad = np.radians(wing_dihedrals_deg[i])
        wing_pos.append([
            prev[0] + dy_seg * np.tan(wing_sweep_rad),
            prev[1] + dy_seg * np.cos(dih_rad),
            prev[2] + dy_seg * np.sin(dih_rad),
        ])

    for sign in [1, -1]:
        # Body
        by = [y * sign * 1000 for y in body_y]
        bx_le = [x * 1000 for x in body_le_x]
        bx_te = [(lx + c) * 1000 for lx, c in zip(body_le_x, body_chords)]
        ax_top.fill_betweenx(by, bx_le, bx_te, alpha=0.12, color=body_color,
                             label="Body" if sign == 1 else None)
        ax_top.plot(bx_le, by, color=body_color, lw=1.5)
        ax_top.plot(bx_te, by, color=body_color, lw=1.5)

        # Outer wing
        wy = [pos[1] * sign * 1000 for pos in wing_pos]
        wx_le = [pos[0] * 1000 for pos in wing_pos]
        wx_te = [(pos[0] + c) * 1000 for pos, c in zip(wing_pos, wing_chords)]
        ax_top.fill_betweenx(wy, wx_le, wx_te, alpha=0.10, color=wing_color,
                             label="Wing" if sign == 1 else None)
        ax_top.plot(wx_le, wy, color=wing_color, lw=1.5)
        ax_top.plot(wx_te, wy, color=wing_color, lw=1.5)
        ax_top.plot([wx_le[-1], wx_te[-1]], [wy[-1], wy[-1]], color=wing_color, lw=1.5)

    ax_top.plot([0, bc * 1000], [0, 0], color=body_color, lw=1.5)

    # Control surfaces
    if controls is not None:
        cs_colors = {'elevon': COLORS['elevon'], 'aileron': COLORS['aileron']}
        for g in controls:
            color = cs_colors.get(g.name, 'gray')
            for sign in [1, -1]:
                fill_x = np.concatenate([g.hinge_line_upper[:, 0], g.te_line_upper[::-1, 0]]) * 1000
                fill_y = np.concatenate([g.hinge_line_upper[:, 1], g.te_line_upper[::-1, 1]]) * sign * 1000
                ax_top.fill(fill_x, fill_y, alpha=0.30, color=color,
                            label=g.name if sign == 1 else None)
                ax_top.plot(g.hinge_line_upper[:, 0] * 1000,
                            g.hinge_line_upper[:, 1] * sign * 1000,
                            '--', color=color, lw=1.5)

    # CG marker
    if cg_data is not None:
        ax_top.plot(cg_data['x_cg'] * 1000, 0, 'go', ms=8, zorder=10,
                    label=f"CG x/c={cg_data['x_cg_frac']:.2f}")

    ax_top.set_xlabel('X [mm]')
    ax_top.set_ylabel('Y [mm]')
    ax_top.set_title(f'Top View  |  span={2*p.half_span*1000:.0f}mm  AR={ar:.1f}  S={s_ref*1e4:.0f}cm²',
                     fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_top.set_aspect('equal')
    ax_top.legend(fontsize=8, loc='upper right')
    ax_top.grid(True, alpha=0.15)
    ax_top.invert_yaxis()

    # ═══════════════════════════════════════════════════════════════════
    # TOP-RIGHT: Side View (XZ, y=0) — body + duct
    # ═══════════════════════════════════════════════════════════════════
    af_orig = build_body_kulfan_at_station(p, 0.0).to_airfoil()
    af_orig_x = af_orig.coordinates[:, 0] * bc * 1000
    af_orig_z = af_orig.coordinates[:, 1] * bc * 1000

    if placement is not None:
        coords_bumped = apply_duct_bump(af_orig.coordinates, placement, bc, bw, 0.0)
        af_bump_x = coords_bumped[:, 0] * bc * 1000
        af_bump_z = coords_bumped[:, 1] * bc * 1000

        ax_side.fill_between(af_bump_x, af_bump_z, alpha=0.08, color='#2c3e50')
        ax_side.plot(af_orig_x, af_orig_z, color='gray', lw=0.8, ls='--', alpha=0.4,
                     label='Original Kulfan')
        ax_side.plot(af_bump_x, af_bump_z, 'k-', lw=1.8, label='With duct bump')

        # Duct envelope
        _, clr = validate_duct_clearance(placement, p, min_clearance_mm=0.0)
        intake_end_x = (placement.intake_x + placement.intake_length) * 1000
        fan_x_mm = placement.fan_x * 1000
        blend_zone = fan_x_mm - intake_end_x
        dx_list, du_list, dl_list = [], [], []
        for r in clr:
            lo = r.centerline_z_mm - r.duct_half_h_mm
            tube_up = r.centerline_z_mm + r.duct_half_h_mm
            oml_up = r.body_z_upper_mm
            if r.x_mm <= intake_end_x:
                up = oml_up
            elif r.x_mm >= fan_x_mm or blend_zone < 1:
                up = tube_up
            else:
                t = (r.x_mm - intake_end_x) / blend_zone
                blend = 0.5 * (1 - np.cos(np.pi * t))
                up = oml_up * (1 - blend) + tube_up * blend
            dx_list.append(r.x_mm)
            du_list.append(up)
            dl_list.append(lo)

        ax_side.fill_between(dx_list, dl_list, du_list, alpha=0.18, color='#e74c3c',
                             label='Duct envelope')
        ax_side.plot(dx_list, du_list, color='#e74c3c', lw=1.2, ls='--')
        ax_side.plot(dx_list, dl_list, color='#e74c3c', lw=1.2, ls='--')

        # Centerline + markers
        cl = compute_duct_centerline(placement, n_pts=80)
        ax_side.plot(cl[:, 0]*1000, cl[:, 2]*1000, 'r-', lw=2, label='Centerline')
        ax_side.plot(placement.intake_x*1000, placement.intake_z*1000, 'go', ms=8)
        ax_side.plot(placement.fan_x*1000, placement.fan_z*1000, 'bs', ms=8)
        ax_side.plot(placement.exhaust_x*1000, placement.exhaust_z*1000, 'r^', ms=8)

        # EDF outline (structural OD — consistent with clearance envelope)
        fan_r = (placement.duct_od / 2 + placement.duct_wall_thickness) * 1000
        ax_side.add_patch(Circle((placement.fan_x*1000, placement.fan_z*1000),
                                  fan_r, fill=False, color='#3498db', lw=1.5))

        # Nozzle slot
        ex = placement.exhaust_x * 1000
        eh = placement.exhaust_height * 1000 / 2
        ez = placement.exhaust_z * 1000
        ax_side.plot([ex, ex], [ez - eh, ez + eh], color='orange', lw=3,
                     solid_capstyle='round', zorder=5, label='Nozzle slot')
    else:
        ax_side.fill_between(af_orig_x, af_orig_z, alpha=0.08, color='#2c3e50')
        ax_side.plot(af_orig_x, af_orig_z, 'k-', lw=1.8)

    ax_side.set_xlabel('X [mm]')
    ax_side.set_ylabel('Z [mm]')
    duct_label = f'  duct OD={placement.duct_od*1000:.0f}mm' if placement else ''
    ax_side.set_title(f'Side View (y=0)  |  chord={bc*1000:.0f}mm{duct_label}',
                      fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_side.set_aspect('equal')
    ax_side.legend(fontsize=7, loc='upper right', ncol=2)
    ax_side.grid(True, alpha=0.15)

    # ═══════════════════════════════════════════════════════════════════
    # BOTTOM-LEFT: Front View (YZ) — gull wing dihedral
    # ═══════════════════════════════════════════════════════════════════
    for sign in [1, -1]:
        by_f = [y * sign * 1000 for y in body_y]
        ax_front.plot(by_f, [0.0] * len(body_y), color=body_color, lw=2.5,
                      label="Body" if sign == 1 else None)

        wy_f = [pos[1] * sign * 1000 for pos in wing_pos]
        wz_f = [pos[2] * 1000 for pos in wing_pos]
        ax_front.plot(wy_f, wz_f, color=wing_color, lw=2.5,
                      label="Wing" if sign == 1 else None)

    ax_front.plot(0, 0, 'go', ms=6)
    dih_str = ', '.join(f'{d:.0f}°' for d in wing_dihedrals_deg)
    ax_front.set_xlabel('Y [mm]')
    ax_front.set_ylabel('Z [mm]')
    ax_front.set_title(f'Front View  |  dihedral [{dih_str}]',
                       fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_front.set_aspect('equal')
    ax_front.legend(fontsize=8)
    ax_front.grid(True, alpha=0.15)

    # ═══════════════════════════════════════════════════════════════════
    # BOTTOM-RIGHT: Airfoil sections (superposed)
    # ═══════════════════════════════════════════════════════════════════
    sections = [
        ("Body center", build_body_kulfan_at_station(p, 0.0, "center"), '#E74C3C', 2.0),
        ("Body blend", build_kulfan_airfoil_at_station(p, 0.0, "blend"), '#F39C12', 1.5),
        ("Wing 50%", build_kulfan_airfoil_at_station(p, 0.5, "mid"), '#2E86DE', 1.5),
        ("Wing tip", build_kulfan_airfoil_at_station(p, 1.0, "tip"), '#85C1E9', 1.0),
    ]
    for label, kaf, color, lw in sections:
        af = kaf.to_airfoil()
        tc = kaf.max_thickness()
        c = af.coordinates
        ax_af.plot(c[:, 0], c[:, 1], color=color, lw=lw,
                   label=f'{label} (t/c={tc:.1%})')
        ax_af.fill(c[:, 0], c[:, 1], alpha=0.06, color=color)

    ax_af.set_xlabel('x/c')
    ax_af.set_ylabel('z/c')
    ax_af.set_title('Airfoil Sections', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_af.set_aspect('equal')
    ax_af.legend(fontsize=8, loc='upper right')
    ax_af.grid(True, alpha=0.15)

    # ═══════════════════════════════════════════════════════════════════
    # Title bar
    # ═══════════════════════════════════════════════════════════════════
    fig.suptitle('MANTA BWB — 32-variable parametric model',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        save_fig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Optimization convergence
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(history: list[dict], save_path: str | None = None,
                     mode: str = "de"):
    """Plot optimization convergence (same logic as v1)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    if mode == "de":
        # Single objective: track L/D and feasibility
        ld_values = [h["result"]["L_over_D"] for h in history]
        feasible = [h["result"]["is_feasible"] for h in history]

        # Running best (feasible only)
        best_ld = []
        current_best = 0
        for ld, feas in zip(ld_values, feasible):
            if feas and ld > current_best:
                current_best = ld
            best_ld.append(current_best)

        colors = ["steelblue" if f else "lightcoral" for f in feasible]
        axes[0].scatter(range(len(ld_values)), ld_values, s=3, c=colors, alpha=0.4)
        axes[0].plot(best_ld, "r-", linewidth=1.5, label="Best feasible L/D")
        axes[0].set_ylabel("L/D")
        axes[0].set_title("Differential Evolution Convergence")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Penalty
        penalties = [h["result"]["penalty"] for h in history]
        axes[1].scatter(range(len(penalties)), penalties, s=3, alpha=0.3, c="coral")
        axes[1].set_ylabel("Penalty")
        axes[1].set_xlabel("Evaluation #")
        axes[1].set_yscale("symlog", linthresh=0.1)
        axes[1].grid(True, alpha=0.3)

    else:
        # NSGA-II mode
        ld_values = [-h["result"]["L_over_D"]
                     if "objectives" not in h else -h["objectives"][0]
                     for h in history]
        best_ld = np.maximum.accumulate([abs(v) for v in ld_values])

        axes[0].scatter(range(len(ld_values)), [abs(v) for v in ld_values],
                        s=3, alpha=0.3, c="steelblue")
        axes[0].plot(best_ld, "r-", linewidth=1.5, label="Running best")
        axes[0].set_ylabel("L/D")
        axes[0].set_title("NSGA-II Convergence")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        mass_values = [h["result"]["struct_mass"] for h in history]
        axes[1].scatter(range(len(mass_values)), mass_values, s=3, alpha=0.3, c="coral")
        axes[1].set_ylabel("Structural Mass [kg]")
        axes[1].set_xlabel("Evaluation #")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Surrogate-assisted convergence
# ═══════════════════════════════════════════════════════════════════════════

def plot_surrogate_convergence(cycle_history: list[dict],
                               save_path: str | None = None):
    """Plot surrogate-assisted optimization convergence.

    cycle_history entries must contain 'cycle', 'best_ld', 'total_vlm_evals'.
    'r2' is optional (may not be available in all cycles).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    cycles = [h["cycle"] for h in cycle_history]
    best_lds = [h["best_ld"] for h in cycle_history]
    vlm_evals = [h["total_vlm_evals"] for h in cycle_history]

    # r2 is optional
    has_r2 = all("r2" in h for h in cycle_history)
    r2s = [h.get("r2", 0.0) for h in cycle_history]

    # L/D per cycle
    axes[0].plot(cycles, best_lds, "bo-", linewidth=2, markersize=8)
    axes[0].set_ylabel("Best Feasible L/D", fontsize=12)
    axes[0].set_title("Surrogate-Assisted Optimization Convergence", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    for c, ld in zip(cycles, best_lds):
        axes[0].annotate(f"{ld:.1f}", (c, ld), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    # R^2 and VLM evals per cycle
    ax_r2 = axes[1]
    ax_evals = ax_r2.twinx()

    if has_r2:
        ax_r2.bar(cycles, r2s, color="steelblue", alpha=0.6, label="Surrogate R\u00b2")
        ax_r2.set_ylabel("Surrogate R\u00b2", fontsize=12, color="steelblue")
        ax_r2.set_ylim(0, 1.05)
    else:
        ax_r2.set_ylabel("(R\u00b2 not available)", fontsize=12, color="gray")

    ax_r2.set_xlabel("Cycle", fontsize=12)

    ax_evals.plot(cycles, vlm_evals, "r^-", linewidth=1.5, markersize=7, label="VLM evals")
    ax_evals.set_ylabel("Cumulative VLM Evals", fontsize=12, color="red")

    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Propulsion balance
# ═══════════════════════════════════════════════════════════════════════════

def plot_propulsion_balance(result: dict, save_path: str | None = None):
    """Plot propulsion thrust vs drag balance with endurance/range annotations.

    Parameters
    ----------
    result : dict
        Must contain keys:
        - T_available : float   — available thrust [N]
        - drag_force  : float   — total drag [N]
        - endurance_h : float   — endurance [hours]
        - range_km    : float   — range [km]
        - P_elec      : float   — electrical power [W]
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    t_avail = result["T_available"]
    drag = result["drag_force"]
    endurance = result["endurance_h"]
    range_km = result["range_km"]
    p_elec = result["P_elec"]

    # Bar chart: thrust vs drag
    bar_labels = ["T available", "Drag"]
    bar_values = [t_avail, drag]
    bar_colors = ["seagreen", "indianred"]

    bars = ax.bar(bar_labels, bar_values, color=bar_colors, width=0.5,
                  edgecolor="black", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f} N", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Surplus / deficit line
    surplus = t_avail - drag
    surplus_label = "surplus" if surplus >= 0 else "deficit"
    surplus_color = "green" if surplus >= 0 else "red"
    ax.axhline(y=drag, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_ylabel("Force [N]", fontsize=12)
    ax.set_title("Propulsion Balance", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Text box with performance
    text_lines = [
        f"Thrust {surplus_label}: {abs(surplus):.3f} N",
        f"Endurance: {endurance:.1f} h",
        f"Range: {range_km:.1f} km",
        f"P_elec: {p_elec:.1f} W",
    ]
    textstr = "\n".join(text_lines)

    props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                 edgecolor="gray", alpha=0.9)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    # Set y-axis to start at 0 with some headroom
    ax.set_ylim(0, max(t_avail, drag) * 1.3)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)
    return fig
