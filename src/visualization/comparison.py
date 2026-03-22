"""Multi-design comparison plots for the Design Catalog.

Provides Pareto plots, radar charts, planform overlays, and summary tables
for comparing multiple BWB designs across aerodynamic and manufacturability
dimensions.
"""

import numpy as np
import matplotlib
if matplotlib.get_backend() == "agg":
    pass  # already set by plots.py or non-interactive context
import matplotlib.pyplot as plt
from .style import COLORS, FONT_SIZES, save_fig
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    OUTER_WING_STATIONS,
    BODY_STATIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Pareto plot: L/D vs. Manufacturability
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto(catalog, obj_x: str = "manufacturability_score",
                obj_y: str = "L_over_D",
                save_path: str | None = None,
                ax=None):
    """Scatter plot of two objectives with Pareto front highlighted.

    Args:
        catalog: DesignCatalog instance
        obj_x: metric name for x-axis (from manufacturing_metrics or aero_metrics)
        obj_y: metric name for y-axis (from aero_metrics or manufacturing_metrics)
    """
    show = ax is None
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    xs, ys, names, colors = [], [], [], []
    color_map = {
        "default": "#2196F3",
        "optimization": "#F44336",
        "pareto": "#FF9800",
        "interpolation": "#4CAF50",
        "manual": "#9C27B0",
    }

    for entry in catalog:
        # Get x value
        x_val = entry.manufacturing_metrics.get(obj_x,
                    entry.aero_metrics.get(obj_x, None))
        y_val = entry.aero_metrics.get(obj_y,
                    entry.manufacturing_metrics.get(obj_y, None))

        if x_val is None or y_val is None:
            continue

        xs.append(float(x_val))
        ys.append(float(y_val))
        names.append(entry.name)
        colors.append(color_map.get(entry.origin, "#757575"))

    if not xs:
        ax.text(0.5, 0.5, "No data: metrics not available\nfor the selected axes",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Design Trade-off Space (no data)", fontsize=13)
        if save_path:
            save_fig(plt.gcf(), save_path)
        return plt.gcf()

    xs, ys = np.array(xs), np.array(ys)

    # Plot points
    for x, y, name, color in zip(xs, ys, names, colors):
        ax.scatter(x, y, c=color, s=100, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, fontweight="bold")

    # Draw Pareto front (maximize both x and y)
    if len(xs) == 1:
        ax.text(0.5, 0.02, "Single design — Pareto front requires ≥ 2 designs",
                ha='center', transform=ax.transAxes, fontsize=9, style='italic', alpha=0.6)
    if len(xs) > 1:
        order = np.argsort(xs)
        pareto_x, pareto_y = [xs[order[0]]], [ys[order[0]]]
        best_y = ys[order[0]]
        for idx in order[1:]:
            if ys[idx] >= best_y:
                pareto_x.append(xs[idx])
                pareto_y.append(ys[idx])
                best_y = ys[idx]
        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, "k--", alpha=0.4, linewidth=1.5,
                    label="Pareto front")

    # Legend for origins
    from matplotlib.lines import Line2D
    handles = []
    for origin, color in color_map.items():
        if any(e.origin == origin for e in catalog):
            handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=10,
                                  label=origin))
    ax.legend(handles=handles, loc="lower right", fontsize=FONT_SIZES["legend"])

    ax.set_xlabel(obj_x.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(obj_y.replace("_", " ").title(), fontsize=11)
    ax.set_title("Design Trade-off Space", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        save_fig(plt.gcf(), save_path)
    if show:
        plt.tight_layout()
        return fig
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# Radar chart: multi-criteria comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_radar(catalog, designs: list[str] | None = None,
               save_path: str | None = None):
    """Radar chart comparing multiple designs across key dimensions.

    Axes (normalized to [0, 1]):
    - L/D (aero performance)
    - Static margin (stability)
    - Manufacturability score
    - Internal volume
    - Endurance
    - Structural mass (inverted: lighter = better)
    """
    entries = [catalog[n] for n in (designs or catalog.names)]

    # Define radar axes with normalization ranges
    axes_def = [
        ("L/D", "L_over_D", "aero", 10, 25),
        ("Static Margin", "static_margin", "aero", 0, 0.25),
        ("Manufacturability", "manufacturability_score", "manuf", 0, 1),
        ("Volume [dm³]", "internal_volume_m3", "manuf", 0, 0.006),
        ("Endurance [min]", "endurance_min", "aero", 0, 15),
        ("Lightness", "struct_mass", "aero_inv", 0.4, 1.0),
    ]

    labels = [a[0] for a in axes_def]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles.append(angles[0])  # close the polygon

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(entries)))

    for entry, color in zip(entries, colors):
        values = []
        for label, key, source, lo, hi in axes_def:
            if source == "aero":
                raw = entry.aero_metrics.get(key, 0)
            elif source == "aero_inv":
                raw = entry.aero_metrics.get(key, hi)
                raw = hi - raw + lo  # invert: lower mass → higher score
            elif source == "manuf":
                raw = entry.manufacturing_metrics.get(key, 0)
            else:
                raw = 0

            # Normalize to [0, 1]
            norm = np.clip((float(raw) - lo) / max(hi - lo, 1e-9), 0, 1)
            values.append(norm)

        values.append(values[0])  # close polygon
        ax.plot(angles, values, 'o-', linewidth=2, label=entry.name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=FONT_SIZES["legend"])
    ax.set_title("Multi-Criteria Design Comparison", fontsize=13,
                 fontweight="bold", pad=20)

    if save_path:
        plt.tight_layout()
        save_fig(plt.gcf(), save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Planform overlay: superimpose multiple designs
# ═══════════════════════════════════════════════════════════════════════════

def plot_planform_overlay(catalog, designs: list[str] | None = None,
                           save_path: str | None = None):
    """Overlay planforms (top view + front view) of multiple designs."""
    entries = [catalog[n] for n in (designs or catalog.names)]

    fig, (ax_top, ax_front) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(entries), 10)))

    for entry, color in zip(entries, colors):
        p = entry.params
        _draw_planform_top(ax_top, p, color, entry.name)
        _draw_planform_front(ax_front, p, color, entry.name)

    ax_top.set_title("Top View (Planform)", fontsize=12, fontweight="bold")
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_aspect("equal")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_front.set_title("Front View (Dihedral)", fontsize=12, fontweight="bold")
    ax_front.set_xlabel("y [m]")
    ax_front.set_ylabel("z [m]")
    ax_front.set_aspect("equal")
    ax_front.grid(True, alpha=0.3)

    plt.suptitle("Planform Comparison", fontsize=14, fontweight="bold")

    if save_path:
        plt.tight_layout()
        save_fig(plt.gcf(), save_path)
    return fig


def _draw_planform_top(ax, p: BWBParams, color, label: str):
    """Draw top-view outline of a single design."""
    # Body LE/TE
    body_le_x = 0.0
    body_te_x = p.body_root_chord
    blend_y = p.body_halfwidth

    body_sweep_rad = np.radians(p.body_sweep_deg)
    blend_le_x = blend_y * np.tan(body_sweep_rad)
    blend_te_x = blend_le_x + p.wing_root_chord

    # Wing LE/TE (outer)
    wing_sweep_rad = np.radians(p.le_sweep_deg)
    outer_span = p.outer_half_span
    tip_y = p.half_span
    tip_le_x = blend_le_x + outer_span * np.tan(wing_sweep_rad)
    tip_te_x = tip_le_x + p.tip_chord

    # Draw right half LE and TE
    le_y = [0, blend_y, tip_y]
    le_x = [body_le_x, blend_le_x, tip_le_x]
    te_y = [0, blend_y, tip_y]
    te_x = [body_te_x, blend_te_x, tip_te_x]

    ax.plot(le_x, le_y, '-', color=color, linewidth=1.5, label=label)
    ax.plot(te_x, te_y, '-', color=color, linewidth=1.5)
    # Tip cap
    ax.plot([tip_le_x, tip_te_x], [tip_y, tip_y], '-', color=color, linewidth=1)
    # Mirror
    ax.plot(le_x, [-y for y in le_y], '-', color=color, linewidth=1.5, alpha=0.5)
    ax.plot(te_x, [-y for y in te_y], '-', color=color, linewidth=1.5, alpha=0.5)
    ax.plot([tip_le_x, tip_te_x], [-tip_y, -tip_y], '-', color=color, linewidth=1, alpha=0.5)


def _draw_planform_front(ax, p: BWBParams, color, label: str):
    """Draw front-view (dihedral profile) of a single design."""
    dihedrals = [p.dihedral_0, p.dihedral_1, p.dihedral_2,
                 p.dihedral_3, p.dihedral_tip]
    outer_span = p.outer_half_span

    # Start at body edge
    y_pts = [p.body_halfwidth]
    z_pts = [0.0]

    for i, dih in enumerate(dihedrals):
        dy = (OUTER_WING_STATIONS[i + 1] - OUTER_WING_STATIONS[i]) * outer_span
        dz = dy * np.tan(np.radians(dih))
        y_pts.append(y_pts[-1] + dy)
        z_pts.append(z_pts[-1] + dz)

    ax.plot(y_pts, z_pts, '-o', color=color, linewidth=1.5, markersize=3, label=label)
    # Mirror
    ax.plot([-y for y in y_pts], z_pts, '-o', color=color, linewidth=1.5,
            markersize=3, alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_table(catalog, designs: list[str] | None = None,
                        save_path: str | None = None):
    """Render a comparison table as a matplotlib figure."""
    entries = [catalog[n] for n in (designs or catalog.names)]

    columns = ["Origin", "L/D", "SM", "Manuf.", "Vol [cm³]", "Mass [g]",
               "Span [m]", "t/c root", "Feasible"]

    cell_text = []
    for e in entries:
        am = e.aero_metrics
        mm = e.manufacturing_metrics
        p = e.params
        row = [
            e.origin,
            f"{am.get('L_over_D', 0):.2f}",
            f"{am.get('static_margin', 0):.3f}",
            f"{mm.get('manufacturability_score', 0):.3f}",
            f"{mm.get('internal_volume_m3', 0) * 1e6:.0f}",
            f"{am.get('struct_mass', 0) * 1000:.0f}",
            f"{2 * p.half_span:.2f}",
            f"{p.body_tc_root:.3f}",
            "OK" if am.get("is_feasible", False) else "X",
        ]
        cell_text.append(row)

    row_labels = [e.name for e in entries]

    fig, ax = plt.subplots(figsize=(14, 0.5 + 0.4 * len(entries)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Color header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")

    # Color feasibility column
    feas_col = columns.index("Feasible")
    for i in range(len(entries)):
        if cell_text[i][feas_col] == "OK":
            table[i + 1, feas_col].set_facecolor("#C8E6C9")
        else:
            table[i + 1, feas_col].set_facecolor("#FFCDD2")

    ax.set_title("Design Catalog Summary", fontsize=13, fontweight="bold", pad=20)

    if save_path:
        save_fig(plt.gcf(), save_path)
    return fig
