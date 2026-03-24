"""Unified matplotlib styling for MANTA visualizations.

Provides a consistent, professional look across all notebooks and
plotting functions: color palette, font sizes, spine management,
grid styling, and standardized figure saving.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


# ═══════════════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    # Aircraft zones
    "body": "#FF8C42",           # Warm orange
    "wing": "#2E86DE",           # Deep blue
    "elevon": "#E67E22",         # Dark orange
    "aileron": "#3498DB",        # Light blue

    # Status
    "feasible": "#27AE60",       # Green
    "infeasible": "#E74C3C",     # Red
    "warning": "#F39C12",        # Amber

    # Catalog origins
    "baseline": "#2196F3",       # Material blue
    "optimized": "#F44336",      # Material red
    "pareto": "#FF9800",         # Material orange
    "interpolation": "#4CAF50",  # Material green

    # General purpose
    "primary": "#2C3E50",        # Dark slate
    "secondary": "#7F8C8D",      # Gray
    "accent": "#8E44AD",         # Purple
    "light": "#BDC3C7",          # Light gray

    # Propulsion
    "thrust": "#27AE60",         # Green
    "drag": "#E74C3C",           # Red
    "duct": "#9B59B6",           # Purple

    # Components
    "battery": "#F1C40F",        # Yellow
    "motor": "#E67E22",          # Orange
    "avionics": "#3498DB",       # Blue
    "structure": "#95A5A6",      # Gray
}

# Sequential palette for multi-line plots
PALETTE = ["#2E86DE", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD", "#1ABC9C"]


# ═══════════════════════════════════════════════════════════════════════════
# Font sizes
# ═══════════════════════════════════════════════════════════════════════════

FONT_SIZES = {
    "title": 14,
    "subtitle": 12,
    "axis_label": 12,
    "tick_label": 11,
    "legend": 10,
    "annotation": 9,
}


# ═══════════════════════════════════════════════════════════════════════════
# Standard figure sizes
# ═══════════════════════════════════════════════════════════════════════════

FIGSIZE = {
    "single": (10, 6),
    "wide": (14, 5),
    "double": (14, 6),
    "triple": (16, 5),
    "square": (8, 8),
    "tall": (10, 8),
}

DPI_SCREEN = 120
DPI_SAVE = 150


# ═══════════════════════════════════════════════════════════════════════════
# Style application
# ═══════════════════════════════════════════════════════════════════════════

def apply_style():
    """Apply MANTA unified style to all matplotlib figures.

    Call once at the top of each notebook or script.
    """
    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": DPI_SCREEN,
        "figure.titlesize": FONT_SIZES["title"],
        "figure.titleweight": "bold",

        # Axes
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.titlesize": FONT_SIZES["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": FONT_SIZES["axis_label"],
        "axes.spines.right": False,
        "axes.spines.top": False,

        # Grid
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Font
        "font.family": "sans-serif",
        "font.size": FONT_SIZES["tick_label"],

        # Ticks
        "xtick.labelsize": FONT_SIZES["tick_label"],
        "ytick.labelsize": FONT_SIZES["tick_label"],

        # Legend
        "legend.fontsize": FONT_SIZES["legend"],
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": True,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 6,

        # Savefig
        "savefig.dpi": DPI_SAVE,
        "savefig.bbox": "tight",
    })


def clean_spines(ax, keep=None):
    """Remove unnecessary spines for a cleaner look.

    Parameters
    ----------
    ax : matplotlib Axes
    keep : list of str, optional
        Spines to keep. Default: ['left', 'bottom'].
    """
    if keep is None:
        keep = ["left", "bottom"]
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(spine in keep)


def save_fig(fig, path, dpi=DPI_SAVE):
    """Standardized figure saving with tight bounding box.

    Does NOT close the figure — in Jupyter inline mode, closing
    before cell completion produces '<Figure with 0 Axes>'.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
