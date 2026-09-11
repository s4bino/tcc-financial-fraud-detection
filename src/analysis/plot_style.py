"""
Shared visual style for every figure in this work.

Centralises palette, typography and matplotlib parameters so that all figures
read as one system. The categorical palette is colourblind-safe; the sequential
and diverging scales derive from the same hues.

Rules applied throughout:
  - thin marks, hairline grid, never dashed;
  - colour follows the entity, never its rank;
  - sequential scales use a single hue, light to dark;
  - diverging scales use two opposite poles with a neutral grey midpoint;
  - text is always ink, never the series colour.

Figure labels are written in Portuguese because the figures are produced for
the monograph, which is written in Portuguese.
"""

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Surfaces and ink
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ---------------------------------------------------------------------------
# Categorical palette, in fixed order
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Class identity, fixed across every figure.
COLOR_LEGIT = BLUE
COLOR_FRAUD = ORANGE
LABEL_LEGIT = "Transação legítima"
LABEL_FRAUD = "Fraude"

# ---------------------------------------------------------------------------
# Continuous scales
# ---------------------------------------------------------------------------
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)

# Diverging: blue and red as poles, neutral grey at the midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a6a5", "#e34948", "#8f2020"],
)

FONT_STACK = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def apply_style():
    """Apply the global matplotlib parameters."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,

        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,

        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.titlecolor": INK,

        # Recessive solid grid; top and right spines removed.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def save(fig, path_without_extension, formats=("png", "pdf")):
    """Write the figure as PNG (quick viewing) and PDF (vector, for LaTeX)."""
    import os

    os.makedirs(os.path.dirname(path_without_extension), exist_ok=True)
    written = []
    for ext in formats:
        target = f"{path_without_extension}.{ext}"
        fig.savefig(target)
        written.append(target)
    return written


def footnote(fig, text):
    """Discreet footnote, used to record data provenance or sampling."""
    fig.text(0.0, -0.02, text, ha="left", va="top",
             fontsize=7.5, color=INK_MUTED)


def thousands(value):
    """Format an integer with the Brazilian thousands separator."""
    return f"{value:,}".replace(",", ".")
