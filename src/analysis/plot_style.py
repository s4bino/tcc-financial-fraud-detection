"""
Shared visual style for every figure in this work.

Centralises palette, typography and matplotlib parameters so that all figures
read as one system, in the clean style of the figures of Campello et al. (2015)
and Neto et al. (2021): serif type, framed axes, thin lines and a discreet grid.

The palette is cold and bluish. Navy and cyan carry the identity of the series
(the bases, the covariance estimates), slate grey carries neutral references,
and a cold violet is reserved for anomalies (frauds and outliers), so that the
anomalous points stand out in every figure with the same colour. The pairs that
share a figure were checked for colour-vision deficiency (protanopia and
deuteranopia, OKLab distance of at least 12) and for contrast against white.

Rules applied throughout:
  - thin marks, hairline grid, never dashed;
  - colour follows the entity, never its rank;
  - the violet accent marks anomalies only, never an ordinary series;
  - sequential scales use a single hue, light to dark;
  - diverging scales use two opposite poles with a neutral grey midpoint;
  - text is always ink, never the series colour.

Figure labels are written in Portuguese because the figures are produced for
the monograph, which is written in Portuguese.

Figures that go into the monograph follow the UFLA layout used by Alcino
(2022): white background, no title or note inside the image (the caption and
the "Fonte" line are written in LaTeX), and a figure size equal to the printed
size, so that fonts keep their nominal size on the page. The type is Times, the
font of the monograph (pslatex and mathptmx in uflamon.cls), with STIX for the
mathematics.
"""

from matplotlib.colors import LinearSegmentedColormap

# Text width of the monograph (uflamon: A4, 3 cm left and 2 cm right margins).
TEXT_WIDTH_IN = 16 / 2.54

# ---------------------------------------------------------------------------
# Surfaces and ink, cold greys
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#1a222e"
INK_SECONDARY = "#424e5e"
INK_MUTED = "#6f7c8c"
GRID = "#e4e8ed"
AXIS = "#424e5e"

# ---------------------------------------------------------------------------
# Cold palette
# ---------------------------------------------------------------------------
NAVY = "#265496"  # azul-marinho, first series
CYAN = "#139cb6"  # ciano, second series
PETROL = "#176c5f"  # azul-petróleo, negative pole of the diverging scale
SLATE = "#7a8798"  # cinza-ardósia, neutral references
SLATE_LIGHT = "#a9b2be"  # hollow points and background observations
NAVY_LIGHT = "#a5c0e5"  # lighter companion of navy, for paired series
CYAN_LIGHT = "#a9d7e2"  # lighter companion of cyan, for paired series
ACCENT = "#7e55f0"  # violeta frio, anomalies only (frauds, outliers)

# Series in fixed order: two hues, each with a lighter companion, so that four
# series pair up by hue (navy and its companion, cyan and its companion).
CATEGORICAL = [NAVY, NAVY_LIGHT, CYAN, CYAN_LIGHT]

# Class identity, fixed across every figure.
COLOR_LEGIT = NAVY
COLOR_FRAUD = ACCENT
LABEL_LEGIT = "Transação legítima"
LABEL_FRAUD = "Fraude"

# ---------------------------------------------------------------------------
# Continuous scales
# ---------------------------------------------------------------------------
# Single hue (OKLCH h about 258), lightness from 0.97 to 0.28.
_BLUE_RAMP = [
    "#eff6fd",
    "#d1e3f9",
    "#adcaee",
    "#85ade3",
    "#6091d4",
    "#4174bf",
    "#295ca5",
    "#204887",
    "#1a3666",
    "#15284b",
]

SEQUENTIAL = LinearSegmentedColormap.from_list("seq_navy", _BLUE_RAMP)

# Diverging: petrol and navy as poles, with matching lightness on both arms and
# a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_petrol_navy",
    [
        "#053c34",
        "#176c5f",
        "#499d8e",
        "#9dccc5",
        "#f1f4f6",
        "#b4c5e2",
        "#7292d0",
        "#395ca9",
        "#1b3168",
    ],
)

# Nested density regions (densest 50%, 90% and 99% of the points), dark to
# light, steps of the sequential ramp.
DENSITY_LEVELS = (0.50, 0.90, 0.99)
DENSITY_COLORS = ("#295ca5", "#6a94cf", "#adcaee")

FONT_STACK = ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"]


def apply_style():
    """Apply the global matplotlib parameters."""
    import matplotlib as mpl
    from cycler import cycler

    if SEQUENTIAL.name not in mpl.colormaps:
        mpl.colormaps.register(SEQUENTIAL)
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # 300 dpi for the PNG and for the rasterised layers inside the PDF.
            "savefig.dpi": 300,
            # TrueType fonts embedded in the PDF, so that the text stays text.
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": FONT_STACK,
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            # Framed axes with outward ticks, as in the reference articles.
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            # Recessive solid grid.
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "image.cmap": SEQUENTIAL.name,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "patch.linewidth": 0.6,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.borderaxespad": 0.3,
            "figure.constrained_layout.use": True,
        }
    )


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
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8, color=INK_MUTED)


def thousands(value):
    """Format an integer with the Brazilian thousands separator."""
    return f"{value:,}".replace(",", ".")


def decimal_comma(*axes):
    """Write the tick labels of numeric axes (ax.xaxis, ax.yaxis) with the decimal comma."""
    from matplotlib.ticker import ScalarFormatter

    class CommaFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    for axis in axes:
        axis.set_major_formatter(CommaFormatter())
