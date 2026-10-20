"""Shared look of the paper's result figures: fonts, sizes and the colors of the five classes.

Every plotting script in this folder imports this module first, so all figures
share one font (the paper's Times), one set of text sizes and one color per
classifier class.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: the paper is set in Times through LaTeX's `times`
# package. Tinos is a TrueType Times clone, so the figures' text reads as part of
# the page. It is listed before Nimbus Roman, the body font itself, because
# matplotlib embeds Nimbus Roman's OpenType outlines in a form that some PDF
# viewers reject, while TrueType fonts embed cleanly.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Tinos", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "text.color": "#0b0b0b",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # NOTE: fonttype 42 embeds TrueType fonts in the PDF instead
    # of Type 3 bitmaps; ACL's checker rejects Type 3 fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Widths of the ACL two-column layout, in inches (7.7 cm column, 16 cm page).
COLUMN_WIDTH = 3.03
PAGE_WIDTH = 6.30

# One color per classifier class, used in every figure, including the diagrams.
# NOTE: the four hues come from a palette checked for
# colorblind readers in this stacking order; model scrambles take a neutral
# gray because they are machine failures, not a kind of speech.
CLASS_COLORS = {
    "control": "#2a78d6",
    "conduction/anomic": "#1baf7a",
    "broca": "#eb6834",
    "wernicke": "#6250d6",
    "model_scrambles": "#898781",
}
CLASS_NAMES = {
    "control": "Control",
    "conduction/anomic": "Conduction/Anomic",
    "broca": "Broca",
    "wernicke": "Wernicke",
    "model_scrambles": "Model scrambles",
}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Sequential blue ramp for magnitudes (light = low, dark = high).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def clean_axes(ax, grid_axis="y"):
    """Hide the top and right spines and draw a recessive grid behind the marks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
