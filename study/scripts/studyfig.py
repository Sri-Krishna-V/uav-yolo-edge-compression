"""The paper's figure style (#49): print widths, fonts, the one encoding every figure shares, and a deterministic save.
The rules are .scratch/uav-yolo-edge/findings/49-research-figures.md, sections 3, 7 and 8.

Widths. 88 mm (one column) and 174 mm (two): IEEE's column, and Springer Applied Intelligence's text width, both inside MDPI's.
A figure is drawn at the size it prints at and included in LaTeX with no width=, so its 8 pt text prints at 8 pt (the floor
of Springer's 8-12 pt and IEEE's 8-10 pt).
Fonts. Arial, embedded as TrueType (pdf.fonttype 42; matplotlib's default is Type 3). No silent fallback: a machine without
Arial stops with an error instead of writing different PDFs.
Encoding. Colour is the model, the marker's shape the configuration, a hollow marker an input resolution the checkpoint was not
trained at (only that), a dagger a throttled cell, a section sign a pair spanning session 6. Colour always has a shape or a
line style beside it, so every figure reads in greyscale; there is no alpha anywhere (EPS cannot draw it).
Saving. A PDF with no CreationDate (so a re-run writes the same bytes) and a 600 dpi PNG, at the figure's size: never
bbox_inches='tight', which changes the width.
Standard library and matplotlib only; no torch, no ultralytics.
"""

from contextlib import contextmanager
from pathlib import Path

MM = 1 / 25.4
SINGLE = 88 * MM
DOUBLE = 174 * MM
MAX_HEIGHT = 200 * MM
PNG_DPI = 600
FONT = "Arial"
SIZE = 8  # pt: the smallest text in any figure

MODEL_COLOR = {"YOLO11s": "#0072B2", "YOLO26s": "#E69F00"}  # Wong (2011) blue and orange, L* 46 and 71: apart in greyscale too
CONFIG_MARKER = {"baseline": "o", "quantized": "s", "pruned": "^", "distilled": "D", "combined": "P"}
CONFIG_NAME = {"baseline": "Baseline", "quantized": "Quantized (INT8)", "pruned": "Pruned", "distilled": "Distilled",
               "combined": "Combined (INT8)"}
RUNTIME_NAME = {"onnxruntime": "ONNX Runtime", "openvino": "OpenVINO", "ncnn": "NCNN"}
RUNTIME_STYLE = {  # Results C and D, where the model is not a series: colour and line style together
    ("onnxruntime", "FP32"): {"color": "#000000", "linestyle": "-"},
    ("openvino", "FP32"): {"color": "#009E73", "linestyle": "--"},
    ("ncnn", "FP16"): {"color": "#CC79A7", "linestyle": "-."},
    ("onnxruntime", "INT8"): {"color": "#D55E00", "linestyle": ":"},
}
HEAD_STYLE = {"one_to_many": {"color": "#000000", "marker": "o"}, "nms_free": {"color": "#CC79A7", "marker": "s"}}
HEAD_NAME = {"one_to_many": "one-to-many head", "nms_free": "NMS-free head"}
INK, MUTED, RULE, TINT = "#222222", "#555555", "#BBBBBB", "#E0E0E0"
DAGGER, SECTION, MINUS = "†", "§", "−"

RC = {
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": SIZE,
    "axes.titlesize": SIZE, "axes.labelsize": SIZE, "xtick.labelsize": SIZE, "ytick.labelsize": SIZE, "legend.fontsize": SIZE,
    "axes.titlelocation": "left", "axes.titlepad": 4,
    "legend.frameon": False, "legend.handlelength": 2.0, "legend.borderaxespad": 0.2, "legend.columnspacing": 1.2,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.5, "grid.color": "#D4D4D4", "axes.axisbelow": True,
    "lines.linewidth": 1.25, "lines.markersize": 4.5, "lines.markeredgewidth": 0.75,
    "axes.unicode_minus": True, "axes.formatter.use_mathtext": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "standard",
    "figure.constrained_layout.h_pad": 2 / 72, "figure.constrained_layout.w_pad": 2 / 72,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


SLIDE = 160 * MM  # the slides' figures (studystory.py): shown about 1,500 px wide on a 1,920 px slide
SLIDE_SIZE = 11  # pt: at that scale about 36 px, above the deck's 24 px floor


@contextmanager
def style(size=SIZE):
    """The paper's rcParams for the figures drawn inside it, on the Agg backend; raises if Arial is not installed. size sets
    every text size (SLIDE_SIZE for the slides' figures)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    font_manager.findfont(FONT, fallback_to_default=False)
    rc = {**RC, **{k: size for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
                                      "legend.fontsize")}}
    with matplotlib.rc_context(rc):
        yield


def figure(width, height_mm, **kw):
    """A figure at its printed size (SINGLE, DOUBLE or SLIDE wide, height in mm), laid out by constrained layout."""
    import matplotlib.pyplot as plt
    if width not in (SINGLE, DOUBLE, SLIDE):
        raise ValueError("a figure is SINGLE, DOUBLE or SLIDE wide")
    if height_mm * MM > MAX_HEIGHT:
        raise ValueError(f"{height_mm} mm is taller than {MAX_HEIGHT / MM:.0f} mm")
    return plt.figure(figsize=(width, height_mm * MM), layout="constrained", **kw)


def panel_label(ax, letter, title=""):
    """'(a)' in bold at the panel's top left, then an optional short title (the caption describes each panel)."""
    ax.annotate(f"({letter})", (0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", fontweight="bold",
                ha="left", va="bottom")
    if title:
        ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(14, 5), textcoords="offset points", ha="left", va="bottom")


def halo(width=2.5):
    """A white outline behind a label, so it stays readable where it crosses a line or a gridline."""
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground="white")]


def text_entry(label):
    """A legend entry that is only text (a mark like the dagger explained in words): an invisible handle."""
    from matplotlib.lines import Line2D
    return Line2D([], [], linestyle="none", marker="none", label=label)


def tint(ax):
    """The provisional panel's background (the phone's latency): a light grey, solid, so EPS draws it."""
    ax.set_facecolor(TINT)


def signed(v, nd=1, unit=""):
    """'+1.2', '−1.2' (a real minus sign, as MDPI asks)."""
    s = f"{v:+.{nd}f}{unit}"
    return s.replace("-", MINUS)


VECTOR = False  # the PDFs are off for now (the user, 1 Nov): PNGs only; studyfigures.py --pdf turns them on


def save(fig, stem, eps=False):
    """stem.png at 600 dpi at the figure's own size; with VECTOR also stem.pdf (TrueType fonts, no CreationDate, so a re-run
    writes the same bytes); stem.eps on request."""
    import matplotlib.pyplot as plt
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    if VECTOR:
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    if eps:
        fig.savefig(stem.with_suffix(".eps"))
    plt.close(fig)
