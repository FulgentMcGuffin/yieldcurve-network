"""Shared dark-theme palette and axis styling for every rendered figure.

Figures across this package are built directly via the ``Figure`` API rather
than ``pyplot`` so rendering can safely run on a background QThread -- pyplot's
state machine is tied to the active GUI backend and warns or fails when touched
off the main thread. Keep that convention in anything added here.

Colours track ``gui/styles.py`` so plots sit in the window without a seam.
"""

from __future__ import annotations

from matplotlib.figure import Figure

BG = "#0f172a"
PANEL = "#1e293b"
FG = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#334155"

# Semantic accents, matching the notebook's palette.
INTRA_COLOR = "#0ea5e9"
INTER_COLOR = "#a78bfa"
ACCENT = "#0ea5e9"
ACCENT_ALT = "#a78bfa"
POSITIVE = "#34d399"
WARN = "#f59e0b"
DANGER = "#f87171"
STRESS = "#ec4899"

# Qualitative palette shared by categorical series (methods, factors, regimes).
CATEGORICAL_10 = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]


def empty_figure(
    message: str, width: float = 12, height: float = 6, dpi: int = 100
) -> Figure:
    """A themed placeholder figure carrying an explanatory message."""
    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=13,
        color=MUTED,
        wrap=True,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def style_axes(ax, *, labelsize: int = 8, grid: bool = False) -> None:
    """Apply the dark theme to one axes."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=labelsize)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    if grid:
        ax.grid(True, color=GRID, alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)


def style_legend(legend) -> None:
    """Apply the dark theme to a legend produced by ``ax.legend()``."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(PANEL)
    frame.set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(FG)
    title = legend.get_title()
    if title is not None:
        title.set_color(MUTED)


def style_date_axis(ax, n_ticks: int = 10) -> None:
    """Thin and slant date tick labels so long spans stay readable."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=n_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
