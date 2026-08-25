"""Plotly 3D factor trajectory: level x slope x curvature, walked over time.

The two "Evo: Resids" factor sub-tabs chart level/slope/curvature as three
separate series against time. That answers "what did each factor do?" but not
"where in factor space is the curve, and how is it moving?" -- the three are
strongly co-dependent, so their joint position is the thing with economic
meaning (a bull steepener and a bear steepener are different points, not
different slope readings). Plotting them as one path through 3-space makes
that joint motion, and any regime loop or reversal in it, directly visible.

Styling deliberately matches the MLN 3D view (``multiplex_plotly``): same dark
canvas, same font, same camera idiom, so the two interactive tabs read as one
family. Colour constants are imported from there rather than restated, so the
two cannot drift apart.

Two deliberate departures from ``multiplex_plotly._apply_dark_layout``:

1. **The axes are drawn and labelled.** In the multiplex view x/y/z are layout
   coordinates with no units, so hiding them removes noise. Here they *are* the
   data, and an unlabelled factor cube is unreadable.
2. **Time is encoded as a single-hue sequential ramp**, not the multiplex's
   sky->slate->amber diverging-ish edge scale. Magnitude (here, elapsed time)
   takes one hue light-to-dark; a multi-hue ramp would read as unordered
   categories. The ramp runs dark-to-light because the canvas is dark -- the
   light end has to be the one that stands off the surface.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from .multiplex_plotly import CANVAS_BG, TEXT_COLOR, TEXT_MUTED

# Sequential single-hue (sky) ramp for elapsed time: earliest window darkest,
# latest lightest. Every step is the same hue family as the app's own sky
# accent, and lightness increases monotonically so ordering survives greyscale
# and colour-vision deficiency alike.
TIME_COLORSCALE = [
    [0.00, "#075985"],  # sky-800
    [0.33, "#0284c7"],  # sky-600
    [0.66, "#38bdf8"],  # sky-400
    [1.00, "#bae6fd"],  # sky-200
]

# The scrubbed point. Amber is the one hue in this app's palette maximally
# separated from the sky ramp, and it is reinforced by size and a surface ring
# rather than carrying the distinction on colour alone.
HIGHLIGHT_COLOR = "#fbbf24"

GRID_COLOR = "#334155"

# Name of the single-point trace the slider moves. The injected JS looks the
# trace up by name rather than by index, so adding traces above it cannot
# silently break the scrubber.
HIGHLIGHT_TRACE_NAME = "factor-highlight"

_AXES = ("level", "slope", "curvature")


def factor_columns(std: bool) -> tuple[str, str, str]:
    """The three ``factors`` columns a trajectory reads, mean or std."""
    suffix = "_std" if std else "_mean"
    return tuple(f"{axis}{suffix}" for axis in _AXES)  # type: ignore[return-value]


def trajectory_points(
    factors: pl.DataFrame, *, std: bool = False
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Extract ``(xs, ys, zs, date_labels)`` in window order.

    Returns empty lists when the frame lacks the columns, so a caller can show
    a placeholder rather than raising -- a partially-failed evolution pass can
    legitimately leave the factor half empty.
    """
    columns = factor_columns(std)
    if factors.is_empty() or any(c not in factors.columns for c in columns):
        return [], [], [], []
    ordered = factors.sort("window_idx") if "window_idx" in factors.columns else factors
    xs, ys, zs = ([float(v) for v in ordered.get_column(c).to_list()] for c in columns)
    if "date_end" in ordered.columns:
        labels = [str(d) for d in ordered.get_column("date_end").to_list()]
    else:
        labels = [str(i) for i in range(len(xs))]
    return list(xs), list(ys), list(zs), labels


def build_factor_trajectory_figure(
    factors: pl.DataFrame,
    regimes: pl.DataFrame | None = None,
    *,
    std: bool = False,
    title: str | None = None,
) -> go.Figure:
    """3D path through level/slope/curvature space, in window order.

    Args:
        factors: The evolution ``factors`` frame (``window_idx``, ``date_end``,
            and the six ``*_mean``/``*_std`` columns).
        regimes: Optional ``window_idx``/``regime`` frame; the regime label is
            folded into each point's hover text when present.
        std: Plot the within-window volatilities instead of the means.
        title: Overrides the default title.

    Returns:
        A figure whose last trace is the single-point highlight the scrubber
        moves (see :data:`HIGHLIGHT_TRACE_NAME`).
    """
    xs, ys, zs, dates = trajectory_points(factors, std=std)
    kind = "Factor volatility" if std else "Factor"
    heading = title or f"{kind} trajectory — level × slope × curvature over time"

    fig = go.Figure()
    if not xs:
        _apply_layout(fig, heading, std=std, empty=True)
        return fig

    regime_of = regime_labels(factors, regimes, len(xs))
    steps = np.arange(len(xs), dtype=float)

    # Hover carries the identity (date, regime) that the axes cannot.
    custom = [[dates[i], regime_of[i]] for i in range(len(xs))]
    axis_names = _axis_titles(std)
    hover = (
        "<b>%{customdata[0]}</b><br>"
        f"{axis_names[0]}=%{{x:.4f}}<br>"
        f"{axis_names[1]}=%{{y:.4f}}<br>"
        f"{axis_names[2]}=%{{z:.4f}}<br>"
        "regime=%{customdata[1]}<extra></extra>"
    )

    # The connecting path. Coloured by the same time ramp as the markers so the
    # direction of travel is legible even where markers overlap.
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(
                color=steps,
                colorscale=TIME_COLORSCALE,
                width=4,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="factor-path",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                size=5,
                color=steps,
                colorscale=TIME_COLORSCALE,
                cmin=0.0,
                cmax=float(max(len(xs) - 1, 1)),
                opacity=0.95,
                line=dict(color=CANVAS_BG, width=1),
                colorbar=dict(
                    title=dict(
                        text="Window",
                        font=dict(family="Arial", size=11, color=TEXT_MUTED),
                    ),
                    thickness=14,
                    len=0.5,
                    x=1.01,
                    tickfont=dict(family="Arial", size=10, color=TEXT_MUTED),
                    outlinecolor=GRID_COLOR,
                ),
            ),
            customdata=custom,
            hovertemplate=hover,
            showlegend=False,
            name="factor-points",
        )
    )

    # Endpoints only -- labelling every window would bury the path it annotates.
    fig.add_trace(
        go.Scatter3d(
            x=[xs[0], xs[-1]],
            y=[ys[0], ys[-1]],
            z=[zs[0], zs[-1]],
            mode="text",
            text=[f"start {dates[0]}", f"end {dates[-1]}"],
            textposition="top center",
            textfont=dict(family="Arial", size=10, color=TEXT_MUTED),
            hoverinfo="skip",
            showlegend=False,
            name="factor-endpoints",
        )
    )

    # The scrubbed window. Must be last: the injected JS restyles it by name,
    # and drawing it last keeps it on top of the path it sits on.
    fig.add_trace(
        go.Scatter3d(
            x=[xs[0]],
            y=[ys[0]],
            z=[zs[0]],
            mode="markers",
            marker=dict(
                size=13,
                color=HIGHLIGHT_COLOR,
                opacity=1.0,
                line=dict(color=CANVAS_BG, width=2),
            ),
            customdata=[custom[0]],
            hovertemplate=hover,
            showlegend=False,
            name=HIGHLIGHT_TRACE_NAME,
        )
    )

    _apply_layout(fig, heading, std=std, empty=False)
    return fig


NO_REGIME = "—"


def regime_labels(
    factors: pl.DataFrame, regimes: pl.DataFrame | None, count: int
) -> list[str]:
    """Per-point regime label in window order, padded to ``count``.

    Regimes are decorative -- ``compute_curve_factors`` returns an empty frame
    when the mixture model cannot be fitted -- so every failure path here
    yields :data:`NO_REGIME` rather than raising.
    """
    blank = [NO_REGIME] * count
    if regimes is None or regimes.is_empty():
        return blank
    if "regime" not in regimes.columns or "window_idx" not in regimes.columns:
        return blank
    if "window_idx" not in factors.columns:
        return blank
    by_window = {
        int(w): str(r)
        for w, r in zip(
            regimes.get_column("window_idx").to_list(),
            regimes.get_column("regime").to_list(),
        )
    }
    ordered = factors.sort("window_idx").get_column("window_idx").to_list()
    labels = [by_window.get(int(w), NO_REGIME) for w in ordered[:count]]
    return labels + blank[len(labels) :]


def _axis_titles(std: bool) -> tuple[str, str, str]:
    suffix = " σ" if std else ""
    return (f"Level{suffix}", f"Slope{suffix}", f"Curvature{suffix}")


def _apply_layout(fig: go.Figure, title: str, *, std: bool, empty: bool) -> None:
    """Dark canvas matching the MLN view, but with the factor axes drawn."""
    x_title, y_title, z_title = _axis_titles(std)
    axis = dict(
        showbackground=False,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        color=TEXT_MUTED,
        title=dict(font=dict(family="Arial", size=11, color=TEXT_MUTED)),
        tickfont=dict(family="Arial", size=9, color=TEXT_MUTED),
    )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=15, color=TEXT_COLOR),
            x=0.02,
            xanchor="left",
        ),
        font=dict(family="Arial", color=TEXT_COLOR),
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=CANVAS_BG,
        plot_bgcolor=CANVAS_BG,
        hoverlabel=dict(bgcolor="#1e293b", font=dict(color=TEXT_COLOR)),
        showlegend=False,
        scene=dict(
            bgcolor=CANVAS_BG,
            xaxis={**axis, "title": {**axis["title"], "text": x_title}},
            yaxis={**axis, "title": {**axis["title"], "text": y_title}},
            zaxis={**axis, "title": {**axis["title"], "text": z_title}},
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
    )
    if empty:
        fig.add_annotation(
            text="No factor windows to plot.",
            showarrow=False,
            font=dict(family="Arial", size=13, color=TEXT_MUTED),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
