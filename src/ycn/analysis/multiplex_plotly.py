"""Plotly 3D multiplex figure from Polars node/edge tables.

Vendored from https://github.com/FulgentMcGuffin/MultiLayerNetViz
(``multiplex_plotly.py``) with three deliberate changes:

1. **Colour constants are defined locally** instead of imported from
   ``multi_layer_NetViz_fcts``. The upstream import is unusable here: the
   installed ``multilayer-netviz`` distribution ships an older revision of that
   module which exports none of ``PLANE_COLORS``/``EDGE_CMAP``/
   ``_vivid_node_color``, and importing it calls ``plt.rcParams.update()`` at
   module scope -- switching ``font.family`` to serif for the whole process and
   silently restyling every Evolution figure. The palette below is this
   project's own dark-theme palette (the same Tailwind hues used in
   ``gui/styles.py``), chosen independently rather than copied, so no
   third-party licence attaches to this file.
2. **Dark theme**, to match the GUI canvas (``#0f172a``). Upstream hardcodes
   ``paper_bgcolor="white"`` and leaves ``scene.bgcolor`` unset.
3. **Synthetic identity links are opt-in** (``add_identity_links``, default
   off). Upstream always draws links between the same node in *consecutive
   displayed* layers. ``mln.build_multilayer_network`` already emits explicit
   inter-layer edges for every pair of layers a node appears in, so leaving
   this on would double-draw them.

``customdata`` and ``hovertemplate`` are preserved verbatim -- the GUI's
click bridge (``gui/mln_bridge.py``) depends on their exact shape:

- node markers:       ``["node", node, layer]``
- intra-edge picks:   ``["intra", source, target, layer, weight]``
- inter-edge picks:   ``["inter", source, target, layer_from, layer_to, weight]``
"""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, to_hex

# --- Dark-theme palette (this project's own; see module docstring) -----------
# Layer planes: sky / violet / emerald / amber, cycled.
PLANE_COLORS = ("#38bdf8", "#a78bfa", "#34d399", "#fbbf24")

# Intra-layer connection strength: sky (weak) -> slate (mid) -> amber (strong).
# A white midpoint (the usual RdBu choice) disappears on a dark canvas.
EDGE_CMAP = LinearSegmentedColormap.from_list(
    "ycn_edge_dark", ["#38bdf8", "#64748b", "#fbbf24"]
)
EDGE_COLORSCALE = [[0.0, "#38bdf8"], [0.5, "#64748b"], [1.0, "#fbbf24"]]

CANVAS_BG = "#0f172a"
TEXT_COLOR = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
INTER_EDGE_COLOR = "#94a3b8"

N_EDGE_BINS = 8
PLANE_PAD = 0.25
LAYER_GAP = 5.4


def _node_color(index: int) -> str:
    """Distinct, saturated hue per node index -- readable on a dark canvas."""
    hue = (index * 0.6180339887) % 1.0  # golden-ratio hop spreads adjacent hues
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.98)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _issuer_xy(issuers: list[str]) -> dict[str, tuple[float, float]]:
    n = max(len(issuers), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return {s: (float(np.cos(a)), float(np.sin(a))) for s, a in zip(issuers, angles)}


def _weight_limits(intra: pl.DataFrame) -> tuple[float, float]:
    if intra.is_empty() or "weight" not in intra.columns:
        return 0.0, 1.0
    weights = intra.get_column("weight").to_numpy()
    wmin, wmax = np.percentile(weights, [10, 90])
    if wmax <= wmin:
        wmin, wmax = float(weights.min()), float(weights.max())
    return float(wmin), float(wmax)


def _edge_hex(weight: float, wmin: float, wmax: float) -> str:
    span = wmax - wmin if wmax > wmin else 1.0
    t = float(np.clip((weight - wmin) / span, 0.0, 1.0))
    return to_hex(EDGE_CMAP(t))


def _bin_index(weight: float, wmin: float, wmax: float) -> int:
    span = wmax - wmin if wmax > wmin else 1.0
    t = float(np.clip((weight - wmin) / span, 0.0, 1.0))
    return min(int(t * N_EDGE_BINS), N_EDGE_BINS - 1)


def build_multiplex_figure(
    nodes: pl.DataFrame,
    intra: pl.DataFrame,
    inter: pl.DataFrame,
    visible_terms: list[str],
    *,
    all_issuers: list[str] | None = None,
    layer_gap: float = LAYER_GAP,
    title: str = "Multi-layer network",
    layer_label: str = "layer",
    add_identity_links: bool = False,
) -> go.Figure:
    """Build the 3D multiplex figure.

    Args:
        nodes/intra/inter: Frames from :func:`multiplex_data.multiplex_tables`.
        visible_terms: Layer values to draw, in stacking order.
        all_issuers: Fixes the circular node layout so it does not jitter when
            layers are toggled. Defaults to the nodes present.
        layer_gap: Vertical spacing between layer planes.
        title: Figure title.
        layer_label: Display name of the layer column (for hover text).
        add_identity_links: Draw synthetic links between the same node in
            consecutive displayed layers. Off by default -- see module
            docstring.
    """
    terms = [t for t in visible_terms if t]
    if all_issuers is None:
        all_issuers = (
            sorted(nodes.get_column("issuer").unique().to_list())
            if not nodes.is_empty()
            else []
        )
    xy_of = _issuer_xy(all_issuers)
    z_of = {term: float(i) * layer_gap for i, term in enumerate(terms)}
    wmin, wmax = _weight_limits(intra)

    fig = go.Figure()
    if not terms:
        _apply_dark_layout(fig, title, layer_gap, 1.0)
        return fig

    xs = [xy_of[s][0] for s in all_issuers if s in xy_of]
    ys = [xy_of[s][1] for s in all_issuers if s in xy_of]
    xmin = (min(xs) if xs else -1.0) - PLANE_PAD
    xmax = (max(xs) if xs else 1.0) + PLANE_PAD
    ymin = (min(ys) if ys else -1.0) - PLANE_PAD
    ymax = (max(ys) if ys else 1.0) + PLANE_PAD

    for i, term in enumerate(terms):
        color = PLANE_COLORS[i % len(PLANE_COLORS)]
        z = z_of[term]
        fig.add_trace(
            go.Mesh3d(
                x=[xmin, xmax, xmax, xmin],
                y=[ymin, ymin, ymax, ymax],
                z=[z, z, z, z],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.16,
                hoverinfo="skip",
                showlegend=False,
                name=f"plane-{term}",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[xmin],
                y=[ymin],
                z=[z],
                mode="text",
                text=[term],
                textfont=dict(family="Arial", size=11, color=TEXT_COLOR),
                hoverinfo="skip",
                showlegend=False,
                name=f"label-{term}",
            )
        )

    issuer_index = {s: idx for idx, s in enumerate(all_issuers)}

    for term in terms:
        layer_nodes = (
            nodes.filter(pl.col("term") == term) if not nodes.is_empty() else nodes
        )
        if layer_nodes.is_empty():
            continue
        issuers = layer_nodes.get_column("issuer").to_list()
        x = [xy_of[s][0] for s in issuers]
        y = [xy_of[s][1] for s in issuers]
        z = [z_of[term]] * len(issuers)
        marker_colors = [_node_color(issuer_index.get(s, 0)) for s in issuers]
        custom = [["node", s, term] for s in issuers]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                text=issuers,
                textposition="top center",
                textfont=dict(family="Arial", size=8, color=TEXT_MUTED),
                marker=dict(
                    size=6,
                    color=marker_colors,
                    line=dict(color="#0f172a", width=1),
                    opacity=1.0,
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + layer_label
                    + "=%{customdata[2]}<extra></extra>"
                ),
                name=term,
                legendgroup=term,
            )
        )

        layer_intra = (
            intra.filter(pl.col("term") == term) if not intra.is_empty() else intra
        )
        if layer_intra.is_empty():
            continue
        bins: list[dict] = [
            {
                "x": [],
                "y": [],
                "z": [],
                "mx": [],
                "my": [],
                "mz": [],
                "cd": [],
                "col": [],
            }
            for _ in range(N_EDGE_BINS)
        ]
        for row in layer_intra.iter_rows(named=True):
            s, t = row["source_issuer"], row["target_issuer"]
            if s not in xy_of or t not in xy_of:
                continue
            w = float(row["weight"])
            b = _bin_index(w, wmin, wmax)
            x0, y0 = xy_of[s]
            x1, y1 = xy_of[t]
            zz = z_of[term]
            bins[b]["x"].extend([x0, x1, None])
            bins[b]["y"].extend([y0, y1, None])
            bins[b]["z"].extend([zz, zz, None])
            bins[b]["mx"].append(0.5 * (x0 + x1))
            bins[b]["my"].append(0.5 * (y0 + y1))
            bins[b]["mz"].append(zz)
            bins[b]["cd"].append(["intra", s, t, term, w])
            bins[b]["col"].append(_edge_hex(w, wmin, wmax))

        for b, bucket in enumerate(bins):
            if not bucket["x"]:
                continue
            t_mid = (b + 0.5) / N_EDGE_BINS
            line_color = to_hex(EDGE_CMAP(t_mid))
            fig.add_trace(
                go.Scatter3d(
                    x=bucket["x"],
                    y=bucket["y"],
                    z=bucket["z"],
                    mode="lines",
                    line=dict(color=line_color, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"intra-lines-{term}-{b}",
                    legendgroup=term,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=bucket["mx"],
                    y=bucket["my"],
                    z=bucket["mz"],
                    mode="markers",
                    marker=dict(size=4, color=bucket["col"], opacity=0.85),
                    customdata=bucket["cd"],
                    hovertemplate=(
                        "intra %{customdata[1]}-%{customdata[2]}"
                        "<br>" + layer_label + "=%{customdata[3]}"
                        "<br>weight=%{customdata[4]:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"intra-picks-{term}-{b}",
                    legendgroup=term,
                )
            )

    # Inter-layer edges from the graph (+ optional synthetic identity links).
    ix, iy, iz = [], [], []
    mx, my, mz, mcol, mcd = [], [], [], [], []

    def _add_inter_segment(s, t, tf, tt, w, extra_kind="inter"):
        if s not in xy_of or t not in xy_of:
            return
        if tf not in z_of or tt not in z_of:
            return
        x0, y0 = xy_of[s]
        x1, y1 = xy_of[t]
        z0, z1 = z_of[tf], z_of[tt]
        ix.extend([x0, x1, None])
        iy.extend([y0, y1, None])
        iz.extend([z0, z1, None])
        mx.append(0.5 * (x0 + x1))
        my.append(0.5 * (y0 + y1))
        mz.append(0.5 * (z0 + z1))
        mcol.append(INTER_EDGE_COLOR)
        mcd.append([extra_kind, s, t, tf, tt, w])

    if not inter.is_empty():
        for row in inter.iter_rows(named=True):
            _add_inter_segment(
                row["source_issuer"],
                row["target_issuer"],
                row["term_from"],
                row["term_to"],
                float(row["weight"]),
            )

    if add_identity_links:
        for i in range(len(terms) - 1):
            t0, t1 = terms[i], terms[i + 1]
            if nodes.is_empty():
                continue
            issuers_0 = set(
                nodes.filter(pl.col("term") == t0).get_column("issuer").to_list()
            )
            issuers_1 = set(
                nodes.filter(pl.col("term") == t1).get_column("issuer").to_list()
            )
            for issuer in sorted(issuers_0 & issuers_1):
                _add_inter_segment(issuer, issuer, t0, t1, 1.0)

    if ix:
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="lines",
                line=dict(color=INTER_EDGE_COLOR, width=2),
                hoverinfo="skip",
                showlegend=False,
                name="inter-lines",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="markers",
                marker=dict(size=4, color=mcol, opacity=0.8),
                customdata=mcd,
                hovertemplate=(
                    "inter %{customdata[1]}-%{customdata[2]}"
                    "<br>%{customdata[3]} -> %{customdata[4]}"
                    "<br>weight=%{customdata[5]:.3f}<extra></extra>"
                ),
                showlegend=False,
                name="inter-picks",
            )
        )

    z_max = max(z_of.values()) if z_of else 1.0
    _apply_dark_layout(fig, title, layer_gap, z_max)

    # Dummy trace so the connection-strength colorbar is shown.
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                size=0.1,
                color=[wmin],
                cmin=wmin,
                cmax=wmax,
                colorscale=EDGE_COLORSCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Intra-layer<br>connection strength",
                        font=dict(family="Arial", size=11, color=TEXT_MUTED),
                    ),
                    thickness=14,
                    len=0.5,
                    x=1.01,
                    tickfont=dict(family="Arial", size=10, color=TEXT_MUTED),
                    outlinecolor="#334155",
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
            name="colorbar",
        )
    )
    return fig


def _apply_dark_layout(
    fig: go.Figure, title: str, layer_gap: float, z_max: float
) -> None:
    """Dark canvas matching the GUI theme (``gui/styles.py``)."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=16, color=TEXT_COLOR),
            x=0.02,
            xanchor="left",
        ),
        font=dict(family="Arial", color=TEXT_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.0,
            font=dict(size=11, color=TEXT_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=CANVAS_BG,
        plot_bgcolor=CANVAS_BG,
        hoverlabel=dict(bgcolor="#1e293b", font=dict(color=TEXT_COLOR)),
        scene=dict(
            bgcolor=CANVAS_BG,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.7),
            camera=dict(eye=dict(x=1.55, y=1.55, z=0.85)),
        ),
    )
    fig.update_layout(scene_zaxis_range=[-0.35 * layer_gap, z_max + 0.35 * layer_gap])
