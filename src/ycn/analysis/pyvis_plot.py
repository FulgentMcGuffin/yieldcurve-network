"""Render a NetworkX graph to a self-contained HTML string for QWebEngineView."""

from __future__ import annotations

import colorsys
import re
from collections.abc import Hashable

import networkx as nx
from pyvis.network import Network

# Match the GUI dark surfaces.
CANVAS_BG = "#0f172a"
NODE_COLOR = "#38bdf8"
NODE_FONT = "#e2e8f0"
TITLE_COLOR = "#e2e8f0"
INTER_EDGE_COLOR = "#64748b"


def _weight_to_color(norm_weight: float) -> str:
    """Map a normalized weight in [0, 1] to a viridis-like hex color."""
    if norm_weight < 0.5:
        r = 0
        g = int(norm_weight * 2 * 255)
        b = 255
    else:
        r = int((norm_weight - 0.5) * 2 * 255)
        g = 255
        b = int(255 - (norm_weight - 0.5) * 2 * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _theme_pyvis_html(html: str) -> str:
    """Remove duplicate headings and force a full-bleed dark canvas."""
    html = re.sub(
        r"<center>\s*<h1>.*?</h1>\s*</center>\s*",
        "",
        html,
        count=1,
        flags=re.S,
    )
    # Percentage heights + pyvis' float:left collapse the canvas in QWebEngineView.
    dark_css = f"""
    <style>
      html, body {{
        background-color: {CANVAS_BG} !important;
        color: {TITLE_COLOR} !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
      }}
      body {{
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
      }}
      .vis-navigation {{
        bottom: 16px !important;
        right: 16px !important;
      }}
      h1 {{
        color: {TITLE_COLOR} !important;
        font-family: "Segoe UI", Helvetica, sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0.5rem 0.75rem !important;
        flex: 0 0 auto;
      }}
      #mynetwork {{
        background-color: {CANVAS_BG} !important;
        border: none !important;
        float: none !important;
        position: relative !important;
        width: 100% !important;
        height: 100% !important;
        flex: 1 1 auto !important;
        min-height: 0 !important;
      }}
      .card, .card-body, .container, .container-fluid {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        width: 100% !important;
        height: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
      }}
    </style>
    """
    # After physics settles, fit the graph to the full canvas.
    fit_script = """
    <script>
      (function () {
        function fitNetwork() {
          try {
            if (typeof network !== "undefined" && network !== null) {
              network.fit({
                animation: { duration: 400, easingFunction: "easeInOutQuad" },
              });
              return true;
            }
          } catch (e) {}
          return false;
        }
        function bindStabilize() {
          try {
            if (typeof network !== "undefined" && network !== null) {
              network.once("stabilized", fitNetwork);
              return true;
            }
          } catch (e) {}
          return false;
        }
        var tries = 0;
        var timer = setInterval(function () {
          tries += 1;
          bindStabilize();
          if (fitNetwork() || tries > 40) {
            clearInterval(timer);
          }
        }, 250);
        window.addEventListener("load", function () {
          setTimeout(fitNetwork, 200);
          setTimeout(fitNetwork, 800);
          setTimeout(fitNetwork, 2000);
        });
      })();
    </script>
    """
    if "</head>" in html:
        html = html.replace("</head>", dark_css + "</head>", 1)
    else:
        html = dark_css + html
    if "</body>" in html:
        html = html.replace("</body>", fit_script + "</body>", 1)
    else:
        html = html + fit_script
    return html


def graph_to_html(
    H: nx.Graph,
    title: str = "Distance Correlation Network",
    height: str = "100%",
    width: str = "100%",
) -> str:
    """Return standalone HTML for the interactive pyvis network."""
    if H.number_of_edges() == 0:
        return (
            "<!DOCTYPE html><html><body style='font-family:Segoe UI,sans-serif;"
            f"padding:2rem;background:{CANVAS_BG};color:{TITLE_COLOR}'>"
            f"<h2>{title}</h2><p>No edges to display for the current settings."
            "</p></body></html>"
        )

    weights = [data["weight"] for _, _, data in H.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)

    def normalize_weight(weight: float) -> float:
        if max_weight == min_weight:
            return 0.5
        return (weight - min_weight) / (max_weight - min_weight)

    n_nodes = H.number_of_nodes()
    spring_length = max(220, min(450, 100 + n_nodes * 3))
    repulsion = -25000 - n_nodes * 150

    net = Network(
        height=height,
        width=width,
        notebook=False,
        cdn_resources="in_line",
        heading=title,
        bgcolor=CANVAS_BG,
        font_color=NODE_FONT,
    )
    net.barnes_hut(
        gravity=repulsion,
        central_gravity=0.03,
        spring_length=spring_length,
        spring_strength=0.04,
        damping=0.25,
        overlap=0.8,
    )

    for node in H.nodes():
        net.add_node(
            node,
            label=str(node),
            size=16,
            color=NODE_COLOR,
            title=str(node),
            font={"size": 14, "face": "arial bold", "color": NODE_FONT},
        )

    for u, v, data in H.edges(data=True):
        weight = data["weight"]
        norm_weight = normalize_weight(weight)
        net.add_edge(
            u,
            v,
            width=0.4 + norm_weight * 0.6,
            color=_weight_to_color(norm_weight),
            title=f"weight: {weight:.3f}",
        )

    net.set_edge_smooth("continuous")
    net.toggle_physics(True)
    return _theme_pyvis_html(net.generate_html(notebook=False))


def _multilayer_node_id(node: Hashable) -> str:
    if isinstance(node, tuple) and len(node) == 2:
        return f"{node[0]}@{node[1]}"
    return str(node)


def _normalize(weight: float, min_weight: float, max_weight: float) -> float:
    if max_weight == min_weight:
        return 0.5
    return (weight - min_weight) / (max_weight - min_weight)


def _term_years(term: Hashable) -> float:
    """Parse Y000p5 → 0.5, Y030p0 → 30.0; unknown terms sort last."""
    text = str(term)
    if len(text) >= 2 and text[0] == "Y" and "p" in text:
        whole, frac = text[1:].split("p", 1)
        try:
            return int(whole) + int(frac) / 10.0
        except ValueError:
            return float("inf")
    return float("inf")


def _hex_rgb(r: float, g: float, b: float) -> str:
    return (
        f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"
    )


def _source_color(source_idx: int, n_sources: int, intensity: float) -> str:
    """Hue from issuer index; saturation/value rise with term intensity in [0, 1]."""
    hue = source_idx / max(n_sources, 1)
    sat = 0.40 + 0.55 * intensity
    val = 0.32 + 0.63 * intensity
    return _hex_rgb(*colorsys.hsv_to_rgb(hue, sat, val))


def _strength_to_color(norm_weight: float) -> str:
    """Bright blue (weak) → white → yellow (strong)."""
    t = max(0.0, min(1.0, norm_weight))
    if t < 0.5:
        u = t * 2.0
        r, g, b = 0.0 + u, 0.71 + 0.29 * u, 1.0
    else:
        u = (t - 0.5) * 2.0
        r, g, b = 1.0 - 0.02 * u, 1.0 - 0.20 * u, 1.0 - 0.92 * u
    return _hex_rgb(r, g, b)


def multilayer_graph_to_html(
    M: nx.Graph,
    title: str = "Multi-Layer Yield Curve Network",
    height: str = "900px",
    width: str = "100%",
) -> str:
    """Render a multiplex graph as a source×term grid (physics off, zoom-to-fit).

    Nodes are ``(source, term)`` tuples. Column hue is the issuer; size and
    brightness increase with maturity. Intra-layer edges use a blue→white→yellow
    scale by weight and curve so non-adjacent issuer pairs stay visible.
    """
    if M.number_of_edges() == 0:
        return (
            "<!DOCTYPE html><html><body style='font-family:Segoe UI,sans-serif;"
            f"padding:2rem;background:{CANVAS_BG};color:{TITLE_COLOR}'>"
            f"<h2>{title}</h2><p>No edges to display for the current settings."
            "</p></body></html>"
        )

    tuple_nodes = [n for n in M.nodes() if isinstance(n, tuple) and len(n) == 2]
    sources = sorted({n[0] for n in tuple_nodes})
    terms = sorted({n[1] for n in tuple_nodes}, key=_term_years)
    src_x = {s: i for i, s in enumerate(sources)}
    term_y = {t: i for i, t in enumerate(terms)}
    n_terms = max(len(terms) - 1, 1)
    x_gap = 180
    y_gap = 140

    intra_weights = [
        float(data.get("weight", 0.5))
        for _, _, data in M.edges(data=True)
        if data.get("layer") == "intra"
    ]
    min_w = min(intra_weights) if intra_weights else 0.0
    max_w = max(intra_weights) if intra_weights else 1.0

    net = Network(
        height=height,
        width=width,
        notebook=False,
        cdn_resources="in_line",
        heading=title,
        bgcolor=CANVAS_BG,
        font_color=NODE_FONT,
    )
    net.set_options("""
        {
          "physics": {"enabled": false},
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true,
            "tooltipDelay": 80
          }
        }
        """)

    id_map: dict[Hashable, str] = {}
    for node in M.nodes():
        node_id = _multilayer_node_id(node)
        id_map[node] = node_id
        if isinstance(node, tuple) and len(node) == 2:
            source, term = node
            intensity = term_y.get(term, 0) / n_terms
            x, y = src_x.get(source, 0) * x_gap, term_y.get(term, 0) * y_gap
            color = _source_color(src_x.get(source, 0), len(sources), intensity)
            size = 14 + 24 * intensity
            label = str(source)
            tip = f"{source} · {term}"
        else:
            x, y, color, size, label, tip = 0, 0, NODE_COLOR, 22, node_id, node_id
        net.add_node(
            node_id,
            label=label,
            size=size,
            color=color,
            title=tip,
            x=x,
            y=y,
            physics=False,
            font={
                "size": 14 + int(6 * (size - 14) / 24),
                "face": "arial",
                "color": NODE_FONT,
            },
        )

    inter_edges = []
    intra_edges = []
    for u, v, data in M.edges(data=True):
        if data.get("layer") == "inter":
            inter_edges.append((u, v, data))
        else:
            intra_edges.append((u, v, data))
    intra_edges.sort(key=lambda e: float(e[2].get("weight", 0.0)))

    for u, v, data in inter_edges:
        weight = float(data.get("weight", 1.0))
        net.add_edge(
            id_map[u],
            id_map[v],
            width=0.8,
            color={"color": INTER_EDGE_COLOR, "opacity": 0.35},
            title=f"inter-layer (same issuer, adjacent terms): {weight:.2f}",
            smooth=False,
        )

    for u, v, data in intra_edges:
        weight = float(data.get("weight", 1.0))
        t = _normalize(weight, min_w, max_w)
        span = 1
        if isinstance(u, tuple) and isinstance(v, tuple):
            span = abs(src_x.get(u[0], 0) - src_x.get(v[0], 0))
        roundness = min(0.65, 0.06 + 0.035 * span)
        net.add_edge(
            id_map[u],
            id_map[v],
            width=1.4 + 7.0 * t,
            color=_strength_to_color(t),
            title=f"intra-layer {data.get('term', '?')}: strength={weight:.3f}",
            smooth={"type": "curvedCW", "roundness": roundness},
        )

    return _theme_pyvis_html(net.generate_html(notebook=False))
