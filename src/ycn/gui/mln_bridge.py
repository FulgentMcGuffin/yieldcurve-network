"""QWebChannel bridge: Plotly clicks in the MLN view -> Qt.

``build_multiplex_figure`` attaches ``customdata`` to every pickable trace:

- node markers:     ``["node", node, layer]``
- intra-edge picks: ``["intra", source, target, layer, weight]``
- inter-edge picks: ``["inter", source, target, layer_from, layer_to, weight]``

The injected script forwards that array as JSON on ``plotly_click``; the Python
side re-emits it as a typed signal the main window can act on.

This module also owns the embedded page itself (:class:`MLNWebPage`), so JS
console output reaches the process log instead of vanishing to stderr.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from PySide6.QtCore import QFile, QIODevice, QObject, Signal, Slot

# Importing QtWebChannel is what registers the ``:/qtwebchannel/`` Qt resource
# that :func:`qwebchannel_js` reads -- without it the QFile open silently fails.
from PySide6.QtWebChannel import QWebChannel  # noqa: F401
from PySide6.QtWebEngineCore import QWebEnginePage

# Chromium performance hints emitted by bundled third-party JS that we have
# already addressed (or cannot address) and that say nothing about this app.
# Matched as substrings of the console message.
_MUTED_JS_MESSAGES = ("willReadFrequently",)


class MLNBridge(QObject):
    """Receives click payloads from the embedded Plotly view."""

    node_clicked = Signal(str, str)  # node, layer
    edge_clicked = Signal(str, str, str)  # kind, source, target

    @Slot(str)
    def on_click(self, payload: str) -> None:
        """Called from JavaScript with the clicked point's ``customdata``."""
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return
        if not isinstance(data, list) or not data:
            return
        kind = str(data[0])
        if kind == "node" and len(data) >= 3:
            self.node_clicked.emit(str(data[1]), str(data[2]))
        elif kind in ("intra", "inter") and len(data) >= 3:
            self.edge_clicked.emit(kind, str(data[1]), str(data[2]))


class MLNWebPage(QWebEnginePage):
    """An embedded Plotly page with JS console output routed into the app.

    Qt's default page prints every JS console message to stderr as ``js: ...``,
    where a real error is indistinguishable from a performance hint and neither
    is visible to someone running the GUI from a shortcut. Messages go to the
    process log instead, minus the muted hints.

    Shared by every Plotly view in the app, so ``label`` names which one a
    message came from -- "MLN view" and "Factor (t)" land in the same log.
    """

    def __init__(
        self,
        parent: QObject | None = None,
        on_message: Callable[[str], None] | None = None,
        label: str = "MLN view",
    ) -> None:
        super().__init__(parent)
        self._on_message = on_message
        self._label = label

    def javaScriptConsoleMessage(  # noqa: N802 -- Qt override
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        line_number: int,
        source_id: str,
    ) -> None:
        if any(muted in message for muted in _MUTED_JS_MESSAGES):
            return
        if self._on_message is None:
            return
        name = getattr(level, "name", str(level)).replace("MessageLevel", "").lower()
        self._on_message(f"{self._label} [{name}]: {message}")


# Plotly's bundled bundle reads pixels back from 2D canvases repeatedly (text
# metrics and glyph atlases), which makes Chromium emit a "willReadFrequently"
# performance hint on every launch. Defaulting the flag on for 2D contexts is
# exactly what that hint asks for, and is what the canvases are actually doing.
# WebGL contexts -- the 3D scene itself -- are untouched, and an explicit
# caller-supplied value always wins.
_CANVAS_SHIM = """
<script>
(function () {
  var original = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attributes) {
    if (type === '2d') {
      attributes = attributes || {};
      if (attributes.willReadFrequently === undefined) {
        attributes.willReadFrequently = true;
      }
    }
    return original.call(this, type, attributes);
  };
})();
</script>
"""


def inject_canvas_shim(html: str) -> str:
    """Insert the 2D-canvas shim ahead of any other script on the page.

    Must land in ``<head>``: overriding ``getContext`` only helps for canvases
    created *after* it runs, and Plotly builds its own during load.
    """
    if "<head>" in html:
        return html.replace("<head>", "<head>" + _CANVAS_SHIM, 1)
    if "<body>" in html:
        return html.replace("<body>", "<body>" + _CANVAS_SHIM, 1)
    return _CANVAS_SHIM + html


def qwebchannel_js() -> str:
    """Return the bundled ``qwebchannel.js`` source from Qt's resources.

    Inlined into the page rather than linked, so the temp HTML file has no
    external dependencies. Returns an empty string if the resource is missing,
    in which case the click bridge is simply inert (the 3D view still works).
    """
    handle = QFile(":/qtwebchannel/qwebchannel.js")
    if not handle.open(QIODevice.OpenModeFlag.ReadOnly):
        return ""
    try:
        return bytes(handle.readAll().data()).decode("utf-8")
    finally:
        handle.close()


_CLICK_SCRIPT = """
<script>
(function () {
  function wire(bridge) {
    var tries = 0;
    var timer = setInterval(function () {
      tries += 1;
      var gd = document.querySelector('.plotly-graph-div');
      if (gd && gd.on) {
        clearInterval(timer);
        gd.on('plotly_click', function (ev) {
          if (!ev || !ev.points || !ev.points.length) { return; }
          var cd = ev.points[0].customdata;
          if (cd) { bridge.on_click(JSON.stringify(cd)); }
        });
      } else if (tries > 60) {
        clearInterval(timer);
      }
    }, 100);
  }
  function start() {
    if (typeof QWebChannel === 'undefined' || !window.qt || !qt.webChannelTransport) {
      return;
    }
    new QWebChannel(qt.webChannelTransport, function (channel) {
      var bridge = channel.objects.mlnBridge;
      if (bridge) { wire(bridge); }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
</script>
"""


def inject_click_bridge(html: str) -> str:
    """Insert ``qwebchannel.js`` and the click hook into a Plotly HTML page."""
    channel_js = qwebchannel_js()
    if not channel_js:
        return html
    payload = f"<script>{channel_js}</script>\n{_CLICK_SCRIPT}"
    if "</body>" in html:
        return html.replace("</body>", payload + "</body>", 1)
    return html + payload
