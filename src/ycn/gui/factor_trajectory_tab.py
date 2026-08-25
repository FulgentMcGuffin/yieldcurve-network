"""The "Factor (t)" / "Factor Std (t)" sub-tabs: an interactive 3D factor path.

A Plotly scene in a ``QWebEngineView``, built the same way the MLN tab is --
figure -> HTML -> temp file next to a shared ``plotly.min.js`` -> ``load()`` --
so the two interactive views share one styling, one JS-console route into the
process log, and one asset directory.

The scrubber moves a **single-point highlight trace** via ``Plotly.restyle``
rather than re-rendering the figure. Rebuilding a 3D scene on every slider tick
would reset the user's camera on each step, which makes a scrub unusable in a
view whose whole point is that you can rotate it.

Selection is bidirectional, matching ``stress_trajectory_tab``: the slider
moves the highlight, and clicking a point moves the slider. Both funnel through
``_select_index`` so the two controls cannot disagree.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import polars as pl
from PySide6.QtCore import QObject, QUrl, Qt, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.factor_trajectory_plotly import (
    HIGHLIGHT_TRACE_NAME,
    build_factor_trajectory_figure,
    regime_labels,
    trajectory_points,
)
from ycn.gui.mln_bridge import MLNWebPage, inject_canvas_shim, qwebchannel_js
from ycn.gui.styles import TEXT, TEXT_MUTED

# Name the bridge is registered under on the web channel; the injected script
# looks it up by exactly this key.
_BRIDGE_NAME = "factorBridge"

# Trace whose points are clickable. Clicks on the path line or the endpoint
# labels carry no meaningful index and are ignored.
_POINTS_TRACE = "factor-points"


class FactorTrajectoryBridge(QObject):
    """Receives point clicks from the embedded Plotly scene."""

    point_clicked = Signal(int)

    @Slot(int)
    def on_point(self, index: int) -> None:
        self.point_clicked.emit(int(index))


def _scrubber_script(points: dict) -> str:
    """JS exposing ``ycnSetHighlight(i)`` and reporting clicks back to Qt.

    The highlight trace is located **by name**, not by index, so adding traces
    to the figure cannot silently point the scrubber at the wrong one.
    """
    return f"""
<script>
(function () {{
  var POINTS = {json.dumps(points)};
  var HL = {json.dumps(HIGHLIGHT_TRACE_NAME)};
  var PICKABLE = {json.dumps(_POINTS_TRACE)};
  var graph = null;
  var pending = null;
  var bridge = null;

  function highlightTrace(gd) {{
    for (var i = 0; i < gd.data.length; i++) {{
      if (gd.data[i].name === HL) {{ return i; }}
    }}
    return -1;
  }}

  window.ycnSetHighlight = function (i) {{
    if (graph === null) {{ pending = i; return; }}
    if (i < 0 || i >= POINTS.x.length) {{ return; }}
    var idx = highlightTrace(graph);
    if (idx < 0) {{ return; }}
    Plotly.restyle(graph, {{
      x: [[POINTS.x[i]]],
      y: [[POINTS.y[i]]],
      z: [[POINTS.z[i]]],
      customdata: [[[POINTS.labels[i], POINTS.regimes[i]]]]
    }}, [idx]);
  }};

  function onClick(ev) {{
    if (!ev || !ev.points || !ev.points.length) {{ return; }}
    var pt = ev.points[0];
    if (!pt.data || pt.data.name !== PICKABLE) {{ return; }}
    if (bridge && typeof pt.pointNumber === 'number') {{
      bridge.on_point(pt.pointNumber);
    }}
  }}

  function ready(gd) {{
    graph = gd;
    gd.on('plotly_click', onClick);
    if (pending !== null) {{
      window.ycnSetHighlight(pending);
      pending = null;
    }}
  }}

  function waitForPlot() {{
    var tries = 0;
    var timer = setInterval(function () {{
      tries += 1;
      var gd = document.querySelector('.plotly-graph-div');
      if (gd && gd.on) {{
        clearInterval(timer);
        ready(gd);
      }} else if (tries > 60) {{
        clearInterval(timer);
      }}
    }}, 100);
  }}

  function start() {{
    waitForPlot();
    if (typeof QWebChannel === 'undefined' || !window.qt || !qt.webChannelTransport) {{
      return;
    }}
    new QWebChannel(qt.webChannelTransport, function (channel) {{
      bridge = channel.objects.{_BRIDGE_NAME} || null;
    }});
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', start);
  }} else {{
    start();
  }}
}})();
</script>
"""


class FactorTrajectoryTab(QFrame):
    """A 3D factor-space path with a date scrubber."""

    def __init__(
        self,
        asset_dir: Callable[[], Path],
        *,
        std: bool = False,
        on_message: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Args:
            asset_dir: Returns the directory holding ``plotly.min.js``. Injected
                rather than created here so every Plotly view in the app shares
                one 4MB sidecar and one cleanup path.
            std: Plot within-window volatilities instead of means.
            on_message: Receives JS console output (the process log).
        """
        super().__init__(parent)
        self.setObjectName("Canvas")
        self._asset_dir = asset_dir
        self._std = std
        self._factors = pl.DataFrame()
        self._regimes = pl.DataFrame()
        self._dates: list[str] = []
        self._temp_html: Path | None = None
        self._ready = False
        self._pending_index: int | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        label = "Factor Std (t)" if std else "Factor (t)"
        self.web = QWebEngineView()
        self.web.setPage(MLNWebPage(self.web, on_message=on_message, label=label))
        self.web.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        outer.addWidget(self.web, stretch=1)

        self._placeholder = QLabel("")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; background: transparent;"
        )
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        outer.addWidget(self._placeholder, stretch=1)

        scrub = QHBoxLayout()
        scrub.setSpacing(8)
        caption = QLabel("Date:")
        caption.setStyleSheet(f"color: {TEXT_MUTED};")
        caption.setToolTip("Drag the slider, or click a point in the scene")
        scrub.addWidget(caption)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        scrub.addWidget(self.slider, stretch=1)
        self.lbl_date = QLabel("—")
        self.lbl_date.setMinimumWidth(160)
        self.lbl_date.setStyleSheet(f"color: {TEXT}; font-weight: 600;")
        scrub.addWidget(self.lbl_date)
        outer.addLayout(scrub)

        self._bridge = FactorTrajectoryBridge()
        self._bridge.point_clicked.connect(self._select_index)
        self._channel = QWebChannel(self.web.page())
        self._channel.registerObject(_BRIDGE_NAME, self._bridge)
        self.web.page().setWebChannel(self._channel)
        self.web.loadFinished.connect(self._on_load_finished)

        self.set_placeholder(
            "The factor trajectory will appear here after the evolution runs."
        )

    # ------------------------------------------------------------- content
    def set_placeholder(self, message: str) -> None:
        """Show ``message`` instead of a scene, and disarm the scrubber."""
        self._placeholder.setText(message)
        self._placeholder.setVisible(True)
        self.web.setVisible(False)
        self._ready = False
        self._pending_index = None
        self._dates = []
        self.slider.blockSignals(True)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.lbl_date.setText("—")

    def set_result(self, factors: pl.DataFrame, regimes: pl.DataFrame) -> None:
        """Adopt a new factor frame and rebuild the scene."""
        self._factors = factors
        self._regimes = regimes
        xs, ys, zs, dates = trajectory_points(factors, std=self._std)
        if not xs:
            self.set_placeholder("No factor windows in this result — nothing to plot.")
            return

        points = {
            "x": xs,
            "y": ys,
            "z": zs,
            "labels": dates,
            "regimes": regime_labels(factors, regimes, len(xs)),
        }
        try:
            figure = build_factor_trajectory_figure(factors, regimes, std=self._std)
            html = inject_canvas_shim(
                figure.to_html(full_html=True, include_plotlyjs="directory")
            )
            html = _inject_before_body_end(
                html,
                f"<script>{qwebchannel_js()}</script>\n" + _scrubber_script(points),
            )
            self._write_html(html)
        except Exception as exc:  # noqa: BLE001 -- a render fault must not kill the tab
            self.set_placeholder(f"Could not draw the factor trajectory:\n{exc}")
            return

        self._dates = dates
        self._placeholder.setVisible(False)
        self.web.setVisible(True)
        self._ready = False
        self._pending_index = 0

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(xs) - 1, 0))
        self.slider.setValue(0)
        self.slider.setEnabled(len(xs) > 1)
        self.slider.blockSignals(False)
        self.lbl_date.setText(dates[0])

    @property
    def factors(self) -> pl.DataFrame:
        return self._factors

    # ------------------------------------------------------------ scrubbing
    def _on_slider(self, value: int) -> None:
        if not self._dates:
            return
        index = max(0, min(value, len(self._dates) - 1))
        self.lbl_date.setText(self._dates[index])
        self._apply_highlight(index)

    def _select_index(self, index: int) -> None:
        """Move the selection from either direction.

        Driving the slider is what moves the highlight, so a scene click sets
        the slider and lets its signal do the rest. When the value is already
        there the signal will not fire, so the highlight is refreshed directly.
        """
        if not self._dates:
            return
        index = max(0, min(int(index), len(self._dates) - 1))
        if self.slider.value() == index:
            self._on_slider(index)
        else:
            self.slider.setValue(index)

    def _apply_highlight(self, index: int) -> None:
        """Ask the page to move the highlight, or queue it until it loads."""
        if not self._ready:
            self._pending_index = index
            return
        page = self.web.page()
        if page is not None:
            page.runJavaScript(f"window.ycnSetHighlight({int(index)});")

    def _on_load_finished(self, ok: bool) -> None:
        self._ready = bool(ok)
        if self._ready and self._pending_index is not None:
            index = self._pending_index
            self._pending_index = None
            self._apply_highlight(index)

    # --------------------------------------------------------------- assets
    def _write_html(self, html: str) -> None:
        """Write the page beside the shared JS sidecar and load it."""
        directory = self._asset_dir()
        handle = tempfile.NamedTemporaryFile(
            prefix="factor_", suffix=".html", dir=str(directory), delete=False
        )
        handle.write(html.encode("utf-8"))
        handle.close()
        old = self._temp_html
        self._temp_html = Path(handle.name)
        self.web.load(QUrl.fromLocalFile(str(self._temp_html.resolve())))
        if old and old.exists():
            try:
                old.unlink()
            except OSError:
                pass


def _inject_before_body_end(html: str, payload: str) -> str:
    if "</body>" in html:
        return html.replace("</body>", payload + "</body>", 1)
    return html + payload
