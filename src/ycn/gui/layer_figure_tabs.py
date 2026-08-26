"""A tab strip holding one matplotlib figure per component network.

Both "MLN: Degree" and "MLN: Centrality" have the same shape -- one sub-tab per
layer, each a single figure with an interactive cursor -- so they share this
widget and differ only in the render callback and the cursor they attach.

The figures are built lazily, one sub-tab at a time, because a 15-layer panel
would otherwise pay for 15 renders on the GUI thread before showing anything.
Only the visible layer is drawn; switching sub-tabs draws that one and caches
it. Cursor objects are parked on the canvas so Python does not collect the only
reference to a live matplotlib event connection.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ycn.gui.styles import TEXT_MUTED

# Renders one layer's figure. Returning None means "this layer has nothing to
# show" and the sub-tab falls back to its placeholder.
LayerRenderer = Callable[[str], Figure | None]

# Builds an optional control strip under one layer's figure (e.g. the degree
# tab's threshold slider). Returning None means that layer gets no controls.
LayerControls = Callable[[str, Figure, FigureCanvas], QWidget | None]


def attach_histogram_cursor(figure: Figure, canvas: FigureCanvas) -> None:
    """Wire the degree-histogram hover tooltip, if the figure carries its data."""
    data = getattr(figure, "_histogram_cursor_data", None)
    if data is None:
        return
    from ycn.analysis.degree_hist import _HistogramCursor

    ax, degrees, bin_edges = data
    # Parked on the canvas: the cursor holds the only mpl_connect callback, and
    # a local would be collected the moment this function returns.
    canvas._ycn_cursor = _HistogramCursor(ax, degrees, bin_edges, canvas)


def attach_line_cursor(figure: Figure, canvas: FigureCanvas) -> None:
    """Wire the centrality-trajectory hover/highlight, one cursor per axis."""
    data = getattr(figure, "_line_cursor_data", None)
    if not data:
        return
    from ycn.analysis.evolution_viz import _LineCursor

    if isinstance(data, dict):
        canvas._ycn_cursor = [
            _LineCursor(ax_obj, line_data, canvas, line_objects, original_styles)
            for ax_obj, line_data, line_objects, original_styles in data.values()
        ]
        return
    ax, line_data, line_objects, original_styles = data
    canvas._ycn_cursor = _LineCursor(
        ax, line_data, canvas, line_objects, original_styles
    )


class LayerFigureTabs(QTabWidget):
    """One sub-tab per layer, each showing a lazily-rendered figure."""

    def __init__(
        self,
        *,
        empty_message: str,
        on_error: Callable[[str], None] | None = None,
        attach_cursor: Callable[[Figure, FigureCanvas], None] | None = None,
        build_controls: LayerControls | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Args:
            empty_message: Shown on the single placeholder tab before a run.
            on_error: Receives a message when one layer fails to render; the
                other layers are unaffected.
            attach_cursor: Wires the figure's interactive cursor to its canvas.
            build_controls: Adds a control strip beneath a layer's figure.
        """
        super().__init__(parent)
        self.setObjectName("ResultTabs")
        self._empty_message = empty_message
        self._on_error = on_error
        self._attach_cursor = attach_cursor
        self._build_controls = build_controls
        self._renderer: LayerRenderer | None = None
        self._layers: list[str] = []
        self._drawn: set[str] = set()
        self._pages: dict[str, QVBoxLayout] = {}
        self._control_slots: dict[str, QVBoxLayout] = {}
        self.currentChanged.connect(self._on_tab_changed)
        self.set_placeholder(empty_message)

    # -------------------------------------------------------------- content
    def set_placeholder(self, message: str | None = None) -> None:
        """Drop every layer tab and show one placeholder in their place."""
        self._renderer = None
        self._layers = []
        self._drawn.clear()
        self._pages.clear()
        self._control_slots.clear()
        self._clear_tabs()
        page, _layout, _controls = self._make_page(message or self._empty_message)
        self.addTab(page, "—")

    def set_layers(self, layers: list[str], renderer: LayerRenderer) -> None:
        """Rebuild the sub-tabs for ``layers``, drawing only the visible one."""
        self._renderer = renderer
        self._layers = list(layers)
        self._drawn.clear()
        self._pages.clear()
        self._control_slots.clear()
        self._clear_tabs()

        if not self._layers:
            page, _layout, _controls = self._make_page(self._empty_message)
            self.addTab(page, "—")
            return

        blocked = self.blockSignals(True)
        try:
            for layer in self._layers:
                page, layout, controls = self._make_page("Rendering…")
                self._pages[layer] = layout
                self._control_slots[layer] = controls
                self.addTab(page, layer)
            self.setCurrentIndex(0)
        finally:
            self.blockSignals(blocked)
        self._draw_current()

    @property
    def layers(self) -> list[str]:
        return list(self._layers)

    def current_layer(self) -> str | None:
        """The layer whose sub-tab is showing, or None while placeholdered."""
        index = self.currentIndex()
        if not self._layers or index < 0 or index >= len(self._layers):
            return None
        return self._layers[index]

    # ------------------------------------------------------------ rendering
    def _on_tab_changed(self, _index: int) -> None:
        self._draw_current()

    def _draw_current(self) -> None:
        """Render the visible layer, once."""
        layer = self.current_layer()
        if layer is None or self._renderer is None or layer in self._drawn:
            return
        layout = self._pages.get(layer)
        if layout is None:
            return
        self._drawn.add(layer)
        try:
            figure = self._renderer(layer)
        except Exception as exc:  # noqa: BLE001 -- one layer must not kill the tab
            if self._on_error is not None:
                self._on_error(f"{layer}: {exc}")
            self._fill(layout, f"Could not render {layer}:\n{exc}")
            return
        if figure is None:
            self._fill(layout, f"No data for {layer}.")
            return
        self._show_figure(layer, layout, figure)

    def _show_figure(self, layer: str, layout: QVBoxLayout, figure: Figure) -> None:
        self._clear_layout(layout)
        canvas = FigureCanvas(figure)
        canvas.setStyleSheet("background-color: transparent;")
        layout.addWidget(canvas)
        canvas.draw()
        if self._attach_cursor is not None:
            try:
                self._attach_cursor(figure, canvas)
            except Exception as exc:  # noqa: BLE001 -- hover is a nicety
                if self._on_error is not None:
                    self._on_error(f"cursor: {exc}")
        if self._build_controls is None:
            return
        slot = self._control_slots.get(layer)
        if slot is None:
            return
        self._clear_layout(slot)
        try:
            controls = self._build_controls(layer, figure, canvas)
        except Exception as exc:  # noqa: BLE001 -- controls are optional
            if self._on_error is not None:
                self._on_error(f"controls: {exc}")
            return
        if controls is not None:
            slot.addWidget(controls)

    def _fill(self, layout: QVBoxLayout, message: str) -> None:
        self._clear_layout(layout)
        layout.addWidget(self._placeholder_label(message))

    # --------------------------------------------------------------- pieces
    def _make_page(self, message: str) -> tuple[QWidget, QVBoxLayout, QVBoxLayout]:
        """A layer page: the figure holder, plus a slot for its controls."""
        page = QFrame()
        page.setObjectName("Canvas")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._placeholder_label(message))
        outer.addWidget(container, stretch=1)

        control_host = QWidget()
        controls = QVBoxLayout(control_host)
        controls.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(control_host, stretch=0)
        return page, layout, controls

    @staticmethod
    def _placeholder_label(message: str) -> QLabel:
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; background: transparent;"
        )
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return label

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            widget = layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

    def _clear_tabs(self) -> None:
        while self.count():
            page = self.widget(0)
            self.removeTab(0)
            if page is not None:
                page.deleteLater()
