"""Main window: configuration sidebar + multi-layer network tabs + process log.

The application builds exactly one artefact: a multi-layer network over a wide
yield-curve panel. Two layerings are possible -- issuers linked within each
term, or terms linked within each issuer -- selected by the NETWORK TYPE
dropdown. Node-name and series-value pickers therefore do not exist; those
roles are inferred from the panel by ``analysis.yield_curve.detect_panel``.
"""

from __future__ import annotations

import tempfile
import threading
from datetime import date
from pathlib import Path

from PySide6.QtCore import QDate, QSize, QThread, QUrl, Qt
from PySide6.QtGui import QColor, QPalette, QTextCursor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import polars as pl

from ycn.analysis.config import PipelineConfig
from ycn.analysis.data_access import (
    column_date_bounds,
    list_columns,
    list_tables,
)
from ycn.analysis.evolution import EvolutionConfig
from ycn.analysis.gui_cache import GuiDataCache
from ycn.analysis.measures import (
    ACE_AVAILABLE,
    ACE_IMPORT_ERROR,
    available_measures,
    measure_short_label,
)
from ycn.analysis.mln import MLNConfig
from ycn.analysis.multiplex_data import filter_tables
from ycn.analysis.multiplex_plotly import build_multiplex_figure
from ycn.analysis.transforms import available_transforms
from ycn.analysis.yield_curve import (
    CurvePanel,
    NetworkKind,
    available_cells,
    detect_panel,
    load_long_panel,
)
from ycn.gui.evolution_settings_dialog import EvolutionSettingsDialog
from ycn.gui.data_table_dialog import DataTableDialog, eye_icon
from ycn.gui.mln_bridge import (
    MLNBridge,
    MLNWebPage,
    inject_canvas_shim,
    inject_click_bridge,
)
from ycn.gui.mln_settings_dialog import MLNSettingsDialog
from ycn.gui.styles import APP_STYLE, BG_SIDEBAR
from ycn.gui.user_filter_dialog import UserFilterDialog
from ycn.gui.workers import MLNResult, MLNWorker

# How long Cancel Render waits for a worker to unwind before detaching from it.
# Long enough for a cooperative worker sitting on a progress checkpoint, short
# enough that the button never feels stuck.
CANCEL_GRACE_MS = 250

# Table and date column selected automatically when a database offers them.
DEFAULT_TABLE = "par_rates"
DEFAULT_DATE_COLUMN = "date"


def _force_dark_surface(widget: QWidget, color: str = BG_SIDEBAR) -> None:
    """Ensure opaque dark fills even when platform styles ignore QSS backgrounds."""
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    qcolor = QColor(color)
    palette.setColor(QPalette.ColorRole.Window, qcolor)
    palette.setColor(QPalette.ColorRole.Base, qcolor)
    palette.setColor(QPalette.ColorRole.Button, qcolor)
    widget.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YieldCurve-Network")
        self.resize(1440, 990)
        self.setStyleSheet(APP_STYLE)

        self._db_path: Path | None = None
        self._last_progress_line: int | None = None
        self._busy = False
        # Shared with MLNWorker: setting this raises inside its progress/status
        # callbacks (checked on every pair) to unwind a long-running computation
        # almost immediately, instead of the GUI thread blocking on
        # QThread.wait() until the computation finishes on its own.
        self._cancel_event = threading.Event()
        # Session-scoped Polars memoization (in-memory SQLite via framecache).
        self._data_cache = GuiDataCache.create()

        # Panel role inference for the selected table, and the user's manual
        # (term, issuer) cell selection. None means "no manual filtering".
        self._panel: CurvePanel | None = None
        self._cell_mask: set[tuple[str, str]] | None = None
        self._cell_mask_key: tuple | None = None

        self._current_measure_tag: str = "measure"

        # Evolution settings are collected now and consumed once MLN evolution
        # is wired up; no evolution tab exists yet.
        self._evolution_config = EvolutionConfig()

        from ycn.gui.edge_settings_dialog import EdgeSettingsConfig

        self._edge_settings = EdgeSettingsConfig()

        self._mln_config = MLNConfig(layer_column="")
        self._mln_worker_thread: QThread | None = None
        self._mln_worker: MLNWorker | None = None
        self._mln_result: MLNResult | None = None
        self._mln_temp_html: Path | None = None
        self._mln_temp_dir: Path | None = None
        # Workers detached by Cancel Render, kept alive until their thread
        # actually exits. Dropping the last Python reference to a still-running
        # QThread/worker lets Qt delete the C++ object underneath it and crash.
        self._retired_workers: list[tuple] = []
        self._last_config: PipelineConfig | None = None
        self._mln_row_of: dict[tuple[str, str], int] = {}
        self._mln_bridge = MLNBridge()
        self._mln_channel: QWebChannel | None = None

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_canvas())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1140])

        self._set_controls_enabled(False)
        self.btn_browse.setEnabled(True)

        if not ACE_AVAILABLE:
            self._append_log(
                "Maximal correlation (ACE) unavailable: "
                f"{ACE_IMPORT_ERROR or 'ace_cream not installed'}. "
                "Install a Fortran compiler (gfortran) and run "
                "`uv sync --extra ace` to enable it."
            )

    # ------------------------------------------------------------------ UI
    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setMinimumWidth(280)
        frame.setMaximumWidth(340)
        _force_dark_surface(frame)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _force_dark_surface(scroll)
        if scroll.viewport() is not None:
            _force_dark_surface(scroll.viewport())
        outer.addWidget(scroll)

        content = QWidget()
        content.setObjectName("SidebarContent")
        _force_dark_surface(content)
        scroll.setWidget(content)
        form = QVBoxLayout(content)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(3)

        brand = QLabel("YieldCurve-Network")
        brand.setObjectName("Brand")
        form.addWidget(brand)

        form.addWidget(self._section("DATA SOURCE"))
        self.lbl_db = QLabel("No database selected")
        self.lbl_db.setObjectName("DbPath")
        self.lbl_db.setWordWrap(True)
        form.addWidget(self.lbl_db)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setObjectName("SecondaryButton")
        self.btn_browse.clicked.connect(self._browse_db)
        form.addWidget(self.btn_browse)

        form.addWidget(self._section("TABLE"))
        self.cmb_table = QComboBox()
        self.cmb_table.currentTextChanged.connect(self._on_table_changed)
        form.addWidget(self.cmb_table)

        form.addWidget(self._section("DATE / DATETIME COLUMN"))
        self.cmb_date = QComboBox()
        self.cmb_date.currentTextChanged.connect(self._on_date_column_changed)
        form.addWidget(self.cmb_date)

        form.addWidget(self._section("NETWORK TYPE"))
        self.cmb_network_kind = QComboBox()
        for kind in (NetworkKind.ISSUER_BY_TERM, NetworkKind.TERM_BY_ISSUER):
            self.cmb_network_kind.addItem(kind.label, kind.value)
        self.cmb_network_kind.setToolTip(
            "Issuer Network by Term: one layer per maturity, nodes are issuers.\n"
            "Term Network by Issuer: one layer per issuer, nodes are maturities."
        )
        form.addWidget(self.cmb_network_kind)

        self.lbl_panel = QLabel("")
        self.lbl_panel.setObjectName("StatusLabel")
        self.lbl_panel.setWordWrap(True)
        form.addWidget(self.lbl_panel)

        filter_body = self._collapsible_section(form, "OPTIONAL FILTER", collapsed=True)
        self.chk_filter_where = QCheckBox("WHERE clause")
        self.chk_filter_where.setToolTip(
            "Filter rows with a raw SQL boolean expression, applied by the "
            "database before anything else runs."
        )
        self.chk_filter_where.toggled.connect(self._on_where_toggled)
        filter_body.addWidget(self.chk_filter_where)
        self.txt_filter_where = QLineEdit()
        self.txt_filter_where.setPlaceholderText("e.g. \"source\" <> 'GRC'")
        self.txt_filter_where.setToolTip(
            "Raw SQL boolean expression inserted after WHERE in the load query."
        )
        filter_body.addWidget(self.txt_filter_where)
        self._on_where_toggled()

        self.btn_mln_settings = QPushButton("⚙ MLN Settings")
        self.btn_mln_settings.setObjectName("SecondaryButton")
        self.btn_mln_settings.clicked.connect(self._show_mln_settings)
        self.btn_mln_settings.setToolTip(
            "Configure MLN centrality, community method and Jaccard threshold"
        )
        form.addWidget(self.btn_mln_settings)

        self.btn_user_filter = QPushButton("▦ User Filter")
        self.btn_user_filter.setObjectName("SecondaryButton")
        self.btn_user_filter.clicked.connect(self._show_user_filter)
        self.btn_user_filter.setToolTip(
            "Pick the individual term × issuer cells that feed the network, "
            "from the data left after the Optional Filter and date range."
        )
        form.addWidget(self.btn_user_filter)

        self.lbl_cell_mask = QLabel("")
        self.lbl_cell_mask.setObjectName("StatusLabel")
        self.lbl_cell_mask.setWordWrap(True)
        form.addWidget(self.lbl_cell_mask)

        transforms_body = self._collapsible_section(
            form, "TRANSFORMS (ORDERED)", collapsed=True
        )
        self.lst_transforms = QListWidget()
        for transform_id, label in available_transforms():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, transform_id)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.lst_transforms.addItem(item)
        self.lst_transforms.setMaximumHeight(56)
        transforms_body.addWidget(self.lst_transforms)
        tf_row = QHBoxLayout()
        tf_row.setSpacing(4)
        self.btn_tf_up = QPushButton("Up")
        self.btn_tf_up.setObjectName("SecondaryButton")
        self.btn_tf_up.clicked.connect(lambda: self._move_transform(-1))
        self.btn_tf_down = QPushButton("Down")
        self.btn_tf_down.setObjectName("SecondaryButton")
        self.btn_tf_down.clicked.connect(lambda: self._move_transform(1))
        tf_row.addWidget(self.btn_tf_up)
        tf_row.addWidget(self.btn_tf_down)
        transforms_body.addLayout(tf_row)

        form.addWidget(self._section("CONNECTION MEASURE"))
        self.cmb_measure = QComboBox()
        for measure_id, label in available_measures():
            self.cmb_measure.addItem(label, measure_id)
        form.addWidget(self.cmb_measure)

        self.btn_edge_settings = QPushButton("⚙ Edge Settings")
        self.btn_edge_settings.setObjectName("SecondaryButton")
        self.btn_edge_settings.clicked.connect(self._show_edge_settings)
        self.btn_edge_settings.setToolTip(
            "Configure measure-specific parameters (e.g., stress regime quantile)"
        )
        form.addWidget(self.btn_edge_settings)

        form.addWidget(self._section("DATE RANGE"))
        dates = QHBoxLayout()
        dates.setSpacing(4)
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        dates.addWidget(self.date_start)
        dates.addWidget(self.date_end)
        form.addLayout(dates)

        form.addWidget(self._section("INDEPENDENCE THRESHOLD"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 1.0)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setValue(0.33)
        form.addWidget(self.spin_threshold)

        evolution_body = self._collapsible_section(form, "EVOLUTION", collapsed=True)
        self.chk_evolution = QCheckBox("Run Evolution")
        # Kept (with its settings) for the MLN-evolution stage, which is the
        # next piece of work. Disabled rather than silently inert so the state
        # of the feature is visible.
        self.chk_evolution.setEnabled(False)
        self.chk_evolution.setToolTip(
            "Evolution of the multi-layer network is not wired up yet. "
            "Settings entered here are retained for it."
        )
        evolution_body.addWidget(self.chk_evolution)
        self.btn_evolution_settings = QPushButton("⚙ Evolution Settings")
        self.btn_evolution_settings.setObjectName("SecondaryButton")
        self.btn_evolution_settings.clicked.connect(self._show_evolution_settings)
        self.btn_evolution_settings.setToolTip(
            "Configure network evolution analysis parameters"
        )
        evolution_body.addWidget(self.btn_evolution_settings)

        form.addSpacing(8)
        self.btn_run = QPushButton("Build network")
        self.btn_run.clicked.connect(self._run_mln)
        form.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        form.addWidget(self.progress)

        self.btn_cancel = QPushButton("✕ Cancel Render")
        self.btn_cancel.setObjectName("CancelButton")
        self.btn_cancel.clicked.connect(self._cancel_render)
        self.btn_cancel.setToolTip("Cancel current analysis and clear the MLN tabs")
        self.btn_cancel.setEnabled(False)
        form.addWidget(self.btn_cancel)

        self.lbl_status = QLabel("Select a DuckDB / SQLite file to begin.")
        self.lbl_status.setObjectName("StatusLabel")
        self.lbl_status.setWordWrap(True)
        form.addWidget(self.lbl_status)

        form.addStretch(1)
        return frame

    def _build_canvas(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("CanvasFrame")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("ResultTabs")
        self.tabs.setMinimumHeight(300)

        self.tabs.addTab(self._build_mln_page(), "MLN")

        mln_metrics_page = QFrame()
        mln_metrics_page.setObjectName("Canvas")
        mln_metrics_outer = QVBoxLayout(mln_metrics_page)
        mln_metrics_outer.setContentsMargins(8, 8, 8, 8)
        self.mln_metrics_container = QWidget()
        self.mln_metrics_layout = QVBoxLayout(self.mln_metrics_container)
        self.mln_metrics_layout.setContentsMargins(0, 0, 0, 0)
        mln_metrics_outer.addWidget(self.mln_metrics_container)
        self._set_mln_metrics_placeholder()
        self.tabs.addTab(mln_metrics_page, "MLN: Metrics")

        mln_comm_page = QFrame()
        mln_comm_page.setObjectName("Canvas")
        mln_comm_outer = QVBoxLayout(mln_comm_page)
        mln_comm_outer.setContentsMargins(8, 8, 8, 8)
        self.mln_community_container = QWidget()
        self.mln_community_layout = QVBoxLayout(self.mln_community_container)
        self.mln_community_layout.setContentsMargins(0, 0, 0, 0)
        mln_comm_outer.addWidget(self.mln_community_container)
        self._set_mln_community_placeholder()
        self.tabs.addTab(mln_comm_page, "MLN: Community")

        self.btn_view_data = QToolButton()
        self.btn_view_data.setObjectName("ViewDataButton")
        self.btn_view_data.setIcon(eye_icon())
        self.btn_view_data.setIconSize(QSize(18, 18))
        self.btn_view_data.setFixedSize(28, 28)
        self.btn_view_data.setToolTip("View data for the selected tab")
        self.btn_view_data.setEnabled(False)
        self.btn_view_data.clicked.connect(self._show_tab_data)
        self.tabs.setCornerWidget(self.btn_view_data, Qt.Corner.TopRightCorner)
        self.tabs.currentChanged.connect(self._update_view_data_button)

        layout.addWidget(self.tabs, stretch=3)

        log_title = QLabel("PROCESS LOG")
        log_title.setObjectName("LogTitle")
        layout.addWidget(log_title)

        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("ProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setMaximumBlockCount(5000)
        self.process_log.setPlaceholderText(
            "Pipeline status, transforms, and pair-progress will appear here…"
        )
        self.process_log.setMinimumHeight(140)
        self.process_log.setMaximumHeight(220)
        layout.addWidget(self.process_log, stretch=0)
        return wrap

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _collapsible_section(
        self, parent: QVBoxLayout, title: str, *, collapsed: bool = True
    ) -> QVBoxLayout:
        """Add a disclosure header; return the inner layout (hidden if collapsed)."""
        toggle = QToolButton()
        toggle.setObjectName("CollapseHeader")
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toggle.setCheckable(True)
        toggle.setAutoRaise(True)
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        body = QWidget()
        body.setObjectName("CollapseBody")
        inner = QVBoxLayout(body)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(3)

        def _sync(expanded: bool) -> None:
            toggle.setText(f"{'▾' if expanded else '▸'}  {title}")
            body.setVisible(expanded)

        # Add to the parent layout BEFORE the first setVisible: hiding a widget
        # while it is still parentless marks it as a hidden top-level window,
        # which on some platforms makes it reappear as a floating panel instead
        # of expanding inline.
        parent.addWidget(toggle)
        parent.addWidget(body)
        toggle.toggled.connect(_sync)
        toggle.setChecked(not collapsed)
        _sync(not collapsed)
        return inner

    # ------------------------------------------------------------- tab data
    def _update_view_data_button(self) -> None:
        """Enable the eye button only when the selected tab has rendered data."""
        if not hasattr(self, "btn_view_data"):
            return
        self.btn_view_data.setEnabled(self._tab_dataframe() is not None)

    def _tab_dataframe(self) -> tuple[str, pl.DataFrame] | None:
        """Return (tab title, frame) for the selected tab, or None if not rendered."""
        title = self.tabs.tabText(self.tabs.currentIndex())
        if self._mln_result is None:
            return None
        if title == "MLN":
            return title, self._mln_edge_frame(self._mln_result)
        if title == "MLN: Metrics":
            return title, self._mln_result.centrality_df
        if title == "MLN: Community":
            return title, self._mln_result.community_df
        return None

    @staticmethod
    def _mln_edge_frame(result: MLNResult) -> pl.DataFrame:
        intra = result.intra.with_columns(pl.lit("intra").alias("edge_type"))
        inter = result.inter.with_columns(pl.lit("inter").alias("edge_type"))
        if intra.is_empty() and inter.is_empty():
            return result.nodes
        return pl.concat([intra, inter], how="diagonal")

    def _show_tab_data(self) -> None:
        """Open a cloned, filterable table of the selected tab's data."""
        payload = self._tab_dataframe()
        if payload is None:
            return
        title, frame = payload
        DataTableDialog(frame, title=title, parent=self).exec()

    def _on_js_message(self, message: str) -> None:
        """Log a JS console message from the MLN view.

        The page can outlive, and load before, the log widget, so this tolerates
        being called while the canvas is still being built.
        """
        if hasattr(self, "process_log"):
            self._append_log(message)

    # ------------------------------------------------------------------ log
    def _append_log(self, message: str, *, replace_last: bool = False) -> None:
        """Append a line, optionally overwriting the previous progress line."""
        cursor = self.process_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if replace_last and self._last_progress_line is not None:
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            cursor.movePosition(
                QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor
            )
            cursor.removeSelectedText()
            cursor.insertText(message)
        else:
            if self.process_log.blockCount() > 1 or self.process_log.toPlainText():
                cursor.insertText("\n")
            cursor.insertText(message)
        self._last_progress_line = cursor.blockNumber() if replace_last else None
        self.process_log.setTextCursor(cursor)
        self.process_log.ensureCursorVisible()

    # ------------------------------------------------------------ data source
    def _browse_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database",
            str(Path.home()),
            "Databases (*.duckdb *.db *.sqlite *.sqlite3);;All files (*.*)",
        )
        if not path:
            return
        self._db_path = Path(path)
        self.lbl_db.setText(str(self._db_path))
        try:
            tables = list_tables(self._db_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Database error", str(exc))
            return

        default = next((t for t in tables if t == DEFAULT_TABLE), None)
        self.cmb_table.blockSignals(True)
        self.cmb_table.clear()
        self.cmb_table.addItems(tables)
        if default is not None:
            self.cmb_table.setCurrentText(default)
        self.cmb_table.blockSignals(False)
        self._set_controls_enabled(True)
        if tables:
            self._on_table_changed(default or tables[0])
        msg = f"Loaded database with {len(tables)} table(s)."
        self.lbl_status.setText(msg)
        self._append_log(msg)

    def _on_table_changed(self, table: str) -> None:
        if not self._db_path or not table:
            return
        try:
            columns = list_columns(self._db_path, table)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Schema error", str(exc))
            return

        # The date column defaults to "date" when present; every other role is
        # inferred from the panel, so this is the only column the user picks.
        date_default = next(
            (c for c in columns if c.lower() == DEFAULT_DATE_COLUMN), None
        )
        self.cmb_date.blockSignals(True)
        self.cmb_date.clear()
        self.cmb_date.addItems(columns)
        if date_default is not None:
            self.cmb_date.setCurrentText(date_default)
        self.cmb_date.blockSignals(False)

        self._reset_cell_mask()
        self._refresh_panel()
        self._guess_date_range()
        self._append_log(f"Selected table {table!r} ({len(columns)} columns).")

    def _on_date_column_changed(self, _text: str) -> None:
        """The date column defines which columns remain available as terms."""
        self._reset_cell_mask()
        self._refresh_panel()
        self._guess_date_range()

    def _refresh_panel(self) -> None:
        """Re-infer issuer/term roles and report them under NETWORK TYPE."""
        self._panel = None
        table = self.cmb_table.currentText()
        date_column = self.cmb_date.currentText()
        if not self._db_path or not table or not date_column:
            self.lbl_panel.setText("")
            self.btn_run.setEnabled(False)
            return
        try:
            panel = detect_panel(self._db_path, table, date_column)
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- a schema quirk must not break the GUI
            self.lbl_panel.setText(f"Could not inspect columns: {exc}")
            self.btn_run.setEnabled(False)
            return

        self._panel = panel
        if not panel.is_usable:
            self.lbl_panel.setText(
                "This table is not a yield-curve panel: it needs an issuer "
                "column and at least two numeric term columns "
                "(e.g. 0.5Y, 1Y, 10Y)."
            )
            self.btn_run.setEnabled(False)
            self._append_log(
                f"Table {table!r}: no usable curve panel "
                f"(issuer={panel.issuer_column or 'none'}, "
                f"{len(panel.term_columns)} term column(s))."
            )
            return

        preview = ", ".join(panel.term_columns[:6])
        if len(panel.term_columns) > 6:
            preview += f", … (+{len(panel.term_columns) - 6})"
        self.lbl_panel.setText(
            f"Issuer column: {panel.issuer_column} · "
            f"{len(panel.term_columns)} terms: {preview}"
        )
        self.btn_run.setEnabled(not self._busy)
        self._append_log(
            f"Panel: issuer={panel.issuer_column!r}, "
            f"terms={list(panel.term_columns)}"
        )

    def _on_where_toggled(self, *_args) -> None:
        """The clause box is live only while the filter is switched on."""
        enabled = self.chk_filter_where.isChecked()
        self.txt_filter_where.setEnabled(enabled)
        if enabled:
            self.txt_filter_where.setFocus()

    def _where_clause(self) -> str | None:
        """The active SQL filter, or None when it is off or blank."""
        if not self.chk_filter_where.isChecked():
            return None
        return self.txt_filter_where.text().strip() or None

    def _guess_date_range(self, *_args) -> None:
        """Seed date widgets from the selected date column when possible."""
        today = QDate.currentDate()
        self.date_end.setDate(today)
        self.date_start.setDate(QDate(2010, 1, 1))
        if not self._db_path:
            return
        table = self.cmb_table.currentText()
        date_col = self.cmb_date.currentText()
        if not table or not date_col:
            return
        try:
            lo, hi = column_date_bounds(self._db_path, table, date_col)
        except Exception:  # noqa: BLE001
            return
        if lo is not None:
            self.date_start.setDate(QDate(lo.year, lo.month, lo.day))
        if hi is not None:
            self.date_end.setDate(QDate(hi.year, hi.month, hi.day))

    def _move_transform(self, delta: int) -> None:
        row = self.lst_transforms.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.lst_transforms.count():
            return
        item = self.lst_transforms.takeItem(row)
        self.lst_transforms.insertItem(new_row, item)
        self.lst_transforms.setCurrentRow(new_row)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable sidebar controls. Browse stays available unless busy."""
        for widget in (
            self.btn_browse,
            self.cmb_table,
            self.cmb_date,
            self.cmb_network_kind,
            self.btn_mln_settings,
            self.btn_user_filter,
            self.chk_filter_where,
            self.lst_transforms,
            self.btn_tf_up,
            self.btn_tf_down,
            self.cmb_measure,
            self.btn_edge_settings,
            self.date_start,
            self.date_end,
            self.spin_threshold,
            self.btn_run,
        ):
            widget.setEnabled(enabled)
        # Evolution settings stay configurable ahead of the feature landing.
        self.btn_evolution_settings.setEnabled(enabled)
        self.txt_filter_where.setEnabled(enabled and self.chk_filter_where.isChecked())
        if enabled and (
            self._db_path is None or self._panel is None or not self._panel.is_usable
        ):
            self.btn_run.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        """Toggle run/cancel only -- every other control stays usable.

        A run happens on a background thread, so the rest of the sidebar can be
        edited while it works (setting up the next build). Only "Build network"
        is blocked, because starting a second run on top of a live one is the
        one genuinely unsafe action.
        """
        self._busy = busy
        usable = self._panel is not None and self._panel.is_usable
        self.btn_run.setEnabled(not busy and self._db_path is not None and usable)
        self.btn_cancel.setEnabled(busy)

    # ------------------------------------------------------------ user filter
    def _mask_context_key(self) -> tuple:
        """Identity of the data the cell mask was picked against.

        A mask is a set of ``(term, issuer)`` labels, so it only stays
        meaningful while the underlying panel and pre-filters do. Changing any
        of these invalidates it rather than silently cutting the new data with
        stale labels.
        """
        return (
            str(self._db_path),
            self.cmb_table.currentText(),
            self.cmb_date.currentText(),
            self._where_clause(),
            self.date_start.date().toPython(),
            self.date_end.date().toPython(),
        )

    def _reset_cell_mask(self) -> None:
        self._cell_mask = None
        self._cell_mask_key = None
        if hasattr(self, "lbl_cell_mask"):
            self.lbl_cell_mask.setText("")

    def _show_user_filter(self) -> None:
        """Open the term × issuer cell picker for the currently filtered data."""
        if self._db_path is None or self._panel is None or not self._panel.is_usable:
            QMessageBox.information(
                self,
                "User Filter",
                "Load a yield-curve panel table first.",
            )
            return
        try:
            config = self._build_config(with_mask=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self.lbl_status.setText("Loading panel for the User Filter…")
        try:
            long = load_long_panel(config, self._panel, apply_cell_mask=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))
            self.lbl_status.setText("User Filter: load failed.")
            return

        cells = available_cells(long, self._panel)
        if not cells:
            QMessageBox.information(
                self,
                "User Filter",
                "No data remains after the Optional Filter and date range, so "
                "there are no cells to choose from.",
            )
            self.lbl_status.setText("User Filter: nothing to show.")
            return

        key = self._mask_context_key()
        current = self._cell_mask if key == self._cell_mask_key else None
        dialog = UserFilterDialog(
            cells,
            current,
            term_label="Term",
            issuer_label=self._panel.issuer_column.title(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.lbl_status.setText("User Filter: unchanged.")
            return

        chosen = dialog.selected_cells()
        self._cell_mask = chosen
        self._cell_mask_key = key
        self._update_cell_mask_label(len(cells))
        self._append_log(f"User Filter: {len(chosen)} of {len(cells)} cells selected.")

    def _update_cell_mask_label(self, total: int | None = None) -> None:
        if self._cell_mask is None:
            self.lbl_cell_mask.setText("")
            return
        suffix = f" of {total}" if total is not None else ""
        self.lbl_cell_mask.setText(
            f"User Filter active: {len(self._cell_mask)}{suffix} cells."
        )

    # -------------------------------------------------------------- settings
    def _show_mln_settings(self) -> None:
        dialog = MLNSettingsDialog(self, initial_config=self._mln_config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._mln_config = dialog.get_config()
            self._append_log(
                f"MLN settings: centrality={self._mln_config.centrality}, "
                f"jaccard={self._mln_config.jaccard_threshold:.2f}, "
                f"method={self._mln_config.community_method.value}, "
                f"max_communities={self._mln_config.max_communities}"
            )

    def _show_evolution_settings(self) -> None:
        dialog = EvolutionSettingsDialog(self, initial_config=self._evolution_config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._evolution_config = dialog.get_config()
            self._append_log(
                f"Evolution settings: window={self._evolution_config.window_size}, "
                f"step={self._evolution_config.step_size}, "
                f"centrality={self._evolution_config.centrality}"
            )

    def _show_edge_settings(self) -> None:
        from ycn.gui.edge_settings_dialog import EdgeSettingsDialog

        dialog = EdgeSettingsDialog(self, initial_config=self._edge_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._edge_settings = dialog.get_config()
            self._append_log(f"Edge settings: {self._edge_settings.to_dict()}")

    # -------------------------------------------------------------------- run
    def _selected_transforms(self) -> list[str]:
        transforms: list[str] = []
        for i in range(self.lst_transforms.count()):
            item = self.lst_transforms.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                transforms.append(item.data(Qt.ItemDataRole.UserRole))
        return transforms

    def _network_kind(self) -> NetworkKind:
        return NetworkKind(self.cmb_network_kind.currentData())

    def _build_config(self, *, with_mask: bool = True) -> PipelineConfig:
        """Assemble the run configuration from the sidebar.

        ``name_column``/``value_column`` describe the *long* frame produced by
        ``yield_curve.load_long_panel``, which is what the MLN analysis
        consumes -- they are derived from NETWORK TYPE, not chosen by the user.
        """
        if self._db_path is None:
            raise ValueError("No database selected.")
        panel = self._panel
        if panel is None or not panel.is_usable:
            raise ValueError(
                "The selected table is not a yield-curve panel (needs an issuer "
                "column and at least two term columns)."
            )

        start = self.date_start.date().toPython()
        end = self.date_end.date().toPython()
        if isinstance(start, date) and isinstance(end, date) and start > end:
            raise ValueError("Start date must be on or before end date.")

        kind = self._network_kind()
        mask = None
        if with_mask and self._cell_mask is not None:
            if self._mask_context_key() == self._cell_mask_key:
                mask = sorted(self._cell_mask)
            else:
                self._append_log(
                    "User Filter cleared: the table, date range or Optional "
                    "Filter changed since the cells were picked."
                )
                self._reset_cell_mask()

        return PipelineConfig(
            db_path=self._db_path,
            table=self.cmb_table.currentText(),
            date_column=panel.date_column,
            name_column=panel.node_column(kind),
            value_column=panel.rate_column,
            where_clause=self._where_clause(),
            transforms=self._selected_transforms(),
            measure=self.cmb_measure.currentData(),
            date_start=start,
            date_end=end,
            independent_threshold=float(self.spin_threshold.value()),
            title=f"{self.cmb_measure.currentText()} ({kind.label})",
            issuer_column=panel.issuer_column,
            term_columns=list(panel.term_columns),
            network_kind=kind.value,
            cell_mask=mask,
        )

    def _run_mln(self) -> None:
        """Build the multi-layer network -- the only pipeline the GUI runs."""
        if self._busy:
            return
        try:
            config = self._build_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        panel = self._panel
        assert panel is not None  # guaranteed by _build_config
        kind = self._network_kind()
        layer_column = panel.layer_column(kind)

        if config.cell_mask is not None and not config.cell_mask:
            QMessageBox.warning(
                self,
                "No cells selected",
                "The User Filter has every cell unchecked, so there is no data "
                "to build from.",
            )
            return

        self._last_config = config

        # A *fresh* event per run. Clearing the shared one would resurrect a
        # worker discarded by a previous cancel, which would then keep running
        # and emit into the new run.
        self._cancel_event = threading.Event()

        self._set_busy(True)
        self.progress.setValue(0)
        self._last_progress_line = None
        self._current_measure_tag = measure_short_label(config.measure)
        self.process_log.clear()
        self.lbl_status.setText("Starting…")
        self._append_log(
            f"Building {kind.label}: layers from {layer_column!r}, "
            f"nodes from {config.name_column!r}."
        )
        if config.cell_mask is not None:
            self._append_log(
                f"User Filter: restricted to {len(config.cell_mask)} cells."
            )
        self._set_mln_building()
        self.tabs.setCurrentIndex(0)
        self._update_view_data_button()

        mln_config = MLNConfig(
            layer_column=layer_column,
            centrality=self._mln_config.centrality,
            jaccard_threshold=self._mln_config.jaccard_threshold,
            community_method=self._mln_config.community_method,
            max_communities=self._mln_config.max_communities,
            min_nodes=self._mln_config.min_nodes,
        )

        # Keep strong Python refs — a local worker is GC'd and the thread dies
        # before run() executes (see "QThread: Destroyed while thread is still running").
        self._mln_worker_thread = QThread(self)
        self._mln_worker = MLNWorker(
            config,
            mln_config,
            edge_settings=self._edge_settings.to_dict(),
            cancel_event=self._cancel_event,
            panel=panel,
        )
        self._mln_worker.moveToThread(self._mln_worker_thread)
        self._mln_worker_thread.started.connect(self._mln_worker.run)
        self._mln_worker.progress.connect(self._on_mln_progress)
        self._mln_worker.status.connect(self._on_mln_status)
        self._mln_worker.finished.connect(self._on_mln_finished)
        self._mln_worker.failed.connect(self._on_mln_failed)
        self._mln_worker.cancelled.connect(self._on_mln_cancelled)
        self._mln_worker.finished.connect(self._mln_worker_thread.quit)
        self._mln_worker.failed.connect(self._mln_worker_thread.quit)
        self._mln_worker.cancelled.connect(self._mln_worker_thread.quit)
        self._mln_worker_thread.start()

    # ------------------------------------------------------- worker lifecycle
    def _is_stale(self, current) -> bool:
        """True when a signal came from a worker we have already detached from.

        Cancel Render stops waiting for threads, so a discarded worker can still
        emit afterwards -- possibly while a *new* run is under way. Acting on
        those signals would overwrite the current run's results, or null its
        thread references (which crashes Qt when the live QThread is later
        garbage-collected).

        Identity comes from ``QObject.sender()`` rather than a bound lambda:
        connecting a lambda gives Qt no receiver object, so it invokes the slot
        DIRECTLY on the worker thread instead of queueing it to the GUI thread,
        and touching widgets from there corrupts the heap. Bound methods keep
        the connection queued, which is required for correctness here.

        ``_retired_workers`` keeps discarded senders alive, so sender() cannot
        dangle. Outside signal delivery it is None, which reads as "current" --
        the right answer for a handler invoked directly.
        """
        sender = self.sender()
        return sender is not None and sender is not current

    def _retire_worker(self, worker, thread) -> None:
        """Hold a detached worker/thread until the thread finishes.

        Cancel Render stops waiting for threads, so without this the last
        Python reference would drop while the thread is still running and Qt
        would delete the underlying C++ objects out from under it.
        """
        if worker is None and thread is None:
            return
        if thread is not None and not thread.isRunning():
            return
        entry = (worker, thread)
        self._retired_workers.append(entry)
        if thread is not None:
            thread.finished.connect(lambda e=entry: self._release_retired(e))

    def _release_retired(self, entry) -> None:
        """Drop a retired worker once its thread has actually exited."""
        try:
            self._retired_workers.remove(entry)
        except ValueError:
            pass

    def _cleanup_mln_worker(self) -> None:
        self._mln_worker = None
        self._mln_worker_thread = None

    def _cancel_render(self) -> None:
        """Cancel the current analysis and clear the MLN tabs.

        Setting ``_cancel_event`` first is what makes this non-blocking: it's
        checked inside the worker's progress/status callbacks (invoked on
        every pair), so the worker unwinds within one iteration rather than
        running to completion. ``QThread.quit()`` alone cannot interrupt a
        synchronous computation already in progress — it only requests the
        thread's event loop to exit once control returns to it.
        """
        self._append_log("Cancelling render...")
        self._cancel_event.set()

        # Give the worker a brief chance to unwind, then carry on regardless.
        # Workers only notice cancellation inside their progress callbacks, so a
        # thread that is momentarily between checkpoints would otherwise stall
        # the GUI here for the whole timeout. Detaching instead keeps the button
        # instant; the discarded thread exits on its next checkpoint and its
        # signals are ignored (see _is_stale).
        thread = self._mln_worker_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            if thread.wait(CANCEL_GRACE_MS):
                self._append_log("MLN cancelled.")
            else:
                self._append_log(
                    "MLN is finishing in the background; its results will "
                    "be discarded."
                )

        self._retire_worker(self._mln_worker, self._mln_worker_thread)
        self._cleanup_mln_worker()

        self._clear_mln_tabs("Render cancelled.")
        self.progress.setValue(0)
        self.lbl_status.setText("Render cancelled. Polars data retained.")
        self._set_busy(False)
        self._append_log("All networks and graphs cleared.")

    # ============================================================================
    # Multi-Layer Network (MLN) tabs
    # ============================================================================

    def _build_mln_page(self) -> QWidget:
        """The 'MLN' tab: 3D multiplex view + layer checklist + node table."""
        page = QFrame()
        page.setObjectName("Canvas")
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.mln_web = QWebEngineView()
        # Own the page so JS console output lands in the process log rather
        # than on stderr. Must precede setHtml and the web-channel wiring.
        self.mln_web.setPage(MLNWebPage(self.mln_web, on_message=self._on_js_message))
        self.mln_web.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.mln_web.setHtml(self._mln_placeholder_html())
        outer.addWidget(self.mln_web, stretch=4)

        side = QWidget()
        side.setMaximumWidth(240)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 6, 6, 6)
        side_layout.setSpacing(4)

        side_layout.addWidget(self._section("VISIBLE LAYERS"))
        self.lst_mln_layers = QListWidget()
        self.lst_mln_layers.setMaximumHeight(130)
        self.lst_mln_layers.itemChanged.connect(self._on_mln_layer_toggled)
        side_layout.addWidget(self.lst_mln_layers)

        side_layout.addWidget(self._section("NODES"))
        self.tbl_mln_nodes = QTableWidget(0, 2)
        self.tbl_mln_nodes.setHorizontalHeaderLabels(["Node", "Layer"])
        self.tbl_mln_nodes.verticalHeader().setVisible(False)
        self.tbl_mln_nodes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_mln_nodes.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        side_layout.addWidget(self.tbl_mln_nodes, stretch=1)
        outer.addWidget(side, stretch=1)

        # Click bridge: Plotly node clicks -> highlight the matching table row.
        self._mln_channel = QWebChannel(self.mln_web.page())
        self._mln_channel.registerObject("mlnBridge", self._mln_bridge)
        self.mln_web.page().setWebChannel(self._mln_channel)
        self._mln_bridge.node_clicked.connect(self._on_mln_node_clicked)
        return page

    @staticmethod
    def _mln_placeholder_html(
        message: str = "Configure the panel on the left, then click Build network.",
    ) -> str:
        return (
            "<!DOCTYPE html><html><body style='font-family:Segoe UI,sans-serif;"
            "display:flex;align-items:center;justify-content:center;height:100%;"
            "margin:0;background:#0f172a;color:#94a3b8'>"
            f"<p>{message}</p></body></html>"
        )

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        """Remove every widget from a canvas layout, closing old figures."""
        while layout.count():
            widget = layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

    def _mln_placeholder_label(self, message: str) -> QLabel:
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return label

    def _set_mln_metrics_placeholder(self, message: str | None = None) -> None:
        self._clear_layout(self.mln_metrics_layout)
        self.mln_metrics_layout.addWidget(
            self._mln_placeholder_label(
                message or "MLN metrics will appear here after you build a network."
            )
        )

    def _set_mln_community_placeholder(self, message: str | None = None) -> None:
        self._clear_layout(self.mln_community_layout)
        self.mln_community_layout.addWidget(
            self._mln_placeholder_label(
                message or "MLN communities will appear here after you build a network."
            )
        )

    def _set_mln_building(self) -> None:
        """Put the MLN tabs into their 'computing' state."""
        self._set_mln_metrics_placeholder("Computing multi-layer network...")
        self._set_mln_community_placeholder("Computing multi-layer network...")
        self.mln_web.setHtml(
            self._mln_placeholder_html("Computing multi-layer network...")
        )

    def _clear_mln_tabs(self, message: str | None = None) -> None:
        """Clear all three MLN tabs and drop any cached result."""
        text = message or "No multi-layer network yet."
        self._set_mln_metrics_placeholder(text)
        self._set_mln_community_placeholder(text)
        self.mln_web.setHtml(self._mln_placeholder_html(text))
        self.lst_mln_layers.blockSignals(True)
        self.lst_mln_layers.clear()
        self.lst_mln_layers.blockSignals(False)
        self.tbl_mln_nodes.setRowCount(0)
        self._mln_row_of = {}
        self._mln_result = None
        self._update_view_data_button()

    def _show_mln_metrics(self, fig: Figure) -> None:
        try:
            self._clear_layout(self.mln_metrics_layout)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.mln_metrics_layout.addWidget(canvas)
            canvas.draw()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"MLN metrics display error: {exc}")

    def _show_mln_community(self, fig: Figure) -> None:
        try:
            self._clear_layout(self.mln_community_layout)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.mln_community_layout.addWidget(canvas)
            canvas.draw()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"MLN community display error: {exc}")

    # ------------------------------------------------------------- MLN signals
    def _on_mln_progress(self, done: int, total: int, desc: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        if total <= 0:
            return
        pct = int(100 * done / total)
        self.progress.setValue(pct)
        if done not in (0, total) and done % max(1, total // 100) != 0:
            self.lbl_status.setText(f"MLN: {desc}")
            return
        bar_w = 24
        filled = int(bar_w * done / total)
        bar = "█" * filled + "░" * (bar_w - filled)
        self._append_log(
            f"{self._current_measure_tag} {bar} {pct:3d}% ({done}/{total})  {desc}",
            replace_last=True,
        )
        self.lbl_status.setText(f"MLN: {desc}")

    def _on_mln_status(self, message: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_mln_finished(self, result: object) -> None:
        if self._is_stale(self._mln_worker):
            return
        if not isinstance(result, MLNResult):
            self._on_mln_failed(f"Unexpected MLN result type: {type(result)!r}")
            return
        self._mln_result = result
        self._populate_mln_layer_list(result)
        self._populate_mln_table(result)
        self._render_mln_view()
        self._show_mln_metrics(result.metrics_fig)
        self._show_mln_community(result.community_fig)
        self.progress.setValue(100)
        self.lbl_status.setText("MLN ready.")
        self._append_log("MLN rendered.")
        self._cleanup_mln_worker()
        self._update_view_data_button()
        self._set_busy(False)

    def _on_mln_failed(self, message: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        self.progress.setValue(0)
        self._append_log(f"MLN ERROR: {message}")
        self.lbl_status.setText("Failed.")
        self._clear_mln_tabs(f"MLN failed: {message}")
        self._cleanup_mln_worker()
        self._set_busy(False)
        QMessageBox.critical(self, "MLN failed", message)

    def _on_mln_cancelled(self) -> None:
        """MLN worker unwound after a cancellation request.

        ``_cancel_render`` performs the UI cleanup, so this only drops the
        worker refs. It may fire slightly after ``_cancel_render`` returns (the
        signal is queued cross-thread), so it must be safe to run on an
        already-cleaned-up UI — no popups here, unlike ``_on_mln_failed``.
        """
        if self._is_stale(self._mln_worker):
            return
        self._cleanup_mln_worker()

    # ------------------------------------------------------- MLN interactivity
    def _populate_mln_layer_list(self, result: MLNResult) -> None:
        """Fill the visible-layers checklist (all layers checked initially)."""
        self.lst_mln_layers.blockSignals(True)
        self.lst_mln_layers.clear()
        for value in result.layer_values:
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.lst_mln_layers.addItem(item)
        self.lst_mln_layers.blockSignals(False)

    def _selected_mln_layers(self) -> list[str]:
        return [
            self.lst_mln_layers.item(i).text()
            for i in range(self.lst_mln_layers.count())
            if self.lst_mln_layers.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _populate_mln_table(self, result: MLNResult) -> None:
        """Fill the node table and the (node, layer) -> row index for O(1) lookup."""
        rows = result.nodes.select(["issuer", "term"]).rows()
        # Suspend repaints while filling: every setItem would otherwise schedule
        # a view update, which is the bulk of the cost for a few thousand rows.
        self.tbl_mln_nodes.setUpdatesEnabled(False)
        try:
            self.tbl_mln_nodes.setRowCount(len(rows))
            self.tbl_mln_nodes.setHorizontalHeaderLabels(
                [result.node_column, result.layer_column]
            )
            self._mln_row_of = {}
            for row_idx, (node, layer) in enumerate(rows):
                self.tbl_mln_nodes.setItem(row_idx, 0, QTableWidgetItem(str(node)))
                self.tbl_mln_nodes.setItem(row_idx, 1, QTableWidgetItem(str(layer)))
                self._mln_row_of[(str(node), str(layer))] = row_idx
            self.tbl_mln_nodes.resizeColumnsToContents()
        finally:
            self.tbl_mln_nodes.setUpdatesEnabled(True)

    def _on_mln_layer_toggled(self, _item) -> None:
        """Re-render the 3D view for the new layer selection.

        Cheap enough for the GUI thread: the multiplex tables are cached on the
        result, so this only re-filters and redraws -- nothing is recomputed.
        """
        self._render_mln_view()

    def _render_mln_view(self) -> None:
        """Draw the cached multiplex for the currently checked layers."""
        result = self._mln_result
        if result is None:
            return
        visible = self._selected_mln_layers()
        if not visible:
            self.mln_web.setHtml(self._mln_placeholder_html("No layers selected."))
            return
        try:
            nodes, intra, inter = filter_tables(
                result.nodes, result.intra, result.inter, visible
            )
            all_nodes = sorted(result.nodes.get_column("issuer").unique().to_list())
            fig = build_multiplex_figure(
                nodes,
                intra,
                inter,
                visible,
                all_issuers=all_nodes,
                title=(
                    f"{result.node_column} x {result.layer_column} "
                    "multi-layer network"
                ),
                layer_label=result.layer_column,
            )
            # 'directory' keeps each page ~50KB by referencing a plotly.min.js
            # written once alongside it; inlining would rewrite 4MB on every
            # layer toggle.
            html = inject_click_bridge(
                inject_canvas_shim(
                    fig.to_html(full_html=True, include_plotlyjs="directory")
                )
            )
            self._write_mln_html(html)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"MLN view render error: {exc}")

    def _mln_asset_dir(self) -> Path:
        """Temp dir holding the MLN page and its one-time plotly.min.js sidecar."""
        if self._mln_temp_dir is None:
            self._mln_temp_dir = Path(tempfile.mkdtemp(prefix="ycn_mln_"))
        js_path = self._mln_temp_dir / "plotly.min.js"
        if not js_path.exists():
            from plotly.offline import get_plotlyjs

            js_path.write_text(get_plotlyjs(), encoding="utf-8")
        return self._mln_temp_dir

    def _write_mln_html(self, html: str) -> None:
        """Write the MLN page next to its JS sidecar and load it."""
        directory = self._mln_asset_dir()
        handle = tempfile.NamedTemporaryFile(
            prefix="view_", suffix=".html", dir=str(directory), delete=False
        )
        handle.write(html.encode("utf-8"))
        handle.close()
        old = self._mln_temp_html
        self._mln_temp_html = Path(handle.name)
        self.mln_web.load(QUrl.fromLocalFile(str(self._mln_temp_html.resolve())))
        if old and old.exists():
            try:
                old.unlink()
            except OSError:
                pass

    def _on_mln_node_clicked(self, node: str, layer: str) -> None:
        """Select and scroll to the clicked node's row in the side table."""
        row = self._mln_row_of.get((node, layer))
        if row is None:
            return
        self.tbl_mln_nodes.selectRow(row)
        item = self.tbl_mln_nodes.item(row, 0)
        if item is not None:
            self.tbl_mln_nodes.scrollToItem(item)
        self.lbl_status.setText(f"MLN: selected {node} in {layer}")

    # ------------------------------------------------------------------ close
    def closeEvent(self, event) -> None:  # noqa: N802
        # Same fix as _cancel_render: set the flag before quit()/wait() so a
        # worker mid-computation unwinds instead of the 2s wait timing out.
        self._cancel_event.set()
        if self._mln_worker_thread is not None and self._mln_worker_thread.isRunning():
            self._mln_worker_thread.quit()
            self._mln_worker_thread.wait(2000)
        if self._mln_temp_dir and self._mln_temp_dir.exists():
            import shutil

            shutil.rmtree(self._mln_temp_dir, ignore_errors=True)
        try:
            self._data_cache.frame_cache.cache_container.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
