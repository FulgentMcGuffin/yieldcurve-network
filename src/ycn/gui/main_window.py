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
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPalette, QPixmap, QShortcut, QTextCursor
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

from ycn.analysis.af_models import NEURAL_AVAILABLE, NEURAL_IMPORT_ERROR
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
from ycn.analysis.degree_hist import render_degree_histogram
from ycn.analysis.evolution_viz import render_centrality_trajectories
from ycn.analysis.mln_evolution_viz import render_factor_evolution, render_stress_quadrants
from ycn.analysis.mln import MLNConfig
from ycn.analysis.mln_layer_metrics import layer_subgraph, metrics_for_layer
from ycn.analysis.session import (
    FILE_FILTER,
    SUFFIX,
    Session,
    SessionError,
    describe,
    load_session,
    save_session,
)
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
from ycn.gui.gui_settings_dialog import GuiSettingsDialog, THEMES
from ycn.gui.data_table_dialog import DataTableDialog, eye_icon
from ycn.gui.degree_threshold_slider import build_threshold_slider
from ycn.gui.factor_trajectory_tab import FactorTrajectoryTab
from ycn.gui.layer_figure_tabs import (
    LayerFigureTabs,
    attach_histogram_cursor,
    attach_line_cursor,
)
from ycn.gui.mln_bridge import (
    MLNBridge,
    MLNWebPage,
    inject_canvas_shim,
    inject_click_bridge,
)
from ycn.gui.mln_settings_dialog import MLNSettingsDialog
from ycn.gui.coverage_dialog import CoverageDialog
from ycn.gui.ns_residuals_tab import NSResidualsTab
from ycn.gui.session_io import (
    capture_evolution,
    capture_mln,
    capture_neural_evolution,
    capture_residual,
    restore_evolution,
    restore_mln,
    restore_neural_evolution,
    restore_residual,
    settings_summary,
)
from ycn.gui.stress_trajectory_tab import StressTrajectoryTab
from ycn.gui.styles import APP_STYLE, BG_SIDEBAR
from ycn.gui.user_filter_dialog import UserFilterDialog
from ycn.gui.workers import (
    MLNEvolutionResult,
    MLNEvolutionWorker,
    MLNResult,
    MLNWorker,
    NeuralEvolutionResult,
    NeuralEvolutionWorker,
    ResidualResult,
    ResidualWorker,
)

# Residual networks keep an edge on |corr| > this. Deliberately independent of
# the sidebar's independence threshold: that one thresholds a similarity
# measure in [0, 1], this one a signed correlation where the sign is noise.
RESIDUAL_THRESHOLD = 0.3

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
        self.setWindowIcon(self._create_network_icon())
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
        # GUI theme name; defaults to "Sky Blue" (the original theme)
        self._current_theme = "Sky Blue"
        self._mln_worker_thread: QThread | None = None
        self._mln_worker: MLNWorker | None = None
        self._mln_result: MLNResult | None = None
        self._mln_temp_html: Path | None = None
        # Shared by every embedded Plotly view (the MLN scene and both factor
        # trajectories): one directory, one 4MB plotly.min.js sidecar, one
        # cleanup in closeEvent.
        self._plotly_temp_dir: Path | None = None
        # Workers detached by Cancel Render, kept alive until their thread
        # actually exits. Dropping the last Python reference to a still-running
        # QThread/worker lets Qt delete the C++ object underneath it and crash.
        self._retired_workers: list[tuple] = []
        self._last_config: PipelineConfig | None = None
        self._mln_row_of: dict[tuple[str, str], int] = {}
        # layer -> (threshold, degrees) once its MLN: Degree slider has been
        # moved, so the eye button reports what is drawn rather than the
        # threshold the network was built at.
        self._degree_at_threshold: dict[str, tuple[float, dict]] = {}
        self._mln_bridge = MLNBridge()
        self._mln_channel: QWebChannel | None = None
        # Which layer values were checked in "VISIBLE LAYERS" last time, so a
        # rebuild (e.g. just to also tick Run Evolution) restores it instead of
        # silently re-checking everything. None means "no prior choice yet".
        self._visible_layers: set[str] | None = None

        # Residual and evolution stages run concurrently after the multiplex
        # lands. ``_active_stages`` is what keeps the busy state honest: the
        # run is only over once every launched stage has reported back.
        self._residual_worker: ResidualWorker | None = None
        self._residual_worker_thread: QThread | None = None
        self._residual_result: ResidualResult | None = None
        self._evolution_worker: MLNEvolutionWorker | None = None
        self._evolution_worker_thread: QThread | None = None
        self._evolution_result: MLNEvolutionResult | None = None
        self._neural_evolution_worker: NeuralEvolutionWorker | None = None
        self._neural_evolution_worker_thread: QThread | None = None
        self._neural_evolution_result: NeuralEvolutionResult | None = None
        self._active_stages: set[str] = set()
        # "Evo: Resids"/"Evo: Cov"/"Evo: Cov(t)" each carry a picker choosing
        # between the NS and Neural-HJM evolution results; they stay in sync
        # via the same _syncing-guard idiom used by the other synced combos.
        self._resid_source_pickers: list[QComboBox] = []
        self._syncing_resid_source: bool = False
        # Second, unlabelled picker on the same three tabs: "Average" (the
        # market-average result, existing behaviour) or a specific issuer's
        # own factor/stress evolution. Same sync idiom, populated once an
        # evolution result actually has per-issuer data.
        self._issuer_pickers: list[QComboBox] = []
        self._syncing_issuer_source: bool = False
        # True while a saved session is being applied, so signal handlers that
        # would recompute or invalidate the restored state can stand down.
        self._loading_settings = False

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

    @staticmethod
    def _create_network_icon() -> QIcon:
        """Create a network-themed icon for the application."""
        # Create a 64x64 pixmap with network graph visualization
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent

        from PySide6.QtGui import QPainter, QPen
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw network nodes (circles)
        node_color = QColor(0, 212, 255)  # Cyan accent
        node_radius = 5

        # Define node positions in a circular pattern
        nodes = [
            (32, 16),   # Top
            (48, 24),   # Top-right
            (52, 40),   # Right
            (48, 56),   # Bottom-right
            (32, 48),   # Bottom
            (16, 56),   # Bottom-left
            (12, 40),   # Left
            (16, 24),   # Top-left
            (32, 32),   # Center
        ]

        # Draw edges (lines between nodes)
        edge_color = QColor(100, 150, 200)
        pen = QPen(edge_color, 1)
        painter.setPen(pen)

        # Connect center to all outer nodes
        center = nodes[-1]
        for node in nodes[:-1]:
            painter.drawLine(center[0], center[1], node[0], node[1])

        # Connect adjacent outer nodes
        for i in range(len(nodes) - 1):
            painter.drawLine(nodes[i][0], nodes[i][1], nodes[(i + 1) % 8][0], nodes[(i + 1) % 8][1])

        # Draw nodes (circles)
        painter.setBrush(node_color)
        for node in nodes:
            painter.drawEllipse(node[0] - node_radius, node[1] - node_radius, node_radius * 2, node_radius * 2)

        painter.end()
        return QIcon(pixmap)

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

        self.btn_user_filter = QPushButton("▦ User Filter")
        self.btn_user_filter.setObjectName("SecondaryButton")
        self.btn_user_filter.clicked.connect(self._show_user_filter)
        self.btn_user_filter.setToolTip(
            "Pick the individual term × issuer cells that feed the network, "
            "from the data left after the Optional Filter and date range."
        )
        form.addWidget(self.btn_user_filter)

        # Kept directly under its button: this reports the User Filter's
        # selection, so it must not read as a caption for MLN Settings.
        self.lbl_cell_mask = QLabel("")
        self.lbl_cell_mask.setObjectName("StatusLabel")
        self.lbl_cell_mask.setWordWrap(True)
        form.addWidget(self.lbl_cell_mask)

        self.btn_mln_settings = QPushButton("⚙ MLN Settings")
        self.btn_mln_settings.setObjectName("SecondaryButton")
        self.btn_mln_settings.clicked.connect(self._show_mln_settings)
        self.btn_mln_settings.setToolTip(
            "Configure MLN centrality, community method and Jaccard threshold"
        )
        form.addWidget(self.btn_mln_settings)

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
        self.chk_evolution.setToolTip(
            "Rebuild the multiplex inside every rolling window and track its "
            "edge composition, community count, curve factors and correlation "
            "stress — for the market average and, additionally, for every "
            "issuer on its own (feeds the issuer picker on 'Evo: Resids'/"
            "'Evo: Cov'/'Evo: Cov(t)'). Much slower than the single multiplex "
            "— the four 'Evo:' tabs stay empty until this is ticked."
        )
        evolution_body.addWidget(self.chk_evolution)
        self.btn_evolution_settings = QPushButton("⚙ Evolution Settings")
        self.btn_evolution_settings.setObjectName("SecondaryButton")
        self.btn_evolution_settings.clicked.connect(self._show_evolution_settings)
        self.btn_evolution_settings.setToolTip(
            "Configure network evolution analysis parameters"
        )
        evolution_body.addWidget(self.btn_evolution_settings)
        self.chk_evolution.toggled.connect(self._on_evolution_checkbox_toggled)

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

        form.addWidget(self._section("ANALYSIS"))
        session_row = QHBoxLayout()
        session_row.setSpacing(4)
        self.btn_save_session = QPushButton("💾 Save…")
        self.btn_save_session.setObjectName("SecondaryButton")
        self.btn_save_session.setToolTip(
            "Save every rendered table and the settings that produced them "
            "to a single file (Ctrl+S)"
        )
        self.btn_save_session.clicked.connect(self._save_session)
        self.btn_save_session.setEnabled(False)
        session_row.addWidget(self.btn_save_session)

        self.btn_load_session = QPushButton("📂 Load…")
        self.btn_load_session.setObjectName("SecondaryButton")
        self.btn_load_session.setToolTip(
            "Reopen a saved analysis: its tabs and settings, without recomputing "
            "(Ctrl+O)"
        )
        self.btn_load_session.clicked.connect(self._load_session)
        session_row.addWidget(self.btn_load_session)
        form.addLayout(session_row)

        self.lbl_session = QLabel("")
        self.lbl_session.setObjectName("StatusLabel")
        self.lbl_session.setWordWrap(True)
        form.addWidget(self.lbl_session)

        self.btn_gui_settings = QPushButton("⚙ GUI Settings")
        self.btn_gui_settings.setObjectName("SecondaryButton")
        self.btn_gui_settings.setToolTip("Configure GUI appearance and themes")
        self.btn_gui_settings.clicked.connect(self._show_gui_settings)
        form.addWidget(self.btn_gui_settings)

        QShortcut(QKeySequence.StandardKey.Save, self, self._save_session)
        QShortcut(QKeySequence.StandardKey.Open, self, self._load_session)

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
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tabs.tabBar().setMinimumHeight(32)

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

        # Per-component-network views: one sub-tab per layer. Degree comes
        # straight off the multiplex edge tables, so it lands with the MLN
        # stage; centrality is a *trajectory* and therefore needs the
        # rolling-window pass, so it stays empty until Run Evolution is ticked.
        self.tab_mln_degree = LayerFigureTabs(
            empty_message=(
                "Per-layer degree histograms will appear here after you build "
                "a network."
            ),
            on_error=lambda msg: self._append_log(f"MLN: Degree — {msg}"),
            attach_cursor=attach_histogram_cursor,
            build_controls=self._build_degree_threshold_slider,
        )
        self.tab_mln_degree.currentChanged.connect(self._update_view_data_button)
        self.tabs.addTab(self.tab_mln_degree, "MLN: Degree")

        self.tab_ns = NSResidualsTab(self._show_coverage)
        self.tabs.addTab(self.tab_ns, "NS Residuals")

        # Per-layer centrality *trajectories*, so this belongs to the
        # rolling-window pass, not the single multiplex -- hence the "Evo:"
        # prefix and its own "Run Centrality" box in Evolution Settings.
        self.tab_evo_centrality = LayerFigureTabs(
            empty_message=(
                "Per-layer centrality trajectories are opt-in: tick “Run "
                "Evolution”, then “Run Centrality” in Evolution Settings."
            ),
            on_error=lambda msg: self._append_log(f"Evo: Centrality — {msg}"),
            attach_cursor=attach_line_cursor,
        )
        self.tab_evo_centrality.currentChanged.connect(self._update_view_data_button)
        self.tabs.addTab(self.tab_evo_centrality, "Evo: Centrality")

        self.evo_links_layout, links_page = self._figure_page()
        self.tabs.addTab(links_page, "Evo: Links")

        # "Evo: Resids" holds the two factor views as sub-tabs so they can be
        # read against each other without another top-level tab each. When
        # the Neural-HJM pass has also run, a picker above the sub-tabs lets
        # the user switch which model's factors these (and the two Cov tabs)
        # show; the picker only becomes selectable once that result exists.
        self.tabs_evo_resids = QTabWidget()
        self.tabs_evo_resids.setObjectName("ResultTabs")
        self.evo_factor_layout, factor_page = self._figure_page()
        self.evo_factor_std_layout, factor_std_page = self._figure_page()
        self.tabs_evo_resids.addTab(factor_page, "Factor")
        self.tabs_evo_resids.addTab(factor_std_page, "Factor Std")

        # The same numbers as the two static sub-tabs, but plotted as one path
        # through level x slope x curvature space instead of three series
        # against time -- the joint position is what carries the curve shape.
        # Interactive (Plotly in a web view), like the MLN tab.
        self.tab_factor_t = FactorTrajectoryTab(
            self._plotly_asset_dir, std=False, on_message=self._on_js_message
        )
        self.tab_factor_std_t = FactorTrajectoryTab(
            self._plotly_asset_dir, std=True, on_message=self._on_js_message
        )
        self.tabs_evo_resids.addTab(self.tab_factor_t, "Factor (t)")
        self.tabs_evo_resids.addTab(self.tab_factor_std_t, "Factor Std (t)")
        self.tabs_evo_resids.currentChanged.connect(self._update_view_data_button)
        self.tabs.addTab(
            self._wrap_with_source_picker(self.tabs_evo_resids), "Evo: Resids"
        )

        self.evo_cov_layout, cov_page = self._figure_page()
        self.tabs.addTab(self._wrap_with_source_picker(cov_page), "Evo: Cov")

        self.tab_cov_t = StressTrajectoryTab()
        cov_t_pickers = QWidget()
        cov_t_pickers_row = QHBoxLayout(cov_t_pickers)
        cov_t_pickers_row.setContentsMargins(0, 0, 0, 0)
        cov_t_pickers_row.setSpacing(6)
        cov_t_pickers_row.addWidget(self._make_source_picker())
        cov_t_pickers_row.addWidget(self._make_issuer_picker())
        self.tab_cov_t.add_toolbar_widget(cov_t_pickers)
        self.tabs.addTab(self.tab_cov_t, "Evo: Cov(t)")

        self._clear_evolution_tabs()

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

    def _figure_page(self) -> tuple[QVBoxLayout, QWidget]:
        """A themed page whose only content is one swappable FigureCanvas."""
        page = QFrame()
        page.setObjectName("Canvas")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)
        return layout, page

    def _show_figure(self, layout: QVBoxLayout, fig: Figure) -> None:
        """Replace whatever is in ``layout`` with a canvas for ``fig``."""
        try:
            self._clear_layout(layout)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            layout.addWidget(canvas)
            canvas.draw()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Figure display error: {exc}")

    # ---------------------------------------------------- residual source picker
    def _make_source_picker(self) -> QComboBox:
        """A combo choosing NS vs. Neural-HJM evolution results.

        "Neural-HJM resids" starts disabled (there is nothing to show until
        that stage has actually run) and every instance across the three
        affected tabs stays in sync via ``_on_resid_source_changed``.
        """
        combo = QComboBox()
        combo.addItem("NS Resids", "ns")
        combo.addItem("Neural-HJM resids", "neural")
        combo.model().item(1).setEnabled(False)
        combo.setToolTip(
            "Choose which curve model's factor/stress evolution these tabs "
            "show. Neural-HJM resids only becomes available after a run with "
            "both 'Run Evolution' and 'Run Neural-HJM' ticked."
        )
        combo.currentIndexChanged.connect(
            lambda _i, c=combo: self._on_resid_source_changed(c)
        )
        self._resid_source_pickers.append(combo)
        return combo

    def _make_issuer_picker(self) -> QComboBox:
        """A second, unlabelled combo choosing "Average" vs. one issuer.

        Sits to the right of the model-source picker on the same three tabs,
        no label of its own -- the model picker's "Show:" label already reads
        naturally across both. Starts with only "Average"; per-issuer items
        are added by :meth:`_populate_issuer_picker` once an evolution result
        actually has per-issuer data. Stays in sync across all three tabs via
        the same guard idiom as the model-source picker.
        """
        combo = QComboBox()
        combo.addItem("Average", "average")
        combo.setToolTip(
            "Choose the market-average result (existing behaviour) or a "
            "single issuer's own factor/stress evolution. Per-issuer entries "
            "appear once 'Run Evolution' has computed them."
        )
        combo.currentIndexChanged.connect(
            lambda _i, c=combo: self._on_issuer_source_changed(c)
        )
        self._issuer_pickers.append(combo)
        return combo

    def _wrap_with_source_picker(self, content: QWidget) -> QWidget:
        """Stack a source-picker row above an existing tab page."""
        page = QFrame()
        page.setObjectName("Canvas")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(8, 6, 8, 0)
        row.setSpacing(6)
        label = QLabel("Show:")
        label.setStyleSheet("color: #94a3b8;")
        row.addWidget(label)
        row.addWidget(self._make_source_picker())
        row.addWidget(self._make_issuer_picker())
        row.addStretch(1)
        outer.addLayout(row)
        outer.addWidget(content, stretch=1)
        return page

    def _on_resid_source_changed(self, changed: QComboBox) -> None:
        if self._syncing_resid_source:
            return
        source = changed.currentData()
        self._syncing_resid_source = True
        try:
            for combo in self._resid_source_pickers:
                if combo is not changed:
                    index = combo.findData(source)
                    if index >= 0:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(index)
                        combo.blockSignals(False)
        finally:
            self._syncing_resid_source = False
        self._render_resid_source()

    def _on_issuer_source_changed(self, changed: QComboBox) -> None:
        if self._syncing_issuer_source:
            return
        issuer = changed.currentData()
        self._syncing_issuer_source = True
        try:
            for combo in self._issuer_pickers:
                if combo is not changed:
                    index = combo.findData(issuer)
                    if index >= 0:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(index)
                        combo.blockSignals(False)
        finally:
            self._syncing_issuer_source = False
        self._render_resid_source()

    def _active_resid_source(self) -> str:
        if not self._resid_source_pickers:
            return "ns"
        return self._resid_source_pickers[0].currentData()

    def _active_issuer_source(self) -> str:
        if not self._issuer_pickers:
            return "average"
        return self._issuer_pickers[0].currentData() or "average"

    def _set_neural_source_available(self, available: bool) -> None:
        """Enable/disable "Neural-HJM resids" on every picker, and fall back."""
        for combo in self._resid_source_pickers:
            combo.model().item(1).setEnabled(available)
        if not available:
            for combo in self._resid_source_pickers:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
        self._render_resid_source()

    def _populate_issuer_picker(self) -> None:
        """(Re)build the issuer entries from whichever evolution result(s) exist.

        Takes the union across both the NS and Neural-HJM results so an
        issuer that succeeded under one model but not the other still shows
        up -- switching the model picker to it then falls back to "no data"
        via the same empty-frame handling ``render_factor_evolution``/
        ``render_stress_quadrants`` already do, rather than the issuer
        vanishing from the list. Keeps the current selection if it is still
        present, otherwise falls back to "Average".
        """
        issuers: set[str] = set()
        for result in (self._evolution_result, self._neural_evolution_result):
            if result is not None:
                issuers.update(result.issuer_factors)
                issuers.update(result.issuer_stress)
        ordered = sorted(issuers)

        current = self._active_issuer_source()
        for combo in self._issuer_pickers:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Average", "average")
            for issuer in ordered:
                combo.addItem(issuer, issuer)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)
        self._render_resid_source()

    def _render_resid_source(self) -> None:
        """Draw whichever evolution result + issuer selection the pickers choose."""
        source = self._active_resid_source()
        result = (
            self._neural_evolution_result
            if source == "neural"
            else self._evolution_result
        )
        if result is None:
            return

        issuer = self._active_issuer_source()
        if issuer == "average":
            factors, regimes, stress = result.factors, result.regimes, result.stress
            factor_fig = result.factor_fig
            factor_std_fig = result.factor_std_fig
            stress_fig = result.stress_fig
        else:
            # Rendered on demand rather than pre-rendered on the worker
            # thread: a run usually only looks at a handful of issuers, so
            # eagerly drawing a figure per issuer per model would pay for
            # every issuer regardless. Both renderers already turn an empty
            # frame (issuer missing under this model, or its fit failed) into
            # a themed placeholder, not an error.
            factors = result.issuer_factors.get(issuer, pl.DataFrame())
            regimes = result.issuer_regimes.get(issuer, pl.DataFrame())
            stress = result.issuer_stress.get(issuer, pl.DataFrame())
            factor_fig = render_factor_evolution(factors, regimes, std=False)
            factor_std_fig = render_factor_evolution(factors, regimes, std=True)
            stress_fig = render_stress_quadrants(stress)

        self._show_figure(self.evo_factor_layout, factor_fig)
        self._show_figure(self.evo_factor_std_layout, factor_std_fig)
        self._show_figure(self.evo_cov_layout, stress_fig)
        # The two 3D sub-tabs read the same frames as the static ones, so the
        # picker moves all four together.
        self.tab_factor_t.set_result(factors, regimes)
        self.tab_factor_std_t.set_result(factors, regimes)
        self.tab_cov_t.set_result(stress)
        self._update_view_data_button()

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
        self._refresh_save_enabled()

    def _tab_dataframe(self) -> tuple[str, pl.DataFrame] | None:
        """Return (tab title, frame) for the selected tab, or None if not rendered.

        Every tab that renders something also exposes the frame behind it, so
        the eye button is never a dead control on a populated tab.
        """
        title = self.tabs.tabText(self.tabs.currentIndex())

        if title in ("MLN", "MLN: Metrics", "MLN: Community"):
            if self._mln_result is None:
                return None
            if title == "MLN":
                return title, self._mln_edge_frame(self._mln_result)
            if title == "MLN: Metrics":
                return title, self._mln_result.centrality_df
            return title, self._mln_result.community_df

        if title == "MLN: Degree":
            layer = self.tab_mln_degree.current_layer()
            if self._mln_result is None or layer is None:
                return None
            # Prefer what the threshold slider last drew: the table must match
            # the histogram on screen, not the threshold the network was built
            # at.
            moved = self._degree_at_threshold.get(layer)
            if moved is not None:
                threshold, degrees = moved
                label = f"{title} — {layer} (threshold {threshold:.2f})"
            else:
                graph = layer_subgraph(
                    self._mln_result.nodes, self._mln_result.intra, layer
                )
                degrees = dict(graph.degree())
                label = f"{title} — {layer}"
            if not degrees:
                return None
            return label, pl.DataFrame(
                {
                    "node": [str(n) for n in degrees],
                    "degree": [int(d) for d in degrees.values()],
                }
            ).sort("degree", descending=True)

        if title == "Evo: Centrality":
            layer = self.tab_evo_centrality.current_layer()
            evo = self._evolution_result
            if evo is None or layer is None:
                return None
            frame = metrics_for_layer(evo.layer_metrics, layer)
            if frame.is_empty():
                return None
            return f"{title} — {layer}", frame

        if title == "NS Residuals":
            residual = self._residual_result
            if residual is None or residual.metrics.is_empty():
                return None
            return title, residual.metrics

        evolution = self._evolution_result
        neural = self._neural_evolution_result

        if title == "Evo: Links":
            if evolution is None:
                return None
            return title, self._links_frame(evolution)

        if title in ("Evo: Resids", "Evo: Cov", "Evo: Cov(t)"):
            source = self._active_resid_source()
            result = neural if source == "neural" else evolution
            if result is None:
                return None
            model_label = "Neural-HJM resids" if source == "neural" else "NS Resids"
            issuer = self._active_issuer_source()
            if issuer == "average":
                factors, regimes, stress = result.factors, result.regimes, result.stress
            else:
                factors = result.issuer_factors.get(issuer, pl.DataFrame())
                regimes = result.issuer_regimes.get(issuer, pl.DataFrame())
                stress = result.issuer_stress.get(issuer, pl.DataFrame())
            label = f"{model_label} — {issuer if issuer != 'average' else 'Average'}"
            if title == "Evo: Resids":
                frame = self._factors_frame(factors, regimes)
                sub = self.tabs_evo_resids.tabText(self.tabs_evo_resids.currentIndex())
                return f"{title} — {label} — {sub}", frame
            if stress.is_empty():
                return None
            return f"{title} — {label}", stress
        return None

    @staticmethod
    def _links_frame(result: MLNEvolutionResult) -> pl.DataFrame:
        """Edge composition per window, widened with each method's chosen k.

        The tab shows both panels, so the data behind it is both -- joined on
        the window rather than offered as two separate tables.
        """
        edges = result.edge_types
        if edges.is_empty() or result.community_k.is_empty():
            return edges
        wide = result.community_k.pivot(
            on="method", index="window_idx", values="n_clusters"
        )
        renamed = {c: f"k_{c}" for c in wide.columns if c != "window_idx"}
        return edges.join(wide.rename(renamed), on="window_idx", how="left")

    @staticmethod
    def _factors_frame(factors: pl.DataFrame, regimes: pl.DataFrame) -> pl.DataFrame:
        """Factor means and volatilities per window, with the regime label.

        Takes the frames directly rather than a result object: the caller may
        be reading the market ``.factors``/``.regimes`` or a per-issuer pair
        out of ``.issuer_factors``/``.issuer_regimes``, and this does not need
        to know which.
        """
        if factors.is_empty() or regimes.is_empty():
            return factors
        return factors.join(
            regimes.select(["window_idx", "regime"]),
            on="window_idx",
            how="left",
        )

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
        # Both this and _on_date_column_changed reset the cell mask and re-infer
        # the panel, which is exactly what a session restore must not trigger.
        if self._loading_settings or not self._db_path or not table:
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
        self._visible_layers = None
        self._refresh_panel()
        self._guess_date_range()
        self._append_log(f"Selected table {table!r} ({len(columns)} columns).")

    def _on_date_column_changed(self, _text: str) -> None:
        """The date column defines which columns remain available as terms."""
        if self._loading_settings:
            return
        self._reset_cell_mask()
        self._visible_layers = None
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
        self.chk_evolution.setEnabled(enabled)
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
    # Field names for _mask_context_key(), in the same order -- used only to
    # turn a mismatch into a readable diff (see _describe_mask_key_mismatch).
    _MASK_KEY_FIELDS = ("db_path", "table", "date_column")

    def _mask_context_key(self) -> tuple:
        """Identity of the *label space* the cell mask's (term, issuer) pairs
        were picked from.

        Deliberately narrow: only the database, table and date column decide
        what a "term" or "issuer" label even means, so only those invalidate
        the mask. The date range and the Optional Filter's WHERE clause change
        which labels currently have data, not what the labels mean -- so
        instead of resetting the whole mask, ``filter_by_cell_mask`` is a
        best-effort intersection: a picked pair that lost its data simply
        contributes nothing (never an error), and the User Filter dialog shows
        any newly-available pair as an unchecked, pickable cell rather than
        silently including it. Resetting on every date-range tweak used to
        make a picked selection feel like it "randomly" reverted to
        everything whenever the range moved even slightly.
        """
        return (
            str(self._db_path),
            self.cmb_table.currentText(),
            self.cmb_date.currentText(),
        )

    def _describe_mask_key_mismatch(self, old: tuple | None, new: tuple) -> str:
        """Which field(s) changed between two mask-context keys, for the log.

        The plain "settings changed" message doesn't say *what* changed, which
        has made a handful of unexplained resets hard to pin down. This turns
        a mismatch into something a bug report can actually be diagnosed from.
        """
        if old is None:
            return "no cells were picked against a stored context yet"
        if len(old) != len(self._MASK_KEY_FIELDS):
            return f"old={old!r} new={new!r}"
        diffs = [
            f"{name} {o!r} -> {n!r}"
            for name, o, n in zip(self._MASK_KEY_FIELDS, old, new)
            if o != n
        ]
        return (
            "; ".join(diffs)
            if diffs
            else "(keys compare unequal but no field differs?!)"
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

    def _sync_evolution_from_sidebar(self) -> None:
        """Copy the sidebar's live values onto the evolution config.

        The edge threshold lives on the sidebar spinbox but is *shown* by the
        Evolution Settings dialog, which reads it off this config. Both the
        dialog and the launcher call this, so the displayed value can never
        drift from the one the run will actually use.
        """
        self._evolution_config.independent_threshold = float(
            self.spin_threshold.value()
        )
        if self._last_config is not None:
            self._evolution_config.measure = self._last_config.measure
        # "Run Centrality" rides on the evolution window loop, so it cannot
        # survive that pass being switched off.
        if not self.chk_evolution.isChecked():
            self._evolution_config.run_centrality = False

    def _show_evolution_settings(self) -> None:
        self._sync_evolution_from_sidebar()
        dialog = EvolutionSettingsDialog(
            self,
            initial_config=self._evolution_config,
            independent_threshold=float(self.spin_threshold.value()),
            evolution_enabled=self.chk_evolution.isChecked(),
            neural_available=NEURAL_AVAILABLE,
            neural_import_error=NEURAL_IMPORT_ERROR,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._evolution_config = dialog.get_config()
            self._append_log(
                f"Evolution settings: window={self._evolution_config.window_size}, "
                f"step={self._evolution_config.step}, "
                f"expanding={self._evolution_config.expanding}, "
                f"edge threshold={self._evolution_config.independent_threshold:.2f}, "
                f"centrality={self._evolution_config.centrality}, "
                f"run_centrality={self._evolution_config.run_centrality}, "
                f"run_neural_hjm={self._evolution_config.run_neural_hjm}"
            )

    def _on_evolution_checkbox_toggled(self, checked: bool) -> None:
        """Clear evolution-dependent settings when "Run Evolution" is unchecked."""
        if not checked:
            self._evolution_config.run_centrality = False
            self._evolution_config.run_neural_hjm = False

    def _show_edge_settings(self) -> None:
        from ycn.gui.edge_settings_dialog import EdgeSettingsDialog

        dialog = EdgeSettingsDialog(self, initial_config=self._edge_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._edge_settings = dialog.get_config()
            self._append_log(f"Edge settings: {self._edge_settings.to_dict()}")

    def _show_gui_settings(self) -> None:
        """Show GUI settings dialog and apply selected theme."""
        dialog = GuiSettingsDialog(self, initial_theme=self._current_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_theme = dialog.get_theme_name()
            if new_theme != self._current_theme:
                self._current_theme = new_theme
                self._apply_theme(new_theme)
                self._append_log(f"Applied theme: {new_theme}")

    def _apply_theme(self, theme_name: str) -> None:
        """Apply a named theme to the entire application."""
        theme = GuiSettingsDialog.get_theme(theme_name)
        if theme is None:
            return
        # Update global stylesheet with theme colors
        theme_css = self._generate_theme_css(theme)
        self.setStyleSheet(theme_css)

    @staticmethod
    def _generate_theme_css(theme) -> str:
        """Generate CSS stylesheet from theme definition."""
        # Base style from APP_STYLE with theme color overrides
        base_css = """
        QWidget { color: #ffffff; }
        QMainWindow, QDialog, QFrame { background-color: %(bg_primary)s; }
        QLabel { color: %(text_primary)s; }
        QPushButton, QToolButton { background-color: %(accent)s; color: #ffffff;
                                    border: 1px solid %(border)s; border-radius: 4px;
                                    padding: 6px 12px; }
        QPushButton:pressed { background-color: %(input_bg)s; }
        QPushButton:hover { background-color: %(accent)s; opacity: 0.8; }
        QPushButton:disabled { background-color: %(input_bg)s; color: %(text_secondary)s; }
        QCheckBox { color: %(text_primary)s; }
        QComboBox { background-color: %(input_bg)s; color: %(text_primary)s;
                   border: 1px solid %(border)s; border-radius: 3px; }
        QSpinBox, QDoubleSpinBox { background-color: %(input_bg)s; color: %(text_primary)s;
                                   border: 1px solid %(border)s; border-radius: 3px; }
        QLineEdit, QPlainTextEdit { background-color: %(input_bg)s; color: %(text_primary)s;
                                   border: 1px solid %(border)s; border-radius: 3px; }
        QTabWidget::pane { border: 1px solid %(border)s; }
        QTabBar::tab { background-color: %(bg_sidebar)s; color: %(text_primary)s;
                       padding: 6px 16px; border: 1px solid %(border)s; }
        QTabBar::tab:selected { background-color: %(accent)s; color: #ffffff; }
        QTableWidget { background-color: %(input_bg)s; color: %(text_primary)s;
                      gridline-color: %(border)s; }
        QTableWidget::item { padding: 4px; }
        QHeaderView::section { background-color: %(bg_sidebar)s; color: %(text_primary)s;
                              padding: 4px; border: 1px solid %(border)s; }
        QScrollBar:vertical { background-color: %(bg_sidebar)s; width: 12px; }
        QScrollBar::handle:vertical { background-color: %(accent)s; border-radius: 6px; }
        #StatusLabel { color: %(text_secondary)s; }
        #SidebarFrame { background-color: %(bg_sidebar)s; }
        #SecondaryButton { background-color: %(accent)s; color: #ffffff; }
        #SecondaryButton:hover { background-color: %(border)s; }
        """
        return base_css % {
            "bg_primary": theme.bg_primary,
            "bg_sidebar": theme.bg_sidebar,
            "text_primary": theme.text_primary,
            "text_secondary": theme.text_secondary,
            "accent": theme.accent,
            "border": theme.border,
            "input_bg": theme.input_bg,
        }

    # -------------------------------------------------------------------- run
    def _selected_transforms(self) -> list[str]:
        transforms: list[str] = []
        for i in range(self.lst_transforms.count()):
            item = self.lst_transforms.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                transforms.append(item.data(Qt.ItemDataRole.UserRole))
        return transforms

    # ------------------------------------------------------- save / load
    def _collect_settings(self) -> dict:
        """Every sidebar value needed to describe and restore a run."""
        kind = self._network_kind()
        panel = self._panel
        return {
            "db_path": str(self._db_path) if self._db_path else "",
            "table": self.cmb_table.currentText(),
            "date_column": self.cmb_date.currentText(),
            "network_kind": kind.value,
            "network_kind_label": kind.label,
            "issuer_column": panel.issuer_column if panel else "",
            "term_columns": list(panel.term_columns) if panel else [],
            "where_enabled": self.chk_filter_where.isChecked(),
            "where_clause": self.txt_filter_where.text(),
            "date_start": self.date_start.date().toPython().isoformat(),
            "date_end": self.date_end.date().toPython().isoformat(),
            "transforms": self._selected_transforms(),
            "measure": self.cmb_measure.currentData(),
            "measure_label": self.cmb_measure.currentText(),
            "independent_threshold": float(self.spin_threshold.value()),
            "cell_mask": (
                sorted(self._cell_mask) if self._cell_mask is not None else None
            ),
            "edge_settings": self._edge_settings.to_dict(),
            "mln": {
                "centrality": self._mln_config.centrality,
                "jaccard_threshold": self._mln_config.jaccard_threshold,
                "community_method": self._mln_config.community_method.value,
                "max_communities": self._mln_config.max_communities,
                "min_nodes": self._mln_config.min_nodes,
                "degree_bins": self._mln_config.degree_bins,
                "centrality_top_n": self._mln_config.centrality_top_n,
            },
            "evolution": {
                "enabled": self.chk_evolution.isChecked(),
                "run_neural_hjm": self._evolution_config.run_neural_hjm,
                "window_size": self._evolution_config.window_size,
                "step": self._evolution_config.step,
                "expanding": self._evolution_config.expanding,
                "min_nodes": self._evolution_config.min_nodes,
                "centrality": self._evolution_config.centrality,
                "n_top_nodes": self._evolution_config.n_top_nodes,
                "max_communities": self._evolution_config.max_communities,
                "community_method": self._evolution_config.community_method.value,
                "run_centrality": self._evolution_config.run_centrality,
            },
        }

    def _apply_settings(self, settings: dict) -> None:
        """Put a saved run's settings back on the sidebar.

        Signals stay blocked throughout: restoring the table would otherwise
        re-run role inference and wipe the very cell mask being restored.
        Nothing here recomputes -- the loaded results are the results.
        """
        self._loading_settings = True
        try:
            db_path = settings.get("db_path") or ""
            self._db_path = Path(db_path) if db_path else None
            self.lbl_db.setText(db_path or "No database selected")

            for combo, value in (
                (self.cmb_table, settings.get("table", "")),
                (self.cmb_date, settings.get("date_column", "")),
            ):
                combo.blockSignals(True)
                if value and combo.findText(value) < 0:
                    combo.addItem(value)
                if value:
                    combo.setCurrentText(value)
                combo.blockSignals(False)

            kind = settings.get("network_kind")
            index = self.cmb_network_kind.findData(kind)
            if index >= 0:
                self.cmb_network_kind.blockSignals(True)
                self.cmb_network_kind.setCurrentIndex(index)
                self.cmb_network_kind.blockSignals(False)

            issuer = settings.get("issuer_column", "")
            terms = list(settings.get("term_columns", []))
            if issuer and len(terms) >= 2:
                self._panel = CurvePanel(
                    date_column=settings.get("date_column", "date"),
                    issuer_column=issuer,
                    term_columns=tuple(terms),
                )
                preview = ", ".join(terms[:6])
                if len(terms) > 6:
                    preview += f", … (+{len(terms) - 6})"
                self.lbl_panel.setText(
                    f"Issuer column: {issuer} · {len(terms)} terms: {preview}"
                )

            self.chk_filter_where.blockSignals(True)
            self.chk_filter_where.setChecked(bool(settings.get("where_enabled")))
            self.chk_filter_where.blockSignals(False)
            self.txt_filter_where.setText(settings.get("where_clause", ""))
            self._on_where_toggled()

            for widget, key in (
                (self.date_start, "date_start"),
                (self.date_end, "date_end"),
            ):
                value = settings.get(key)
                if value:
                    widget.setDate(QDate.fromString(str(value), "yyyy-MM-dd"))

            wanted = set(settings.get("transforms", []))
            for i in range(self.lst_transforms.count()):
                item = self.lst_transforms.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if item.data(Qt.ItemDataRole.UserRole) in wanted
                    else Qt.CheckState.Unchecked
                )

            measure = settings.get("measure")
            index = self.cmb_measure.findData(measure)
            if index >= 0:
                self.cmb_measure.setCurrentIndex(index)
            self.spin_threshold.setValue(
                float(settings.get("independent_threshold", 0.33))
            )

            mask = settings.get("cell_mask")
            if mask is None:
                self._reset_cell_mask()
            else:
                self._cell_mask = {(str(t), str(i)) for t, i in mask}
                self._cell_mask_key = self._mask_context_key()
                self._update_cell_mask_label()

            from ycn.gui.edge_settings_dialog import EdgeSettingsConfig

            edge = settings.get("edge_settings") or {}
            if edge:
                try:
                    self._edge_settings = EdgeSettingsConfig(**edge)
                except TypeError:
                    # An archive from a build with different edge settings:
                    # keep the defaults rather than refusing to open it.
                    self._append_log(
                        "Session: edge settings from this archive are not "
                        "recognised by this build; defaults kept."
                    )

            mln = settings.get("mln") or {}
            if mln:
                self._mln_config = MLNConfig(
                    layer_column="",
                    centrality=mln.get("centrality", "eigenvector"),
                    jaccard_threshold=float(mln.get("jaccard_threshold", 0.6)),
                    community_method=mln.get("community_method", "fixed"),
                    max_communities=int(mln.get("max_communities", 10)),
                    min_nodes=int(mln.get("min_nodes", 3)),
                    degree_bins=int(mln.get("degree_bins", 15)),
                    centrality_top_n=int(mln.get("centrality_top_n", 10)),
                )

            evo = settings.get("evolution") or {}
            if evo:
                self._evolution_config = EvolutionConfig(
                    window_size=int(evo.get("window_size", 252)),
                    step=int(evo.get("step", 21)),
                    expanding=bool(evo.get("expanding", False)),
                    min_nodes=int(evo.get("min_nodes", 5)),
                    centrality=evo.get("centrality", "eigenvector"),
                    n_top_nodes=int(evo.get("n_top_nodes", 10)),
                    max_communities=int(evo.get("max_communities", 10)),
                    community_method=evo.get("community_method", "fixed"),
                    independent_threshold=float(
                        settings.get("independent_threshold", 0.33)
                    ),
                    run_centrality=bool(evo.get("run_centrality", False)),
                    run_neural_hjm=bool(evo.get("run_neural_hjm", False)) and NEURAL_AVAILABLE,
                )
                self.chk_evolution.blockSignals(True)
                self.chk_evolution.setChecked(bool(evo.get("enabled")))
                self.chk_evolution.blockSignals(False)
        finally:
            self._loading_settings = False

    def _save_session(self) -> None:
        """Write every rendered table plus its settings to one archive."""
        if not self._has_results():
            QMessageBox.information(
                self,
                "Nothing to save",
                "Build a network first — there are no results to save yet.",
            )
            return

        suggested = (
            Path.home() / f"{self.cmb_table.currentText() or 'analysis'}{SUFFIX}"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save analysis", str(suggested), FILE_FILTER
        )
        if not path:
            return

        session = Session(settings=self._collect_settings())
        if self._mln_result is not None:
            capture_mln(session, self._mln_result)
        if self._residual_result is not None:
            capture_residual(session, self._residual_result)
        if self._evolution_result is not None:
            capture_evolution(session, self._evolution_result)
        if self._neural_evolution_result is not None:
            capture_neural_evolution(session, self._neural_evolution_result)

        try:
            written = save_session(path, session)
        except SessionError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            self._append_log(f"Session save failed: {exc}")
            return

        size_mb = written.stat().st_size / 1e6
        self.lbl_session.setText(f"Saved: {written.name} ({size_mb:.1f} MB)")
        self.lbl_status.setText("Analysis saved.")
        self._append_log(
            f"Session saved to {written} ({size_mb:.1f} MB): {describe(session)}"
        )

    def _load_session(self) -> None:
        """Reopen a saved analysis into the tabs and the sidebar."""
        if self._busy:
            QMessageBox.information(
                self,
                "Build in progress",
                "Wait for the current build to finish, or press Cancel Render, "
                "before loading a saved analysis.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Load analysis", str(Path.home()), FILE_FILTER
        )
        if not path:
            return
        try:
            session = load_session(path)
        except SessionError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self._append_log(f"Session load failed: {exc}")
            return

        self.process_log.clear()
        self._append_log(f"Loading {Path(path).name} (saved {session.saved_at})…")
        self._apply_settings(session.settings)

        # Clear everything first: a session that lacks a stage must not leave
        # the previous run's tabs on screen pretending to belong to it.
        self._clear_mln_tabs("Not stored in this analysis.")
        self.tab_ns.set_placeholder("Not stored in this analysis.")
        self._clear_evolution_tabs("Not stored in this analysis.")

        self._mln_result = restore_mln(session)
        if self._mln_result is not None:
            self._populate_mln_layer_list(self._mln_result)
            self._populate_mln_table(self._mln_result)
            self._render_mln_view()
            self._show_mln_metrics(self._mln_result.metrics_fig)
            self._show_mln_community(self._mln_result.community_fig)
            # Rebuilt from the restored edge tables, so a loaded analysis gets
            # the same histograms as a fresh run with nothing extra stored.
            self._populate_mln_degree(self._mln_result)

        self._residual_result = restore_residual(session)
        if self._residual_result is not None:
            self.tab_ns.set_result(
                self._residual_result.metrics,
                self._residual_result.label_order,
                self._residual_result.label_column,
            )

        self._evolution_result = restore_evolution(session)
        if self._evolution_result is not None:
            self._show_figure(self.evo_links_layout, self._evolution_result.links_fig)
            self._populate_evo_centrality(self._evolution_result)

        self._neural_evolution_result = restore_neural_evolution(session)
        self._set_neural_source_available(self._neural_evolution_result is not None)
        self._render_resid_source()

        self._set_controls_enabled(True)
        self._update_view_data_button()
        self.tabs.setCurrentIndex(0)
        summary = settings_summary(session.settings)
        self.lbl_session.setText(f"Loaded: {Path(path).name}")
        self.lbl_status.setText(f"Loaded analysis — {summary}")
        self._append_log(
            f"Loaded {describe(session)} — {summary}. "
            "Settings restored; press Build network to recompute."
        )

    def _has_results(self) -> bool:
        return any(
            r is not None
            for r in (
                self._mln_result,
                self._residual_result,
                self._evolution_result,
                self._neural_evolution_result,
            )
        )

    def _refresh_save_enabled(self) -> None:
        if hasattr(self, "btn_save_session"):
            self.btn_save_session.setEnabled(self._has_results())

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
            current_key = self._mask_context_key()
            if current_key == self._cell_mask_key:
                mask = sorted(self._cell_mask)
            else:
                # Only reaches here if the database, table or date column
                # changed without going through _on_table_changed /
                # _on_date_column_changed (which already reset the mask
                # themselves) -- e.g. a session load. Date range and the
                # Optional Filter deliberately do NOT land here any more; see
                # _mask_context_key. Easy to miss as just another log line
                # once a run is under way, so it also goes on the status
                # label, and the diff says exactly what changed.
                message = (
                    "User Filter cleared: the table or date column changed "
                    "since the cells were picked."
                )
                diff = self._describe_mask_key_mismatch(
                    self._cell_mask_key, current_key
                )
                self._append_log(f"WARNING: {message} ({diff})")
                self.lbl_status.setText(message)
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
        self._active_stages = {"mln"}
        self._mln_worker_thread.start()

    # ------------------------------------------------------- follow-on stages
    def _stage_done(self, stage: str) -> None:
        """Mark one stage complete; leave busy only when all of them are."""
        self._active_stages.discard(stage)
        if not self._active_stages:
            self._set_busy(False)
            self.progress.setValue(100)

    def _launch_follow_on_stages(self) -> None:
        """Queue the residual and evolution passes once the multiplex is up.

        They run **one after the other**, not together. Both are CPU-bound and
        overwhelmingly pure Python (scipy curve fits, networkx traversals), so
        under the GIL running them concurrently buys no throughput -- it just
        adds a second thread competing with the GUI for the interpreter, which
        is what made the window crawl. Sequential costs the same wall time,
        and the cheap residual pass lands first.
        """
        if self._cancel_event.is_set() or self._last_config is None:
            return
        if self._panel is None:
            return

        self._active_stages.add("residual")
        if self.chk_evolution.isChecked():
            # Reserved now, started when the residual pass reports back, so the
            # run does not look finished in between.
            self._active_stages.add("evolution")
            if self._evolution_config.run_neural_hjm and NEURAL_AVAILABLE:
                self._active_stages.add("neural_evolution")
            self._set_evolution_building()
        else:
            self._clear_evolution_tabs(
                "Tick “Run Evolution” in the sidebar to compute this."
            )

        self.tab_ns.set_placeholder("Computing residual networks…")
        self._residual_worker_thread = QThread(self)
        self._residual_worker = ResidualWorker(
            self._last_config,
            self._panel,
            NetworkKind(self._last_config.network_kind),
            threshold=RESIDUAL_THRESHOLD,
            cancel_event=self._cancel_event,
        )
        self._wire_stage(
            self._residual_worker,
            self._residual_worker_thread,
            self._on_residual_progress,
            self._on_residual_status,
            self._on_residual_finished,
            self._on_residual_failed,
            self._on_residual_cancelled,
        )

    def _start_evolution_if_pending(self) -> None:
        """Begin the evolution pass, once the residual pass has stepped aside."""
        if "evolution" not in self._active_stages:
            return
        if self._evolution_worker is not None or self._cancel_event.is_set():
            return
        if self._last_config is None or self._panel is None:
            self._stage_done("evolution")
            return

        panel = self._panel
        kind = NetworkKind(self._last_config.network_kind)
        self._sync_evolution_from_sidebar()
        self._append_log(
            f"Evolution: window={self._evolution_config.window_size}, "
            f"step={self._evolution_config.step}, "
            f"expanding={self._evolution_config.expanding}, "
            f"edge threshold={self._evolution_config.independent_threshold:.2f}"
        )
        self._evolution_worker_thread = QThread(self)
        self._evolution_worker = MLNEvolutionWorker(
            self._last_config,
            panel,
            MLNConfig(
                layer_column=panel.layer_column(kind),
                centrality=self._mln_config.centrality,
                jaccard_threshold=self._mln_config.jaccard_threshold,
                community_method=self._mln_config.community_method,
                max_communities=self._mln_config.max_communities,
                min_nodes=self._mln_config.min_nodes,
            ),
            self._evolution_config,
            edge_settings=self._edge_settings.to_dict(),
            cancel_event=self._cancel_event,
        )
        self._wire_stage(
            self._evolution_worker,
            self._evolution_worker_thread,
            self._on_evolution_progress,
            self._on_evolution_status,
            self._on_evolution_finished,
            self._on_evolution_failed,
            self._on_evolution_cancelled,
        )

    def _start_neural_evolution_if_pending(self) -> None:
        """Begin the Neural-HJM pass, once the NS evolution pass has finished.

        Called from both the NS evolution finish and failure handlers: a
        failed NS pass must not silently cancel this one, mirroring how the
        residual pass's failure does not cancel evolution.
        """
        if "neural_evolution" not in self._active_stages:
            return
        if self._neural_evolution_worker is not None or self._cancel_event.is_set():
            return
        if self._last_config is None or self._panel is None:
            self._stage_done("neural_evolution")
            return

        panel = self._panel
        self._sync_evolution_from_sidebar()
        self._append_log(
            f"Neural-HJM evolution: window={self._evolution_config.window_size}, "
            f"step={self._evolution_config.step}"
        )
        self._neural_evolution_worker_thread = QThread(self)
        self._neural_evolution_worker = NeuralEvolutionWorker(
            self._last_config,
            panel,
            self._evolution_config,
            cancel_event=self._cancel_event,
        )
        self._wire_stage(
            self._neural_evolution_worker,
            self._neural_evolution_worker_thread,
            self._on_neural_evolution_progress,
            self._on_neural_evolution_status,
            self._on_neural_evolution_finished,
            self._on_neural_evolution_failed,
            self._on_neural_evolution_cancelled,
        )

    @staticmethod
    def _wire_stage(
        worker, thread, on_progress, on_status, on_finished, on_failed, on_cancelled
    ) -> None:
        """Standard worker/thread wiring, then start at low priority.

        Low priority matters here: the analysis threads are compute-bound and
        would otherwise be scheduled on equal terms with the thread that has to
        repaint the window and service clicks.
        """
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(on_progress)
        worker.status.connect(on_status)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.start(QThread.Priority.LowPriority)

    # ------------------------------------------------------- residual signals
    def _on_residual_progress(self, done: int, total: int, desc: str) -> None:
        if self._is_stale(self._residual_worker) or total <= 0:
            return
        self.lbl_status.setText(f"NS residuals: {desc}")

    def _on_residual_status(self, message: str) -> None:
        if self._is_stale(self._residual_worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_residual_finished(self, result: object) -> None:
        if self._is_stale(self._residual_worker):
            return
        if not isinstance(result, ResidualResult):
            self._on_residual_failed(f"Unexpected residual result: {type(result)!r}")
            return
        self._residual_result = result
        self.tab_ns.set_result(result.metrics, result.label_order, result.label_column)
        self._append_log(
            f"NS residuals: {result.metrics.height} component network(s) rendered."
        )
        self._residual_worker = None
        self._residual_worker_thread = None
        self._update_view_data_button()
        self._stage_done("residual")
        self._start_evolution_if_pending()

    def _on_residual_failed(self, message: str) -> None:
        if self._is_stale(self._residual_worker):
            return
        self._append_log(f"NS ERROR: {message}")
        self.tab_ns.set_placeholder(f"Residual networks failed:\n{message}")
        self._residual_result = None
        self._residual_worker = None
        self._residual_worker_thread = None
        self._update_view_data_button()
        self._stage_done("residual")
        # The evolution pass does not depend on the residual networks, so a
        # failure here must not silently cancel it.
        self._start_evolution_if_pending()

    def _on_residual_cancelled(self) -> None:
        if self._is_stale(self._residual_worker):
            return
        self._residual_worker = None
        self._residual_worker_thread = None

    def _show_coverage(self) -> None:
        """Open the coverage-by-issuer pop-up for the current residual run."""
        result = self._residual_result
        if result is None or result.coverage.is_empty():
            QMessageBox.information(
                self,
                "Coverage",
                "Build a network first — coverage is computed from the loaded panel.",
            )
            return
        CoverageDialog(result.coverage, result.issuer_column, parent=self).exec()

    # ------------------------------------------------------ evolution signals
    def _on_evolution_progress(self, done: int, total: int, desc: str) -> None:
        if self._is_stale(self._evolution_worker) or total <= 0:
            return
        self.progress.setValue(int(100 * done / total))
        self.lbl_status.setText(f"Evolution: {desc}")

    def _on_evolution_status(self, message: str) -> None:
        if self._is_stale(self._evolution_worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_evolution_finished(self, result: object) -> None:
        if self._is_stale(self._evolution_worker):
            return
        if not isinstance(result, MLNEvolutionResult):
            self._on_evolution_failed(f"Unexpected evolution result: {type(result)!r}")
            return
        self._evolution_result = result
        self._show_figure(self.evo_links_layout, result.links_fig)
        self._populate_evo_centrality(result)
        self._populate_issuer_picker()
        self._append_log(
            f"Evolution rendered: {result.edge_types.height} window(s), "
            f"{result.factors.height} factor window(s), "
            f"{result.stress.height} stress window(s), "
            f"{len(result.issuer_factors)} issuer(s) with per-issuer factors, "
            f"{len(result.issuer_stress)} issuer(s) with per-issuer stress."
        )
        self._evolution_worker = None
        self._evolution_worker_thread = None
        self._update_view_data_button()
        self._stage_done("evolution")
        self._start_neural_evolution_if_pending()

    def _on_evolution_failed(self, message: str) -> None:
        if self._is_stale(self._evolution_worker):
            return
        self._append_log(f"EVOLUTION ERROR: {message}")
        self._clear_evolution_tabs(f"Evolution failed:\n{message}")
        self._evolution_worker = None
        self._evolution_worker_thread = None
        self._update_view_data_button()
        self._stage_done("evolution")
        self._start_neural_evolution_if_pending()

    def _on_evolution_cancelled(self) -> None:
        if self._is_stale(self._evolution_worker):
            return
        self._evolution_worker = None
        self._evolution_worker_thread = None

    # ------------------------------------------------------ neural evolution signals
    def _on_neural_evolution_progress(self, done: int, total: int, desc: str) -> None:
        if self._is_stale(self._neural_evolution_worker) or total <= 0:
            return
        self.progress.setValue(int(100 * done / total))
        self.lbl_status.setText(f"Neural: {desc}")

    def _on_neural_evolution_status(self, message: str) -> None:
        if self._is_stale(self._neural_evolution_worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_neural_evolution_finished(self, result: object) -> None:
        if self._is_stale(self._neural_evolution_worker):
            return
        if not isinstance(result, NeuralEvolutionResult):
            self._on_neural_evolution_failed(
                f"Unexpected Neural-HJM result: {type(result)!r}"
            )
            return
        self._neural_evolution_result = result
        self._set_neural_source_available(True)
        self._populate_issuer_picker()
        self._append_log(
            f"Neural-HJM evolution rendered: {result.factors.height} factor "
            f"window(s), {result.stress.height} stress window(s), "
            f"{len(result.issuer_factors)} issuer(s) with per-issuer factors, "
            f"{len(result.issuer_stress)} issuer(s) with per-issuer stress."
        )
        self._neural_evolution_worker = None
        self._neural_evolution_worker_thread = None
        self._update_view_data_button()
        self._stage_done("neural_evolution")

    def _on_neural_evolution_failed(self, message: str) -> None:
        if self._is_stale(self._neural_evolution_worker):
            return
        self._append_log(f"NEURAL-HJM ERROR: {message}")
        self._neural_evolution_result = None
        self._set_neural_source_available(False)
        self._neural_evolution_worker = None
        self._neural_evolution_worker_thread = None
        self._update_view_data_button()
        self._stage_done("neural_evolution")

    def _on_neural_evolution_cancelled(self) -> None:
        if self._is_stale(self._neural_evolution_worker):
            return
        self._neural_evolution_worker = None
        self._neural_evolution_worker_thread = None

    # ------------------------------------------------------ evolution canvases
    def _evolution_placeholder(self, layout: QVBoxLayout, message: str) -> None:
        self._clear_layout(layout)
        layout.addWidget(self._mln_placeholder_label(message))

    def _clear_evolution_tabs(self, message: str | None = None) -> None:
        text = message or "Evolution results will appear here after a build."
        for layout in (
            self.evo_links_layout,
            self.evo_factor_layout,
            self.evo_factor_std_layout,
            self.evo_cov_layout,
        ):
            self._evolution_placeholder(layout, text)
        # Evo: Centrality is gated on a second box, so the generic "tick Run
        # Evolution" text would send the user to the wrong control.
        if not self._evolution_config.run_centrality:
            self.tab_evo_centrality.set_placeholder(
                "Tick “Run Evolution” in the sidebar and “Run Centrality” in "
                "Evolution Settings to compute this."
            )
        else:
            self.tab_evo_centrality.set_placeholder(text)
        self.tab_factor_t.set_placeholder(text)
        self.tab_factor_std_t.set_placeholder(text)
        self.tab_cov_t.set_placeholder(text)
        self._evolution_result = None
        self._neural_evolution_result = None
        self._set_neural_source_available(False)
        self._populate_issuer_picker()
        self._update_view_data_button()

    def _set_evolution_building(self) -> None:
        self._clear_evolution_tabs("Computing evolution…")

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
        for thread, label in (
            (self._mln_worker_thread, "MLN"),
            (self._residual_worker_thread, "NS residuals"),
            (self._evolution_worker_thread, "Evolution"),
            (self._neural_evolution_worker_thread, "Neural-HJM"),
        ):
            if thread is None or not thread.isRunning():
                continue
            thread.quit()
            if thread.wait(CANCEL_GRACE_MS):
                self._append_log(f"{label} cancelled.")
            else:
                self._append_log(
                    f"{label} is finishing in the background; its results will "
                    "be discarded."
                )

        self._retire_worker(self._mln_worker, self._mln_worker_thread)
        self._retire_worker(self._residual_worker, self._residual_worker_thread)
        self._retire_worker(self._evolution_worker, self._evolution_worker_thread)
        self._retire_worker(
            self._neural_evolution_worker, self._neural_evolution_worker_thread
        )
        self._cleanup_mln_worker()
        self._residual_worker = None
        self._residual_worker_thread = None
        self._evolution_worker = None
        self._evolution_worker_thread = None
        self._neural_evolution_worker = None
        self._neural_evolution_worker_thread = None
        self._active_stages.clear()

        self._clear_mln_tabs("Render cancelled.")
        self._residual_result = None
        self.tab_ns.set_placeholder("Render cancelled.")
        self._clear_evolution_tabs("Render cancelled.")
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
        self.tab_mln_degree.set_placeholder("Computing multi-layer network...")
        self.mln_web.setHtml(
            self._mln_placeholder_html("Computing multi-layer network...")
        )

    def _clear_mln_tabs(self, message: str | None = None) -> None:
        """Clear every MLN tab and drop any cached result."""
        text = message or "No multi-layer network yet."
        self._set_mln_metrics_placeholder(text)
        self._set_mln_community_placeholder(text)
        self._degree_at_threshold.clear()
        self.tab_mln_degree.set_placeholder(text)
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

    # ------------------------------------------------- per-layer (component) tabs
    def _populate_mln_degree(self, result: MLNResult) -> None:
        """Point the MLN: Degree sub-tabs at this run's component networks.

        The renderer closes over the multiplex tables rather than pre-building
        every layer's graph, so a 15-layer panel costs one histogram, not 15,
        until the user actually opens the other sub-tabs.
        """

        def render(layer: str) -> Figure | None:
            graph = layer_subgraph(result.nodes, result.intra, layer)
            if graph.number_of_nodes() == 0:
                return None
            return render_degree_histogram(
                graph,
                f"Degree Histogram — {result.layer_column} = {layer}",
                bins=self._mln_config.degree_bins,
            )

        self._degree_at_threshold.clear()
        self.tab_mln_degree.set_layers(list(result.layer_values), render)

    def _build_degree_threshold_slider(
        self, layer: str, figure: Figure, canvas: FigureCanvas
    ) -> QWidget | None:
        """The live threshold slider under one MLN: Degree histogram.

        Returns None when this layer's measure matrix is unavailable — which
        is the case for a restored session, since the archive stores
        thresholded edges rather than measures. The histogram itself still
        works; only the slider is absent.
        """
        result = self._mln_result
        if result is None:
            return None
        measure_df = result.layer_measures.get(layer)
        if measure_df is None or measure_df.is_empty():
            return None

        def on_changed(threshold: float, degrees: dict) -> None:
            # The eye button must show the table that is on screen, not the
            # one from the threshold the network was built at.
            self._degree_at_threshold[layer] = (threshold, dict(degrees))
            self._update_view_data_button()

        return build_threshold_slider(
            measure_df=measure_df,
            measure=self._last_config.measure if self._last_config else "",
            initial=result.independent_threshold,
            figure=figure,
            canvas=canvas,
            bins=self._mln_config.degree_bins,
            title=f"Degree Histogram — {result.layer_column} = {layer}",
            on_changed=on_changed,
            on_error=lambda msg: self._append_log(f"MLN: Degree — {layer}: {msg}"),
        )

    def _populate_evo_centrality(self, result: MLNEvolutionResult) -> None:
        """Point the Evo: Centrality sub-tabs at the per-layer metric paths."""
        metrics = result.layer_metrics
        if metrics.is_empty() or "layer" not in metrics.columns:
            # Distinguish "you did not ask for it" from "it was asked for and
            # produced nothing" -- the first is the common case and the user
            # needs to be told which box to tick.
            if not self._evolution_config.run_centrality:
                self.tab_evo_centrality.set_placeholder(
                    "Not computed: tick “Run Centrality” in Evolution "
                    "Settings, then build again."
                )
            else:
                self.tab_evo_centrality.set_placeholder(
                    "The evolution pass produced no per-layer centrality data."
                )
            return
        layers = [
            layer
            for layer in self._mln_layer_order()
            if layer in set(metrics.get_column("layer").to_list())
        ]
        if not layers:
            layers = sorted(set(metrics.get_column("layer").to_list()))
        centrality = self._mln_config.centrality
        top_n = self._mln_config.centrality_top_n

        def render(layer: str) -> Figure | None:
            frame = metrics_for_layer(metrics, layer)
            if frame.is_empty():
                return None
            return render_centrality_trajectories(
                frame, centrality_metric=centrality, n_nodes=top_n
            )

        self.tab_evo_centrality.set_layers(layers, render)

    def _mln_layer_order(self) -> list[str]:
        """Layer values in the multiplex's own order, when a result exists."""
        if self._mln_result is None:
            return []
        return list(self._mln_result.layer_values)

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
        self._populate_mln_degree(result)
        self.lbl_status.setText("MLN ready.")
        self._append_log("MLN rendered.")
        self._cleanup_mln_worker()
        self._update_view_data_button()
        # The residual and evolution passes only make sense once the multiplex
        # is up, but neither depends on its output -- they re-derive from the
        # same panel, on their own threads.
        self._launch_follow_on_stages()
        self._stage_done("mln")

    def _on_mln_failed(self, message: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        self.progress.setValue(0)
        self._append_log(f"MLN ERROR: {message}")
        self.lbl_status.setText("Failed.")
        self._clear_mln_tabs(f"MLN failed: {message}")
        self.tab_ns.set_placeholder("MLN failed — residual networks not run.")
        self._clear_evolution_tabs("MLN failed — evolution not run.")
        self._cleanup_mln_worker()
        self._active_stages.clear()
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
        """Fill the visible-layers checklist.

        Restores whichever subset was checked last time, when the same layer
        values reappear -- otherwise a rebuild (e.g. just to also tick "Run
        Evolution") silently re-checks everything and discards a deliberate
        selection. Falls back to "all checked" the first time, or when none of
        the remembered values still apply (e.g. NETWORK TYPE flipped which set
        of values are layers).
        """
        keep: set[str] | None = None
        if self._visible_layers is not None:
            intersection = set(result.layer_values) & self._visible_layers
            if intersection:
                keep = intersection

        self.lst_mln_layers.blockSignals(True)
        self.lst_mln_layers.clear()
        for value in result.layer_values:
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = value in keep if keep is not None else True
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            self.lst_mln_layers.addItem(item)
        self.lst_mln_layers.blockSignals(False)
        self._visible_layers = set(self._selected_mln_layers())

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
        Also remembers the choice, so the next rebuild can restore it instead
        of resetting to "all checked" (see ``_populate_mln_layer_list``).
        """
        self._visible_layers = set(self._selected_mln_layers())
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

    def _plotly_asset_dir(self) -> Path:
        """Temp dir holding every Plotly page and their one plotly.min.js sidecar.

        Shared with the two factor-trajectory tabs (they take this method as a
        callable), so the 4MB sidecar is written once per session rather than
        once per view.
        """
        if self._plotly_temp_dir is None:
            self._plotly_temp_dir = Path(tempfile.mkdtemp(prefix="ycn_plotly_"))
        js_path = self._plotly_temp_dir / "plotly.min.js"
        if not js_path.exists():
            from plotly.offline import get_plotlyjs

            js_path.write_text(get_plotlyjs(), encoding="utf-8")
        return self._plotly_temp_dir

    def _write_mln_html(self, html: str) -> None:
        """Write the MLN page next to its JS sidecar and load it."""
        directory = self._plotly_asset_dir()
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
        for thread in (
            self._mln_worker_thread,
            self._residual_worker_thread,
            self._evolution_worker_thread,
            self._neural_evolution_worker_thread,
        ):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(2000)
        if self._plotly_temp_dir and self._plotly_temp_dir.exists():
            import shutil

            shutil.rmtree(self._plotly_temp_dir, ignore_errors=True)
        try:
            self._data_cache.frame_cache.cache_container.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
