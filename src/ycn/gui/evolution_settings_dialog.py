"""Settings dialog for network evolution analysis configuration."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.evolution import CommunityMethod, EvolutionConfig
from ycn.gui.styles import APP_STYLE, BG_SIDEBAR


def _community_method_label(method: CommunityMethod | str) -> str:
    """Return combo-box label for a community method (enum or plain string)."""
    if isinstance(method, CommunityMethod):
        return method.value
    return method


def _coerce_community_method(method: CommunityMethod | str | None) -> CommunityMethod:
    """Normalize combo-box data back to CommunityMethod."""
    if isinstance(method, CommunityMethod):
        return method
    if isinstance(method, str):
        return CommunityMethod(method)
    return CommunityMethod.FIXED


class EvolutionSettingsDialog(QDialog):
    """Dialog for configuring temporal network evolution parameters."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_config: EvolutionConfig | None = None,
        independent_threshold: float = 0.33,
        max_nodes: int | None = None,
        evolution_enabled: bool = True,
        neural_available: bool = False,
        neural_import_error: str = "",
    ) -> None:
        """Initialize evolution settings dialog.

        Args:
            evolution_enabled: Whether the sidebar's "Run Evolution" is ticked.
                "Run Centrality" and "Run Neural-HJM" are meaningless without it.
            neural_available: Whether the optional 'neural' extra is installed.
            neural_import_error: Error message if neural is not available.
        """
        super().__init__(parent)
        self.setWindowTitle("Evolution Analysis Settings")
        self.setModal(True)
        self.resize(400, 420)
        self.setStyleSheet(APP_STYLE)
        self._force_dark_bg()

        self.initial_config = initial_config or EvolutionConfig(
            independent_threshold=independent_threshold
        )
        self.max_nodes = max_nodes or 20
        self.evolution_enabled = evolution_enabled
        self.neural_available = neural_available
        self.neural_import_error = neural_import_error

        self._build_form()

    def _force_dark_bg(self) -> None:
        """Force dark background."""
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_SIDEBAR))
        self.setPalette(palette)

    def _build_form(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Section label
        title = QLabel("Network Evolution Configuration")
        title.setStyleSheet("font-weight: 600; color: #ffffff; font-size: 13px;")
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(10)

        # Window size
        self.spin_window_size = QSpinBox()
        self.spin_window_size.setRange(2, 504)
        self.spin_window_size.setValue(self.initial_config.window_size)
        self.spin_window_size.setSuffix(" observations")
        form.addRow("Window size:", self.spin_window_size)

        # Step size
        self.spin_step = QSpinBox()
        self.spin_step.setRange(2, 252)
        self.spin_step.setValue(self.initial_config.step)
        self.spin_step.setSuffix(" observations")
        form.addRow("Step size:", self.spin_step)

        # Min nodes
        self.spin_min_nodes = QSpinBox()
        self.spin_min_nodes.setRange(2, self.max_nodes)
        self.spin_min_nodes.setValue(self.initial_config.min_nodes)
        form.addRow("Min nodes/window:", self.spin_min_nodes)

        # Centrality
        self.cmb_centrality = QComboBox()
        self.cmb_centrality.addItems(["eigenvector", "betweenness", "degree"])
        self.cmb_centrality.setCurrentText(self.initial_config.centrality)
        form.addRow("Centrality measure:", self.cmb_centrality)

        # Num nodes (for top/bottom centrality trajectories plots)
        self.spin_n_nodes = QSpinBox()
        self.spin_n_nodes.setRange(1, 20)
        self.spin_n_nodes.setValue(self.initial_config.n_top_nodes)
        form.addRow("Num nodes to plot:", self.spin_n_nodes)

        # Community detection method (optimization strategy for k selection per window)
        self.cmb_community_method = QComboBox()
        for method in CommunityMethod:
            self.cmb_community_method.addItem(method.value, method)
        self.cmb_community_method.setCurrentText(
            _community_method_label(self.initial_config.community_method)
        )
        form.addRow("Community method:", self.cmb_community_method)

        # Communities control: exact k for FIXED, search upper bound for other methods
        self._lbl_max_communities = QLabel()
        self.spin_max_communities = QSpinBox()
        self.spin_max_communities.setRange(2, 15)
        self.spin_max_communities.setValue(self.initial_config.max_communities)
        form.addRow(self._lbl_max_communities, self.spin_max_communities)
        self.cmb_community_method.currentIndexChanged.connect(
            self._update_community_spin_state
        )
        self._update_community_spin_state()

        # Per-layer centrality trajectories
        self.chk_run_centrality = QCheckBox("Run Centrality")
        self.chk_run_centrality.setChecked(
            self.initial_config.run_centrality and self.evolution_enabled
        )
        self.chk_run_centrality.setEnabled(self.evolution_enabled)
        self.chk_run_centrality.setToolTip(
            "Also track each component network's per-node centrality across "
            "the windows, filling the Evo: Centrality tab. Adds one "
            "centrality solve per layer per window; nothing else in the "
            "evolution pass needs it, so it is off by default."
            if self.evolution_enabled
            else "Tick Run Evolution in the sidebar first — this rides on "
            "that pass's window loop."
        )
        form.addRow("Centrality trajectories:", self.chk_run_centrality)

        # Neural-HJM evolution
        self.chk_run_neural_hjm = QCheckBox("Run Neural-HJM")
        self.chk_run_neural_hjm.setChecked(
            self.initial_config.run_neural_hjm and self.evolution_enabled
        )
        self.chk_run_neural_hjm.setEnabled(self.evolution_enabled and self.neural_available)
        if self.neural_available:
            self.chk_run_neural_hjm.setToolTip(
                "Also fit the experimental Neural HJM model (needs the neural "
                "extra) and compute its factor/stress evolution alongside the "
                "Nelson-Siegel one. Only runs when Run Evolution is also ticked; "
                "adds a fourth, sequential analysis stage — this is the slowest of "
                "them, since it trains one small neural net per issuer."
            )
        else:
            self.chk_run_neural_hjm.setToolTip(
                "Needs the optional neural extra (uv sync --extra neural): "
                f"{self.neural_import_error}"
            )
        form.addRow("Neural-HJM evolution:", self.chk_run_neural_hjm)

        # Independent threshold (read-only, informational)
        lbl_threshold = QLabel(f"{self.initial_config.independent_threshold:.2f}")
        form.addRow("Edge threshold:", lbl_threshold)

        layout.addLayout(form)
        layout.addSpacing(12)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_community_spin_state(self) -> None:
        """Reflect how max_communities is interpreted for the selected method."""
        method = _coerce_community_method(
            self.cmb_community_method.currentData()
            or self.cmb_community_method.currentText()
        )
        if method == CommunityMethod.FIXED:
            self._lbl_max_communities.setText("Communities per window:")
            self.spin_max_communities.setToolTip(
                "Fixed number of communities used in every rolling window."
            )
        else:
            self._lbl_max_communities.setText("Max communities (search bound):")
            self.spin_max_communities.setToolTip(
                f"The {method.value} method chooses the optimal community count "
                "independently for each window, searching from 2 up to this maximum."
            )

    def get_config(self) -> EvolutionConfig:
        """Return configured EvolutionConfig."""
        return EvolutionConfig(
            window_size=self.spin_window_size.value(),
            step=self.spin_step.value(),
            expanding=False,  # Always rolling for now
            min_nodes=self.spin_min_nodes.value(),
            independent_threshold=self.initial_config.independent_threshold,
            centrality=self.cmb_centrality.currentText(),
            n_top_nodes=self.spin_n_nodes.value(),
            max_communities=self.spin_max_communities.value(),
            community_method=_coerce_community_method(
                self.cmb_community_method.currentData()
                or self.cmb_community_method.currentText()
            ),
            measure=self.initial_config.measure,
            edge_settings=self.initial_config.edge_settings,
            run_centrality=self.chk_run_centrality.isChecked(),
            run_neural_hjm=self.chk_run_neural_hjm.isChecked(),
        )
