"""Settings dialog for the multi-layer network workstream."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.evolution import CommunityMethod
from ycn.analysis.mln import MLNConfig
from ycn.gui.styles import APP_STYLE, BG_SIDEBAR


def _coerce_community_method(method: CommunityMethod | str | None) -> CommunityMethod:
    """Normalize combo-box data back to CommunityMethod."""
    if isinstance(method, CommunityMethod):
        return method
    if isinstance(method, str):
        return CommunityMethod(method)
    return CommunityMethod.FIXED


class MLNSettingsDialog(QDialog):
    """Dialog for configuring multi-layer network parameters."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_config: MLNConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multi-Layer Network Settings")
        self.setModal(True)
        self.resize(400, 300)
        self.setStyleSheet(APP_STYLE)
        self._force_dark_bg()

        self.initial_config = initial_config or MLNConfig(layer_column="")
        self._build_form()

    def _force_dark_bg(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_SIDEBAR))
        self.setPalette(palette)

    def _build_form(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Multi-Layer Network Configuration")
        title.setStyleSheet("font-weight: 600; color: #ffffff; font-size: 13px;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)

        # Centrality -- same choices as Evolution Settings.
        self.cmb_centrality = QComboBox()
        self.cmb_centrality.addItems(["eigenvector", "betweenness", "degree"])
        self.cmb_centrality.setCurrentText(self.initial_config.centrality)
        self.cmb_centrality.setToolTip(
            "Centrality shown in the node x layer heatmap on the MLN: Metrics tab."
        )
        form.addRow("Centrality measure:", self.cmb_centrality)

        # Jaccard threshold for cross-layer community alignment.
        self.spin_jaccard = QDoubleSpinBox()
        self.spin_jaccard.setRange(0.0, 1.0)
        self.spin_jaccard.setSingleStep(0.05)
        self.spin_jaccard.setDecimals(2)
        self.spin_jaccard.setValue(self.initial_config.jaccard_threshold)
        self.spin_jaccard.setToolTip(
            "Two communities in different layers are treated as the same community "
            "when the Jaccard overlap of their members reaches this level. Higher "
            "values demand more overlap, producing more distinct communities."
        )
        form.addRow("Jaccard similarity:", self.spin_jaccard)

        # Community detection method -- same choices as Evolution Settings.
        self.cmb_community_method = QComboBox()
        for method in CommunityMethod:
            self.cmb_community_method.addItem(method.value, method)
        self.cmb_community_method.setCurrentText(
            self.initial_config.community_method.value
        )
        form.addRow("Community method:", self.cmb_community_method)

        self._lbl_max_communities = QLabel()
        self.spin_max_communities = QSpinBox()
        self.spin_max_communities.setRange(2, 15)
        self.spin_max_communities.setValue(self.initial_config.max_communities)
        form.addRow(self._lbl_max_communities, self.spin_max_communities)
        self.cmb_community_method.currentIndexChanged.connect(
            self._update_community_spin_state
        )
        self._update_community_spin_state()

        # --- Per-component-network tabs -------------------------------------
        self.spin_degree_bins = QSpinBox()
        self.spin_degree_bins.setRange(3, 60)
        self.spin_degree_bins.setValue(self.initial_config.degree_bins)
        self.spin_degree_bins.setToolTip(
            "Histogram bins on each MLN: Degree sub-tab. Too many bins on a "
            "small layer leaves mostly-empty bars; too few hide the shape."
        )
        form.addRow("Degree histogram bins:", self.spin_degree_bins)

        self.spin_centrality_top_n = QSpinBox()
        self.spin_centrality_top_n.setRange(1, 20)
        self.spin_centrality_top_n.setValue(self.initial_config.centrality_top_n)
        self.spin_centrality_top_n.setToolTip(
            "Nodes drawn in each panel of an MLN: Centrality sub-tab — the "
            "most-variable on top, the least-variable below. Capped per layer "
            "at half that layer's nodes so the two panels never overlap."
        )
        form.addRow("Centrality nodes per panel:", self.spin_centrality_top_n)

        layout.addLayout(form)
        layout.addSpacing(12)

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
            self._lbl_max_communities.setText("Communities per layer:")
            self.spin_max_communities.setToolTip(
                "Fixed number of communities used in every layer."
            )
        else:
            self._lbl_max_communities.setText("Max communities (search bound):")
            self.spin_max_communities.setToolTip(
                f"The {method.value} method chooses the optimal community count "
                "independently for each layer, searching from 2 up to this maximum."
            )

    def get_config(self) -> MLNConfig:
        """Return configured MLNConfig (layer_column is set by the caller)."""
        return MLNConfig(
            layer_column=self.initial_config.layer_column,
            centrality=self.cmb_centrality.currentText(),
            jaccard_threshold=float(self.spin_jaccard.value()),
            community_method=_coerce_community_method(
                self.cmb_community_method.currentData()
                or self.cmb_community_method.currentText()
            ),
            max_communities=self.spin_max_communities.value(),
            min_nodes=self.initial_config.min_nodes,
            degree_bins=self.spin_degree_bins.value(),
            centrality_top_n=self.spin_centrality_top_n.value(),
        )
