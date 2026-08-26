"""Edge/connection measure settings dialog."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QMessageBox,
)


@dataclass
class EdgeSettingsConfig:
    """Configuration for edge/connection measure parameters."""

    # Conditional correlation settings
    conditional_quantile: float = 0.90

    # FastDTW settings
    fastdtw_radius: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeSettingsConfig:
        """Create from dictionary."""
        # Only extract known fields
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


class EdgeSettingsDialog(QDialog):
    """Dialog for configuring edge/connection measure parameters."""

    def __init__(self, parent=None, initial_config: EdgeSettingsConfig | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edge Settings")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            QLabel {
                color: #cbd5e1;
            }
            QDoubleSpinBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
            }
            QGroupBox {
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #0ea5e9;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
            QPushButton:pressed {
                background-color: #0369a1;
            }
        """)

        self.config = initial_config or EdgeSettingsConfig()

        # Build UI
        layout = QVBoxLayout(self)

        # Conditional Correlation Group
        conditional_group = QGroupBox("Conditional Correlation Settings")
        conditional_layout = QFormLayout(conditional_group)

        lbl_quantile = QLabel("Quantile Threshold (stress regime):")
        lbl_quantile.setToolTip(
            "Only correlations computed on days where abs(return) exceeds this quantile. "
            "0.90 = top 10% extreme days. Higher = more extreme stress regime."
        )
        self.spin_conditional_quantile = QDoubleSpinBox()
        self.spin_conditional_quantile.setMinimum(0.0)
        self.spin_conditional_quantile.setMaximum(1.0)
        self.spin_conditional_quantile.setSingleStep(0.05)
        self.spin_conditional_quantile.setDecimals(2)
        self.spin_conditional_quantile.setValue(self.config.conditional_quantile)

        conditional_layout.addRow(lbl_quantile, self.spin_conditional_quantile)

        layout.addWidget(conditional_group)

        # No FastDTW group: both DTW variants are excluded from the GUI's
        # connection-measure dropdown (see measures._MEASURE_GUI_DISABLED), so
        # a radius control here could never affect a run. The
        # ``fastdtw_radius`` field is still carried on the config -- it round
        # trips through saved sessions and is read by API callers of
        # ``compute_measure`` -- it simply has no widget.

        # Info text
        info = QLabel(
            "Configure parameters for connection measures that accept optional settings.\n\n"
            "• Conditional Correlation: Isolates correlation during market stress "
            "(extreme return days)"
        )
        info.setStyleSheet("color: #94a3b8; font-size: 9px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setObjectName("SecondaryButton")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #64748b;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        btn_reset.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(btn_reset)

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        layout.addLayout(button_layout)

    def _reset_to_defaults(self) -> None:
        """Reset every setting that has a widget."""
        defaults = EdgeSettingsConfig()
        self.spin_conditional_quantile.setValue(defaults.conditional_quantile)

    def get_config(self) -> EdgeSettingsConfig:
        """Return configured EdgeSettingsConfig.

        ``fastdtw_radius`` has no widget (see ``_build``) so it is carried
        through from the incoming config rather than reset, keeping a value
        set via the API or restored from a session intact.
        """
        return EdgeSettingsConfig(
            conditional_quantile=float(self.spin_conditional_quantile.value()),
            fastdtw_radius=self.config.fastdtw_radius,
        )
