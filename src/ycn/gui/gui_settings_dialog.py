"""Settings dialog for GUI appearance and theme management."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ycn.gui.styles import APP_STYLE, BG_SIDEBAR


@dataclass
class Theme:
    """Color theme definition."""

    name: str
    bg_primary: str
    bg_sidebar: str
    text_primary: str
    text_secondary: str
    accent: str
    border: str
    input_bg: str


# Professional themes: classic dark, modern minimalist, and corporate styles
THEMES = [
    # Original Theme
    Theme(
        "Sky Blue",
        bg_primary="#0f172a",
        bg_sidebar="#1a1c24",
        text_primary="#e5e7eb",
        text_secondary="#94a3b8",
        accent="#2563eb",
        border="#7dd3fc",
        input_bg="#2b3340",
    ),
    # Classic Dark Themes
    Theme(
        "Midnight Blue",
        bg_primary="#1a1a2e",
        bg_sidebar="#16213e",
        text_primary="#ffffff",
        text_secondary="#b0b0b0",
        accent="#00d4ff",
        border="#2a3a4e",
        input_bg="#0f3460",
    ),
    Theme(
        "Forest Green",
        bg_primary="#1a2d1a",
        bg_sidebar="#0f1f0f",
        text_primary="#ffffff",
        text_secondary="#a8d5a8",
        accent="#4ade80",
        border="#2d4a2d",
        input_bg="#1d3a1d",
    ),
    Theme(
        "Royal Purple",
        bg_primary="#1a0f2e",
        bg_sidebar="#120924",
        text_primary="#ffffff",
        text_secondary="#c8b5e0",
        accent="#b85eff",
        border="#342353",
        input_bg="#1f1237",
    ),
    Theme(
        "Burnt Orange",
        bg_primary="#2d1810",
        bg_sidebar="#1f0f08",
        text_primary="#ffffff",
        text_secondary="#d4a574",
        accent="#ff8c42",
        border="#4a2817",
        input_bg="#3a1f11",
    ),
    Theme(
        "Deep Red",
        bg_primary="#2d1a1a",
        bg_sidebar="#1f0f0f",
        text_primary="#ffffff",
        text_secondary="#e0a8a8",
        accent="#ff5555",
        border="#4a2a2a",
        input_bg="#3a1a1a",
    ),
    # Modern Minimalist Themes
    Theme(
        "Slate Gray",
        bg_primary="#2c3e50",
        bg_sidebar="#34495e",
        text_primary="#ecf0f1",
        text_secondary="#bdc3c7",
        accent="#3498db",
        border="#404854",
        input_bg="#1e2936",
    ),
    Theme(
        "Charcoal Steel",
        bg_primary="#2b2b2b",
        bg_sidebar="#1e1e1e",
        text_primary="#e8e8e8",
        text_secondary="#a0a0a0",
        accent="#00bcd4",
        border="#3a3a3a",
        input_bg="#1a1a1a",
    ),
    Theme(
        "Ocean Depth",
        bg_primary="#0d1b2a",
        bg_sidebar="#1b263b",
        text_primary="#e0e1dd",
        text_secondary="#a8b0bf",
        accent="#06aed5",
        border="#27556a",
        input_bg="#1d2d3d",
    ),
    Theme(
        "Emerald Noir",
        bg_primary="#1b2a2a",
        bg_sidebar="#0f1f1f",
        text_primary="#d4e8e8",
        text_secondary="#a0c4c4",
        accent="#2dd4bf",
        border="#2a3f3f",
        input_bg="#161f1f",
    ),
    Theme(
        "Indigo Night",
        bg_primary="#1a1f3a",
        bg_sidebar="#0f1629",
        text_primary="#e8e8ff",
        text_secondary="#b8b8e0",
        accent="#6366f1",
        border="#2a2f4a",
        input_bg="#151a2f",
    ),
    # Professional Corporate Themes
    Theme(
        "Navy Professional",
        bg_primary="#1e3a5f",
        bg_sidebar="#122e4a",
        text_primary="#ffffff",
        text_secondary="#b0c4d4",
        accent="#2196f3",
        border="#2a4a6a",
        input_bg="#172a45",
    ),
    Theme(
        "Black & White",
        bg_primary="#1f1f1f",
        bg_sidebar="#0a0a0a",
        text_primary="#ffffff",
        text_secondary="#cccccc",
        accent="#4a4a4a",
        border="#2a2a2a",
        input_bg="#121212",
    ),
    Theme(
        "Teal Corporate",
        bg_primary="#1a3a3f",
        bg_sidebar="#0f2a2d",
        text_primary="#ffffff",
        text_secondary="#a8d4d8",
        accent="#17a2b8",
        border="#2a4a50",
        input_bg="#132a2f",
    ),
    Theme(
        "Graphite Executive",
        bg_primary="#2a2a2a",
        bg_sidebar="#1a1a1a",
        text_primary="#f5f5f5",
        text_secondary="#c0c0c0",
        accent="#5a9fd4",
        border="#3a3a3a",
        input_bg="#141414",
    ),
]


class GuiSettingsDialog(QDialog):
    """Dialog for configuring GUI appearance and themes."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_theme: str = "Midnight Blue",
    ) -> None:
        """Initialize the GUI settings dialog.

        Args:
            initial_theme: Name of the currently active theme.
        """
        super().__init__(parent)
        self.setWindowTitle("GUI Settings")
        self.setModal(True)
        self.resize(300, 150)
        self.setStyleSheet(APP_STYLE)
        self._force_dark_bg()

        self.initial_theme = initial_theme
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
        title = QLabel("Application Appearance")
        title.setStyleSheet("font-weight: 600; color: #ffffff; font-size: 13px;")
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(10)

        # Theme selector
        self.cmb_theme = QComboBox()
        theme_names = [theme.name for theme in THEMES]
        self.cmb_theme.addItems(theme_names)
        self.cmb_theme.setCurrentText(self.initial_theme)
        self.cmb_theme.setToolTip("Select an application color theme")
        form.addRow("Theme:", self.cmb_theme)

        layout.addLayout(form)
        layout.addSpacing(12)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_theme_name(self) -> str:
        """Return selected theme name."""
        return self.cmb_theme.currentText()

    @staticmethod
    def get_theme(theme_name: str) -> Theme | None:
        """Get a theme by name."""
        for theme in THEMES:
            if theme.name == theme_name:
                return theme
        return None
