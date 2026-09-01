"""Application-wide stylesheet matching the dark data-tool aesthetic."""

# Shared dark surfaces used by QSS and a few programmatic palette fills.
BG_APP = "#0f172a"
BG_SIDEBAR = "#1a1c24"
BG_CONTROL = "#2b3340"
BG_CONTROL_HOVER = "#334155"
BORDER = "#7dd3fc"  # sky blue — matches process-log accent, delineates controls
BORDER_MUTED = "#38bdf8"
TEXT = "#e5e7eb"
TEXT_MUTED = "#94a3b8"
TEXT_LOG = "#7dd3fc"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"


def build_app_style(
    *,
    bg_app: str = BG_APP,
    bg_sidebar: str = BG_SIDEBAR,
    bg_control: str = BG_CONTROL,
    bg_control_hover: str = BG_CONTROL_HOVER,
    border: str = BORDER,
    border_muted: str = BORDER_MUTED,
    text: str = TEXT,
    text_muted: str = TEXT_MUTED,
    text_log: str = TEXT_LOG,
    accent: str = ACCENT,
    accent_hover: str = ACCENT_HOVER,
) -> str:
    """The full stylesheet, parameterized by its eleven colour roles.

    ``APP_STYLE`` below is this called with the defaults -- the startup
    theme. GUI theme switching (``main_window._apply_theme``) calls this
    again with a different colour set instead of maintaining a second,
    shorter stylesheet: an independently hand-rolled theme stylesheet drifts
    out of sync with this one over time, silently losing rules (button/
    progress-bar corner radii, the collapsible sections' transparent
    background) that then fall back to Qt's plain default look the moment a
    theme is applied. One template, always fully applied, cannot drift.
    """
    return f"""
QWidget {{
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
    color: {text};
    background-color: {bg_app};
}}
QMainWindow, QWidget#Root {{
    background-color: {bg_app};
    color: {text};
}}

/* Sidebar containers only — avoid universal descendant border:none. */
QFrame#Sidebar {{
    background-color: {bg_sidebar};
    color: {text};
    border: none;
    border-right: 1px solid #2a2f3a;
}}
QFrame#Sidebar QScrollArea {{
    background-color: {bg_sidebar};
    border: none;
}}
QWidget#SidebarContent {{
    background-color: {bg_sidebar};
    color: {text};
    border: none;
}}

QLabel {{
    background-color: transparent;
    color: {text};
}}
QLabel#Brand {{
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding-bottom: 2px;
}}
QLabel#SectionTitle {{
    color: {text_muted};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    padding-top: 4px;
    padding-bottom: 0px;
}}
QToolButton#CollapseHeader {{
    color: {text_muted};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    padding: 4px 0 0 0;
    background-color: transparent;
    border: none;
    text-align: left;
}}
QToolButton#CollapseHeader:hover {{
    color: {text};
}}
QWidget#CollapseBody {{
    background-color: transparent;
    border: none;
}}
QLabel#DbPath, QLabel#StatusLabel {{
    color: {text_muted};
    font-size: 11px;
}}
QLabel#LogTitle {{
    color: {text_muted};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    background-color: transparent;
}}

QCheckBox {{
    background-color: transparent;
    color: {text};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid {border};
    background-color: {bg_control};
}}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {border};
}}

QRadioButton {{
    background-color: transparent;
    color: {text};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 1px solid {border};
    background-color: {bg_control};
}}
QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {border};
}}
QRadioButton:disabled {{
    color: {text_muted};
}}

QComboBox, QLineEdit, QDateEdit, QDoubleSpinBox, QListWidget {{
    background-color: {bg_control};
    color: #ffffff;
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px 7px;
    selection-background-color: {accent};
    min-height: 22px;
}}
QDateEdit, QDoubleSpinBox {{
    padding-right: 22px;
}}
QComboBox:hover, QLineEdit:hover, QDateEdit:hover, QListWidget:hover,
QDoubleSpinBox:hover {{
    border-color: #bae6fd;
    background-color: {bg_control_hover};
}}
QComboBox:focus, QLineEdit:focus, QDateEdit:focus, QDoubleSpinBox:focus,
QListWidget:focus {{
    border: 1px solid #e0f2fe;
}}
QComboBox:disabled, QLineEdit:disabled, QDateEdit:disabled,
QDoubleSpinBox:disabled, QListWidget:disabled {{
    color: #6b7280;
    background-color: #151820;
    border-color: #475569;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {border};
    width: 0;
    height: 0;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {bg_control};
    color: #ffffff;
    border: 1px solid {border};
    selection-background-color: {accent};
    outline: none;
}}
QDateEdit::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border: none;
    border-left: 1px solid #475569;
    border-bottom: 1px solid #475569;
    background-color: {bg_control_hover};
    border-top-right-radius: 5px;
}}
QDateEdit::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border: none;
    border-left: 1px solid #475569;
    border-top: 1px solid #475569;
    background-color: {bg_control_hover};
    border-bottom-right-radius: 5px;
}}
QDateEdit::up-button:hover, QDoubleSpinBox::up-button:hover,
QDateEdit::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #475569;
}}
QDateEdit::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {border};
}}
QDateEdit::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {border};
}}

QListWidget {{
    outline: none;
    padding: 2px;
}}
QListWidget::item {{
    padding: 2px 4px;
    border-radius: 3px;
    color: #ffffff;
}}
QListWidget::item:selected {{
    background-color: {accent};
}}

QPushButton {{
    background-color: {accent};
    color: #ffffff;
    border: 1px solid {border_muted};
    border-radius: 16px;
    padding: 7px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {accent_hover};
    border-color: {border};
}}
QPushButton:pressed {{
    background-color: #1e40af;
}}
QPushButton:disabled {{
    background-color: #334155;
    color: {text_muted};
    border-color: #475569;
}}
QPushButton#SecondaryButton {{
    background-color: {bg_control};
    color: {text_muted};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 5px 10px;
}}
QPushButton#SecondaryButton:hover {{
    background-color: {bg_control_hover};
    color: {text};
    border-color: #bae6fd;
}}
QToolButton#ViewDataButton {{
    background-color: {bg_control};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 0;
    margin: 2px 6px 0 0;
}}
QToolButton#ViewDataButton:hover {{
    background-color: {bg_control_hover};
    border-color: #bae6fd;
}}
QToolButton#ViewDataButton:disabled {{
    background-color: #151820;
    border-color: #475569;
}}
QTableView {{
    background-color: {bg_control};
    color: #ffffff;
    border: 1px solid {border};
    border-radius: 6px;
    gridline-color: #334155;
    selection-background-color: {accent};
    selection-color: #ffffff;
    alternate-background-color: #243044;
}}
QHeaderView::section {{
    background-color: {bg_sidebar};
    color: {text};
    border: 1px solid #334155;
    padding: 4px 18px 4px 6px;
    font-weight: 600;
}}
QTableView::item:selected {{
    background-color: {accent};
    color: #ffffff;
}}
QPushButton#CancelButton {{
    background-color: #7f1d1d;
    color: #fecaca;
    border: 1px solid #991b1b;
    border-radius: 8px;
    padding: 5px 10px;
}}
QPushButton#CancelButton:hover {{
    background-color: #991b1b;
    color: #fca5a5;
    border-color: #dc2626;
}}
QPushButton#CancelButton:disabled {{
    background-color: #374151;
    color: #6b7280;
    border-color: #4b5563;
}}

QProgressBar {{
    background-color: {bg_control};
    border: 1px solid {border};
    border-radius: 6px;
    text-align: center;
    color: {text};
    height: 14px;
}}
QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 5px;
}}

QFrame#CanvasFrame {{
    background-color: {bg_app};
    border: none;
}}
QFrame#Canvas {{
    background-color: {bg_app};
    border: 1px solid #2a2f3a;
    border-radius: 10px;
}}
QTabWidget#ResultTabs {{
    background-color: {bg_app};
    border: none;
}}
QTabWidget#ResultTabs::pane {{
    background-color: {bg_app};
    border: 1px solid #2a2f3a;
    border-radius: 8px;
    top: 0px;
    border-top: none;
}}
QTabWidget#ResultTabs QTabBar::tab {{
    background-color: {bg_sidebar};
    color: {text_muted};
    border: 1px solid #2a2f3a;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 14px;
    margin-right: 3px;
}}
QTabWidget#ResultTabs QTabBar::tab:selected {{
    background-color: #24292e;
    color: {text_log};
    border-color: {border};
}}
QTabWidget#ResultTabs QTabBar::tab:hover {{
    color: #ffffff;
}}
QLabel#HistLabel {{
    background-color: transparent;
    color: {text_muted};
}}
QPlainTextEdit#ProcessLog {{
    background-color: {bg_sidebar};
    color: {text_log};
    border: 1px solid {border};
    border-radius: 8px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    padding: 8px;
}}

QScrollBar:vertical {{
    background: {bg_sidebar};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #3a4150;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {bg_sidebar};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #3a4150;
    border-radius: 5px;
}}
QSplitter::handle {{
    background-color: #2a2f3a;
    width: 1px;
}}
QCalendarWidget QWidget {{
    background-color: {bg_control};
    color: #ffffff;
}}
QCalendarWidget QToolButton {{
    color: #ffffff;
    background-color: {bg_control};
}}
QCalendarWidget QAbstractItemView:enabled {{
    color: #ffffff;
    background-color: {bg_control};
    selection-background-color: {accent};
}}
"""


APP_STYLE = build_app_style()
