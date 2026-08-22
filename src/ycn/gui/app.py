"""Application entry point for the YieldCurve-Network GUI."""

from __future__ import annotations

import sys

# Set matplotlib backend before any GUI imports
import matplotlib

matplotlib.use("Qt5Agg")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ycn.gui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    # Helps high-DPI displays look crisp on Windows.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv)
    app.setApplicationName("YieldCurve-Network")
    app.setOrganizationName("ycn")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
