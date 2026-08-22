"""Read-only data viewer: Excel-style filters, copy, and CSV/Parquet export.

The dialog always works on a cloned frame, so nothing here mutates GUI state.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    QPointF,
    QRect,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ycn.gui.styles import APP_STYLE, TEXT, TEXT_MUTED


def eye_icon(size: int = 64) -> QIcon:
    """Draw a clear eye (outline + iris + pupil) as a QIcon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor(TEXT)
    painter.setPen(QPen(color, max(2.0, size / 14.0)))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    lid = QPainterPath()
    lid.moveTo(size * 0.10, size * 0.50)
    lid.quadTo(size * 0.50, size * 0.10, size * 0.90, size * 0.50)
    lid.quadTo(size * 0.50, size * 0.90, size * 0.10, size * 0.50)
    painter.drawPath(lid)
    iris_r = size * 0.16
    painter.setBrush(QBrush(color))
    painter.drawEllipse(QPointF(size * 0.50, size * 0.50), iris_r, iris_r)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor("#0f172a")))
    painter.drawEllipse(QPointF(size * 0.52, size * 0.48), iris_r * 0.45, iris_r * 0.45)
    painter.end()
    return QIcon(pixmap)


def _fmt(value: object) -> str:
    """Format a cell for display, filtering, and copy."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.6g}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


class PolarsTableModel(QAbstractTableModel):
    """Read-only table model backed by a cloned Polars frame."""

    def __init__(self, df: pl.DataFrame, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df = df
        self._columns = list(df.columns)
        self._cells: list[list[str]] = [
            [_fmt(value) for value in row] for row in df.iter_rows()
        ]

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        return 0 if parent and parent.isValid() else len(self._cells)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        return 0 if parent and parent.isValid() else len(self._columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
        ):
            return None
        return self._cells[index.row()][index.column()]

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section]
        return str(section + 1)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def displayed_frame(self, source_rows: list[int]) -> pl.DataFrame:
        """Return the original-typed rows at ``source_rows`` (cloned)."""
        if not source_rows:
            return self._df.head(0)
        return self._df.gather(source_rows)


class ColumnFilterProxy(QSortFilterProxyModel):
    """Per-column allowed-value filters (Excel autofilter)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._allowed: dict[int, frozenset[str] | None] = {}
        self.setDynamicSortFilter(True)

    def set_allowed(self, column: int, values: frozenset[str] | None) -> None:
        """Restrict ``column`` to ``values``; ``None`` clears that filter."""
        self._allowed[column] = values
        self.invalidateFilter()

    def allowed(self, column: int) -> frozenset[str] | None:
        return self._allowed.get(column)

    def is_filtered(self, column: int) -> bool:
        return self._allowed.get(column) is not None

    def filterAcceptsRow(  # noqa: N802
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        model = self.sourceModel()
        if model is None:
            return True
        for column, allowed in self._allowed.items():
            if allowed is None:
                continue
            index = model.index(source_row, column, source_parent)
            if str(model.data(index) or "") not in allowed:
                return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, PolarsTableModel):
            return super().lessThan(left, right)
        lhs = model._df.row(left.row())[left.column()]
        rhs = model._df.row(right.row())[right.column()]
        if lhs is None:
            return True
        if rhs is None:
            return False
        try:
            return lhs < rhs
        except TypeError:
            return str(lhs) < str(rhs)


class FilterHeader(QHeaderView):
    """Horizontal header with an Excel-style filter button on each section."""

    filter_clicked = Signal(int, QPoint)

    def __init__(self, proxy: ColumnFilterProxy, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._proxy = proxy
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setStretchLastSection(True)

    def paintSection(
        self, painter: QPainter, rect: QRect, logical_index: int
    ) -> None:  # noqa: N802
        super().paintSection(painter, rect, logical_index)
        button = self._button_rect(rect)
        painter.save()
        painter.setPen(
            QPen(
                QColor(
                    TEXT_MUTED
                    if not self._proxy.is_filtered(logical_index)
                    else "#38bdf8"
                )
            )
        )
        painter.drawText(button, Qt.AlignmentFlag.AlignCenter, "▾")
        painter.restore()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self.logicalIndexAt(event.position().toPoint())
        if index >= 0:
            section = QRect(
                self.sectionViewportPosition(index),
                0,
                self.sectionSize(index),
                self.height(),
            )
            if self._button_rect(section).contains(event.position().toPoint()):
                global_pos = self.mapToGlobal(self._button_rect(section).bottomLeft())
                self.filter_clicked.emit(index, global_pos)
                event.accept()
                return
        super().mousePressEvent(event)

    @staticmethod
    def _button_rect(section: QRect) -> QRect:
        width = 16
        return QRect(section.right() - width - 2, section.center().y() - 8, width, 16)


class FilterPopup(QWidget):
    """Excel-like value list for one column."""

    applied = Signal(int, object)  # column, frozenset[str] | None
    sort_requested = Signal(int, Qt.SortOrder)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self._column = 0
        self._all_values: list[str] = []
        self.setStyleSheet(APP_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.resize(240, 360)

        self.btn_az = QPushButton("Sort A to Z")
        self.btn_za = QPushButton("Sort Z to A")
        for btn in (self.btn_az, self.btn_za):
            btn.setObjectName("SecondaryButton")
        self.btn_az.clicked.connect(
            lambda: self.sort_requested.emit(self._column, Qt.SortOrder.AscendingOrder)
        )
        self.btn_za.clicked.connect(
            lambda: self.sort_requested.emit(self._column, Qt.SortOrder.DescendingOrder)
        )
        layout.addWidget(self.btn_az)
        layout.addWidget(self.btn_za)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.textChanged.connect(self._repopulate)
        layout.addWidget(self.search)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)

        buttons = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("SecondaryButton")
        self.btn_ok.clicked.connect(self._apply)
        self.btn_cancel.clicked.connect(self.hide)
        buttons.addWidget(self.btn_ok)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.list_widget.itemChanged.connect(self._on_item_changed)
        self._checked: set[str] = set()

    def open_for(
        self,
        column: int,
        values: list[str],
        selected: frozenset[str] | None,
        global_pos: QPoint,
    ) -> None:
        """Show unique ``values`` with ``selected`` checked (None = all)."""
        self._column = column
        self._all_values = values
        self._checked = set(values) if selected is None else set(selected)
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self._repopulate()
        self.move(global_pos)
        self.show()
        self.search.setFocus()

    def _visible_values(self) -> list[str]:
        query = self.search.text().casefold()
        return [v for v in self._all_values if query in v.casefold()]

    def _repopulate(self) -> None:
        visible = self._visible_values()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        select_all = QListWidgetItem("(Select All)")
        select_all.setFlags(
            select_all.flags()
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
        )
        self.list_widget.addItem(select_all)
        checked_count = 0
        for value in visible:
            item = QListWidgetItem(value if value else "(Blanks)")
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            state = (
                Qt.CheckState.Checked
                if value in self._checked
                else Qt.CheckState.Unchecked
            )
            item.setCheckState(state)
            if state == Qt.CheckState.Checked:
                checked_count += 1
            self.list_widget.addItem(item)
        if not visible:
            select_all.setCheckState(Qt.CheckState.Unchecked)
        elif checked_count == len(visible):
            select_all.setCheckState(Qt.CheckState.Checked)
        elif checked_count == 0:
            select_all.setCheckState(Qt.CheckState.Unchecked)
        else:
            select_all.setCheckState(Qt.CheckState.PartiallyChecked)
        self.list_widget.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        self.list_widget.blockSignals(True)
        if item is self.list_widget.item(0):
            state = (
                Qt.CheckState.Checked
                if item.checkState() != Qt.CheckState.Unchecked
                else Qt.CheckState.Unchecked
            )
            item.setCheckState(state)
            for i in range(1, self.list_widget.count()):
                child = self.list_widget.item(i)
                child.setCheckState(state)
                value = str(child.data(Qt.ItemDataRole.UserRole) or "")
                if state == Qt.CheckState.Checked:
                    self._checked.add(value)
                else:
                    self._checked.discard(value)
        else:
            value = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if item.checkState() == Qt.CheckState.Checked:
                self._checked.add(value)
            else:
                self._checked.discard(value)
            visible = self.list_widget.count() - 1
            checked = sum(
                1
                for i in range(1, self.list_widget.count())
                if self.list_widget.item(i).checkState() == Qt.CheckState.Checked
            )
            header = self.list_widget.item(0)
            if checked == visible and visible:
                header.setCheckState(Qt.CheckState.Checked)
            elif checked == 0:
                header.setCheckState(Qt.CheckState.Unchecked)
            else:
                header.setCheckState(Qt.CheckState.PartiallyChecked)
        self.list_widget.blockSignals(False)

    def _apply(self) -> None:
        if self._checked == set(self._all_values):
            self.applied.emit(self._column, None)
        else:
            self.applied.emit(self._column, frozenset(self._checked))
        self.hide()


class DataTableView(QTableView):
    """Table that copies the selection as TSV on Ctrl+C."""

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def copy_selection(self) -> None:
        """Copy the bounding rectangle of the selection as tab-separated text."""
        indexes = self.selectedIndexes()
        if not indexes:
            return
        rows = sorted({i.row() for i in indexes})
        cols = sorted({i.column() for i in indexes})
        selected = {(i.row(), i.column()) for i in indexes}
        model = self.model()
        if model is None:
            return
        include_header = len(rows) > 1 or len(cols) > 1
        lines: list[str] = []
        if include_header:
            lines.append(
                "\t".join(
                    str(model.headerData(c, Qt.Orientation.Horizontal) or "")
                    for c in cols
                )
            )
        for row in rows:
            lines.append(
                "\t".join(
                    (
                        str(model.data(model.index(row, col)) or "")
                        if (row, col) in selected
                        else ""
                    )
                    for col in cols
                )
            )
        QApplication.clipboard().setText("\n".join(lines))


class DataTableDialog(QDialog):
    """Popup viewer for a cloned Polars frame (filters do not touch the source)."""

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"View data — {title}")
        self.resize(960, 620)
        self.setStyleSheet(APP_STYLE)
        self._source = df.clone()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "Excel-style ▾ filters on headers. Click a row/column header to select it. "
            "Ctrl+C copies the selection. Filters and exports here never change the main view."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(hint)

        self.model = PolarsTableModel(self._source, self)
        self.proxy = ColumnFilterProxy(self)
        self.proxy.setSourceModel(self.model)

        self.table = DataTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setSectionsClickable(True)
        self.table.verticalHeader().sectionClicked.connect(self._select_row)
        header = FilterHeader(self.proxy, self.table)
        self.table.setHorizontalHeader(header)
        header.sectionClicked.connect(self._select_column)
        header.filter_clicked.connect(self._open_filter)
        layout.addWidget(self.table, 1)

        self._popup = FilterPopup(self)
        self._popup.applied.connect(self._on_filter_applied)
        self._popup.sort_requested.connect(self._on_sort)

        status = QHBoxLayout()
        self.lbl_rows = QLabel()
        status.addWidget(self.lbl_rows)
        status.addStretch(1)
        self.btn_csv = QPushButton("Download CSV")
        self.btn_csv.setObjectName("SecondaryButton")
        self.btn_parquet = QPushButton("Download Parquet")
        self.btn_parquet.setObjectName("SecondaryButton")
        self.btn_close = QPushButton("Close")
        self.btn_csv.clicked.connect(lambda: self._download("csv"))
        self.btn_parquet.clicked.connect(lambda: self._download("parquet"))
        self.btn_close.clicked.connect(self.accept)
        status.addWidget(self.btn_csv)
        status.addWidget(self.btn_parquet)
        status.addWidget(self.btn_close)
        layout.addLayout(status)

        self._title_slug = "".join(ch if ch.isalnum() else "_" for ch in title).strip(
            "_"
        )
        self._refresh_count()
        self.proxy.modelReset.connect(self._refresh_count)
        self.proxy.layoutChanged.connect(self._refresh_count)

    def _refresh_count(self) -> None:
        shown = self.proxy.rowCount()
        total = self.model.rowCount()
        self.lbl_rows.setText(f"{shown:,} of {total:,} rows")

    def _select_column(self, column: int) -> None:
        self._select_line(column, rows=False)

    def _select_row(self, row: int) -> None:
        self._select_line(row, rows=True)

    def _select_line(self, index: int, *, rows: bool) -> None:
        model = self.table.selectionModel()
        if model is None or self.proxy.rowCount() == 0 or self.proxy.columnCount() == 0:
            return
        if rows:
            top = self.proxy.index(index, 0)
            bottom = self.proxy.index(index, self.proxy.columnCount() - 1)
            flag_axis = QItemSelectionModel.SelectionFlag.Rows
        else:
            top = self.proxy.index(0, index)
            bottom = self.proxy.index(self.proxy.rowCount() - 1, index)
            flag_axis = QItemSelectionModel.SelectionFlag.Columns
        selection = QItemSelection(top, bottom)
        ctrl = bool(
            QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier
        )
        flags = (
            QItemSelectionModel.SelectionFlag.Toggle
            if ctrl
            else QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        model.select(selection, flags | flag_axis)

    def _open_filter(self, column: int, global_pos: QPoint) -> None:
        values = self._unique_values(column)
        self._popup.open_for(column, values, self.proxy.allowed(column), global_pos)

    def _unique_values(self, column: int) -> list[str]:
        """Distinct display values in ``column`` after other columns' filters."""
        seen: dict[str, None] = {}
        source = self.model
        for row in range(source.rowCount()):
            if not self._passes_other_filters(row, column):
                continue
            text = str(source.data(source.index(row, column)) or "")
            seen.setdefault(text, None)
        return list(seen)

    def _passes_other_filters(self, source_row: int, skip_column: int) -> bool:
        for col in range(self.model.columnCount()):
            if col == skip_column:
                continue
            allowed = self.proxy.allowed(col)
            if allowed is None:
                continue
            text = str(self.model.data(self.model.index(source_row, col)) or "")
            if text not in allowed:
                return False
        return True

    def _on_filter_applied(self, column: int, allowed: object) -> None:
        self.proxy.set_allowed(
            column, allowed if isinstance(allowed, frozenset) else None
        )
        self.table.horizontalHeader().viewport().update()
        self._refresh_count()

    def _on_sort(self, column: int, order: Qt.SortOrder) -> None:
        self.proxy.sort(column, order)
        header = self.table.horizontalHeader()
        header.setSortIndicatorShown(True)
        header.setSortIndicator(column, order)
        self._popup.hide()

    def _visible_source_rows(self) -> list[int]:
        return [
            self.proxy.mapToSource(self.proxy.index(row, 0)).row()
            for row in range(self.proxy.rowCount())
        ]

    def _download(self, kind: str) -> None:
        displayed = self.model.displayed_frame(self._visible_source_rows())
        suffix = "csv" if kind == "csv" else "parquet"
        caption = "Save CSV" if kind == "csv" else "Save Parquet"
        filt = "CSV (*.csv)" if kind == "csv" else "Parquet (*.parquet)"
        path, _chosen = QFileDialog.getSaveFileName(
            self,
            caption,
            str(Path.home() / f"ycn_{self._title_slug}.{suffix}"),
            filt,
        )
        if not path:
            return
        try:
            if kind == "csv":
                displayed.write_csv(path)
            else:
                displayed.write_parquet(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(
            self, "Saved", f"Wrote {displayed.height:,} rows to\n{path}"
        )
