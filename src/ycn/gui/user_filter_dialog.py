"""Mouse-driven ``(term, issuer)`` cell picker for the loaded panel.

Shows the term x issuer intersection of the *already loaded and filtered*
panel. A cell exists only where the data does; missing combinations render as
inert empty space. Only checked cells reach the multi-layer network.

Everything here is operable with the mouse alone: cells toggle on click
anywhere in the cell, header clicks toggle a whole row or column (flashing it
so the extent of the change is visible), Shift and Ctrl extend that to several
rows/columns at once, and the row/column orders are flipped from combo boxes
rather than by typing. Each header carries a ``checked/available`` count so a
partially selected row or column is visible at a glance rather than having to
be inferred from the grid.

Two Qt behaviours are deliberately not used:

* ``ItemIsUserCheckable`` -- with it set, Qt toggles the item itself when a
  click lands on the indicator, and this dialog's ``cellClicked`` handler would
  then toggle it straight back, so clicking the checkbox (the obvious target)
  silently did nothing. :class:`_CheckCellDelegate` owns both the painting and
  the hit handling instead, which also lets the indicator be centred and drawn
  heavier.
* The "check unless fully checked" rule for header toggles. It needs *two*
  clicks to clear a partially selected row -- the first fills it -- which reads
  as the click not registering, and silently reinstates cells the user had
  removed one at a time. Header toggles now clear whenever anything in the
  group is checked, and only fill when the group is completely empty.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.yield_curve import sort_terms
from ycn.gui.styles import APP_STYLE, BG_CONTROL, TEXT_MUTED

# Marks an item as a combination the panel actually contains. Kept separate
# from the item flags so the "is this a real cell" test never depends on Qt's
# check handling.
_PRESENT_ROLE = Qt.ItemDataRole.UserRole

# Header items display "<label>  <n>/<total>"; the bare label lives here so the
# term/issuer identity never has to be parsed back out of the display text.
_LABEL_ROLE = Qt.ItemDataRole.UserRole

# Raw value behind Qt.CheckState.Checked, for comparing against the plain int
# that QModelIndex.data() returns.
_CHECKED = Qt.CheckState.Checked.value

# Cells that exist in the data; distinct from the flat background used for
# combinations the panel simply does not contain.
_PRESENT_BG = QColor(BG_CONTROL)
_MISSING_BG = QColor("#151821")
_BOX_BORDER = QColor("#64748b")
_BOX_FILL = QColor("#2563eb")
_TICK = QColor("#ffffff")
_FLASH = QColor(125, 211, 252, 125)

# Two on/off pulses at this interval -- long enough to register, short enough
# not to delay the next click.
_FLASH_PULSE_MS = 95
_FLASH_PULSES = 2

# Indicator geometry.
_BOX_MAX = 20
_BOX_MIN = 12
_BOX_MARGIN = 8
_TICK_WIDTH = 2.8


class _CheckCellDelegate(QStyledItemDelegate):
    """Paints a centred, heavy check indicator and the row/column flash."""

    def __init__(self, dialog: "UserFilterDialog") -> None:
        super().__init__(dialog)
        self._dialog = dialog

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        present = bool(index.data(_PRESENT_ROLE))
        painter.fillRect(option.rect, _PRESENT_BG if present else _MISSING_BG)

        if self._dialog.is_flashing(index.row(), index.column()):
            painter.fillRect(option.rect, _FLASH)

        if present:
            # QModelIndex.data() hands back a plain int for CheckStateRole, not
            # a Qt.CheckState, so compare against the enum's value.
            state = index.data(Qt.ItemDataRole.CheckStateRole)
            checked = state is not None and int(state) == _CHECKED
            self._paint_box(painter, option.rect, checked)

        painter.restore()

    @staticmethod
    def _paint_box(painter: QPainter, rect: QRect, checked: bool) -> None:
        side = max(
            _BOX_MIN,
            min(_BOX_MAX, min(rect.width(), rect.height()) - _BOX_MARGIN),
        )
        box = QRect(0, 0, side, side)
        box.moveCenter(rect.center())

        if checked:
            painter.setPen(QPen(_BOX_FILL, 1.0))
            painter.setBrush(QBrush(_BOX_FILL))
        else:
            painter.setPen(QPen(_BOX_BORDER, 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(box, 3.5, 3.5)

        if not checked:
            return

        # Proportional tick so it stays centred and balanced at any box size.
        path = QPainterPath()
        path.moveTo(box.left() + side * 0.24, box.top() + side * 0.52)
        path.lineTo(box.left() + side * 0.43, box.top() + side * 0.71)
        path.lineTo(box.left() + side * 0.77, box.top() + side * 0.30)
        pen = QPen(_TICK, _TICK_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)


class UserFilterDialog(QDialog):
    """Check/uncheck the ``(term, issuer)`` cells that feed the network."""

    def __init__(
        self,
        available: set[tuple[str, str]],
        selected: set[tuple[str, str]] | None,
        *,
        term_label: str = "Term",
        issuer_label: str = "Issuer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("User Filter — select data cells")
        self.setStyleSheet(APP_STYLE)
        self.resize(1040, 660)

        self._available = set(available)
        self._selected = (
            set(self._available)
            if selected is None
            else set(selected) & self._available
        )
        self._term_label = term_label
        self._issuer_label = issuer_label

        self._terms = sort_terms({t for t, _ in self._available})
        self._issuers = sorted({i for _, i in self._available})
        self._syncing = False

        # Shift extends from the last plainly-clicked header; Ctrl repeats the
        # direction of the last header action rather than toggling per section.
        self._anchor_row: int | None = None
        self._anchor_col: int | None = None
        self._last_header_target: bool | None = None

        # Rows/columns currently flashing after a header toggle.
        self._flash_rows: set[int] = set()
        self._flash_cols: set[int] = set()
        self._flash_on = False
        self._flash_step = 0
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._on_flash_tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        intro = QLabel(
            f"Rows are {term_label.lower()}s, columns are "
            f"{issuer_label.lower()}s. Click a cell to toggle it. Click a "
            "header to clear that whole row/column (or fill it, if it is "
            "already empty) — then <b>Shift</b>+click another header to repeat "
            "that over a range, or <b>Ctrl</b>+click to repeat it on scattered "
            "rows/columns. Each header shows how many of its cells are "
            "selected. Blank cells have no data in the current selection."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(intro)

        layout.addLayout(self._build_toolbar())

        self.table = QTableWidget(self)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(True)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setItemDelegate(_CheckCellDelegate(self))
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.horizontalHeader().sectionClicked.connect(self._on_column_clicked)
        self.table.verticalHeader().sectionClicked.connect(self._on_row_clicked)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        # Room for the centred indicator and the two-line header count.
        self.table.horizontalHeader().setMinimumSectionSize(52)
        self.table.horizontalHeader().setMinimumHeight(42)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.verticalHeader().setSectionsClickable(True)
        layout.addWidget(self.table, stretch=1)

        self.lbl_count = QLabel()
        self.lbl_count.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self.lbl_count)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._rebuild()

    # ------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        btn_all = QPushButton("Check all")
        btn_all.setObjectName("SecondaryButton")
        btn_all.clicked.connect(lambda: self._set_all(True))
        row.addWidget(btn_all)

        btn_none = QPushButton("Uncheck all")
        btn_none.setObjectName("SecondaryButton")
        btn_none.clicked.connect(lambda: self._set_all(False))
        row.addWidget(btn_none)

        btn_invert = QPushButton("Invert")
        btn_invert.setObjectName("SecondaryButton")
        btn_invert.clicked.connect(self._invert)
        row.addWidget(btn_invert)

        row.addSpacing(16)

        row.addWidget(QLabel(f"{self._term_label} order:"))
        self.cmb_row_order = QComboBox()
        self.cmb_row_order.addItem("Ascending", False)
        self.cmb_row_order.addItem("Descending", True)
        self.cmb_row_order.currentIndexChanged.connect(self._rebuild)
        row.addWidget(self.cmb_row_order)

        row.addWidget(QLabel(f"{self._issuer_label} order:"))
        self.cmb_col_order = QComboBox()
        self.cmb_col_order.addItem("A → Z", False)
        self.cmb_col_order.addItem("Z → A", True)
        self.cmb_col_order.currentIndexChanged.connect(self._rebuild)
        row.addWidget(self.cmb_col_order)

        row.addStretch(1)
        return row

    # -------------------------------------------------------------- render
    def _ordered_terms(self) -> list[str]:
        terms = list(self._terms)
        if self.cmb_row_order.currentData():
            terms.reverse()
        return terms

    def _ordered_issuers(self) -> list[str]:
        issuers = list(self._issuers)
        if self.cmb_col_order.currentData():
            issuers.reverse()
        return issuers

    def _rebuild(self) -> None:
        """Repaint the whole grid for the current row/column ordering."""
        self._stop_flash()
        self._anchor_row = None
        self._anchor_col = None
        terms = self._ordered_terms()
        issuers = self._ordered_issuers()

        self._syncing = True
        self.table.clear()
        self.table.setRowCount(len(terms))
        self.table.setColumnCount(len(issuers))

        for c, issuer in enumerate(issuers):
            head = QTableWidgetItem(issuer)
            head.setData(_LABEL_ROLE, issuer)
            self.table.setHorizontalHeaderItem(c, head)
        for r, term in enumerate(terms):
            head = QTableWidgetItem(term)
            head.setData(_LABEL_ROLE, term)
            self.table.setVerticalHeaderItem(r, head)

        for r, term in enumerate(terms):
            for c, issuer in enumerate(issuers):
                item = QTableWidgetItem()
                present = (term, issuer) in self._available
                # Never ItemIsUserCheckable: Qt would toggle the item itself on
                # an indicator click and fight _on_cell_clicked (see module
                # docstring). This dialog owns every state change.
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled if present else Qt.ItemFlag.NoItemFlags
                )
                item.setData(_PRESENT_ROLE, present)
                if present:
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if (term, issuer) in self._selected
                        else Qt.CheckState.Unchecked
                    )
                self.table.setItem(r, c, item)

        self._syncing = False
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        """Update the running total and every header's checked/available count."""
        self.lbl_count.setText(
            f"{len(self._selected)} of {len(self._available)} available cells "
            f"selected  ·  {len(self._terms)} {self._term_label.lower()}s × "
            f"{len(self._issuers)} {self._issuer_label.lower()}s"
        )
        for r in range(self.table.rowCount()):
            head = self.table.verticalHeaderItem(r)
            cells = [
                c for c in range(self.table.columnCount()) if self._is_present(r, c)
            ]
            done = sum(1 for c in cells if self._is_checked(r, c))
            head.setText(f"{head.data(_LABEL_ROLE)}  {done}/{len(cells)}")
        for c in range(self.table.columnCount()):
            head = self.table.horizontalHeaderItem(c)
            cells = [r for r in range(self.table.rowCount()) if self._is_present(r, c)]
            done = sum(1 for r in cells if self._is_checked(r, c))
            head.setText(f"{head.data(_LABEL_ROLE)}\n{done}/{len(cells)}")

    def _is_present(self, row: int, col: int) -> bool:
        item = self.table.item(row, col)
        return item is not None and bool(item.data(_PRESENT_ROLE))

    def _is_checked(self, row: int, col: int) -> bool:
        item = self.table.item(row, col)
        return item is not None and item.checkState() == Qt.CheckState.Checked

    def _apply_state(self, row: int, col: int, checked: bool) -> None:
        """Set one existing cell's state in both the model and the view."""
        if not self._is_present(row, col):
            return
        term = self.table.verticalHeaderItem(row).data(_LABEL_ROLE)
        issuer = self.table.horizontalHeaderItem(col).data(_LABEL_ROLE)
        if checked:
            self._selected.add((term, issuer))
        else:
            self._selected.discard((term, issuer))
        self.table.item(row, col).setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )

    # --------------------------------------------------------------- flash
    def is_flashing(self, row: int, col: int) -> bool:
        """True while ``(row, col)`` belongs to a flashing row/column."""
        if not self._flash_on:
            return False
        return row in self._flash_rows or col in self._flash_cols

    def _start_flash(self, rows: set[int] | None, cols: set[int] | None) -> None:
        self._flash_rows = set(rows or ())
        self._flash_cols = set(cols or ())
        self._flash_step = 0
        self._flash_on = True
        self.table.viewport().update()
        self._flash_timer.start(_FLASH_PULSE_MS)

    def _on_flash_tick(self) -> None:
        self._flash_step += 1
        if self._flash_step >= _FLASH_PULSES * 2:
            self._stop_flash()
            return
        self._flash_on = self._flash_step % 2 == 0
        self.table.viewport().update()

    def _stop_flash(self) -> None:
        self._flash_timer.stop()
        self._flash_rows = set()
        self._flash_cols = set()
        self._flash_on = False
        if self.table.viewport() is not None:
            self.table.viewport().update()

    # ------------------------------------------------------------ handlers
    def _on_cell_clicked(self, row: int, col: int) -> None:
        if self._syncing or not self._is_present(row, col):
            return
        self._apply_state(row, col, not self._is_checked(row, col))
        self._refresh_labels()

    @staticmethod
    def _resolve_sections(
        clicked: int, anchor: int | None, modifiers
    ) -> tuple[list[int], bool, int | None]:
        """Which header sections a click acts on.

        Returns ``(sections, repeat_last, new_anchor)``. A plain click decides
        its own direction and records it; Shift then spans the anchor to the
        clicked section and Ctrl picks out a single scattered section, both
        *repeating* that recorded direction. Recomputing per group instead
        would make Shift clear a range the user was plainly extending a fill
        across, and would let Ctrl flip sections independently.
        """
        if modifiers & Qt.KeyboardModifier.ShiftModifier and anchor is not None:
            lo, hi = sorted((anchor, clicked))
            return list(range(lo, hi + 1)), True, anchor
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return [clicked], True, clicked if anchor is None else anchor
        return [clicked], False, clicked

    def _apply_sections(
        self, sections: list[int], *, is_row: bool, repeat_last: bool
    ) -> None:
        """Clear or fill every present cell in the given rows/columns.

        The whole group resolves to ONE target state, so a mixed selection ends
        up uniform rather than each section flipping on its own.
        """
        if is_row:
            cells = [
                (s, c)
                for s in sections
                for c in range(self.table.columnCount())
                if self._is_present(s, c)
            ]
        else:
            cells = [
                (r, s)
                for s in sections
                for r in range(self.table.rowCount())
                if self._is_present(r, s)
            ]
        if not cells:
            return

        if repeat_last and self._last_header_target is not None:
            target = self._last_header_target
        else:
            # Clear whenever anything is checked; only fill a wholly empty
            # group. The old "fill unless completely full" rule needed two
            # clicks to clear a partial row, which read as a dropped click.
            target = not any(self._is_checked(r, c) for r, c in cells)

        for r, c in cells:
            self._apply_state(r, c, target)
        self._last_header_target = target
        self._refresh_labels()
        self._start_flash(
            set(sections) if is_row else None,
            None if is_row else set(sections),
        )

    def _on_row_clicked(self, row: int) -> None:
        if self._syncing:
            return
        sections, repeat, anchor = self._resolve_sections(
            row, self._anchor_row, QApplication.keyboardModifiers()
        )
        self._anchor_row = anchor
        self._apply_sections(sections, is_row=True, repeat_last=repeat)

    def _on_column_clicked(self, col: int) -> None:
        if self._syncing:
            return
        sections, repeat, anchor = self._resolve_sections(
            col, self._anchor_col, QApplication.keyboardModifiers()
        )
        self._anchor_col = anchor
        self._apply_sections(sections, is_row=False, repeat_last=repeat)

    def _set_all(self, checked: bool) -> None:
        self._selected = set(self._available) if checked else set()
        self._last_header_target = None
        self._rebuild()

    def _invert(self) -> None:
        self._selected = self._available - self._selected
        self._last_header_target = None
        self._rebuild()

    # -------------------------------------------------------------- result
    def selected_cells(self) -> set[tuple[str, str]]:
        """The checked ``(term, issuer)`` pairs."""
        return set(self._selected)
