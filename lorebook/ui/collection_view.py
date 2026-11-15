"""Collection tab: browse a game's <Game>List.csv and export it."""

import logging
import os
import shutil

from PySide6.QtCore import QStandardPaths, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lorebook.core.card_names import name_for
from lorebook.core.csv_manager import CsvReadError, clear_collection, read_collection_rows
from lorebook.core.game_types import csv_for_game
from lorebook.core.image_fetcher import selectable_games
from lorebook.ui.icons import get_icon

logger = logging.getLogger(__name__)

_COLUMNS = ["Set", "Card", "Variant", "Count", "Name"]


def _export_folder() -> str:
    """Return the folder to suggest for exports: Documents, or the home folder if there is none."""
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return os.path.normpath(documents or os.path.expanduser("~"))  # Qt hands back forward slashes


class CollectionView(QWidget):
    """
    Read-only view of a game's collection CSV with a game selector, a
    sortable table, and an "Export CSV…" (save-a-copy) button. The CSV on
    disk stays the single source of truth — this widget never writes to it.
    """

    cleared = Signal(str)  # game whose collection CSV was emptied

    def __init__(self, initial_game: str | None = None, parent: QWidget | None = None) -> None:
        """Build the toolbar and table, then load initial_game's collection."""
        super().__init__(parent)

        # Toolbar row
        self.game_combo = QComboBox()
        games = selectable_games()
        self.game_combo.addItems(games)
        if initial_game and initial_game in games:
            self.game_combo.setCurrentText(initial_game)
        self.game_combo.currentTextChanged.connect(lambda _: self.refresh())

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("statusLabel")

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)

        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export_csv)

        self.clear_btn = QPushButton("Clear…")
        self.clear_btn.setObjectName("dangerBtn")
        self.clear_btn.setToolTip("Delete every entry in this game's collection CSV")
        self.clear_btn.clicked.connect(self._clear_csv)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("Game:"))
        toolbar.addWidget(self.game_combo)
        toolbar.addStretch()
        toolbar.addWidget(self.summary_label)
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addWidget(self.clear_btn)

        # Table
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)  # Name fills
        # Short values ("011") would otherwise shrink columns below their
        # header text + sort arrow, truncating "Set" to garbage.
        header.setMinimumSectionSize(72)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.refresh()

    def apply_icons(self, color: str, danger_color: str) -> None:
        """Tint the toolbar icons — called by MainWindow on every theme change."""
        self.refresh_btn.setIcon(get_icon("refresh", color))
        self.export_btn.setIcon(get_icon("export", color))
        self.clear_btn.setIcon(get_icon("trash", danger_color))

    def current_game(self) -> str:
        """Return the game selected in the dropdown."""
        return self.game_combo.currentText()

    def _sync_games(self) -> None:
        """Pick up game folders created (or removed) since the dropdown was filled, keeping the selection."""
        games = selectable_games()
        if games == [self.game_combo.itemText(i) for i in range(self.game_combo.count())]:
            return
        current = self.current_game()
        self.game_combo.blockSignals(True)  # refresh() is already running; don't re-enter it
        try:
            self.game_combo.clear()
            self.game_combo.addItems(games)
            if current in games:
                self.game_combo.setCurrentText(current)
        finally:
            self.game_combo.blockSignals(False)

    def refresh(self) -> None:
        """Reload the table from the selected game's CSV."""
        self._sync_games()
        game = self.current_game()
        try:
            rows = read_collection_rows(game, strict=True) if game else []
        except CsvReadError as e:
            logger.error(str(e))
            self.table.setRowCount(0)
            self.summary_label.setText(f"Couldn't read {csv_for_game(game)} — check the file.")
            return

        self.table.setSortingEnabled(False)  # sorting mid-fill scrambles rows
        self.table.setRowCount(len(rows))
        total = 0
        for r, (set_code, card_code, variant, count) in enumerate(rows):
            try:
                count_val = int(count)
            except (TypeError, ValueError):
                count_val = 0
            total += count_val

            count_item = QTableWidgetItem()
            count_item.setData(Qt.ItemDataRole.DisplayRole, count_val)  # int → numeric sort

            name = name_for(set_code, card_code, game) or ""
            for c, item in enumerate(
                [
                    QTableWidgetItem(set_code),
                    QTableWidgetItem(card_code),
                    QTableWidgetItem(variant),
                    count_item,
                    QTableWidgetItem(name),
                ]
            ):
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)

        self.summary_label.setText(f"{len(rows)} unique · {total} cards")

    def _export_csv(self) -> None:
        """Save a copy of the selected game's CSV wherever the user chooses."""
        game = self.current_game()
        if not game:
            return
        source = csv_for_game(game)
        if not os.path.exists(source):
            self.summary_label.setText("Nothing to export — no collection file yet.")
            return
        # Suggest the Documents folder: pre-filling the collection file's own
        # path made the default answer "copy the file onto itself".
        suggested = os.path.join(_export_folder(), os.path.basename(source))
        dest, _ = QFileDialog.getSaveFileName(self, "Export collection CSV", suggested, "CSV files (*.csv)")
        if not dest:
            return  # user cancelled
        if os.path.exists(dest) and os.path.samefile(source, dest):
            self.summary_label.setText("That is the collection file itself — pick another name or folder.")
            return
        try:
            shutil.copyfile(source, dest)
            self.summary_label.setText(f"Exported to {dest}")
            logger.info("Exported %s to %s", source, dest)
        except OSError as e:
            self.summary_label.setText(f"Export failed: {e}")
            logger.error("Error exporting %s to %s: %s", source, dest, e)

    def _clear_csv(self) -> None:
        """Empty the selected game's collection CSV after an explicit confirm."""
        game = self.current_game()
        if not game:
            return
        try:
            rows = read_collection_rows(game, strict=True)
        except CsvReadError:
            self.summary_label.setText(f"Couldn't read {csv_for_game(game)} — not clearing it.")
            return
        if not rows:
            self.summary_label.setText("Collection is already empty.")
            return
        total = sum(int(r[3]) for r in rows if str(r[3]).lstrip("-").isdigit())
        answer = QMessageBox.warning(
            self,
            "Clear collection",
            f"Delete all {len(rows)} entries ({total} cards) from "
            f"{os.path.basename(csv_for_game(game))}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            clear_collection(game)
        except OSError as e:  # locked (e.g. open in Excel) — the file is untouched
            logger.error("Could not clear %s: %s", csv_for_game(game), e)
            self.summary_label.setText(
                f"Not cleared — {os.path.basename(csv_for_game(game))} could not be updated "
                "(open in another program?)"
            )
            return
        logger.info("Cleared collection CSV for %s", game)
        # The scanner's single-level Undo now points at rows that no longer
        # exist — whoever owns it should disable it.
        self.cleared.emit(game)
        self.refresh()
        self.summary_label.setText("Collection cleared.")
