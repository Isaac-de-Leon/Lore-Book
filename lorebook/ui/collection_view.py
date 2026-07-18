# collection_view.py — Collection tab: browse a game's <Game>List.csv and export it.

import logging
import os
import shutil
from typing import List, Optional

from PySide6.QtCore import Qt
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
from lorebook.core.csv_manager import clear_collection, read_collection_rows
from lorebook.core.game_types import BASE_DATABASE_PATH, csv_for_game
from lorebook.ui.icons import get_icon

_COLUMNS = ["Set", "Card", "Variant", "Count", "Name"]


def _game_folders() -> List[str]:
    """Game folder names under Card_Images/ (same discovery as the settings tree)."""
    if not os.path.exists(BASE_DATABASE_PATH):
        return []
    return sorted(
        item for item in os.listdir(BASE_DATABASE_PATH)
        if os.path.isdir(os.path.join(BASE_DATABASE_PATH, item))
        and item not in ("__pycache__",)
    )


class CollectionView(QWidget):
    """
    Read-only view of a game's collection CSV with a game selector, a
    sortable table, and an "Export CSV…" (save-a-copy) button. The CSV on
    disk stays the single source of truth — this widget never writes to it.
    """

    def __init__(self, initial_game: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Toolbar row
        self.game_combo = QComboBox()
        games = _game_folders()
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
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(_COLUMNS) - 1, QHeaderView.Stretch)  # Name fills
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
        return self.game_combo.currentText()

    def refresh(self) -> None:
        """Reload the table from the selected game's CSV."""
        game = self.current_game()
        rows = read_collection_rows(game) if game else []

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
            count_item.setData(Qt.DisplayRole, count_val)  # int → numeric sort

            name = name_for(set_code, card_code, game) or ""
            for c, item in enumerate([
                QTableWidgetItem(set_code),
                QTableWidgetItem(card_code),
                QTableWidgetItem(variant),
                count_item,
                QTableWidgetItem(name),
            ]):
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
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export collection CSV", source, "CSV files (*.csv)"
        )
        if not dest:
            return  # user cancelled
        try:
            shutil.copyfile(source, dest)
            self.summary_label.setText(f"Exported to {dest}")
            self.logger.info(f"Exported {source} to {dest}")
        except OSError as e:
            self.summary_label.setText(f"Export failed: {e}")
            self.logger.error(f"Error exporting {source} to {dest}: {e}")

    def _clear_csv(self) -> None:
        """Empty the selected game's collection CSV after an explicit confirm."""
        game = self.current_game()
        if not game:
            return
        rows = read_collection_rows(game)
        if not rows:
            self.summary_label.setText("Collection is already empty.")
            return
        total = sum(int(r[3]) for r in rows if str(r[3]).lstrip("-").isdigit())
        answer = QMessageBox.warning(
            self,
            "Clear collection",
            f"Delete all {len(rows)} entries ({total} cards) from "
            f"{os.path.basename(csv_for_game(game))}?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        clear_collection(game)
        self.logger.info(f"Cleared collection CSV for {game}")
        # The scanner's single-level Undo now points at rows that no longer
        # exist — disable it rather than let it "undo" into the empty file.
        win = self.window()
        if hasattr(win, "undo_btn"):
            win.undo_btn.setEnabled(False)
            win._last_add = None
        self.refresh()
        self.summary_label.setText("Collection cleared.")
