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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lorebook.core.card_names import name_for
from lorebook.core.csv_manager import read_collection_rows
from lorebook.core.game_types import BASE_DATABASE_PATH, csv_for_game

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

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)

        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(self._export_csv)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("Game:"))
        toolbar.addWidget(self.game_combo)
        toolbar.addStretch()
        toolbar.addWidget(self.summary_label)
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(export_btn)

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

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.refresh()

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
