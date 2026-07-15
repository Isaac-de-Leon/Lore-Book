# settings_window.py — Settings dialog for camera, UI, and game/set selection.

import logging
import os
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
)

from lorebook.core.card_prices import SUPPORTED_CURRENCIES
from lorebook.core.csv_manager import get_available_sets
from lorebook.core.game_types import BASE_DATABASE_PATH
from lorebook.ui.styles import APP_STYLESHEET


class SettingsWindow(QDialog):
    """
    Settings dialog for camera and UI options.

    Handles camera selection, foil/rotation/crop preferences, confidence
    threshold, debug mode, and per-game/set filtering.  Configuration is
    saved to ui_settings.json via the parent MainWindow on Apply.
    """

    logger = logging.getLogger("SettingsWindow")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(390, 675)
        self.setStyleSheet(APP_STYLESHEET)
        self.logger.info("Initializing settings window")

        layout = QFormLayout()

        # Camera index selector (0–4)
        self.camera_combo = QComboBox()
        for i in range(5):
            self.camera_combo.addItem(f"Camera {i}", userData=i)
        self.camera_combo.setCurrentIndex(min(getattr(parent, "camera_index", 0), 4))
        layout.addRow("Camera:", self.camera_combo)

        help_label = QLabel("Note: If camera doesn't work, try a different index\nand restart the application.")
        help_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow(help_label)

        self.keep_foil_checked = QCheckBox("Keep Foil Checked")
        self.keep_foil_checked.setChecked(getattr(parent, "keep_foil_checked", False))
        layout.addRow(self.keep_foil_checked)

        self.rotate_checkbox = QCheckBox("Rotate preview & capture 180°")
        self.rotate_checkbox.setChecked(getattr(parent, "rotate_display", False))
        layout.addRow(self.rotate_checkbox)

        self.crop_checkbox = QCheckBox("Scan only inside focus box")
        self.crop_checkbox.setChecked(getattr(parent, "crop_to_focus", True))
        layout.addRow(self.crop_checkbox)

        self.auto_scan_checkbox = QCheckBox("Auto-scan when a card settles in the focus box")
        self.auto_scan_checkbox.setChecked(getattr(parent, "auto_scan", False))
        layout.addRow(self.auto_scan_checkbox)

        self.confidence_input = QLineEdit(
            str(int(100 * getattr(parent, "confidence_threshold", 0.90)))
        )
        layout.addRow("Confidence Threshold (%):", self.confidence_input)

        self.foil_threshold_input = QLineEdit(
            str(round(100 * getattr(parent, "foil_threshold", 0.08), 1))
        )
        layout.addRow("Foil Threshold (%):", self.foil_threshold_input)

        # Display currency for scanned-card market prices (source data is USD)
        self.currency_combo = QComboBox()
        for code in SUPPORTED_CURRENCIES:
            self.currency_combo.addItem(code, userData=code)
        current_currency = getattr(parent, "currency", "USD")
        idx = self.currency_combo.findData(current_currency)
        self.currency_combo.setCurrentIndex(max(0, idx))
        layout.addRow("Price currency:", self.currency_combo)

        self.debug_mode_checkbox = QCheckBox("Enable Debug Mode (heatmap overlay)")
        self.debug_mode_checkbox.setChecked(getattr(parent, "debug_mode", False))
        layout.addRow(self.debug_mode_checkbox)

        # Game / set tree
        self.set_tree = QTreeWidget()
        self.set_tree.setHeaderHidden(True)
        self.set_tree.setMinimumHeight(200)

        selected_games = getattr(parent, "selected_games", {})
        selected_sets = getattr(parent, "selected_sets", {})

        game_folders: List[str] = []
        if os.path.exists(BASE_DATABASE_PATH):
            game_folders = [
                item for item in os.listdir(BASE_DATABASE_PATH)
                if os.path.isdir(os.path.join(BASE_DATABASE_PATH, item))
                and item not in ("__pycache__",)
            ]

        for game_name in game_folders:
            game_root = QTreeWidgetItem(self.set_tree, [game_name])
            game_root.setFlags(
                game_root.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable
            )
            is_selected = selected_games.get(game_name.lower(), False)
            game_root.setCheckState(0, Qt.Checked if is_selected else Qt.Unchecked)

            # Pass db_path explicitly — avoids touching the global databasePath
            game_db_path = os.path.join(BASE_DATABASE_PATH, game_name)
            try:
                game_sets = get_available_sets(db_path=game_db_path)
                game_selected_sets = selected_sets.get(game_name.lower(), [])
                for set_code in game_sets:
                    set_item = QTreeWidgetItem(game_root, [f"Set {set_code}"])
                    set_item.setFlags(set_item.flags() | Qt.ItemIsUserCheckable)
                    set_item.setCheckState(
                        0, Qt.Checked if set_code in game_selected_sets else Qt.Unchecked
                    )
                    set_item.setData(0, Qt.UserRole, (game_name.lower(), set_code))
            except Exception as e:
                self.logger.warning(f"Error getting sets for {game_name}: {e}")

        self.set_tree.expandAll()
        layout.addRow("Filter Sets:", self.set_tree)

        rebuild_button = QPushButton("Rebuild Database")
        rebuild_button.setToolTip("Re-scan all card images and rebuild the feature cache")
        rebuild_button.clicked.connect(self._rebuild_database)
        layout.addRow(rebuild_button)

        apply_button = QPushButton("Apply")
        # lambda guards against QPushButton.clicked's `checked` bool binding to `close`
        apply_button.clicked.connect(lambda: self.apply_settings())
        layout.addRow(apply_button)
        self.setLayout(layout)

    def apply_settings(self, close: bool = True):
        """Apply settings to parent MainWindow and persist them.

        Set close=False to apply without dismissing the dialog (used by
        the Rebuild Database button).
        """
        parent = self.parent()
        if not parent:
            if close:
                self.close()
            return

        parent.keep_foil_checked = self.keep_foil_checked.isChecked()
        parent.rotate_display = self.rotate_checkbox.isChecked()
        parent.crop_to_focus = self.crop_checkbox.isChecked()
        parent.auto_scan = self.auto_scan_checkbox.isChecked()
        parent.debug_mode = self.debug_mode_checkbox.isChecked()
        parent.camera_index = self.camera_combo.currentData()
        parent.currency = self.currency_combo.currentData()

        try:
            pct = float(self.confidence_input.text())
            parent.confidence_threshold = max(0.0, min(1.0, pct / 100.0))
        except Exception:
            parent.confidence_threshold = 0.90

        try:
            fpct = float(self.foil_threshold_input.text())
            parent.foil_threshold = max(0.0, min(1.0, fpct / 100.0))
        except Exception:
            parent.foil_threshold = 0.08

        # Collect game / set selections from tree
        selected_games = {"lorcana": False, "riftbound": False}
        selected_sets = {"lorcana": [], "riftbound": []}

        for i in range(self.set_tree.topLevelItemCount()):
            root = self.set_tree.topLevelItem(i)
            game_name = root.text(0).lower()
            if game_name not in selected_games:
                continue
            selected_games[game_name] = root.checkState(0) != Qt.Unchecked
            if selected_games[game_name]:
                for j in range(root.childCount()):
                    child = root.child(j)
                    if child.checkState(0) == Qt.Checked:
                        set_code = child.text(0).split(" ")[-1]
                        selected_sets[game_name].append(set_code)

        self.logger.info(f"Applying settings — games: {selected_games}, sets: {selected_sets}")
        parent.selected_games = selected_games
        parent.selected_sets = selected_sets

        # Load the first selected game's database (rebuilds the match index
        # and keeps MainWindow's cache tracker in sync)
        for game_name, is_selected in selected_games.items():
            if is_selected:
                parent.load_game_database(game_name.capitalize())
                break

        parent.save_settings()
        if close:
            self.close()

    def _rebuild_database(self):
        """Apply current settings, trigger a background DB rebuild, then close."""
        self.apply_settings(close=False)
        parent = self.parent()
        if parent and hasattr(parent, "start_db_build_in_background"):
            parent.start_db_build_in_background()
        self.close()
