# settings_window.py — Settings dialog for camera, UI, and game/set selection.

import logging
import os
from typing import Dict, List, Optional

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
from lorebook.core.game_types import (
    BASE_DATABASE_PATH,
    game_folders,
    sets_to_display,
    sets_to_store,
)


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
        self.setFixedSize(390, 715)
        # Stylesheet is inherited from the parent MainWindow, so the dialog
        # always matches the active theme.
        self.logger.info("Initializing settings window")

        layout = QFormLayout()

        # Theme selector
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", userData="dark")
        self.theme_combo.addItem("Light", userData="light")
        if getattr(parent, "theme", "dark") == "light":
            self.theme_combo.setCurrentIndex(1)
        layout.addRow("Theme:", self.theme_combo)

        # Camera index selector (0–4)
        self.camera_combo = QComboBox()
        for i in range(5):
            self.camera_combo.addItem(f"Camera {i}", userData=i)
        self.camera_combo.setCurrentIndex(min(getattr(parent, "camera_index", 0), 4))
        layout.addRow("Camera:", self.camera_combo)

        help_label = QLabel("Note: If camera doesn't work, try a different index\nand restart the application.")
        help_label.setObjectName("statusLabel")  # themed muted text
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

        for game_name in game_folders():
            key = game_name.lower()
            is_selected = bool(selected_games.get(key, False))
            game_root = QTreeWidgetItem(self.set_tree, [game_name])
            game_root.setFlags(
                game_root.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable
            )
            # Only used as-is when the folder has no sets yet; with children
            # the box is derived from them (Qt auto-tristate).
            game_root.setCheckState(0, Qt.Checked if is_selected else Qt.Unchecked)

            # Pass db_path explicitly — avoids touching the global databasePath
            game_db_path = os.path.join(BASE_DATABASE_PATH, game_name)
            try:
                game_sets = get_available_sets(db_path=game_db_path)
                ticked = set(sets_to_display(selected_sets.get(key, []), game_sets, is_selected))
                for set_code in game_sets:
                    set_item = QTreeWidgetItem(game_root, [f"Set {set_code}"])
                    set_item.setFlags(set_item.flags() | Qt.ItemIsUserCheckable)
                    set_item.setCheckState(0, Qt.Checked if set_code in ticked else Qt.Unchecked)
                    set_item.setData(0, Qt.UserRole, set_code)
            except Exception as e:
                self.logger.warning(f"Error getting sets for {game_name}: {e}")

        # One active game at a time: ticking a game (or any of its sets)
        # unticks the others, so the dialog never shows a selection that
        # scanning would silently ignore.
        self.set_tree.itemChanged.connect(self._enforce_single_game)

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

        if hasattr(parent, "apply_theme"):
            parent.apply_theme(self.theme_combo.currentData())
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

        # Collect game / set selections from tree — every game folder, not
        # just the built-in two, so new games are selectable too.
        selected_games: Dict[str, bool] = {}
        selected_sets: Dict[str, List[str]] = {}
        active_folder: Optional[str] = None
        for i in range(self.set_tree.topLevelItemCount()):
            root = self.set_tree.topLevelItem(i)
            folder = root.text(0)
            key = folder.lower()
            selected = root.checkState(0) != Qt.Unchecked
            selected_games[key] = selected
            available = [root.child(j).data(0, Qt.UserRole) for j in range(root.childCount())]
            checked = [
                root.child(j).data(0, Qt.UserRole)
                for j in range(root.childCount())
                if root.child(j).checkState(0) == Qt.Checked
            ]
            selected_sets[key] = sets_to_store(checked, available) if selected else []
            if selected and active_folder is None:
                active_folder = folder

        self.logger.info(f"Applying settings — games: {selected_games}, sets: {selected_sets}")
        parent.selected_games = selected_games
        parent.selected_sets = selected_sets

        # Load the selected game's database (rebuilds the match index and
        # keeps MainWindow's cache tracker in sync)
        if active_folder is not None:
            parent.load_game_database(active_folder)

        parent.save_settings()
        if close:
            self.close()

    def _enforce_single_game(self, item: QTreeWidgetItem, column: int) -> None:
        """Untick every other game when a game (or one of its sets) is ticked."""
        root = item.parent() or item
        if root.checkState(0) == Qt.Unchecked:
            return
        self.set_tree.blockSignals(True)
        try:
            for i in range(self.set_tree.topLevelItemCount()):
                other = self.set_tree.topLevelItem(i)
                if other is not root and other.checkState(0) != Qt.Unchecked:
                    other.setCheckState(0, Qt.Unchecked)
        finally:
            self.set_tree.blockSignals(False)

    def _rebuild_database(self):
        """Apply current settings, trigger a background DB rebuild, then close."""
        self.apply_settings(close=False)
        parent = self.parent()
        if parent and hasattr(parent, "start_db_build_in_background"):
            parent.start_db_build_in_background()
        self.close()
