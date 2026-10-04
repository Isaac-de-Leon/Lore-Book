# settings_window.py — Settings dialog for camera, UI, and game/set selection.

import logging
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lorebook.core.card_prices import SUPPORTED_CURRENCIES
from lorebook.core.csv_manager import get_available_sets
from lorebook.core.game_types import (
    BASE_DATABASE_PATH,
    game_folders,
    sets_to_display,
    sets_to_store,
)
from lorebook.core.settings import AppSettings


class SettingsWindow(QDialog):
    """
    Settings dialog for camera and UI options.

    Edits a *copy* of the given AppSettings (camera, foil/rotation/crop,
    thresholds, currency, theme, debug, per-game/set filter). Apply emits
    ``applied`` with the new settings; the owner decides what to do with them
    (store, save, re-theme, load the game's database). The dialog never
    touches its parent's state directly.
    """

    logger = logging.getLogger("SettingsWindow")

    applied = Signal(object)        # AppSettings
    rebuild_requested = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self._settings = settings.copy()
        self.setWindowTitle("Settings")
        # Not a fixed size: the form scrolls and the buttons stay pinned, so
        # Apply is reachable on short screens (1366x768, 1080p at 150%).
        self.setMinimumWidth(390)
        # Stylesheet is inherited from the parent MainWindow, so the dialog
        # always matches the active theme.
        self.logger.info("Initializing settings window")

        layout = QFormLayout()

        # Theme selector
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", userData="dark")
        self.theme_combo.addItem("Light", userData="light")
        if settings.theme == "light":
            self.theme_combo.setCurrentIndex(1)
        layout.addRow("Theme:", self.theme_combo)

        # Camera index selector (0–4)
        self.camera_combo = QComboBox()
        for i in range(5):
            self.camera_combo.addItem(f"Camera {i}", userData=i)
        self.camera_combo.setCurrentIndex(min(settings.camera_index, 4))
        layout.addRow("Camera:", self.camera_combo)

        help_label = QLabel("Note: If camera doesn't work, try a different index\nand restart the application.")
        help_label.setObjectName("statusLabel")  # themed muted text
        layout.addRow(help_label)

        self.keep_foil_checked = QCheckBox("Keep Foil Checked")
        self.keep_foil_checked.setChecked(settings.keep_foil_checked)
        layout.addRow(self.keep_foil_checked)

        self.rotate_checkbox = QCheckBox("Rotate preview & capture 180°")
        self.rotate_checkbox.setChecked(settings.rotate_display)
        layout.addRow(self.rotate_checkbox)

        self.crop_checkbox = QCheckBox("Scan only inside focus box")
        self.crop_checkbox.setChecked(settings.crop_to_focus)
        layout.addRow(self.crop_checkbox)

        self.auto_scan_checkbox = QCheckBox("Auto-scan when a card settles in the focus box")
        self.auto_scan_checkbox.setChecked(settings.auto_scan)
        layout.addRow(self.auto_scan_checkbox)

        self.confidence_input = QLineEdit(
            str(int(100 * settings.confidence_threshold))
        )
        layout.addRow("Confidence Threshold (%):", self.confidence_input)

        self.foil_threshold_input = QLineEdit(
            str(round(100 * settings.foil_threshold, 1))
        )
        layout.addRow("Foil Threshold (%):", self.foil_threshold_input)

        # Display currency for scanned-card market prices (source data is USD)
        self.currency_combo = QComboBox()
        for code in SUPPORTED_CURRENCIES:
            self.currency_combo.addItem(code, userData=code)
        idx = self.currency_combo.findData(settings.currency)
        self.currency_combo.setCurrentIndex(max(0, idx))
        layout.addRow("Price currency:", self.currency_combo)

        self.debug_mode_checkbox = QCheckBox("Enable Debug Mode (heatmap overlay)")
        self.debug_mode_checkbox.setChecked(settings.debug_mode)
        layout.addRow(self.debug_mode_checkbox)

        # Game / set tree
        self.set_tree = QTreeWidget()
        self.set_tree.setHeaderHidden(True)
        self.set_tree.setMinimumHeight(200)

        selected_games = settings.selected_games
        selected_sets = settings.selected_sets

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

            # Pass db_path explicitly — avoids touching the global database path
            game_db_path = os.path.join(BASE_DATABASE_PATH, game_name)
            try:
                game_sets = get_available_sets(db_path=game_db_path)
                ticked = set(sets_to_display(selected_sets.get(key, []), game_sets, is_selected))
                for set_code in game_sets:
                    set_item = QTreeWidgetItem(game_root, [f"Set {set_code}"])
                    set_item.setFlags(set_item.flags() | Qt.ItemIsUserCheckable)
                    set_item.setCheckState(0, Qt.Checked if set_code in ticked else Qt.Unchecked)
                    set_item.setData(0, Qt.UserRole, set_code)
            except OSError as e:
                self.logger.warning("Error getting sets for %s: %s", game_name, e)

        # One active game at a time: ticking a game (or any of its sets)
        # unticks the others, so the dialog never shows a selection that
        # scanning would silently ignore.
        self.set_tree.itemChanged.connect(self._enforce_single_game)

        self.set_tree.expandAll()
        layout.addRow("Filter Sets:", self.set_tree)

        rebuild_button = QPushButton("Rebuild Database")
        rebuild_button.setToolTip("Re-scan all card images and rebuild the feature cache")
        rebuild_button.clicked.connect(self._rebuild_database)

        apply_button = QPushButton("Apply")
        # lambda guards against QPushButton.clicked's `checked` bool binding to `close`
        apply_button.clicked.connect(lambda: self.apply_settings())

        form = QWidget()
        form.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(form)

        outer = QVBoxLayout()
        outer.addWidget(scroll, stretch=1)
        outer.addWidget(rebuild_button)
        outer.addWidget(apply_button)
        self.setLayout(outer)
        self.apply_button = apply_button

        # Preferred 715px tall, capped to the screen's usable height.
        screen = (parent.screen() if parent is not None else None) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry().height() if screen is not None else 715
        self.resize(420, max(320, min(715, avail - 60)))

    def result_settings(self) -> AppSettings:
        """The settings as currently entered in the dialog."""
        new = self._settings.copy()
        new.theme = self.theme_combo.currentData()
        new.keep_foil_checked = self.keep_foil_checked.isChecked()
        new.rotate_display = self.rotate_checkbox.isChecked()
        new.crop_to_focus = self.crop_checkbox.isChecked()
        new.auto_scan = self.auto_scan_checkbox.isChecked()
        new.debug_mode = self.debug_mode_checkbox.isChecked()
        new.camera_index = self.camera_combo.currentData()
        new.currency = self.currency_combo.currentData()
        new.confidence_threshold = _percent(self.confidence_input.text(), AppSettings.confidence_threshold)
        new.foil_threshold = _percent(self.foil_threshold_input.text(), AppSettings.foil_threshold)

        # Game / set selections from the tree — every game folder, not just
        # the built-in two, so new games are selectable too.
        new.selected_games, new.selected_sets = {}, {}
        for i in range(self.set_tree.topLevelItemCount()):
            root = self.set_tree.topLevelItem(i)
            key = root.text(0).lower()
            selected = root.checkState(0) != Qt.Unchecked
            available = [root.child(j).data(0, Qt.UserRole) for j in range(root.childCount())]
            checked = [
                root.child(j).data(0, Qt.UserRole)
                for j in range(root.childCount())
                if root.child(j).checkState(0) == Qt.Checked
            ]
            new.selected_games[key] = selected
            new.selected_sets[key] = sets_to_store(checked, available) if selected else []
        return new

    def apply_settings(self, close: bool = True) -> AppSettings:
        """Emit ``applied`` with the entered settings (and close, by default).

        close=False applies without dismissing the dialog (Rebuild Database).
        """
        new = self.result_settings()
        self.logger.info(
            "Applying settings — games: %s, sets: %s", new.selected_games, new.selected_sets
        )
        self.applied.emit(new)
        if close:
            self.close()
        return new

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
        """Apply current settings, request a background DB rebuild, then close."""
        self.apply_settings(close=False)
        self.rebuild_requested.emit()
        self.close()


def _percent(text: str, fallback: float) -> float:
    """A 0–100 percent field as a 0.0–1.0 fraction (fallback if not a number)."""
    try:
        return max(0.0, min(1.0, float(text) / 100.0))
    except ValueError:
        return fallback
