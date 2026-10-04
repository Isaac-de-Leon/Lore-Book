# scanner_view.py — the Scanner page: camera preview, match results, Add/Undo.
#
# A view plus its own result-panel logic (Prev/Next, foil-only lock, writing
# the current match to the collection CSV with single-level Undo). It owns no
# camera, model or thread: MainWindow feeds it frames (show_frame), status
# text (set_status_text) and scan results (show_matches), and listens for
# scan_requested / collection_changed.

import logging
import os

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIntValidator, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from lorebook.core.card_names import name_for
from lorebook.core.card_prices import format_price, price_for, rate_for
from lorebook.core.csv_manager import split_filename, update_cardlist
from lorebook.core.game_types import BASE_DATABASE_PATH, csv_for_game, game_type_from_name, is_foil_only_card
from lorebook.core.settings import AppSettings
from lorebook.ui.icons import get_icon
from lorebook.ui.qt_images import show_bgr

logger = logging.getLogger(__name__)


class ScannerView(QWidget):
    """Scanner page widget; see module docstring."""

    # Top-2 score gap below which a match is flagged as ambiguous (foil
    # variants, reprints, and alt arts often score within a couple percent).
    AMBIGUOUS_GAP = 0.02

    scan_requested = Signal()
    collection_changed = Signal()   # a CSV was written (Add / Undo)

    def __init__(self, settings: AppSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings = settings
        self._tokens: dict[str, str] = {}

        self.last_matches: list[tuple[str, float]] = []
        self.current_match_idx = 0
        self.matches_game: str | None = None   # the game the shown matches belong to
        self._last_add: tuple[str, bool, int, str] | None = None  # (fname, foil, count, game)
        self._foil_locked = False        # Foil checkbox forced on for foil-only rarities
        self._foil_before_lock = False   # user's checkbox state to restore on unlock

        self._build_widgets()

    # ------------------------------------------------------------------ layout

    def _build_widgets(self) -> None:
        self.preview_label = QLabel("Camera stopped")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_label.setMinimumSize(640, 360)

        # Status / DB-build message (below camera, above result panel)
        self.match_label = QLabel("—")
        self.match_label.setObjectName("statusLabel")
        self.match_label.setAlignment(Qt.AlignCenter)

        self.result_panel = QFrame()
        self.result_panel.setObjectName("resultPanel")  # height set by apply_scale

        self.image_label = QLabel("—")
        self.image_label.setObjectName("thumbLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(92, 128)
        self.match_name_label = QLabel("—")
        self.match_name_label.setObjectName("matchName")
        self.match_detail_label = QLabel("")
        self.match_detail_label.setObjectName("matchDetail")
        self.match_price_label = QLabel("")  # market value; empty when unknown
        self.match_price_label.setObjectName("matchDetail")

        # ‹ position › grouped in one compact pager pill
        self.prev_btn = QPushButton()
        self.prev_btn.setFixedSize(24, 24)
        self.prev_btn.clicked.connect(self.prev_match)
        self.match_pos_label = QLabel("")
        self.match_pos_label.setObjectName("matchDetail")
        self.match_pos_label.setAlignment(Qt.AlignCenter)
        self.next_btn = QPushButton()
        self.next_btn.setFixedSize(24, 24)
        self.next_btn.clicked.connect(self.next_match)
        pager = QFrame()
        pager.setObjectName("pagerPill")
        pager_row = QHBoxLayout(pager)
        pager_row.setContentsMargins(3, 3, 3, 3)
        pager_row.setSpacing(2)
        for w in (self.prev_btn, self.match_pos_label, self.next_btn):
            pager_row.addWidget(w)
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.addWidget(pager)
        nav_row.addStretch()

        self.foil_check = QCheckBox("Foil")
        self.foil_check.setChecked(self.settings.keep_foil_checked)
        self.count_edit = QLineEdit("1")
        self.count_edit.setFixedWidth(48)
        self.count_edit.setValidator(QIntValidator(1, 999, self))
        self.add_csv_btn = QPushButton("Add to Collection")
        self.add_csv_btn.setObjectName("addBtn")
        self.add_csv_btn.clicked.connect(self.add_to_csv)
        self.add_csv_btn.setEnabled(False)
        self.undo_btn = QPushButton()  # icon-only ghost button; Add is the prominent action
        self.undo_btn.clicked.connect(self.undo_last_add)
        self.undo_btn.setEnabled(False)
        self.csv_status = QLabel("")
        self.csv_status.setObjectName("statusLabel")
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        for w in (self.foil_check, QLabel("×"), self.count_edit, self.add_csv_btn, self.undo_btn):
            actions_row.addWidget(w)
        actions_row.addStretch()
        actions_row.addWidget(self.csv_status)

        info_col = QVBoxLayout()
        info_col.setContentsMargins(10, 10, 10, 10)
        info_col.setSpacing(4)
        for w in (self.match_name_label, self.match_detail_label, self.match_price_label):
            info_col.addWidget(w)
        info_col.addLayout(nav_row)
        info_col.addStretch()
        info_col.addLayout(actions_row)
        panel_inner = QHBoxLayout(self.result_panel)
        panel_inner.setContentsMargins(10, 10, 10, 10)
        panel_inner.setSpacing(10)
        panel_inner.addWidget(self.image_label)
        panel_inner.addLayout(info_col, stretch=1)

        self.capture_btn = QPushButton("SCAN CARD")
        self.capture_btn.setObjectName("scanBtn")
        self.capture_btn.clicked.connect(self.scan_requested.emit)
        scan_row = QHBoxLayout()
        scan_row.addStretch()
        scan_row.addWidget(self.capture_btn)
        scan_row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.preview_label, stretch=1)
        layout.addWidget(self.match_label)
        layout.addWidget(self.result_panel)
        layout.addLayout(scan_row)

        self.capture_btn.setToolTip("Scan Card (C)")
        self.add_csv_btn.setToolTip("Add to Collection (Ctrl+S or Alt+S)")
        self.undo_btn.setToolTip("Undo last add (Ctrl+Z)")
        self.prev_btn.setToolTip("Previous match (A)")
        self.next_btn.setToolTip("Next match (D)")
        self.foil_check.setToolTip("Toggle Foil (F)")
        # Keyboard shortcuts belong to the window; only the count field takes focus.
        for w in (self.capture_btn, self.add_csv_btn, self.undo_btn, self.prev_btn,
                  self.next_btn, self.foil_check):
            w.setFocusPolicy(Qt.NoFocus)
        self.count_edit.setFocusPolicy(Qt.StrongFocus)

    def apply_theme(self, tokens: dict[str, str]) -> None:
        """(Re)tint icons and status colors for the current theme."""
        self._tokens = tokens
        self.capture_btn.setIcon(get_icon("scan", tokens["on_accent"]))
        self.add_csv_btn.setIcon(get_icon("plus", tokens["on_success"]))
        self.undo_btn.setIcon(get_icon("undo", tokens["text"]))
        self.prev_btn.setIcon(get_icon("chevron-left", tokens["text"]))
        self.next_btn.setIcon(get_icon("chevron-right", tokens["text"]))

    def apply_scale(self, window_height: int) -> None:
        """Scale the fixed-size chrome with the window height (clamped)."""
        h = max(window_height, 1)
        # Scan FAB: ~7% of window height; pill radius and font follow.
        fab_h = int(min(max(h * 0.07, 48), 76))
        fab_font = int(min(max(fab_h * 0.29, 14), 19))
        self.capture_btn.setMinimumSize(int(fab_h * 3.6), fab_h)
        self.capture_btn.setIconSize(QSize(fab_font + 5, fab_font + 5))
        self.capture_btn.setStyleSheet(
            f"QPushButton#scanBtn {{ border-radius: {fab_h // 2}px; "
            f"font-size: {fab_font}px; padding: 0 {fab_h // 2}px; }}"
        )
        btn_h = int(min(max(h * 0.035, 28), 40))
        self.add_csv_btn.setMinimumHeight(btn_h)
        self.undo_btn.setMinimumHeight(btn_h)
        # Result panel and its card thumbnail grow together (92x128 aspect).
        panel_h = int(min(max(h * 0.20, 140), 200))
        self.result_panel.setFixedHeight(panel_h)
        thumb_h = panel_h - 24
        self.image_label.setFixedSize(int(thumb_h * 92 / 128), thumb_h)

    # ------------------------------------------------------------------ inputs from MainWindow

    def apply_settings(self, settings: AppSettings) -> None:
        """Adopt new settings; clears results that belong to another game."""
        self.settings = settings
        game = settings.active_game()
        if self.last_matches and game != self.matches_game:
            self.clear_results()
            self.match_label.setText("Game changed — scan a card.")
        if self._foil_locked:
            self._foil_before_lock = settings.keep_foil_checked
        else:
            self.foil_check.setChecked(settings.keep_foil_checked)

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        show_bgr(self.preview_label, frame_bgr, fill=True)

    def show_camera_message(self, text: str) -> None:
        self.preview_label.setText(text)

    def set_status_text(self, text: str) -> None:
        self.match_label.setText(text)

    def set_scanning(self, busy: bool) -> None:
        self.capture_btn.setEnabled(not busy)
        self.capture_btn.setText("SCANNING…" if busy else "SCAN CARD")
        if busy:
            self.match_label.setText("Scanning…")

    def prepare_scan(self, probably_foil: bool) -> None:
        """Before a scan: drop any foil-only lock and pre-set the Foil box."""
        if self._foil_locked:
            self._foil_locked = False
            self.foil_check.setEnabled(True)
        self.foil_check.setChecked(self.settings.keep_foil_checked or probably_foil)

    def scan_failed(self, text: str) -> None:
        self.match_label.setText(text)
        self.add_csv_btn.setEnabled(False)

    def show_matches(self, matches: list[tuple[str, float]], game: str) -> None:
        """Show scan results (already filtered to the selected sets)."""
        if not matches:
            self.scan_failed("No match found.")
            return
        self.matches_game = game
        self.last_matches = matches[:20]
        self.current_match_idx = 0
        self._show_match_at(0)
        self.add_csv_btn.setEnabled(True)
        # After _show_match_at so the warning isn't overwritten by its label text.
        if len(matches) >= 2 and (matches[0][1] - matches[1][1]) < self.AMBIGUOUS_GAP:
            self.match_label.setText(
                f"⚠ Close match ({matches[0][1]:.1%} vs {matches[1][1]:.1%}) — "
                "check alternatives with A/D before adding."
            )

    def refresh_current_match(self) -> None:
        """Re-render the shown match (e.g. after prices were refreshed)."""
        if self.last_matches:
            self._show_match_at(self.current_match_idx)

    def clear_results(self) -> None:
        self.last_matches = []
        self.current_match_idx = 0
        self.matches_game = None
        self.add_csv_btn.setEnabled(False)
        if self._foil_locked:
            self._foil_locked = False
            self.foil_check.setEnabled(True)
            self.foil_check.setChecked(self._foil_before_lock)
        self.match_name_label.setText("—")
        self.match_detail_label.setText("")
        self.match_price_label.setText("")
        self.match_pos_label.setText("")
        self.image_label.clear()
        self.image_label.setText("—")

    # ------------------------------------------------------------------ result panel

    def _show_match_at(self, idx: int) -> None:
        if not self.last_matches:
            return
        idx = max(0, min(idx, len(self.last_matches) - 1))
        self.current_match_idx = idx
        fname, score = self.last_matches[idx]
        set_code, card_code = split_filename(fname)
        game = self.matches_game or self.settings.active_game()
        code = f"{set_code}-{card_code}" if set_code else fname
        card_name = name_for(set_code, card_code, game) if game else None
        self.match_name_label.setText(f"{card_name}  ·  {code}" if card_name else code)
        self.match_detail_label.setText(
            f"Set {set_code}  ·  {score:.1%} match" if set_code else f"{score:.1%} match"
        )
        self._apply_foil_lock(set_code, card_code, game)
        currency = self.settings.currency
        price = price_for(set_code, card_code, game) if game else None
        self.match_price_label.setText(format_price(price, currency, rate_for(currency)))
        self.match_pos_label.setText(f"{idx + 1} / {len(self.last_matches)}")
        self.match_label.setText(f"{score:.3f}  {fname}")

        if game:
            path = os.path.join(BASE_DATABASE_PATH, game, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                show_bgr(self.image_label, img, fill=False)  # replaces the placeholder text
                return
            logger.error("Failed to load image: %s", path)
        self.image_label.setText("—")

    def _apply_foil_lock(self, set_code: str, card_code: str, game: str | None) -> None:
        """Force (and lock) Foil while a foil-only rarity (Lorcana Enchanted/
        Epic/Iconic) is shown — Dreamborn.ink rejects a "normal" row for them.
        Restores the user's checkbox state when a regular card is shown again."""
        foil_only = game is not None and is_foil_only_card(set_code, card_code, game_type_from_name(game))
        if foil_only and not self._foil_locked:
            self._foil_before_lock = self.foil_check.isChecked()
            self._foil_locked = True
            self.foil_check.setChecked(True)
            self.foil_check.setEnabled(False)
        elif not foil_only and self._foil_locked:
            self._foil_locked = False
            self.foil_check.setEnabled(True)
            self.foil_check.setChecked(self._foil_before_lock)

    def prev_match(self) -> None:
        if self.last_matches:
            self._show_match_at(self.current_match_idx - 1)

    def next_match(self) -> None:
        if self.last_matches:
            self._show_match_at(self.current_match_idx + 1)

    # ------------------------------------------------------------------ collection writes

    def set_status(self, message: str, is_error: bool = False, timeout_ms: int = 3500) -> None:
        color = self._tokens.get("danger" if is_error else "success", "")
        self.csv_status.setStyleSheet(f"color: {color}; padding-left: 8px;")
        self.csv_status.setText(message)
        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, lambda: self.csv_status.setText(""))

    def add_to_csv(self) -> None:
        """Write the current match to its game's collection CSV."""
        if not self.last_matches:
            self.set_status("No match to add.", is_error=True)
            return
        fname = self.last_matches[self.current_match_idx][0]
        try:
            count = int(self.count_edit.text())
        except ValueError:
            count = 0
        if count < 1:
            # Refuse rather than silently adding 1 — a typo'd count ("1o",
            # empty field) must not corrupt the collection.
            self.set_status("Invalid count — enter a number from 1 to 999.", is_error=True)
            return
        is_foil = self.foil_check.isChecked()
        # The game the match came from — not whatever Settings says now.
        game = self.matches_game or self.settings.active_game() or "Lorcana"
        target = csv_for_game(game)
        try:
            update_cardlist(fname, is_foil, count, game=game)
        except OSError as e:  # unreadable or locked (e.g. open in Excel) — file untouched
            logger.error("Could not add %s to %s: %s", fname, target, e)
            self.set_status(f"Not added — {target} could not be updated (open in another program?)",
                            is_error=True, timeout_ms=8000)
            return
        self._last_add = (fname, is_foil, count, game)
        self.undo_btn.setEnabled(True)
        self.set_status(f"Added {count}× {fname} to {target}")
        self.add_csv_btn.setEnabled(False)
        self.collection_changed.emit()

    def undo_last_add(self) -> None:
        """Reverse the most recent Add (single level)."""
        if not self._last_add:
            return
        fname, is_foil, count, game = self._last_add
        try:
            update_cardlist(fname, is_foil, -count, game=game, allow_negative=True)
        except OSError as e:  # keep _last_add so the user can retry
            logger.error("Could not undo %s: %s", fname, e)
            self.set_status("Undo failed — the collection file could not be updated.",
                            is_error=True, timeout_ms=8000)
            return
        self._last_add = None
        self.undo_btn.setEnabled(False)
        self.set_status(f"Removed {count}× {fname}")
        self.collection_changed.emit()

    def forget_last_add(self) -> None:
        """The collection was cleared elsewhere: Undo would point at nothing."""
        self._last_add = None
        self.undo_btn.setEnabled(False)

    # ------------------------------------------------------------------ keyboard

    def handle_key(self, event: QKeyEvent) -> bool:
        """Scanner shortcuts; True if the key was handled."""
        key, mods = event.key(), event.modifiers()
        if key == Qt.Key_C:
            self.scan_requested.emit()
        elif key == Qt.Key_S and (mods & (Qt.ControlModifier | Qt.AltModifier)):
            if self.add_csv_btn.isEnabled():
                self.add_to_csv()
        elif key == Qt.Key_Z and (mods & Qt.ControlModifier):
            self.undo_last_add()
        elif key == Qt.Key_A:
            self.prev_match()
        elif key == Qt.Key_D:
            self.next_match()
        elif key == Qt.Key_F:
            if self.foil_check.isEnabled():
                self.foil_check.setChecked(not self.foil_check.isChecked())
        else:
            return False
        return True
