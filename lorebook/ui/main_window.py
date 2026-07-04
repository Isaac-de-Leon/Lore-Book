# main_window.py — MainWindow: camera preview, card matching, CSV export.

import gc
import json
import logging
import logging.handlers
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

from lorebook.core.card_database import (
    build_feature_database,
    load_cache,
    set_database_path,
)
from lorebook.core.csv_manager import split_filename, update_cardlist
from lorebook.core.game_types import csv_for_game
from lorebook.core.features import extract_features, visualize_activation_overlay
from lorebook.core.image_utils import is_probably_foil
from lorebook.core.matching import find_best_matches
from lorebook.hardware.camera import open_capture
from lorebook.ui.settings_window import SettingsWindow
from lorebook.ui.styles import APP_STYLESHEET

SETTINGS_FILE = "ui_settings.json"


def setup_logging(log_file: str = "card_scanner.log") -> None:
    """Configure application-wide logging with rotating file + console handlers."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    ch.setLevel(logging.WARNING)

    root.addHandler(fh)
    root.addHandler(ch)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.info(f"Logging initialized — {log_path}")


class MainWindow(QWidget):
    """
    Main application window.

    Responsibilities:
    - Live camera preview with focus-box overlay
    - On-demand card scan: feature extraction → DB match → display results
    - Previous / Next navigation across top matches
    - One-click "Add to card list" writing to the appropriate CSV
    - Background DB build with progress bar
    - Settings persistence via ui_settings.json
    """

    logger = logging.getLogger("MainWindow")

    # Signals for marshalling background-thread updates onto the main thread.
    build_status = Signal(str)      # status text for the status label
    build_done = Signal(bool)       # True = all builds succeeded

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _split_filename(filename: str) -> Tuple[str, str]:
        return split_filename(filename)

    def show_error(self, message: str, title: str = "Error", details: Optional[str] = None) -> None:
        self.logger.error(message + (f": {details}" if details else ""))
        QMessageBox.critical(self, title, message)

    def set_status(self, message: str, is_error: bool = False, timeout_ms: int = 3500) -> None:
        self.csv_status.setStyleSheet(
            f"color: {'#F85149' if is_error else '#3FB950'}; padding-left: 8px;"
        )
        self.csv_status.setText(message)
        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, lambda: self.csv_status.setText(""))

    def get_active_game(self) -> Optional[str]:
        """Return the capitalized name of the currently selected game, or None."""
        try:
            for name, selected in self.selected_games.items():
                if selected:
                    return name.capitalize()
        except Exception as e:
            self.logger.error(f"Error getting active game: {e}")
        return None

    # ------------------------------------------------------------------ init

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lore Book")
        self.resize(960, 820)
        self.setStyleSheet(APP_STYLESHEET)

        # State
        self.featureDB: Dict[str, np.ndarray] = load_cache()
        self._loaded_game: Optional[str] = None  # game whose featureDB is in memory
        self.selected_games: Dict[str, bool] = {"lorcana": False, "riftbound": False}
        self.selected_sets: Dict[str, List[str]] = {"lorcana": [], "riftbound": []}
        self.keep_foil_checked = False
        self.confidence_threshold = 0.90
        self.foil_threshold = 0.08
        self.debug_mode = False
        self.camera_index = 0
        self.rotate_display = False
        self.crop_to_focus = True

        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: Optional[float] = None
        self.read_fail_count = 0
        self.max_read_fail = 20

        self.last_matches: List[Tuple[str, float]] = []
        self.current_match_idx: int = 0

        self.load_settings()

        # ── Widgets ──────────────────────────────────────────────────────────

        # Top bar
        title_label = QLabel("LORE BOOK")
        title_label.setObjectName("appTitle")

        self.start_btn = QPushButton("▶  Start")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.settings_btn = QPushButton("⚙  Settings")
        self.settings_btn.clicked.connect(self.open_settings)

        # Thin progress bar (DB build indicator)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        self.progress.hide()

        # Camera preview
        self.preview_label = QLabel("Camera stopped")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_label.setMinimumSize(640, 360)

        # Status / DB-build message (below camera, above result panel)
        self.match_label = QLabel("—")
        self.match_label.setObjectName("statusLabel")
        self.match_label.setAlignment(Qt.AlignCenter)

        # ── Result panel ─────────────────────────────────────────────────────
        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_panel.setFixedHeight(152)

        # Thumbnail (card image)
        self.image_label = QLabel()
        self.image_label.setObjectName("thumbLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(92, 128)
        self.image_label.setText("—")

        # Card name / code
        self.match_name_label = QLabel("—")
        self.match_name_label.setObjectName("matchName")

        # Score + set detail
        self.match_detail_label = QLabel("")
        self.match_detail_label.setObjectName("matchDetail")

        # Navigation
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedSize(30, 30)
        self.prev_btn.clicked.connect(self.prev_match)
        self.match_pos_label = QLabel("")
        self.match_pos_label.setObjectName("matchDetail")
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(30, 30)
        self.next_btn.clicked.connect(self.next_match)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.match_pos_label)
        nav_row.addWidget(self.next_btn)
        nav_row.addStretch()

        # Foil / count / add / status
        self.foil_check = QCheckBox("Foil")
        self.foil_check.setChecked(self.keep_foil_checked)

        self.count_edit = QLineEdit("1")
        self.count_edit.setFixedWidth(48)

        self.add_csv_btn = QPushButton("+ Add to Collection")
        self.add_csv_btn.setObjectName("addBtn")
        self.add_csv_btn.clicked.connect(self.add_to_csv)
        self.add_csv_btn.setEnabled(False)

        self.csv_status = QLabel("")
        self.csv_status.setObjectName("statusLabel")

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        actions_row.addWidget(self.foil_check)
        actions_row.addWidget(QLabel("×"))
        actions_row.addWidget(self.count_edit)
        actions_row.addWidget(self.add_csv_btn)
        actions_row.addStretch()
        actions_row.addWidget(self.csv_status)

        info_col = QVBoxLayout()
        info_col.setContentsMargins(10, 10, 10, 10)
        info_col.setSpacing(4)
        info_col.addWidget(self.match_name_label)
        info_col.addWidget(self.match_detail_label)
        info_col.addLayout(nav_row)
        info_col.addStretch()
        info_col.addLayout(actions_row)

        panel_inner = QHBoxLayout()
        panel_inner.setContentsMargins(10, 10, 10, 10)
        panel_inner.setSpacing(10)
        panel_inner.addWidget(self.image_label)
        panel_inner.addLayout(info_col, stretch=1)
        result_panel.setLayout(panel_inner)

        # ── Scan FAB ─────────────────────────────────────────────────────────
        self.capture_btn = QPushButton("●  SCAN CARD")
        self.capture_btn.setObjectName("scanBtn")
        self.capture_btn.clicked.connect(self.capture_and_match)

        scan_row = QHBoxLayout()
        scan_row.addStretch()
        scan_row.addWidget(self.capture_btn)
        scan_row.addStretch()

        # ── Top bar layout ────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(title_label)
        top.addStretch()
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.settings_btn)

        # ── Root layout ───────────────────────────────────────────────────────
        root = QVBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        root.addLayout(top)
        root.addWidget(self.progress)
        root.addWidget(self.preview_label, stretch=1)
        root.addWidget(self.match_label)
        root.addWidget(result_panel)
        root.addLayout(scan_row)
        self.setLayout(root)

        # Signals from background build threads → main-thread slots
        self.build_status.connect(self.match_label.setText)
        self.build_done.connect(self._on_build_done)

        # Timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab_frame)

        self.watchdog = QTimer(self)
        self.watchdog.setInterval(1000)
        self.watchdog.timeout.connect(self._watchdog_tick)

        self._db_progress_pct = 0
        self._db_progress_timer = QTimer(self)
        self._db_progress_timer.setInterval(100)
        self._db_progress_timer.timeout.connect(self._tick_db_progress)

        # Focus / keyboard
        self.setFocusPolicy(Qt.StrongFocus)
        self._setup_tooltips()
        for widget in [
            self.capture_btn, self.start_btn, self.stop_btn,
            self.settings_btn, self.add_csv_btn, self.prev_btn,
            self.next_btn, self.foil_check,
        ]:
            widget.setFocusPolicy(Qt.NoFocus)
        self.count_edit.setFocusPolicy(Qt.StrongFocus)

        self.start_db_build_in_background()

    # ------------------------------------------------------------------ settings

    def load_settings(self) -> None:
        """Load settings from SETTINGS_FILE; fall back to defaults on any error."""
        defaults: Dict[str, Any] = {
            "camera_index": 0,
            "keep_foil_checked": False,
            "confidence_threshold": 0.90,
            "foil_threshold": 0.08,
            "debug_mode": False,
            "selected_games": {"lorcana": True, "riftbound": False},
            "selected_sets": {"lorcana": [], "riftbound": []},
            "rotate_display": False,
            "crop_to_focus": True,
        }

        def apply_defaults():
            for k, v in defaults.items():
                setattr(self, k, v)

        if not os.path.exists(SETTINGS_FILE):
            self.logger.info("No settings file found — using defaults")
            apply_defaults()
            return

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)

            for key, default in defaults.items():
                try:
                    value = settings.get(key, default)
                    if isinstance(default, bool) and not isinstance(value, bool):
                        value = default
                    elif isinstance(default, (int, float)) and not isinstance(value, (int, float)):
                        value = default
                    elif isinstance(default, dict) and not isinstance(value, dict):
                        value = default
                    if key in ("confidence_threshold", "foil_threshold"):
                        value = max(0.0, min(1.0, float(value)))
                    elif key == "camera_index":
                        value = max(0, int(value))
                    setattr(self, key, value)
                except Exception as e:
                    self.logger.error(f"Error loading setting {key}: {e}")
                    setattr(self, key, default)

            self.logger.info("Settings loaded successfully")

        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Error loading settings file: {e}")
            apply_defaults()

    def save_settings(self) -> None:
        """Persist current settings to SETTINGS_FILE."""
        s = {
            "camera_index": self.camera_index,
            "keep_foil_checked": self.keep_foil_checked,
            "confidence_threshold": float(self.confidence_threshold),
            "foil_threshold": float(self.foil_threshold),
            "debug_mode": self.debug_mode,
            "selected_games": getattr(self, "selected_games", {}),
            "selected_sets": getattr(self, "selected_sets", {}),
            "rotate_display": self.rotate_display,
            "crop_to_focus": self.crop_to_focus,
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------ DB build

    def start_db_build_in_background(self) -> None:
        """Discover game folders and kick off one background build thread for all of them."""
        card_images_dir = "Card_Images"
        try:
            os.makedirs(card_images_dir, exist_ok=True)
            game_folders = [
                item for item in os.listdir(card_images_dir)
                if os.path.isdir(os.path.join(card_images_dir, item))
                and item not in ("__pycache__", "logs")
            ]
        except Exception as e:
            self.logger.error(f"Error scanning {card_images_dir}: {e}")
            self.match_label.setText("Error scanning Card_Images directory")
            return

        if not game_folders:
            self.match_label.setText("No game folders found in Card_Images directory")
            return

        self._db_progress_pct = 0
        self.progress.setValue(0)
        self.progress.show()
        self.match_label.setText(f"Building {game_folders[0]} database, please wait…")

        threading.Thread(
            target=self._build_all_games_worker, args=(game_folders,), daemon=True
        ).start()
        self._db_progress_timer.start()

    def _build_all_games_worker(self, game_folders: List[str]) -> None:
        """Build every game's feature DB sequentially in one background thread.

        Runs off the main thread and must NOT touch Qt widgets directly — all
        UI updates go through signals (build_status, build_done) which Qt
        delivers on the main thread. The active database path global is left
        alone; each build gets its db_path explicitly.
        """
        start_time = time.time()
        failed: List[str] = []

        def progress_callback(pct: int, current_file: Optional[str]) -> None:
            self._db_progress_pct = pct
            if current_file:
                self.build_status.emit(f"Processing {current_file}…")

        for game_name in game_folders:
            game_path = os.path.join("Card_Images", game_name)
            self._db_progress_pct = 0
            self.build_status.emit(f"Building {game_name} database…")

            for attempt in range(1, 4):  # initial try + 2 retries
                try:
                    self.logger.info(
                        f"Building DB for {game_name} at {game_path} (attempt {attempt})"
                    )
                    if not os.path.exists(game_path):
                        raise RuntimeError(f"Game folder not found: {game_path}")
                    db = build_feature_database(
                        progress_callback=progress_callback,
                        db_path=game_path,
                    )
                    if not db:
                        raise RuntimeError("Database build returned no entries")
                    self.logger.info(f"DB ready for {game_name}: {len(db)} entries")
                    break
                except Exception as e:
                    self.logger.error(f"Error building DB for {game_name}: {e}")
            else:
                failed.append(game_name)
                self.build_status.emit(f"Error building {game_name} database")

        elapsed = time.time() - start_time
        self.logger.info(f"DB builds finished in {elapsed:.1f}s ({len(failed)} failed)")
        self.build_done.emit(not failed)

    def _on_build_done(self, success: bool) -> None:
        """Main-thread slot: finalize UI after the background build finishes."""
        self._db_progress_timer.stop()
        self.progress.hide()
        if success:
            self.match_label.setText("All databases ready. Scan a card to begin.")
        # The in-memory featureDB may be stale after a rebuild — force reload on next scan
        self._loaded_game = None

    def _tick_db_progress(self) -> None:
        """Poll _db_progress_pct and update the progress bar.

        The timer keeps running until _on_build_done — a single game hitting
        100% must not kill progress display for the games after it.
        """
        progress = int(getattr(self, "_db_progress_pct", 0))
        if progress != self.progress.value():
            self.progress.setValue(progress)

    # ------------------------------------------------------------------ camera

    def start_camera(self) -> None:
        """Open the camera via the shared backend-probe helper, then start the preview."""
        self.stop_camera()
        cv2.destroyAllWindows()

        # open_capture() (lorebook.hardware.camera) holds the platform backend
        # probe + warm-up logic shared with the headless sorter.
        try:
            self.cap = open_capture(self.camera_index)
        except Exception as e:
            self.cap = None
            self._fail_and_stop(str(e))
            return

        self.read_fail_count = 0
        self.last_frame_time = time.monotonic()
        self.timer.start(30)
        self.watchdog.start()
        self.match_label.setText("Camera running …")

    def _fail_and_stop(self, message: str) -> None:
        self.stop_camera()
        QMessageBox.warning(self, "Camera Error", f"{message}\n\nTry a different camera index in Settings.")

    def stop_camera(self) -> None:
        """Stop timers and release camera resources."""
        self.timer.stop()
        self.watchdog.stop()
        if self.cap is not None:
            try:
                for _ in range(5):
                    self.cap.grab()
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None
        self.last_frame = None
        self.last_frame_time = None
        self.read_fail_count = 0
        self.preview_label.setText("Camera stopped")
        gc.collect()

    def _watchdog_tick(self) -> None:
        if self.cap is not None and self.last_frame_time is not None:
            if (time.monotonic() - self.last_frame_time) > 3.0:
                self._fail_and_stop("No frames received for 3 seconds. Camera stopped.")

    def _focus_rect(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Return (fx, fy, fw, fh) for a 63:88 portrait focus box at ~60% of frame height."""
        card_ratio = 63 / 88.0
        fh = min(int(h * 0.6), h - 4)
        fw = min(int(fh * card_ratio), w - 4)
        fx = max((w - fw) // 2, 2)
        fy = max((h - fh) // 2, 2)
        return fx, fy, fw, fh

    def _grab_frame(self) -> None:
        """Grab a camera frame, draw the focus overlay, and update the preview label."""
        if not self.cap:
            return
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self.read_fail_count += 1
                if self.read_fail_count > self.max_read_fail:
                    self._fail_and_stop("Camera not delivering frames.")
                return

            self.read_fail_count = 0
            self.last_frame_time = time.monotonic()

            if self.rotate_display:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.last_frame = frame.copy()

            overlay = frame.copy()
            h, w = overlay.shape[:2]
            fx, fy, fw, fh = self._focus_rect(h, w)

            color, thickness = (0, 255, 0), 2
            cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), color, thickness)
            cl = 40  # corner line length
            for (sx, sy), (ex, ey) in [
                ((fx, fy), (fx + cl, fy)), ((fx, fy), (fx, fy + cl)),
                ((fx + fw, fy), (fx + fw - cl, fy)), ((fx + fw, fy), (fx + fw, fy + cl)),
                ((fx, fy + fh), (fx + cl, fy + fh)), ((fx, fy + fh), (fx, fy + fh - cl)),
                ((fx + fw, fy + fh), (fx + fw - cl, fy + fh)), ((fx + fw, fy + fh), (fx + fw, fy + fh - cl)),
            ]:
                cv2.line(overlay, (sx, sy), (ex, ey), color, thickness)

            preview = visualize_activation_overlay(overlay) if self.debug_mode else overlay
            self._show_on_label(self.preview_label, preview, fill=True)

        except Exception as e:
            self.read_fail_count += 1
            if self.read_fail_count > self.max_read_fail:
                self._fail_and_stop(f"Camera error: {e}")

    # ------------------------------------------------------------------ matching

    def capture_and_match(self) -> None:
        """Capture the current frame, extract features, and find the best matches."""
        if self.last_frame is None:
            QMessageBox.warning(self, "Scan Card", "Camera is not running or no frame available.")
            return

        img_bgr = self.last_frame.copy()

        if self.crop_to_focus:
            h, w = img_bgr.shape[:2]
            fx, fy, fw, fh = self._focus_rect(h, w)
            fx, fy = max(fx, 0), max(fy, 0)
            fw, fh = min(fw, w - fx), min(fh, h - fy)
            if fw > 10 and fh > 10:
                img_bgr = img_bgr[fy:fy + fh, fx:fx + fw].copy()

        self.foil_check.setChecked(self.keep_foil_checked or is_probably_foil(img_bgr, threshold=self.foil_threshold))

        active_game = self.get_active_game()
        if not active_game:
            self.match_label.setText("Please select a game type in Settings.")
            self.add_csv_btn.setEnabled(False)
            return

        # Reload featureDB only when the active game changed (or after a rebuild)
        if self._loaded_game != active_game or not self.featureDB:
            set_database_path(active_game)
            self.featureDB = load_cache()
            self._loaded_game = active_game if self.featureDB else None
            self.logger.info(f"Loaded {len(self.featureDB)} entries for {active_game}")
        if not self.featureDB:
            self.match_label.setText("No feature database found. Build the DB first.")
            self.add_csv_btn.setEnabled(False)
            return

        features = extract_features(img_bgr)
        if features is None:
            self.match_label.setText("Could not extract features from image.")
            self.add_csv_btn.setEnabled(False)
            return

        matches = find_best_matches(features, self.featureDB, threshold=self.confidence_threshold)
        matches = self._filter_matches(matches)
        self.logger.info(f"Found {len(matches)} matches after filtering")

        if not matches:
            self.match_label.setText("No match found.")
            self.add_csv_btn.setEnabled(False)
            return

        self.last_matches = matches[:20]
        self.current_match_idx = 0
        self._show_match_at(0)
        self.add_csv_btn.setEnabled(True)

    def _show_match_at(self, idx: int) -> None:
        """Display the match at position idx in the result panel."""
        if not self.last_matches:
            return
        idx = max(0, min(idx, len(self.last_matches) - 1))
        self.current_match_idx = idx
        fname, score = self.last_matches[idx]

        set_code, card_code = split_filename(fname)
        self.match_name_label.setText(f"{set_code}-{card_code}" if set_code else fname)
        self.match_detail_label.setText(
            f"Set {set_code}  ·  {score:.1%} match" if set_code else f"{score:.1%} match"
        )
        self.match_pos_label.setText(f"{idx + 1} / {len(self.last_matches)}")
        self.match_label.setText(f"{score:.3f}  {fname}")

        active_game = self.get_active_game()
        if active_game:
            match_path = os.path.join("Card_Images", active_game, fname)
            img = cv2.imread(match_path, cv2.IMREAD_COLOR)
            if img is not None:
                # setPixmap clears any placeholder text automatically
                self._show_on_label(self.image_label, img, fill=False)
                return
            self.logger.error(f"Failed to load image: {match_path}")
        self.image_label.setText("—")

    def prev_match(self) -> None:
        if self.last_matches:
            self._show_match_at(self.current_match_idx - 1)

    def next_match(self) -> None:
        if self.last_matches:
            self._show_match_at(self.current_match_idx + 1)

    def _filter_matches(self, matches: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Keep only matches that belong to the active game's selected sets."""
        if not matches:
            return []
        active_game = None
        for name, selected in self.selected_games.items():
            if selected:
                active_game = name
                break
        if not active_game:
            return matches

        game_selected_sets = self.selected_sets.get(active_game, [])
        filtered = []
        for fname, score in matches:
            set_code, _ = split_filename(os.path.basename(fname))
            if not game_selected_sets or set_code in game_selected_sets:
                filtered.append((fname, score))
        return filtered

    def open_settings(self) -> None:
        dlg = SettingsWindow(self)
        dlg.exec()
        self.foil_check.setChecked(self.keep_foil_checked)

    def add_to_csv(self) -> None:
        """Write the current match to the game-appropriate CSV."""
        if not self.last_matches:
            self.set_status("No match to add.", is_error=True)
            return

        fname = self.last_matches[self.current_match_idx][0]
        try:
            cnt = int(self.count_edit.text())
        except Exception:
            cnt = 1
        is_foil = self.foil_check.isChecked()

        active_game = self.get_active_game() or "Lorcana"
        target_file = csv_for_game(active_game)

        update_cardlist(fname, is_foil, cnt, game=active_game)
        self.set_status(f"Added {cnt}× {fname} to {target_file}")
        self.add_csv_btn.setEnabled(False)

    # ------------------------------------------------------------------ display helpers

    def _crop_to_fill(self, img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Center-crop img to match target aspect ratio (cover mode)."""
        if img_bgr is None or img_bgr.size == 0 or target_w <= 0 or target_h <= 0:
            return np.ascontiguousarray(img_bgr) if img_bgr is not None else img_bgr
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.ascontiguousarray(img_bgr)
        if (w / float(h)) > (target_w / float(target_h)):
            new_w = int(h * target_w / target_h)
            x0 = max((w - new_w) // 2, 0)
            cropped = img_bgr[:, x0:x0 + new_w]
        else:
            new_h = int(w * target_h / target_w)
            y0 = max((h - new_h) // 2, 0)
            cropped = img_bgr[y0:y0 + new_h, :]
        return np.ascontiguousarray(cropped)

    def _show_on_label(self, label: QLabel, img_bgr: np.ndarray, fill: bool = False) -> None:
        """Render a BGR image onto a QLabel, optionally crop-to-fill."""
        if img_bgr is None or img_bgr.size == 0:
            return
        if fill:
            img_bgr = self._crop_to_fill(img_bgr, label.width(), label.height())
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[-1] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        if img_bgr.dtype != np.uint8 or not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
        h, w = img_bgr.shape[:2]
        qimg = QImage(img_bgr.data, w, h, img_bgr.strides[0], QImage.Format_BGR888)
        label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                label.size(),
                Qt.IgnoreAspectRatio if fill else Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # ------------------------------------------------------------------ keyboard

    def focusInEvent(self, event):
        self.setFocus()
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_C:
            event.accept()
            self.capture_and_match()
        elif key == Qt.Key_S and (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            if self.add_csv_btn.isEnabled():
                self.add_to_csv()
        elif key == Qt.Key_A:
            self.prev_match()
        elif key == Qt.Key_D:
            self.next_match()
        elif key == Qt.Key_F:
            self.foil_check.setChecked(not self.foil_check.isChecked())
        else:
            super().keyPressEvent(event)

    def _setup_tooltips(self):
        self.capture_btn.setToolTip("Scan Card (C)")
        self.add_csv_btn.setToolTip("Add to Collection (Ctrl+S or Alt+S)")
        self.prev_btn.setToolTip("Previous match (A)")
        self.next_btn.setToolTip("Next match (D)")
        self.foil_check.setToolTip("Toggle Foil (F)")
