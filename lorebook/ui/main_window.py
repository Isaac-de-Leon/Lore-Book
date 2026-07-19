# main_window.py — MainWindow: camera preview, card matching, CSV export.

import faulthandler
import gc
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lorebook import __version__
from lorebook.core.card_database import (
    build_feature_database,
    load_cache,
    set_database_path,
)
from lorebook.core.card_names import name_for
from lorebook.core.card_prices import (
    RATES_FILE,
    SUPPORTED_CURRENCIES,
    clear_price_cache,
    format_price,
    price_for,
    prices_stale,
    rate_for,
)
from lorebook.core.csv_manager import split_filename, update_cardlist
from lorebook.core.game_types import (
    BASE_DATABASE_PATH,
    csv_for_game,
    game_type_from_name,
    is_foil_only_card,
)
from lorebook.core.paths import data_path
from lorebook.core.features import extract_features, visualize_activation_overlay
from lorebook.core.image_fetcher import download_new_images
from lorebook.core.image_utils import MotionGate, crop_to_card, focus_rect, is_probably_foil
from lorebook.core.matching import MatchIndex
from lorebook.core.price_fetcher import download_card_prices, download_currency_rates
from lorebook.hardware.camera import open_capture
from lorebook.ui.collection_view import CollectionView
from lorebook.ui.icons import get_icon
from lorebook.ui.progress_dialog import BuildProgressDialog
from lorebook.ui.settings_window import SettingsWindow
from lorebook.ui.styles import DEFAULT_THEME, THEMES, build_stylesheet, theme_tokens

SETTINGS_FILE = data_path("ui_settings.json")

# Held at module level so the faulthandler target file is never GC-closed.
_crash_log_file = None


def setup_logging(log_file: str = "card_scanner.log") -> None:
    """Configure application-wide logging with rotating file + console handlers."""
    log_dir = data_path("logs")
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

    # Crash diagnostics. Native faults (e.g. access violations inside Qt or
    # OpenCV) kill the process with no Python traceback — faulthandler dumps
    # every thread's Python stack to logs/crash_native.log at fault time so
    # the crash site is identifiable afterwards.
    global _crash_log_file
    try:
        _crash_log_file = open(
            os.path.join(log_dir, "crash_native.log"), "a", encoding="utf-8"
        )
        faulthandler.enable(file=_crash_log_file)
    except OSError as e:
        logging.warning(f"Could not enable native crash logging: {e}")

    # Uncaught Python exceptions land in the main log too (PySide6 prints
    # them to stderr, which is invisible when launched outside a terminal).
    def _log_excepthook(exc_type, exc, tb):
        logging.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_excepthook

    logging.info(f"Logging initialized — {log_path}")


class MainWindow(QWidget):
    """
    Main application window.

    Responsibilities:
    - Live camera preview with focus-box overlay (optional auto-scan)
    - On-demand card scan: feature extraction → DB match → display results
    - Previous / Next navigation across top matches
    - One-click "Add to card list" writing to the appropriate CSV (with Undo)
    - Background DB build with a progress dialog (status, percent, Cancel)
    - Settings persistence via ui_settings.json

    Threading rules
    ---------------
    Two kinds of background daemon threads exist: the DB build
    (_build_all_games_worker) and the camera opener (_open_camera_worker,
    which keeps the slow open_capture() backend probe off the paint path).
    From any non-main thread, the ONLY permitted interactions with this
    object are:
      - writing the plain attribute ``_db_progress_pct`` (polled by a QTimer)
      - emitting the ``build_status`` / ``build_done`` / ``prices_refreshed`` /
        ``camera_ready`` / ``camera_failed`` signals (Qt delivers them on the
        main thread)
      - reading the ``cancel_event`` / ``token`` passed in as worker arguments
    Everything else — widgets (including the progress dialog),
    featureDB/_match_index, settings, the set_database_path global — is
    main-thread-only. Camera-open results carry a generation token checked
    against ``_cam_open_token`` so a Stop/restart discards stale opens.
    """

    logger = logging.getLogger("MainWindow")

    # Top-2 score gap below which a match is flagged as ambiguous (foil
    # variants, reprints, and alt arts often score within a couple percent).
    AMBIGUOUS_GAP = 0.02

    # Signals for marshalling background-thread updates onto the main thread.
    build_status = Signal(str)      # status text for the status label
    build_done = Signal(bool)       # True = all builds succeeded
    prices_refreshed = Signal()     # price/rate files were rewritten on disk
    camera_ready = Signal(int, object)   # (open token, cv2.VideoCapture)
    camera_failed = Signal(int, str)     # (open token, error message)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _split_filename(filename: str) -> Tuple[str, str]:
        return split_filename(filename)

    def show_error(self, message: str, title: str = "Error", details: Optional[str] = None) -> None:
        self.logger.error(message + (f": {details}" if details else ""))
        QMessageBox.critical(self, title, message)

    def set_status(self, message: str, is_error: bool = False, timeout_ms: int = 3500) -> None:
        tokens = theme_tokens(self.theme)
        color = tokens["danger"] if is_error else tokens["success"]
        self.csv_status.setStyleSheet(f"color: {color}; padding-left: 8px;")
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

    def load_game_database(self, game_name: str) -> bool:
        """
        Make game_name the active database: load its feature cache and rebuild
        the match index. The single place that owns the featureDB/_match_index/
        _loaded_game invariant — every game switch must go through here.
        Returns True if the loaded cache has entries.
        """
        set_database_path(game_name)
        self.featureDB = load_cache()
        self._match_index = MatchIndex(self.featureDB)
        self._loaded_game = game_name if self.featureDB else None
        self.logger.info(f"Loaded {len(self.featureDB)} entries for {game_name}")
        return bool(self.featureDB)

    # ------------------------------------------------------------------ init

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Lore Book v{__version__}")
        self.resize(1000, 840)

        # State
        self.featureDB: Dict[str, np.ndarray] = load_cache()
        self._match_index = MatchIndex(self.featureDB)  # vectorized matcher over featureDB
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
        self.auto_scan = False
        self.theme = DEFAULT_THEME
        # min_std suppresses triggers on an empty (near-uniform) focus box
        self._motion_gate = MotionGate(min_std=12.0)

        self.cap: Optional[cv2.VideoCapture] = None
        # Bumped on every start/stop; camera-open worker results carrying an
        # older token are stale (user hit Stop or restarted) and get discarded.
        self._cam_open_token = 0
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: Optional[float] = None
        self.read_fail_count = 0
        self.max_read_fail = 20

        self.last_matches: List[Tuple[str, float]] = []
        self.current_match_idx: int = 0
        self._last_add: Optional[Tuple[str, bool, int, str]] = None  # (fname, foil, count, game)
        self._foil_locked = False        # Foil checkbox forced on for foil-only rarities
        self._foil_before_lock = False   # user's checkbox state to restore on unlock

        # DB build state (dialog created lazily on first build)
        self._progress_dialog: Optional[BuildProgressDialog] = None
        self._build_cancel_event = threading.Event()
        self._build_thread: Optional[threading.Thread] = None

        self.load_settings()

        # ── Widgets ──────────────────────────────────────────────────────────

        # Header bar (top of the content column, right of the sidebar).
        # One smart camera toggle instead of a Start/Stop pair — its label,
        # icon and style track the camera lifecycle via _set_camera_state.
        title_label = QLabel("LORE BOOK")
        title_label.setObjectName("appTitle")

        self._camera_state = "idle"  # idle | starting | running
        self.camera_btn = QPushButton("Start")
        self.camera_btn.setObjectName("cameraBtn")
        self.camera_btn.setToolTip("Start / stop the camera")
        self.camera_btn.clicked.connect(self._on_camera_btn)

        # Sidebar: logo + page navigation on top, settings at the bottom.
        # Icons are set (and re-tinted) by _apply_icons via apply_theme.
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setToolTip("Lore Book")

        def _nav_button(tooltip: str, checkable: bool = False) -> QPushButton:
            btn = QPushButton()
            btn.setObjectName("navBtn")
            btn.setIconSize(QSize(22, 22))
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(checkable)
            return btn

        self.nav_scanner_btn = _nav_button("Scanner", checkable=True)
        self.nav_scanner_btn.setChecked(True)
        self.nav_collection_btn = _nav_button("Collection", checkable=True)
        self.settings_btn = _nav_button("Settings")
        self.settings_btn.clicked.connect(self.open_settings)

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
        # Height is set by _apply_scale (scales with the window, 140-200px).
        self.result_panel = result_panel

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

        # Market value (empty when no price data is available)
        self.match_price_label = QLabel("")
        self.match_price_label.setObjectName("matchDetail")

        # Navigation — ‹ position › grouped in one compact pager pill
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
        pager_row = QHBoxLayout()
        pager_row.setContentsMargins(3, 3, 3, 3)
        pager_row.setSpacing(2)
        pager_row.addWidget(self.prev_btn)
        pager_row.addWidget(self.match_pos_label)
        pager_row.addWidget(self.next_btn)
        pager.setLayout(pager_row)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.addWidget(pager)
        nav_row.addStretch()

        # Foil / count / add / status
        self.foil_check = QCheckBox("Foil")
        self.foil_check.setChecked(self.keep_foil_checked)

        self.count_edit = QLineEdit("1")
        self.count_edit.setFixedWidth(48)
        self.count_edit.setValidator(QIntValidator(1, 999, self))

        self.add_csv_btn = QPushButton("Add to Collection")
        self.add_csv_btn.setObjectName("addBtn")
        self.add_csv_btn.clicked.connect(self.add_to_csv)
        self.add_csv_btn.setEnabled(False)

        # Icon-only ghost button — Add is the sole prominent action
        self.undo_btn = QPushButton()
        self.undo_btn.clicked.connect(self.undo_last_add)
        self.undo_btn.setEnabled(False)

        self.csv_status = QLabel("")
        self.csv_status.setObjectName("statusLabel")

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        actions_row.addWidget(self.foil_check)
        actions_row.addWidget(QLabel("×"))
        actions_row.addWidget(self.count_edit)
        actions_row.addWidget(self.add_csv_btn)
        actions_row.addWidget(self.undo_btn)
        actions_row.addStretch()
        actions_row.addWidget(self.csv_status)

        info_col = QVBoxLayout()
        info_col.setContentsMargins(10, 10, 10, 10)
        info_col.setSpacing(4)
        info_col.addWidget(self.match_name_label)
        info_col.addWidget(self.match_detail_label)
        info_col.addWidget(self.match_price_label)
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
        self.capture_btn = QPushButton("SCAN CARD")
        self.capture_btn.setObjectName("scanBtn")
        self.capture_btn.clicked.connect(self.capture_and_match)

        scan_row = QHBoxLayout()
        scan_row.addStretch()
        scan_row.addWidget(self.capture_btn)
        scan_row.addStretch()

        # ── Header row ────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(title_label)
        top.addStretch()
        top.addWidget(self.camera_btn)

        # ── Pages: Scanner / Collection, switched by the sidebar nav ─────────
        self.scanner_page = QWidget()
        scanner_layout = QVBoxLayout()
        scanner_layout.setContentsMargins(0, 8, 0, 0)
        scanner_layout.setSpacing(8)
        scanner_layout.addWidget(self.preview_label, stretch=1)
        scanner_layout.addWidget(self.match_label)
        scanner_layout.addWidget(result_panel)
        scanner_layout.addLayout(scan_row)
        self.scanner_page.setLayout(scanner_layout)

        self.collection_view = CollectionView(initial_game=self.get_active_game())

        self.stack = QStackedWidget()
        self.stack.addWidget(self.scanner_page)
        self.stack.addWidget(self.collection_view)
        # Pick up CSV changes made while the page was hidden (scans, undo, edits)
        self.stack.currentChanged.connect(self._on_page_changed)

        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        nav_group.addButton(self.nav_scanner_btn, 0)
        nav_group.addButton(self.nav_collection_btn, 1)
        nav_group.idClicked.connect(self.stack.setCurrentIndex)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(64)
        side = QVBoxLayout()
        side.setContentsMargins(10, 14, 10, 14)
        side.setSpacing(10)
        side.addWidget(self.logo_label, 0, Qt.AlignHCenter)
        side.addSpacing(8)
        side.addWidget(self.nav_scanner_btn, 0, Qt.AlignHCenter)
        side.addWidget(self.nav_collection_btn, 0, Qt.AlignHCenter)
        side.addStretch(1)
        side.addWidget(self.settings_btn, 0, Qt.AlignHCenter)
        sidebar.setLayout(side)

        # ── Root layout: sidebar | content column ────────────────────────────
        content = QVBoxLayout()
        content.setContentsMargins(16, 12, 16, 12)
        content.setSpacing(8)
        content.addLayout(top)
        content.addWidget(self.stack, stretch=1)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(sidebar)
        root.addLayout(content, stretch=1)
        self.setLayout(root)

        # Signals from background threads → main-thread slots
        self.build_status.connect(self._on_build_status)
        self.build_done.connect(self._on_build_done)
        self.prices_refreshed.connect(self._on_prices_refreshed)
        self.camera_ready.connect(self._on_camera_ready)
        self.camera_failed.connect(self._on_camera_failed)

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
            self.capture_btn, self.camera_btn,
            self.settings_btn, self.add_csv_btn, self.undo_btn,
            self.prev_btn, self.next_btn, self.foil_check,
            self.nav_scanner_btn, self.nav_collection_btn,
        ]:
            widget.setFocusPolicy(Qt.NoFocus)
        self.count_edit.setFocusPolicy(Qt.StrongFocus)

        self.apply_theme(self.theme)  # stylesheet + icon tints (also scales chrome)
        # Deferred until the event loop runs so the main window paints before
        # the progress dialog appears — showing it here leaves both windows
        # as unpainted white rectangles until the first paint event.
        QTimer.singleShot(0, self.start_db_build_in_background)

    # ------------------------------------------------------------------ theme

    def apply_theme(self, theme: str) -> None:
        """Switch the app-wide theme at runtime: stylesheet + icon tints.

        Child dialogs (Settings, build progress) inherit the stylesheet
        automatically, so re-setting it here re-themes them too.
        """
        self.theme = theme if theme in THEMES else DEFAULT_THEME
        self.setStyleSheet(build_stylesheet(self.theme))
        self._apply_icons()
        self._apply_scale()  # re-applies the scan FAB's inline geometry QSS

    def _apply_icons(self) -> None:
        """(Re)tint every icon for the current theme."""
        tokens = theme_tokens(self.theme)
        text, muted, accent = tokens["text"], tokens["muted"], tokens["accent"]

        self.logo_label.setPixmap(get_icon("book", accent).pixmap(26, 26))
        self.nav_scanner_btn.setIcon(get_icon("camera", muted, checked_color=accent))
        self.nav_collection_btn.setIcon(get_icon("grid", muted, checked_color=accent))
        self.settings_btn.setIcon(get_icon("sliders", muted))

        self._set_camera_state(self._camera_state)  # re-tints the camera toggle
        self.capture_btn.setIcon(get_icon("scan", tokens["on_accent"]))
        self.add_csv_btn.setIcon(get_icon("plus", tokens["on_success"]))
        self.undo_btn.setIcon(get_icon("undo", text))
        self.prev_btn.setIcon(get_icon("chevron-left", text))
        self.next_btn.setIcon(get_icon("chevron-right", text))
        self.collection_view.apply_icons(text, tokens["danger"])

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
            "auto_scan": False,
            "currency": "USD",
            "theme": DEFAULT_THEME,
        }

        def apply_defaults():
            for k, v in defaults.items():
                setattr(self, k, v)

        if not os.path.exists(SETTINGS_FILE):
            self.logger.info("No settings file found — using defaults")
            apply_defaults()
            return

        try:
            # utf-8-sig tolerates a BOM (editors/PowerShell often add one)
            with open(SETTINGS_FILE, "r", encoding="utf-8-sig") as f:
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
                    elif key == "currency":
                        value = str(value).upper()
                        if value not in SUPPORTED_CURRENCIES:
                            value = default
                    elif key == "theme":
                        value = str(value).lower()
                        if value not in THEMES:
                            value = default
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
            "auto_scan": self.auto_scan,
            "currency": getattr(self, "currency", "USD"),
            "theme": self.theme,
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
        card_images_dir = BASE_DATABASE_PATH
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

        # One build at a time — Settings → Rebuild while the startup build is
        # still running must not spawn a second worker thread.
        if self._build_thread is not None and self._build_thread.is_alive():
            if self._progress_dialog is not None:
                self._progress_dialog.show()
                self._progress_dialog.raise_()
            return

        # Fresh Event per build: never clear() a shared one — a lingering old
        # worker could be un-cancelled, or a new build instantly cancelled.
        self._build_cancel_event = threading.Event()

        if self._progress_dialog is None:
            self._progress_dialog = BuildProgressDialog(self)
            self._progress_dialog.cancel_requested.connect(self._cancel_db_build)
        self._progress_dialog.reset_for_new_build()
        self._progress_dialog.set_status(f"Building {game_folders[0]} database, please wait…")
        self._progress_dialog.show()

        self._db_progress_pct = 0
        self._build_thread = threading.Thread(
            target=self._build_all_games_worker,
            args=(game_folders, self._build_cancel_event),
            daemon=True,
        )
        self._build_thread.start()
        self._db_progress_timer.start()

    def _build_all_games_worker(self, game_folders: List[str], cancel_event: threading.Event) -> None:
        """Build every game's feature DB sequentially in one background thread.

        Runs off the main thread and must NOT touch Qt widgets directly — all
        UI updates go through signals (build_status, build_done) which Qt
        delivers on the main thread. The active database path global is left
        alone; each build gets its db_path explicitly. cancel_event stops the
        build between images/batches/games; partial caches stay valid.
        """
        start_time = time.time()
        failed: List[str] = []
        refreshed_prices = False

        def progress_callback(pct: int, current_file: Optional[str]) -> None:
            self._db_progress_pct = pct
            if current_file:
                self.build_status.emit(f"Processing {current_file}…")

        # Refresh the shared USD exchange rates once per run when stale. A
        # failure never blocks anything — non-USD display falls back to USD.
        if prices_stale(path=RATES_FILE):
            try:
                download_currency_rates()
                refreshed_prices = True
            except Exception as e:
                self.logger.warning(f"Currency-rate refresh failed (continuing): {e}")

        for game_name in game_folders:
            if cancel_event.is_set():
                break
            game_path = os.path.join(BASE_DATABASE_PATH, game_name)
            # -1 = indeterminate: the download phase only reports text lines
            self._db_progress_pct = -1

            # Download any newly released card images first (no-op for games
            # without a registered fetcher). A fetch failure — offline, source
            # down, format change — must never block the build itself.
            try:
                self.build_status.emit(f"Checking for new {game_name} card images…")
                stats = download_new_images(
                    game_name,
                    out_dir=game_path,
                    progress_callback=self.build_status.emit,
                    cancel_event=cancel_event,
                )
                if stats and stats.downloaded:
                    self.logger.info(
                        f"Downloaded {stats.downloaded} new {game_name} images "
                        f"({stats.failed} failed)"
                    )
                    self.build_status.emit(
                        f"Downloaded {stats.downloaded} new {game_name} card images"
                    )
            except Exception as e:
                self.logger.warning(f"Image fetch for {game_name} failed (continuing build): {e}")

            # Refresh market prices when the cached file is missing or older
            # than a day (no-op for games without a registered price source).
            # Same failure contract as the image fetch: warn and move on.
            if prices_stale(game_name):
                try:
                    self.build_status.emit(f"Updating {game_name} card prices…")
                    count = download_card_prices(game_name)
                    if count is not None:
                        self.logger.info(f"Refreshed {count} {game_name} price entries")
                        refreshed_prices = True
                except Exception as e:
                    self.logger.warning(f"Price fetch for {game_name} failed (continuing build): {e}")

            self.build_status.emit(f"Building {game_name} database…")
            self._db_progress_pct = 0

            for attempt in range(1, 4):  # initial try + 2 retries
                if cancel_event.is_set():
                    break
                try:
                    self.logger.info(
                        f"Building DB for {game_name} at {game_path} (attempt {attempt})"
                    )
                    if not os.path.exists(game_path):
                        raise RuntimeError(f"Game folder not found: {game_path}")
                    db = build_feature_database(
                        progress_callback=progress_callback,
                        db_path=game_path,
                        cancel_event=cancel_event,
                    )
                    # A cancelled build legitimately returns few/no entries —
                    # don't count it as a failure or burn retries on it.
                    if cancel_event.is_set():
                        break
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
        cancelled = cancel_event.is_set()
        self.logger.info(
            f"DB builds finished in {elapsed:.1f}s "
            f"({len(failed)} failed{', cancelled' if cancelled else ''})"
        )
        if refreshed_prices:
            self.prices_refreshed.emit()
        self.build_done.emit(not failed and not cancelled)

    def _on_build_status(self, text: str) -> None:
        """Main-thread slot: route worker status lines into the progress dialog."""
        if self._progress_dialog is not None:
            self._progress_dialog.set_status(text)

    def _cancel_db_build(self) -> None:
        """Main-thread slot: the dialog's Cancel was pressed (or it was closed)."""
        self.logger.info("Database build cancel requested")
        self._build_cancel_event.set()

    def _on_build_done(self, success: bool) -> None:
        """Main-thread slot: finalize UI after the background build finishes."""
        self._db_progress_timer.stop()
        if self._progress_dialog is not None:
            self._progress_dialog.hide()
        if self._build_cancel_event.is_set():
            self.match_label.setText("Database build cancelled.")
        elif success:
            self.match_label.setText("All databases ready. Scan a card to begin.")
        # The in-memory featureDB may be stale after a rebuild — force reload on next scan
        self._loaded_game = None
        clear_price_cache()

    def _on_prices_refreshed(self) -> None:
        """Main-thread slot: re-read price files and re-render the current match."""
        clear_price_cache()
        if self.last_matches:
            self._show_match_at(self.current_match_idx)

    def _tick_db_progress(self) -> None:
        """Poll _db_progress_pct and update the dialog's progress bar.

        The timer keeps running until _on_build_done — a single game hitting
        100% must not kill progress display for the games after it.
        """
        if self._progress_dialog is not None:
            self._progress_dialog.set_progress(int(getattr(self, "_db_progress_pct", 0)))

    # ------------------------------------------------------------------ camera

    def _on_camera_btn(self) -> None:
        """The smart toggle: Start when idle, Stop when running."""
        if self._camera_state == "running":
            self.stop_camera()
        elif self._camera_state == "idle":
            self.start_camera()
        # "starting" → button is disabled, nothing to do

    def _set_camera_state(self, state: str) -> None:
        """Sync the camera toggle's text/icon/enabled + [cam=...] QSS state."""
        self._camera_state = state
        tokens = theme_tokens(self.theme)
        if state == "starting":
            self.camera_btn.setText("Starting…")
            self.camera_btn.setIcon(get_icon("camera", tokens["muted"]))
            self.camera_btn.setEnabled(False)
        elif state == "running":
            self.camera_btn.setText("Stop")
            self.camera_btn.setIcon(get_icon("stop", tokens["danger"]))
            self.camera_btn.setEnabled(True)
        else:  # idle
            self.camera_btn.setText("Start")
            self.camera_btn.setIcon(get_icon("play", tokens["on_accent"]))
            self.camera_btn.setEnabled(True)
        # Property-based QSS needs an explicit repolish to take effect
        self.camera_btn.setProperty("cam", state)
        self.camera_btn.style().unpolish(self.camera_btn)
        self.camera_btn.style().polish(self.camera_btn)

    def start_camera(self) -> None:
        """Open the camera in a background thread, then start the preview.

        open_capture() (lorebook.hardware.camera, shared with the headless
        sorter) probes backends and reads warm-up frames — several seconds of
        blocking work that must stay off the GUI thread or the window sits
        unpainted. The worker hands the opened capture back via camera_ready/
        camera_failed; a stale token (Stop or another Start meanwhile) means
        the result is discarded.
        """
        self.stop_camera()
        cv2.destroyAllWindows()

        self._cam_open_token += 1
        token = self._cam_open_token
        self._set_camera_state("starting")
        self.preview_label.setText("Starting camera…")
        self.match_label.setText("Starting camera…")

        threading.Thread(
            target=self._open_camera_worker,
            args=(token, self.camera_index),
            daemon=True,
        ).start()

    def _open_camera_worker(self, token: int, camera_index: int) -> None:
        """Background thread: open the camera and report back via signals only."""
        try:
            cap = open_capture(camera_index)
        except Exception as e:
            self.camera_failed.emit(token, str(e))
            return
        self.camera_ready.emit(token, cap)

    def _on_camera_ready(self, token: int, cap) -> None:
        """Main-thread slot: the background open succeeded — start the preview."""
        if token != self._cam_open_token:
            # Stop (or a newer Start) happened while this open was in flight.
            try:
                cap.release()
            except Exception:
                pass
            return
        self.cap = cap
        self.read_fail_count = 0
        self.last_frame_time = time.monotonic()
        self.timer.start(30)
        self.watchdog.start()
        self._set_camera_state("running")
        self.match_label.setText("Camera running …")

    def _on_camera_failed(self, token: int, message: str) -> None:
        """Main-thread slot: the background open failed."""
        if token != self._cam_open_token:
            return
        self._fail_and_stop(message)

    def _fail_and_stop(self, message: str) -> None:
        self.stop_camera()
        QMessageBox.warning(self, "Camera Error", f"{message}\n\nTry a different camera index in Settings.")

    def stop_camera(self) -> None:
        """Stop timers and release camera resources."""
        self._cam_open_token += 1  # invalidate any in-flight background open
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
        self._set_camera_state("idle")
        self.preview_label.setText("Camera stopped")
        gc.collect()

    def _watchdog_tick(self) -> None:
        if self.cap is not None and self.last_frame_time is not None:
            if (time.monotonic() - self.last_frame_time) > 3.0:
                self._fail_and_stop("No frames received for 3 seconds. Camera stopped.")

    def _focus_rect(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Return (fx, fy, fw, fh) for a 63:88 portrait focus box at ~60% of frame height."""
        return focus_rect(h, w)

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

            if self.auto_scan:
                small = cv2.resize(crop_to_card(frame), (64, 64), interpolation=cv2.INTER_AREA)
                if self._motion_gate.update(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)):
                    self.capture_and_match()

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
            img_bgr = crop_to_card(img_bgr)

        # Release any foil-only lock from the previous match before the fresh
        # auto-detect; _show_match_at re-locks if the new match is foil-only.
        if self._foil_locked:
            self._foil_locked = False
            self.foil_check.setEnabled(True)
        self.foil_check.setChecked(self.keep_foil_checked or is_probably_foil(img_bgr, threshold=self.foil_threshold))

        active_game = self.get_active_game()
        if not active_game:
            self.match_label.setText("Please select a game type in Settings.")
            self.add_csv_btn.setEnabled(False)
            return

        # Reload featureDB only when the active game changed (or after a rebuild)
        if self._loaded_game != active_game or not self.featureDB:
            self.load_game_database(active_game)
        if not self.featureDB:
            self.match_label.setText("No feature database found. Build the DB first.")
            self.add_csv_btn.setEnabled(False)
            return

        features = extract_features(img_bgr)
        if features is None:
            self.match_label.setText("Could not extract features from image.")
            self.add_csv_btn.setEnabled(False)
            return

        matches = self._match_index.find(features, threshold=self.confidence_threshold)
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

        # After _show_match_at so the warning isn't overwritten by its label text.
        if len(matches) >= 2 and (matches[0][1] - matches[1][1]) < self.AMBIGUOUS_GAP:
            self.match_label.setText(
                f"⚠ Close match ({matches[0][1]:.1%} vs {matches[1][1]:.1%}) — "
                "check alternatives with A/D before adding."
            )

    def _show_match_at(self, idx: int) -> None:
        """Display the match at position idx in the result panel."""
        if not self.last_matches:
            return
        idx = max(0, min(idx, len(self.last_matches) - 1))
        self.current_match_idx = idx
        fname, score = self.last_matches[idx]

        set_code, card_code = split_filename(fname)
        active_game = self.get_active_game()
        code = f"{set_code}-{card_code}" if set_code else fname
        card_name = name_for(set_code, card_code, active_game) if active_game else None
        self.match_name_label.setText(f"{card_name}  ·  {code}" if card_name else code)
        self.match_detail_label.setText(
            f"Set {set_code}  ·  {score:.1%} match" if set_code else f"{score:.1%} match"
        )
        self._apply_foil_lock(set_code, card_code, active_game)
        price = price_for(set_code, card_code, active_game) if active_game else None
        self.match_price_label.setText(
            format_price(price, self.currency, rate_for(self.currency))
        )
        self.match_pos_label.setText(f"{idx + 1} / {len(self.last_matches)}")
        self.match_label.setText(f"{score:.3f}  {fname}")

        if active_game:
            match_path = os.path.join(BASE_DATABASE_PATH, active_game, fname)
            img = cv2.imread(match_path, cv2.IMREAD_COLOR)
            if img is not None:
                # setPixmap clears any placeholder text automatically
                self._show_on_label(self.image_label, img, fill=False)
                return
            self.logger.error(f"Failed to load image: {match_path}")
        self.image_label.setText("—")

    def _apply_foil_lock(self, set_code: str, card_code: str, active_game: Optional[str]) -> None:
        """
        Force the Foil checkbox on (and lock it) while a foil-only rarity
        (Lorcana Enchanted/Epic/Iconic) is displayed — those cards have no
        normal printing, and Dreamborn.ink rejects a "normal" row for them.
        Restores the user's checkbox state when a regular card is shown again.
        """
        foil_only = bool(active_game) and is_foil_only_card(
            set_code, card_code, game_type_from_name(active_game)
        )
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
        if self._foil_locked:
            self._foil_before_lock = self.keep_foil_checked
        else:
            self.foil_check.setChecked(self.keep_foil_checked)

    def add_to_csv(self) -> None:
        """Write the current match to the game-appropriate CSV."""
        if not self.last_matches:
            self.set_status("No match to add.", is_error=True)
            return

        fname = self.last_matches[self.current_match_idx][0]
        try:
            cnt = int(self.count_edit.text())
        except ValueError:
            cnt = 0
        if cnt < 1:
            # Refuse rather than silently adding 1 — a typo'd count ("1o",
            # empty field) must not corrupt the collection.
            self.set_status("Invalid count — enter a number from 1 to 999.", is_error=True)
            return
        is_foil = self.foil_check.isChecked()

        active_game = self.get_active_game() or "Lorcana"
        target_file = csv_for_game(active_game)

        update_cardlist(fname, is_foil, cnt, game=active_game)
        self._last_add = (fname, is_foil, cnt, active_game)
        self.undo_btn.setEnabled(True)
        self.set_status(f"Added {cnt}× {fname} to {target_file}")
        self.add_csv_btn.setEnabled(False)
        self.collection_view.refresh()

    def undo_last_add(self) -> None:
        """Reverse the most recent Add (single-level undo)."""
        if not self._last_add:
            return
        fname, is_foil, cnt, game = self._last_add
        update_cardlist(fname, is_foil, -cnt, game=game, allow_negative=True)
        self._last_add = None
        self.undo_btn.setEnabled(False)
        self.set_status(f"Removed {cnt}× {fname}")
        self.collection_view.refresh()

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

    # ------------------------------------------------------------------ scaling

    def _apply_scale(self) -> None:
        """
        Scale the fixed-size chrome — scan FAB, Add/Undo buttons, result panel,
        thumbnail — with the window height, so the layout works maximized and
        small alike. Clamps keep everything usable at the extremes.
        """
        h = max(self.height(), 1)

        # Scan FAB: ~7% of window height; pill radius and font follow.
        fab_h = int(min(max(h * 0.07, 48), 76))
        fab_font = int(min(max(fab_h * 0.29, 14), 19))
        self.capture_btn.setMinimumSize(int(fab_h * 3.6), fab_h)
        self.capture_btn.setIconSize(QSize(fab_font + 5, fab_font + 5))
        self.capture_btn.setStyleSheet(
            f"QPushButton#scanBtn {{ border-radius: {fab_h // 2}px; "
            f"font-size: {fab_font}px; padding: 0 {fab_h // 2}px; }}"
        )

        # Add to Collection / Undo: modestly taller on big windows.
        btn_h = int(min(max(h * 0.035, 28), 40))
        self.add_csv_btn.setMinimumHeight(btn_h)
        self.undo_btn.setMinimumHeight(btn_h)

        # Result panel and its card thumbnail grow together (92x128 aspect).
        panel_h = int(min(max(h * 0.20, 140), 200))
        self.result_panel.setFixedHeight(panel_h)
        thumb_h = panel_h - 24
        self.image_label.setFixedSize(int(thumb_h * 92 / 128), thumb_h)

    def resizeEvent(self, event):
        self._apply_scale()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ keyboard

    def focusInEvent(self, event):
        self.setFocus()
        super().focusInEvent(event)

    def _on_page_changed(self, index: int) -> None:
        """Refresh the collection table whenever its page becomes visible."""
        if self.stack.widget(index) is self.collection_view:
            self.collection_view.refresh()

    def keyPressEvent(self, event):
        # Scan shortcuts are Scanner-page-only: browsing the collection table
        # must not trigger captures or foil toggles.
        if self.stack.currentWidget() is not self.scanner_page:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_C:
            event.accept()
            self.capture_and_match()
        elif key == Qt.Key_S and (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            if self.add_csv_btn.isEnabled():
                self.add_to_csv()
        elif key == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undo_last_add()
        elif key == Qt.Key_A:
            self.prev_match()
        elif key == Qt.Key_D:
            self.next_match()
        elif key == Qt.Key_F:
            if self.foil_check.isEnabled():
                self.foil_check.setChecked(not self.foil_check.isChecked())
        else:
            super().keyPressEvent(event)

    def _setup_tooltips(self):
        self.capture_btn.setToolTip("Scan Card (C)")
        self.add_csv_btn.setToolTip("Add to Collection (Ctrl+S or Alt+S)")
        self.undo_btn.setToolTip("Undo last add (Ctrl+Z)")
        self.prev_btn.setToolTip("Previous match (A)")
        self.next_btn.setToolTip("Next match (D)")
        self.foil_check.setToolTip("Toggle Foil (F)")
