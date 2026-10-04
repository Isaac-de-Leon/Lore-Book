# main_window.py — MainWindow: camera preview, card matching, CSV export.

import faulthandler
import logging
import logging.handlers
import os
import sys
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIntValidator
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
from lorebook.core.build import BuildReport
from lorebook.core.card_database import (
    GameDatabase,
)
from lorebook.core.card_names import name_for
from lorebook.core.card_prices import (
    clear_price_cache,
    format_price,
    price_for,
    rate_for,
)
from lorebook.core.csv_manager import split_filename, update_cardlist
from lorebook.core.features import extract_features, get_extractor, visualize_activation_overlay
from lorebook.core.game_types import (
    BASE_DATABASE_PATH,
    csv_for_game,
    game_folders,
    game_type_from_name,
    is_foil_only_card,
)
from lorebook.core.image_utils import (
    card_motion_gate,
    crop_to_card,
    draw_focus_overlay,
    is_probably_foil,
    motion_sample,
)
from lorebook.core.matching import MatchIndex
from lorebook.core.paths import data_path
from lorebook.core.settings import AppSettings
from lorebook.ui.collection_view import CollectionView
from lorebook.ui.icons import get_icon
from lorebook.ui.progress_dialog import BuildProgressDialog
from lorebook.ui.qt_images import show_bgr
from lorebook.ui.settings_window import SettingsWindow
from lorebook.ui.styles import DEFAULT_THEME, THEMES, build_stylesheet, theme_tokens
from lorebook.ui.threads import WorkerThread
from lorebook.ui.workers import BuildWorker, CameraWorker

logger = logging.getLogger(__name__)


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
        logger.warning("Could not enable native crash logging: %s", e)

    # Uncaught Python exceptions land in the main log too (PySide6 prints
    # them to stderr, which is invisible when launched outside a terminal).
    def _log_excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_excepthook

    logger.info("Logging initialized — %s", log_path)


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
    Background daemon threads: the DB build (_build_all_games_worker), the
    camera opener (_open_camera_worker, which keeps the slow open_capture()
    backend probe off the paint path), the scan worker (_scan_worker:
    feature extraction + matching) and a one-shot model warm-up. Model use
    is serialized by features._model_lock, so a scan during a build waits
    for the current batch instead of racing it.
    From any non-main thread, the ONLY permitted interactions with this
    object are:
      - writing the plain attribute ``_db_progress_pct`` (polled by a QTimer)
      - emitting the ``build_status`` / ``build_done`` / ``prices_refreshed`` /
        ``camera_ready`` / ``camera_failed`` / ``scan_done`` signals (Qt delivers them on the
        main thread)
      - reading the ``cancel_event`` / ``token`` passed in as worker arguments
    Everything else — widgets (including the progress dialog),
    game_db, settings — is
    main-thread-only. Camera-open results carry a generation token checked
    against ``_cam_open_token`` so a Stop/restart discards stale opens;
    scan results likewise carry ``_scan_token``.
    """

    logger = logging.getLogger("MainWindow")

    # Top-2 score gap below which a match is flagged as ambiguous (foil
    # variants, reprints, and alt arts often score within a couple percent).
    AMBIGUOUS_GAP = 0.02

    # Signals for marshalling background-thread updates onto the main thread.
    build_requested = Signal(object, object)  # (game folders, cancel Event) → BuildWorker.run
    camera_start_requested = Signal(int)     # camera index → CameraWorker.start
    camera_stop_requested = Signal()         # → CameraWorker.stop
    scan_done = Signal(int, object)      # (scan token, (matches | None, error | None))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _split_filename(filename: str) -> tuple[str, str]:
        return split_filename(filename)

    def show_error(self, message: str, title: str = "Error", details: str | None = None) -> None:
        if details:
            self.logger.error("%s: %s", message, details)
        else:
            self.logger.error("%s", message)
        QMessageBox.critical(self, title, message)

    def set_status(self, message: str, is_error: bool = False, timeout_ms: int = 3500) -> None:
        tokens = theme_tokens(self.settings.theme)
        color = tokens["danger"] if is_error else tokens["success"]
        self.csv_status.setStyleSheet(f"color: {color}; padding-left: 8px;")
        self.csv_status.setText(message)
        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, lambda: self.csv_status.setText(""))

    def get_active_game(self) -> str | None:
        """Card_Images folder name of the selected game (real case), or None."""
        return self.settings.active_game()

    def load_game_database(self, game_name: str) -> bool:
        """Make game_name the active database. True if its cache has entries."""
        return self.game_db.load(game_name)

    # ------------------------------------------------------------------ init

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Lore Book v{__version__}")
        self.resize(1000, 840)

        # State
        self.game_db = GameDatabase()  # active game's vectors + match index
        self.settings = AppSettings.load()
        # min_std suppresses triggers on an empty (near-uniform) focus box
        self._motion_gate = card_motion_gate()

        # Camera: opened and read on its own thread (CameraWorker); the GUI
        # keeps only the latest frame and when it arrived (for the watchdog).
        self._camera = WorkerThread(CameraWorker(), "lorebook-camera")
        self.last_frame: np.ndarray | None = None
        self.last_frame_time: float | None = None

        self.last_matches: list[tuple[str, float]] = []
        self.current_match_idx: int = 0
        self._last_add: tuple[str, bool, int, str] | None = None  # (fname, foil, count, game)
        self._foil_locked = False        # Foil checkbox forced on for foil-only rarities
        self._foil_before_lock = False   # user's checkbox state to restore on unlock

        # DB build state (dialog created lazily on first build)
        self._progress_dialog: BuildProgressDialog | None = None
        self._build_cancel_event = threading.Event()
        self._build_running = False
        self._build = WorkerThread(BuildWorker(), "lorebook-build")

        # Scan state: extraction runs on a worker thread; results carry a
        # token so a superseded scan's result is dropped. _matches_game is
        # the game the displayed matches belong to (Add/Undo/Prev/Next use
        # it, never whatever Settings says now).
        self._scan_token = 0
        self._scan_busy = False
        self._matches_game: str | None = None

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
        self.foil_check.setChecked(self.settings.keep_foil_checked)

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
        build_worker = self._build.worker
        self.build_requested.connect(build_worker.run)
        build_worker.status.connect(self._on_build_status)
        build_worker.progress.connect(self._on_build_progress)
        build_worker.done.connect(self._on_build_done)
        self._build.start()
        camera_worker = self._camera.worker
        self.camera_start_requested.connect(camera_worker.start)
        self.camera_stop_requested.connect(camera_worker.stop)
        camera_worker.state_changed.connect(self._on_camera_state)
        camera_worker.frame_ready.connect(self._on_frame)
        camera_worker.failed.connect(self._on_camera_failed)
        self._camera.start()
        self.scan_done.connect(self._on_scan_done)

        # Watchdog: frames stopped arriving (e.g. a driver hung inside read()).
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(1000)
        self.watchdog.timeout.connect(self._watchdog_tick)

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

        self.apply_theme(self.settings.theme)  # stylesheet + icon tints (also scales chrome)
        # Deferred until the event loop runs so the main window paints before
        # the progress dialog appears — showing it here leaves both windows
        # as unpainted white rectangles until the first paint event.
        QTimer.singleShot(0, self.start_db_build_in_background)
        # Load the feature model in the background so the first scan doesn't
        # stall for seconds (or a first-run weights download).
        threading.Thread(target=self._warm_model_worker, daemon=True).start()

    # ------------------------------------------------------------------ theme

    def apply_theme(self, theme: str) -> None:
        """Switch the app-wide theme at runtime: stylesheet + icon tints.

        Child dialogs (Settings, build progress) inherit the stylesheet
        automatically, so re-setting it here re-themes them too.
        """
        self.settings.theme = theme if theme in THEMES else DEFAULT_THEME
        self.setStyleSheet(build_stylesheet(self.settings.theme))
        self._apply_icons()
        self._apply_scale()  # re-applies the scan FAB's inline geometry QSS

    def _apply_icons(self) -> None:
        """(Re)tint every icon for the current theme."""
        tokens = theme_tokens(self.settings.theme)
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

    def save_settings(self) -> None:
        """Persist the current settings (atomic; see AppSettings.save)."""
        self.settings.save()

    def closeEvent(self, event):
        self.save_settings()
        # Clean shutdown: release the camera, stop a running build at its next
        # safe point (downloads and cache writes are atomic, so nothing is left
        # half-written), and drop any in-flight scan result.
        self.stop_camera()
        self._build_cancel_event.set()
        self._scan_token += 1
        self.stop_workers()
        super().closeEvent(event)

    def stop_workers(self, timeout_ms: int = 3000) -> bool:
        """Stop every worker thread; True if all finished in time.

        A worker stuck in a long blocking call (e.g. a network request) can
        outlive the timeout — app.main then exits without destroying it.
        """
        stopped = all([w.stop(timeout_ms) for w in self._workers()])
        if not self._camera.is_running():
            self._camera.worker.release()  # thread finished: safe to touch from here
        return stopped

    def _workers(self) -> list[WorkerThread]:
        return [self._camera, self._build]

    # ------------------------------------------------------------------ DB build

    def start_db_build_in_background(self) -> None:
        """Discover game folders and kick off one background build thread for all of them."""
        card_images_dir = BASE_DATABASE_PATH
        try:
            os.makedirs(card_images_dir, exist_ok=True)
            folders = game_folders(card_images_dir)
        except OSError as e:
            self.logger.error("Error scanning %s: %s", card_images_dir, e)
            self.match_label.setText("Error scanning Card_Images directory")
            return

        if not folders:
            self.match_label.setText("No game folders found in Card_Images directory")
            return

        # One build at a time — Settings → Rebuild while the startup build is
        # still running just re-shows its progress dialog.
        if self._build_running:
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
        self._progress_dialog.set_status(f"Building {folders[0]} database, please wait…")
        self._progress_dialog.show()

        self._build_running = True
        self.build_requested.emit(folders, self._build_cancel_event)

    def _on_build_status(self, text: str) -> None:
        """Main-thread slot: route worker status lines into the progress dialog."""
        if self._progress_dialog is not None:
            self._progress_dialog.set_status(text)

    def _cancel_db_build(self) -> None:
        """Main-thread slot: the dialog's Cancel was pressed (or it was closed)."""
        self.logger.info("Database build cancel requested")
        self._build_cancel_event.set()

    def _on_build_progress(self, pct: int) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.set_progress(pct)

    def _on_build_done(self, report: BuildReport) -> None:
        """Main-thread slot: finalize UI after the background build finishes."""
        self._build_running = False
        if self._progress_dialog is not None:
            self._progress_dialog.hide()
        if report.cancelled:
            self.match_label.setText("Database build cancelled.")
        elif report.ok:
            self.match_label.setText("All databases ready. Scan a card to begin.")
        else:
            # The progress dialog is hidden now — don't let the failure vanish with it.
            self.match_label.setText(
                f"Database build failed for {', '.join(report.failed)} — see logs/card_scanner.log."
            )
        # The in-memory vectors may be stale after a rebuild — reload on next scan
        self.game_db.invalidate()
        # Price/rate files may have been rewritten: re-read and re-render.
        clear_price_cache()
        if report.prices_refreshed and self.last_matches:
            self._show_match_at(self.current_match_idx)

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
        tokens = theme_tokens(self.settings.theme)
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
        """Ask the camera worker to open the camera (a few seconds of backend
        probing and warm-up, all on the camera thread)."""
        self.last_frame = None
        self._set_camera_state("starting")
        self.preview_label.setText("Starting camera…")
        self.match_label.setText("Starting camera…")
        self.camera_start_requested.emit(self.settings.camera_index)

    def stop_camera(self) -> None:
        """Ask the camera worker to release the camera; ignore frames from now on."""
        self.camera_stop_requested.emit()
        self._show_camera_stopped()

    def _show_camera_stopped(self) -> None:
        self.watchdog.stop()
        self.last_frame = None
        self.last_frame_time = None
        self._set_camera_state("idle")
        self.preview_label.setText("Camera stopped")

    def _on_camera_state(self, state: str) -> None:
        """Main-thread slot: the camera worker changed state."""
        if state == "running":
            self.last_frame_time = time.monotonic()
            self.watchdog.start()
            self._set_camera_state("running")
            self.match_label.setText("Camera running …")
        elif state == "idle":
            self._show_camera_stopped()
        else:
            self._set_camera_state(state)

    def _on_camera_failed(self, message: str) -> None:
        """Main-thread slot: the camera couldn't open or stopped delivering frames."""
        self._show_camera_stopped()
        QMessageBox.warning(self, "Camera Error", f"{message}\n\nTry a different camera index in Settings.")

    def _watchdog_tick(self) -> None:
        if self.last_frame_time is not None and (time.monotonic() - self.last_frame_time) > 3.0:
            self.stop_camera()
            self._on_camera_failed("No frames received for 3 seconds. Camera stopped.")

    def _on_frame(self, frame: np.ndarray) -> None:
        """Main-thread slot: a new camera frame — overlay it and show it."""
        try:
            if self._camera_state != "running":
                return  # a frame sent before a Stop/failure arrived late
            self.last_frame_time = time.monotonic()
            if self.settings.rotate_display:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.last_frame = frame
            if self.settings.auto_scan and self._motion_gate.update(motion_sample(frame)):
                self.capture_and_match()
            overlay = draw_focus_overlay(frame)
            if self.settings.debug_mode:
                overlay = visualize_activation_overlay(overlay, blocking=False)
            show_bgr(self.preview_label, overlay, fill=True)
        finally:
            self._camera.worker.frame_consumed()  # ready for the next frame

    # ------------------------------------------------------------------ matching

    def capture_and_match(self) -> None:
        """Capture the current frame, extract features, and find the best matches."""
        if self._scan_busy:
            return  # one scan at a time; auto-scan simply skips this trigger
        if self.last_frame is None:
            QMessageBox.warning(self, "Scan Card", "Camera is not running or no frame available.")
            return

        img_bgr = self.last_frame.copy()

        if self.settings.crop_to_focus:
            img_bgr = crop_to_card(img_bgr)

        # Release any foil-only lock from the previous match before the fresh
        # auto-detect; _show_match_at re-locks if the new match is foil-only.
        if self._foil_locked:
            self._foil_locked = False
            self.foil_check.setEnabled(True)
        self.foil_check.setChecked(self.settings.keep_foil_checked or is_probably_foil(img_bgr, threshold=self.settings.foil_threshold))

        active_game = self.get_active_game()
        if not active_game:
            self.match_label.setText("Please select a game type in Settings.")
            self.add_csv_btn.setEnabled(False)
            return

        # Reload only when the active game changed (or after a rebuild)
        if not self.game_db.ensure(active_game):
            self.match_label.setText("No feature database found. Build the DB first.")
            self.add_csv_btn.setEnabled(False)
            return

        # Extraction (and the first-call model load) is seconds of work —
        # run it off the GUI thread and finish in _on_scan_done.
        self._scan_token += 1
        self._set_scan_busy(True)
        threading.Thread(
            target=self._scan_worker,
            args=(self._scan_token, img_bgr, self.game_db.index, self.settings.confidence_threshold, active_game),
            daemon=True,
        ).start()

    def _scan_worker(self, token: int, img_bgr: np.ndarray, index: MatchIndex,
                     threshold: float, game: str) -> None:
        """Background thread: extract + match, report via scan_done only.

        Uses the index object it was handed, so a game switch on the main
        thread mid-scan can't mix games.
        """
        try:
            features = extract_features(img_bgr)
            matches = index.find(features, threshold=threshold) if features is not None else None
            self.scan_done.emit(token, (game, matches, None))
        # Boundary: never let the worker die silently — the UI stays in
        # "Scanning…" until scan_done arrives.
        except Exception as e:  # noqa: BLE001
            self.scan_done.emit(token, (game, None, str(e)))

    def _warm_model_worker(self) -> None:
        try:
            get_extractor("keras").ensure_ready()
        except Exception as e:  # noqa: BLE001 — optional warm-up; the scan reports real errors
            self.logger.warning("Background model load failed (will retry on scan): %s", e)

    def _set_scan_busy(self, busy: bool) -> None:
        self._scan_busy = busy
        self.capture_btn.setEnabled(not busy)
        self.capture_btn.setText("SCANNING…" if busy else "SCAN CARD")
        if busy:
            self.match_label.setText("Scanning…")

    def _on_scan_done(self, token: int, result) -> None:
        """Main-thread slot: display the worker's matches."""
        if token != self._scan_token:
            return  # superseded
        self._set_scan_busy(False)
        game, matches, error = result
        if error is not None:
            self.logger.error("Scan failed: %s", error)
        if matches is None:
            self.match_label.setText("Could not extract features from image.")
            self.add_csv_btn.setEnabled(False)
            return

        matches = self._filter_matches(matches, game)
        self.logger.info("Found %s matches after filtering", len(matches))

        if not matches:
            self.match_label.setText("No match found.")
            self.add_csv_btn.setEnabled(False)
            return

        self._matches_game = game
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

    def _clear_results(self) -> None:
        """Empty the result panel (e.g. the active game changed under it)."""
        self.last_matches = []
        self.current_match_idx = 0
        self._matches_game = None
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

    def _show_match_at(self, idx: int) -> None:
        """Display the match at position idx in the result panel."""
        if not self.last_matches:
            return
        idx = max(0, min(idx, len(self.last_matches) - 1))
        self.current_match_idx = idx
        fname, score = self.last_matches[idx]

        set_code, card_code = split_filename(fname)
        active_game = self._matches_game or self.get_active_game()
        code = f"{set_code}-{card_code}" if set_code else fname
        card_name = name_for(set_code, card_code, active_game) if active_game else None
        self.match_name_label.setText(f"{card_name}  ·  {code}" if card_name else code)
        self.match_detail_label.setText(
            f"Set {set_code}  ·  {score:.1%} match" if set_code else f"{score:.1%} match"
        )
        self._apply_foil_lock(set_code, card_code, active_game)
        price = price_for(set_code, card_code, active_game) if active_game else None
        self.match_price_label.setText(
            format_price(price, self.settings.currency, rate_for(self.settings.currency))
        )
        self.match_pos_label.setText(f"{idx + 1} / {len(self.last_matches)}")
        self.match_label.setText(f"{score:.3f}  {fname}")

        if active_game:
            match_path = os.path.join(BASE_DATABASE_PATH, active_game, fname)
            img = cv2.imread(match_path, cv2.IMREAD_COLOR)
            if img is not None:
                # setPixmap clears any placeholder text automatically
                show_bgr(self.image_label, img, fill=False)
                return
            self.logger.error("Failed to load image: %s", match_path)
        self.image_label.setText("—")

    def _apply_foil_lock(self, set_code: str, card_code: str, active_game: str | None) -> None:
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

    def _filter_matches(self, matches: list[tuple[str, float]], game: str | None) -> list[tuple[str, float]]:
        """Keep only matches in the selected sets of the game they were scanned for."""
        if not matches or not game:
            return matches or []
        game_selected_sets = self.settings.sets_for(game)
        filtered = []
        for fname, score in matches:
            set_code, _ = split_filename(os.path.basename(fname))
            if not game_selected_sets or set_code in game_selected_sets:
                filtered.append((fname, score))
        return filtered

    def _make_settings_dialog(self) -> SettingsWindow:
        dlg = SettingsWindow(self.settings, self)
        dlg.applied.connect(self._apply_settings)
        dlg.rebuild_requested.connect(self.start_db_build_in_background)
        return dlg

    def open_settings(self) -> None:
        self._make_settings_dialog().exec()

    def _apply_settings(self, new: AppSettings) -> None:
        """Adopt settings from the dialog: re-theme, load the game, persist."""
        self.settings = new
        self.apply_theme(new.theme)
        game = new.active_game()
        if game is not None:
            self.load_game_database(game)
        self.save_settings()
        if self.last_matches and game != self._matches_game:
            self._clear_results()
            self.match_label.setText("Game changed — scan a card.")
        if self._foil_locked:
            self._foil_before_lock = new.keep_foil_checked
        else:
            self.foil_check.setChecked(new.keep_foil_checked)

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

        # The game the match came from — not whatever Settings says now.
        active_game = self._matches_game or self.get_active_game() or "Lorcana"
        target_file = csv_for_game(active_game)

        try:
            update_cardlist(fname, is_foil, cnt, game=active_game)
        except OSError as e:  # unreadable or locked (e.g. open in Excel) — file untouched
            self.logger.error("Could not add %s to %s: %s", fname, target_file, e)
            self.set_status(f"Not added — {target_file} could not be updated (open in another program?)",
                            is_error=True, timeout_ms=8000)
            return
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
        try:
            update_cardlist(fname, is_foil, -cnt, game=game, allow_negative=True)
        except OSError as e:  # keep _last_add so the user can retry
            self.logger.error("Could not undo %s: %s", fname, e)
            self.set_status("Undo failed — the collection file could not be updated.",
                            is_error=True, timeout_ms=8000)
            return
        self._last_add = None
        self.undo_btn.setEnabled(False)
        self.set_status(f"Removed {cnt}× {fname}")
        self.collection_view.refresh()

    # ------------------------------------------------------------------ display helpers

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
