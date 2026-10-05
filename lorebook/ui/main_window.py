# main_window.py — MainWindow: app shell that wires settings, data and workers to the pages.

import logging
import os
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lorebook import __version__
from lorebook.core.build import BuildReport
from lorebook.core.card_database import GameDatabase
from lorebook.core.card_prices import clear_price_cache
from lorebook.core.csv_manager import split_filename
from lorebook.core.features import visualize_activation_overlay
from lorebook.core.game_types import BASE_DATABASE_PATH, game_folders
from lorebook.core.image_utils import (
    card_motion_gate,
    crop_to_card,
    draw_focus_overlay,
    is_probably_foil,
    motion_sample,
)
from lorebook.core.settings import AppSettings
from lorebook.ui.collection_view import CollectionView
from lorebook.ui.icons import get_icon
from lorebook.ui.progress_dialog import BuildProgressDialog
from lorebook.ui.scanner_view import ScannerView
from lorebook.ui.settings_window import SettingsWindow
from lorebook.ui.styles import DEFAULT_THEME, THEMES, build_stylesheet, theme_tokens
from lorebook.ui.threads import WorkerThread
from lorebook.ui.workers import BuildWorker, CameraWorker, ScanResult, ScanWorker

logger = logging.getLogger(__name__)

class MainWindow(QWidget):
    """
    The app shell: sidebar + header + pages (ScannerView, CollectionView),
    and the coordinator between them, the settings, the active game's
    database and three worker threads.

    Threading
    ---------
    Each worker is a QObject on its own QThread (ui/threads.py, ui/workers.py):
      - CameraWorker  opens the camera and reads frames
      - ScanWorker    extracts features and matches (long-lived; warms the model)
      - BuildWorker   downloads art/prices and builds the feature caches
    The window talks to workers only through queued signals (the *_requested
    signals below) and hears back through the workers' own signals, which Qt
    delivers on the GUI thread. The only direct cross-thread touches are
    thread-safe by design: setting the build's cancel Event and calling
    CameraWorker.frame_consumed(). Everything else — widgets, settings,
    game_db — is GUI-thread-only. Scan results carry a token so a superseded
    scan's result is dropped. Model use is serialized by features._model_lock
    (a scan during a build waits for the current batch).
    """

    # Queued requests to the workers (connected in _start_workers).
    camera_start_requested = Signal(int)                       # camera index
    camera_stop_requested = Signal()
    scan_job_requested = Signal(int, object, object, float, str)  # token, image, index, threshold, game
    warm_up_requested = Signal()
    build_requested = Signal(object, object)                   # game folders, cancel Event

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Lore Book v{__version__}")
        self.resize(1000, 840)

        self.settings = AppSettings.load()
        self.game_db = GameDatabase()  # active game's vectors + match index
        self._motion_gate = card_motion_gate()

        self.last_frame: np.ndarray | None = None
        self.last_frame_time: float | None = None
        self._camera_state = "idle"  # idle | starting | running
        self._scan_token = 0
        self._scan_busy = False
        self._build_running = False
        self._build_cancel_event = threading.Event()
        self._progress_dialog: BuildProgressDialog | None = None

        self._build_ui()
        self._start_workers()

        self.watchdog = QTimer(self)  # frames stopped arriving (e.g. a driver hung in read())
        self.watchdog.setInterval(1000)
        self.watchdog.timeout.connect(self._watchdog_tick)

        self.setFocusPolicy(Qt.StrongFocus)
        self.apply_theme(self.settings.theme)
        # Deferred until the event loop runs so the main window paints before
        # the progress dialog appears.
        QTimer.singleShot(0, self.start_db_build_in_background)
        self.warm_up_requested.emit()  # load the model now, not on the first scan

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        title_label = QLabel("LORE BOOK")
        title_label.setObjectName("appTitle")
        # One smart camera toggle; label/icon/style track _camera_state.
        self.camera_btn = QPushButton("Start")
        self.camera_btn.setObjectName("cameraBtn")
        self.camera_btn.setToolTip("Start / stop the camera")
        self.camera_btn.setFocusPolicy(Qt.NoFocus)
        self.camera_btn.clicked.connect(self._on_camera_btn)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setToolTip("Lore Book")

        def nav_button(tooltip: str, checkable: bool = False) -> QPushButton:
            btn = QPushButton()
            btn.setObjectName("navBtn")
            btn.setIconSize(QSize(22, 22))
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(checkable)
            btn.setFocusPolicy(Qt.NoFocus)
            return btn

        self.nav_scanner_btn = nav_button("Scanner", checkable=True)
        self.nav_scanner_btn.setChecked(True)
        self.nav_collection_btn = nav_button("Collection", checkable=True)
        self.settings_btn = nav_button("Settings")
        self.settings_btn.clicked.connect(self.open_settings)

        self.scanner = ScannerView(self.settings)
        self.scanner.scan_requested.connect(self.capture_and_match)
        self.collection_view = CollectionView(initial_game=self.get_active_game())
        self.scanner.collection_changed.connect(self.collection_view.refresh)
        self.collection_view.cleared.connect(lambda _game: self.scanner.forget_last_add())

        self.stack = QStackedWidget()
        self.stack.addWidget(self.scanner)
        self.stack.addWidget(self.collection_view)
        # Pick up CSV changes made while the page was hidden.
        self.stack.currentChanged.connect(self._on_page_changed)
        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        nav_group.addButton(self.nav_scanner_btn, 0)
        nav_group.addButton(self.nav_collection_btn, 1)
        nav_group.idClicked.connect(self.stack.setCurrentIndex)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(64)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(10, 14, 10, 14)
        side.setSpacing(10)
        side.addWidget(self.logo_label, 0, Qt.AlignHCenter)
        side.addSpacing(8)
        side.addWidget(self.nav_scanner_btn, 0, Qt.AlignHCenter)
        side.addWidget(self.nav_collection_btn, 0, Qt.AlignHCenter)
        side.addStretch(1)
        side.addWidget(self.settings_btn, 0, Qt.AlignHCenter)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(title_label)
        top.addStretch()
        top.addWidget(self.camera_btn)
        content = QVBoxLayout()
        content.setContentsMargins(16, 12, 16, 12)
        content.setSpacing(8)
        content.addLayout(top)
        content.addWidget(self.stack, stretch=1)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(sidebar)
        root.addLayout(content, stretch=1)

    def _start_workers(self) -> None:
        self._camera = WorkerThread(CameraWorker(), "lorebook-camera")
        self._scan = WorkerThread(ScanWorker(), "lorebook-scan")
        self._build = WorkerThread(BuildWorker(), "lorebook-build")

        camera = self._camera.worker
        self.camera_start_requested.connect(camera.start)
        self.camera_stop_requested.connect(camera.stop)
        camera.state_changed.connect(self._on_camera_state)
        camera.frame_ready.connect(self._on_frame)
        camera.failed.connect(self._on_camera_failed)

        scan = self._scan.worker
        self.scan_job_requested.connect(scan.scan)
        self.warm_up_requested.connect(scan.warm_up)
        scan.done.connect(self._on_scan_done)

        build = self._build.worker
        self.build_requested.connect(build.run)
        build.status.connect(self._on_build_status)
        build.progress.connect(self._on_build_progress)
        build.done.connect(self._on_build_done)

        for worker in self._workers():
            worker.start()

    def _workers(self) -> list[WorkerThread]:
        return [self._camera, self._scan, self._build]

    # ------------------------------------------------------------------ theme / scale

    def apply_theme(self, theme: str) -> None:
        """Switch the app-wide theme at runtime (child dialogs inherit it)."""
        self.settings.theme = theme if theme in THEMES else DEFAULT_THEME
        self.setStyleSheet(build_stylesheet(self.settings.theme))
        tokens = theme_tokens(self.settings.theme)
        self.logo_label.setPixmap(get_icon("book", tokens["accent"]).pixmap(26, 26))
        self.nav_scanner_btn.setIcon(get_icon("camera", tokens["muted"], checked_color=tokens["accent"]))
        self.nav_collection_btn.setIcon(get_icon("grid", tokens["muted"], checked_color=tokens["accent"]))
        self.settings_btn.setIcon(get_icon("sliders", tokens["muted"]))
        self._set_camera_state(self._camera_state)  # re-tints the camera toggle
        self.scanner.apply_theme(tokens)
        self.collection_view.apply_icons(tokens["text"], tokens["danger"])
        self.scanner.apply_scale(self.height())

    def resizeEvent(self, event) -> None:
        self.scanner.apply_scale(self.height())
        super().resizeEvent(event)

    # ------------------------------------------------------------------ settings

    def get_active_game(self) -> str | None:
        """Card_Images folder name of the selected game (real case), or None."""
        return self.settings.active_game()

    def load_game_database(self, game_name: str) -> bool:
        """Make game_name the active database. True if its cache has entries."""
        return self.game_db.load(game_name)

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
        self.scanner.apply_settings(new)

    def save_settings(self) -> None:
        self.settings.save()

    # ------------------------------------------------------------------ shutdown

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings()
        # Release the camera, stop a running build at its next safe point
        # (downloads and cache writes are atomic), drop any in-flight scan.
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

    # ------------------------------------------------------------------ DB build

    def start_db_build_in_background(self) -> None:
        """Build every game folder's database on the build worker."""
        try:
            os.makedirs(BASE_DATABASE_PATH, exist_ok=True)
            folders = game_folders(BASE_DATABASE_PATH)
        except OSError as e:
            logger.error("Error scanning %s: %s", BASE_DATABASE_PATH, e)
            self.scanner.set_status_text("Error scanning Card_Images directory")
            return
        if not folders:
            self.scanner.set_status_text("No game folders found in Card_Images directory")
            return

        if self._progress_dialog is None:
            self._progress_dialog = BuildProgressDialog(self)
            self._progress_dialog.cancel_requested.connect(self._cancel_db_build)
        # One build at a time — Rebuild during the startup build re-shows its dialog.
        if self._build_running:
            self._progress_dialog.show()
            self._progress_dialog.raise_()
            return
        # Fresh Event per build: never clear() a shared one.
        self._build_cancel_event = threading.Event()
        self._progress_dialog.reset_for_new_build()
        self._progress_dialog.set_status(f"Building {folders[0]} database, please wait…")
        self._progress_dialog.show()
        self._build_running = True
        self.build_requested.emit(folders, self._build_cancel_event)

    def _cancel_db_build(self) -> None:
        logger.info("Database build cancel requested")
        self._build_cancel_event.set()  # thread-safe; read by the build job

    def _on_build_status(self, text: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.set_status(text)

    def _on_build_progress(self, pct: int) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.set_progress(pct)

    def _on_build_done(self, report: BuildReport) -> None:
        self._build_running = False
        if self._progress_dialog is not None:
            self._progress_dialog.hide()
        if report.cancelled:
            self.scanner.set_status_text("Database build cancelled.")
        elif report.ok:
            self.scanner.set_status_text("All databases ready. Scan a card to begin.")
        else:
            # The progress dialog is hidden now — don't let the failure vanish with it.
            self.scanner.set_status_text(
                f"Database build failed for {', '.join(report.failed)} — see logs/card_scanner.log."
            )
        self.game_db.invalidate()  # vectors may be stale after a rebuild
        clear_price_cache()        # price/rate files may have been rewritten
        if report.prices_refreshed:
            self.scanner.refresh_current_match()

    # ------------------------------------------------------------------ camera

    def _on_camera_btn(self) -> None:
        """The smart toggle: Start when idle, Stop when running."""
        if self._camera_state == "running":
            self.stop_camera()
        elif self._camera_state == "idle":
            self.start_camera()

    def _set_camera_state(self, state: str) -> None:
        """Sync the camera toggle's text/icon/enabled + [cam=...] QSS state."""
        self._camera_state = state
        tokens = theme_tokens(self.settings.theme)
        text, icon, color, enabled = {
            "starting": ("Starting…", "camera", tokens["muted"], False),
            "running": ("Stop", "stop", tokens["danger"], True),
        }.get(state, ("Start", "play", tokens["on_accent"], True))
        self.camera_btn.setText(text)
        self.camera_btn.setIcon(get_icon(icon, color))
        self.camera_btn.setEnabled(enabled)
        # Property-based QSS needs an explicit repolish to take effect
        self.camera_btn.setProperty("cam", state)
        self.camera_btn.style().unpolish(self.camera_btn)
        self.camera_btn.style().polish(self.camera_btn)

    def start_camera(self) -> None:
        """Ask the camera worker to open the camera (probing + warm-up run there)."""
        self.last_frame = None
        self._set_camera_state("starting")
        self.scanner.show_camera_message("Starting camera…")
        self.scanner.set_status_text("Starting camera…")
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
        self.scanner.show_camera_message("Camera stopped")

    def _on_camera_state(self, state: str) -> None:
        if state == "running":
            self.last_frame_time = time.monotonic()
            self.watchdog.start()
            self._set_camera_state("running")
            self.scanner.set_status_text("Camera running …")
        elif state == "idle":
            self._show_camera_stopped()
        else:
            self._set_camera_state(state)

    def _on_camera_failed(self, message: str) -> None:
        self._show_camera_stopped()
        QMessageBox.warning(self, "Camera Error", f"{message}\n\nTry a different camera index in Settings.")

    def _watchdog_tick(self) -> None:
        if self.last_frame_time is not None and (time.monotonic() - self.last_frame_time) > 3.0:
            self.stop_camera()
            self._on_camera_failed("No frames received for 3 seconds. Camera stopped.")

    def _on_frame(self, frame: np.ndarray) -> None:
        """A new camera frame: keep it, maybe auto-scan, show it with the overlay."""
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
            self.scanner.show_frame(overlay)
        finally:
            self._camera.worker.frame_consumed()  # ready for the next frame

    # ------------------------------------------------------------------ scanning

    def capture_and_match(self) -> None:
        """Scan the current frame on the scan worker; results arrive in _on_scan_done."""
        if self._scan_busy:
            return  # one scan at a time; auto-scan simply skips this trigger
        if self.last_frame is None:
            QMessageBox.warning(self, "Scan Card", "Camera is not running or no frame available.")
            return
        img = crop_to_card(self.last_frame) if self.settings.crop_to_focus else self.last_frame.copy()
        self.scanner.prepare_scan(is_probably_foil(img, threshold=self.settings.foil_threshold))

        game = self.get_active_game()
        if not game:
            self.scanner.scan_failed("Please select a game type in Settings.")
            return
        if not self.game_db.ensure(game):  # reloads only after a game switch or rebuild
            self.scanner.scan_failed("No feature database found. Build the DB first.")
            return

        self._scan_token += 1
        self._scan_busy = True
        self.scanner.set_scanning(True)
        self.scan_job_requested.emit(self._scan_token, img, self.game_db.index,
                                     self.settings.confidence_threshold, game)

    def _on_scan_done(self, token: int, result: ScanResult) -> None:
        if token != self._scan_token:
            return  # superseded (or the window is closing)
        self._scan_busy = False
        self.scanner.set_scanning(False)
        if result.error is not None:
            logger.error("Scan failed: %s", result.error)
        if result.matches is None:
            self.scanner.scan_failed("Could not extract features from image.")
            return
        matches = self._filter_matches(result.matches, result.game)
        logger.info("Found %s matches after filtering", len(matches))
        self.scanner.show_matches(matches, result.game)

    def _filter_matches(self, matches: list[tuple[str, float]], game: str) -> list[tuple[str, float]]:
        """Keep only matches in the selected sets of the game they were scanned for."""
        wanted = self.settings.sets_for(game)
        if not wanted:
            return list(matches)
        return [(f, s) for f, s in matches if split_filename(os.path.basename(f))[0] in wanted]

    # ------------------------------------------------------------------ pages / keyboard

    def _on_page_changed(self, index: int) -> None:
        if self.stack.widget(index) is self.collection_view:
            self.collection_view.refresh()

    def focusInEvent(self, event) -> None:
        self.setFocus()
        super().focusInEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Scan shortcuts are Scanner-page-only: browsing the collection table
        # must not trigger captures or foil toggles.
        if self.stack.currentWidget() is self.scanner and self.scanner.handle_key(event):
            event.accept()
            return
        super().keyPressEvent(event)
