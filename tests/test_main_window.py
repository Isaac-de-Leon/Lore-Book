# tests/test_main_window.py — the main window: settings, builds, camera, scanning, shortcuts (headless Qt).
#
# The `window` fixture (tests/conftest.py) gives a real MainWindow with its
# three worker threads, a fake camera and a fake build job.

import shutil
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from lorebook.core import feature_cache
from lorebook.core.build import BuildReport
from lorebook.core.csv_manager import update_cardlist
from lorebook.core.features import registry
from lorebook.ui import workers
from lorebook.ui.progress_dialog import BuildProgressDialog
from lorebook.ui.workers import CameraState, ScanResult

FRAME = np.zeros((72, 128, 3), np.uint8)


def _tree_roots(dialog):
    """The Settings dialog's game items, by game name."""
    tree = dialog.set_tree
    return {tree.topLevelItem(i).text(0): tree.topLevelItem(i) for i in range(tree.topLevelItemCount())}


def _status(window):
    return window.scanner.status_label.text()


def _running(window):
    """Put the window in the camera-running state without starting the camera worker."""
    window._on_camera_state(CameraState.RUNNING)
    return window


# ---------------------------------------------------------------- settings


def test_applying_settings_unchanged_keeps_the_game_and_all_sets(window):
    assert window.get_active_game() == "Lorcana"
    window._make_settings_dialog().apply_settings()
    assert window.settings.selected_games["lorcana"] and window.settings.selected_sets["lorcana"] == []


def test_downloadable_game_gets_its_folder_on_rebuild_not_before(window, tmp_path):
    # A game we can download is selectable with no folder yet; Rebuild creates
    # the folder, so its art can download. Opening Settings alone must not.
    dialog = window._make_settings_dialog()
    roots = _tree_roots(dialog)
    assert roots["Riftbound"].childCount() == 0
    assert not (tmp_path / "Card_Images" / "Riftbound").exists()

    roots["Riftbound"].setCheckState(0, Qt.CheckState.Checked)
    dialog._rebuild_database()
    assert window.get_active_game() == "Riftbound" and window.builds == [1]
    assert (tmp_path / "Card_Images" / "Riftbound").is_dir()


def test_selecting_a_new_game_folder_keeps_its_case_and_deselects_the_rest(window):
    dialog = window._make_settings_dialog()
    _tree_roots(dialog)["MTG"].setCheckState(0, Qt.CheckState.Checked)
    dialog.apply_settings()
    assert window.get_active_game() == "MTG"  # not "Mtg"
    assert not window.settings.selected_games["lorcana"] and not window.settings.selected_games["riftbound"]


def test_theme_switches_at_runtime_and_unknown_themes_fall_back(window):
    window.apply_theme("light")
    assert window.settings.theme == "light" and "#F6F8FA" in window.styleSheet()
    window.apply_theme("neon")
    assert window.settings.theme == "dark"


def test_resizing_rescales_the_scanner_page(window, qt_app):
    def panel_height_for(window_height):  # ScannerView.apply_scale: a fifth of the window, 140 to 200
        return int(min(max(window_height * 0.20, 140), 200))

    window.show()  # Qt only delivers resize events to a window that has been shown
    for height in (1000, 900):  # the window won't go below its minimum, so read back what it took
        window.resize(1000, height)
        qt_app.processEvents()
        assert window.scanner.result_panel.height() == panel_height_for(window.height())


# ---------------------------------------------------------------- database builds


def test_startup_build_covers_every_game_folder(window, wait_until):
    window.use_real_builds()
    window.start_db_build_in_background()
    assert window._build_running and not window._progress_dialog.isHidden()
    assert wait_until(lambda: not window._build_running)
    assert window.build_jobs == [["Lorcana", "MTG"]]
    assert _status(window) == "All databases ready. Scan a card to begin."
    assert window._progress_dialog.isHidden()


def test_no_game_folders_is_reported_instead_of_building(window, tmp_path):
    shutil.rmtree(tmp_path / "Card_Images")
    window.use_real_builds()
    window.start_db_build_in_background()
    assert _status(window) == "No game folders found in Card_Images directory"
    assert not window._build_running and window.build_jobs == []


def test_rebuild_during_a_build_runs_again_afterwards(window, wait_until):
    window.use_real_builds()
    window.settings.selected_games = {"lorcana": False, "riftbound": True}
    window._build_running = True  # as if the startup build were still going
    window._rebuild_database()
    assert window._rebuild_queued and window.build_jobs == []  # nothing starts yet...

    window._on_build_done(BuildReport())  # ...until that build finishes
    assert wait_until(lambda: window.build_jobs and not window._build_running)
    assert window.build_jobs == [["Lorcana", "MTG", "Riftbound"]]  # includes the folder Rebuild created
    assert not window._rebuild_queued


def test_cancelling_drops_a_queued_rebuild(window, qt_app):
    window.use_real_builds()
    window._build_running = True
    window._rebuild_database()
    window._cancel_db_build()
    window._on_build_done(BuildReport(cancelled=True))
    qt_app.processEvents()
    assert window.build_jobs == [] and not window._build_running
    assert _status(window) == "Database build cancelled."


def test_build_progress_and_failures_reach_the_user(window):
    window._on_build_status("ignored: no dialog yet")  # before the first build: nothing to update
    window._on_build_progress(10)
    window._progress_dialog = BuildProgressDialog(window)
    window._on_build_status("Processing 001-001.webp…")
    window._on_build_progress(40)
    assert window._progress_dialog.status_label.text() == "Processing 001-001.webp…"
    assert window._progress_dialog.bar.value() == 40

    window._on_build_done(BuildReport(failed=["Riftbound"]))
    assert _status(window) == "Database build failed for Riftbound — see logs/card_scanner.log."


# ---------------------------------------------------------------- camera


def test_camera_button_starts_and_stops_the_camera(window, wait_until):
    window.camera_btn.click()
    assert window._camera_state == CameraState.STARTING and not window.camera_btn.isEnabled()
    assert wait_until(lambda: window._camera_state == CameraState.RUNNING)
    assert window.camera_btn.text() == "Stop" and _status(window) == "Camera running …"
    assert wait_until(lambda: window.last_frame is not None)  # frames arrive from the worker thread
    assert window.watchdog.isActive()

    window.camera_btn.click()
    assert window._camera_state == CameraState.IDLE and window.camera_btn.text() == "Start"
    assert window.last_frame is None and not window.watchdog.isActive()
    assert window.scanner.preview_label.text() == "Camera stopped"


def test_camera_that_will_not_open_resets_the_button_and_explains(
    window, wait_until, message_boxes, monkeypatch
):
    def no_camera(index):
        raise RuntimeError("Failed to open camera 0 with any backend")

    monkeypatch.setattr(workers, "open_capture", no_camera)
    window.start_camera()
    assert wait_until(lambda: len(message_boxes) == 1)
    title, text = message_boxes[0]
    assert title == "Camera Error" and "Failed to open camera 0" in text
    assert "Try a different camera index in Settings." in text
    assert window._camera_state == CameraState.IDLE and window.camera_btn.isEnabled()


def test_camera_is_stopped_when_frames_stop_arriving(window, message_boxes):
    _running(window)
    window._watchdog_tick()  # a frame arrived a moment ago: fine
    assert window._camera_state == CameraState.RUNNING and message_boxes == []

    window.last_frame_time = time.monotonic() - 4.0
    window._watchdog_tick()
    assert window._camera_state == CameraState.IDLE
    assert message_boxes[0][1].startswith("No frames received for 3 seconds.")


def test_frames_are_ignored_unless_the_camera_is_running(window):
    window._camera.worker._frame_pending.set()
    window._on_frame(FRAME)  # a frame that was already on its way when Stop was pressed
    assert window.last_frame is None
    assert not window._camera.worker._frame_pending.is_set()  # the worker may still send the next one


def test_rotate_setting_turns_the_frame_upside_down(window):
    _running(window)
    frame = FRAME.copy()
    frame[0, 0] = 255
    window._on_frame(frame)
    assert window.last_frame[0, 0].max() == 255

    window.settings.rotate_display = True
    window._on_frame(frame)
    assert window.last_frame[-1, -1].max() == 255 and window.last_frame[0, 0].max() == 0


def test_auto_scan_fires_when_a_card_settles(window, monkeypatch):
    _running(window)
    scans = []
    monkeypatch.setattr(window, "request_scan", lambda: scans.append(1))
    monkeypatch.setattr(window._motion_gate, "update", lambda sample: True)  # "a card has settled"

    window._on_frame(FRAME)
    assert scans == []  # auto-scan is off by default
    window.settings.auto_scan = True
    window._on_frame(FRAME)
    assert scans == [1]


def test_debug_mode_requests_one_heatmap_at_a_time_and_shows_the_latest(window, monkeypatch):
    _running(window)
    window.settings.debug_mode = True
    requested, shown = [], []
    window.heatmap_requested.connect(requested.append)
    monkeypatch.setattr(window.scanner, "show_frame", shown.append)

    window._on_frame(FRAME)
    window._on_frame(FRAME)  # the first heatmap isn't back yet: no second request
    assert len(requested) == 1

    heatmap = np.full((72, 128, 3), 9, np.uint8)
    window._on_heatmap(heatmap)
    window._on_frame(FRAME)
    assert len(requested) == 2 and shown[-1] is heatmap  # the newest heatmap replaces the plain frame

    window.settings.debug_mode = False
    window._on_heatmap(heatmap)  # a late reply after debug mode was switched off is dropped
    window._on_frame(FRAME)
    assert shown[-1] is not heatmap


# ---------------------------------------------------------------- scanning


def _seed_cache(game, filename):
    vec = np.zeros(1280, np.float32)
    vec[0] = 1
    feature_cache._init_db(feature_cache._cache_path(game))
    with feature_cache.sqlite3.connect(feature_cache._cache_path(game)) as conn:
        conn.execute("INSERT INTO features (filename, vector) VALUES (?, ?)", (filename, vec.tobytes()))
    return vec


def test_scan_runs_on_the_worker_thread_and_fills_the_result_panel(window, wait_until, monkeypatch):
    vec = _seed_cache("Lorcana", "001-001.webp")

    class FakeExtractor:
        def extract(self, img):
            return vec

        def ensure_ready(self):
            pass

    monkeypatch.setitem(registry._extractors, "keras", FakeExtractor())
    window.last_frame = np.zeros((720, 1280, 3), np.uint8)
    window.request_scan()
    assert window._scan_busy and not window.scanner.capture_btn.isEnabled()
    assert wait_until(lambda: not window._scan_busy)
    # The result is tied to the game it was scanned for.
    assert window.scanner.last_matches[0][0] == "001-001.webp"
    assert window.scanner.matches_game == "Lorcana" and window.scanner.add_csv_btn.isEnabled()


def test_scan_without_a_feature_database_says_to_build_it(window, wait_until):
    window.last_frame = FRAME
    window.request_scan()
    assert wait_until(lambda: not window._scan_busy)
    assert _status(window) == "No feature database found. Build the DB first."


def test_scan_needs_a_frame_and_a_selected_game(window, message_boxes):
    jobs = []
    window.scan_job_requested.connect(lambda *job: jobs.append(job))

    window.request_scan()
    assert message_boxes == [("Scan Card", "Camera is not running or no frame available.")]

    window.last_frame = FRAME
    window.settings.selected_games = {"lorcana": False}
    window.request_scan()
    assert _status(window) == "Please select a game type in Settings."
    assert jobs == [] and not window._scan_busy


def test_only_one_scan_runs_at_a_time(window):
    jobs = []
    window.scan_job_requested.connect(lambda *job: jobs.append(job))
    window.last_frame = FRAME  # a frame and a game are available, so only the busy check can stop it
    window._scan_busy = True
    window.request_scan()  # e.g. auto-scan firing while the last scan is still being matched
    assert jobs == [] and window._scan_token == 0


def test_scan_sends_the_focus_box_or_the_whole_frame(window, wait_until):
    jobs = []
    window.scan_job_requested.connect(lambda token, image, threshold, game: jobs.append(image.shape))
    window.last_frame = np.zeros((720, 1280, 3), np.uint8)

    window.request_scan()  # crop_to_focus is on by default
    assert wait_until(lambda: not window._scan_busy)
    window.settings.crop_to_focus = False
    window.request_scan()
    assert wait_until(lambda: not window._scan_busy)
    assert jobs == [(432, 309, 3), (720, 1280, 3)]


def test_scan_results_are_filtered_and_stale_ones_dropped(window):
    matches = [("001-001.webp", 0.97), ("002-001.webp", 0.95)]
    window._scan_token = 5

    window._on_scan_done(4, ScanResult("Lorcana", matches))  # an older scan answering late
    assert window.scanner.last_matches == []

    window._on_scan_done(5, ScanResult("Lorcana", matches))
    assert window.scanner.last_matches == matches

    window.settings.selected_sets = {"lorcana": ["002"]}  # only set 002 is selected
    window._on_scan_done(5, ScanResult("Lorcana", matches))
    assert window.scanner.last_matches == [("002-001.webp", 0.95)]

    window.settings.selected_sets = {"lorcana": ["009"]}
    window._on_scan_done(5, ScanResult("Lorcana", matches))
    assert _status(window) == "No match found."


def test_scan_that_could_not_read_the_image_says_so(window):
    window._scan_token = 1
    window._on_scan_done(1, ScanResult("Lorcana", None, error="model exploded"))
    assert _status(window) == "Could not extract features from image."
    assert window.scanner.capture_btn.isEnabled()  # ready for another try


# ---------------------------------------------------------------- pages, keyboard, shutdown


def test_collection_page_reloads_when_it_is_shown(window):
    assert window.collection_view.table.rowCount() == 0
    update_cardlist("001-001.webp", False, 1, game="Lorcana")  # written while the page was hidden
    window.nav_collection_btn.click()
    assert window.collection_view.table.rowCount() == 1


def test_scan_shortcuts_work_on_the_scanner_page_only(window, message_boxes):
    QTest.keyClick(window, Qt.Key.Key_C)  # no frame yet, so the scan request shows the "no frame" box
    assert len(message_boxes) == 1

    window.nav_collection_btn.click()
    QTest.keyClick(window, Qt.Key.Key_C)  # browsing the collection must not trigger a capture
    QTest.keyClick(window, Qt.Key.Key_F)
    assert len(message_boxes) == 1 and not window.scanner.foil_check.isChecked()

    window.nav_scanner_btn.click()
    QTest.keyClick(window, Qt.Key.Key_F)
    assert window.scanner.foil_check.isChecked()
    QTest.keyClick(window, Qt.Key.Key_X)  # not a shortcut: ignored


def test_clearing_the_collection_disables_the_scanners_undo(window, message_boxes):
    from PySide6.QtWidgets import QMessageBox

    window.scanner.show_matches([("001-001.webp", 0.97)], "Lorcana")
    window.scanner.add_to_csv()
    assert window.scanner.undo_btn.isEnabled()
    message_boxes.answer = QMessageBox.StandardButton.Yes
    window.nav_collection_btn.click()
    window.collection_view.clear_btn.click()
    assert not window.scanner.undo_btn.isEnabled()  # Undo would point at a row that no longer exists


def test_closing_the_window_stops_the_camera_and_every_thread(window, tmp_path):
    assert all(w.is_running() for w in window._workers())
    window.settings.camera_index = 2
    window.close()
    assert window._camera_state == CameraState.IDLE
    assert not any(w.is_running() for w in window._workers())
    assert '"camera_index": 2' in (tmp_path / "ui_settings.json").read_text(
        encoding="utf-8"
    )  # settings saved
