# tests/test_gui_smoke.py — headless Qt smoke test (offscreen platform).
#
# Runs in CI's GUI job (PySide6 installed, QT_QPA_PLATFORM=offscreen); the
# core test job has no Qt, so it skips. Guards the Settings regressions that
# once stopped scanning altogether.

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def test_settings_and_main_window(tmp_path, monkeypatch):
    import lorebook.ui.main_window as mw

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    for game, files in (("Lorcana", ("001-001.webp", "002-001.webp")), ("MTG", ("DOM-001.webp",))):
        (tmp_path / "Card_Images" / game).mkdir(parents=True)
        for name in files:
            (tmp_path / "Card_Images" / game / name).touch()
    # No network: skip the startup build. (TensorFlow is stubbed by conftest.)
    monkeypatch.setattr(mw.MainWindow, "start_db_build_in_background", lambda self: None)

    window = mw.MainWindow()
    assert window.get_active_game() == "Lorcana"

    # Apply without changes keeps an "all sets" game selected.
    window._make_settings_dialog().apply_settings()
    assert window.settings.selected_games["lorcana"] and window.settings.selected_sets["lorcana"] == []

    def tree_roots(dialog):
        return {
            dialog.set_tree.topLevelItem(i).text(0): dialog.set_tree.topLevelItem(i)
            for i in range(dialog.set_tree.topLevelItemCount())
        }

    # A game we can download is selectable with no folder yet; Rebuild creates
    # the folder, so its art can download. Opening Settings alone must not.
    builds = []
    monkeypatch.setattr(mw.MainWindow, "start_db_build_in_background", lambda self: builds.append(1))
    dialog = window._make_settings_dialog()
    roots = tree_roots(dialog)
    assert roots["Riftbound"].childCount() == 0
    assert not (tmp_path / "Card_Images" / "Riftbound").exists()
    roots["Riftbound"].setCheckState(0, Qt.Checked)
    dialog._rebuild_database()
    assert window.get_active_game() == "Riftbound" and builds == [1]
    assert (tmp_path / "Card_Images" / "Riftbound").is_dir()

    # A new game folder is selectable, keeps its case, and replaces the old one.
    dialog = window._make_settings_dialog()
    roots = tree_roots(dialog)
    roots["MTG"].setCheckState(0, Qt.Checked)
    dialog.apply_settings()
    assert window.get_active_game() == "MTG"
    assert not window.settings.selected_games["lorcana"] and not window.settings.selected_games["riftbound"]

    # A scan runs on the scan worker thread and lands in the result panel,
    # tied to the game it was scanned for.
    import time

    import numpy as np

    from lorebook.core import feature_cache
    from lorebook.core.features import registry

    vec = np.zeros(1280, np.float32)
    vec[0] = 1
    feature_cache._init_db(feature_cache._cache_path("MTG"))
    with feature_cache.sqlite3.connect(feature_cache._cache_path("MTG")) as conn:
        conn.execute("INSERT INTO features (filename, vector) VALUES (?, ?)", ("DOM-001.webp", vec.tobytes()))

    class FakeExtractor:
        def extract(self, img):
            return vec

        def ensure_ready(self):
            pass

    monkeypatch.setitem(registry._extractors, "keras", FakeExtractor())
    window.last_frame = np.zeros((720, 1280, 3), np.uint8)
    window.request_scan()
    deadline = time.time() + 5
    while window._scan_busy and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.scanner.last_matches[0][0] == "DOM-001.webp"
    assert window.scanner.matches_game == "MTG" and window.scanner.add_csv_btn.isEnabled()

    window.close()
    assert window._camera_state == "idle"
    assert not any(w.is_running() for w in window._workers())  # every QThread stopped
    app.processEvents()


def _wait_until(app, condition, timeout=5.0):
    """Process Qt events until condition() is true (or the timeout passes)."""
    import time

    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_scanner_add_undo_and_status_message(tmp_path, monkeypatch):
    from PySide6.QtTest import QTest

    from lorebook.core.csv_manager import read_collection_rows
    from lorebook.core.settings import AppSettings
    from lorebook.ui.scanner_view import ScannerView

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    scanner = ScannerView(AppSettings())
    scanner.show_matches([("001-001.webp", 0.97), ("002-001.webp", 0.90)], "Lorcana")
    add = scanner.add_csv_btn
    assert add.isEnabled()

    # Adding disables Add (no double count); Undo makes the card addable again.
    scanner.add_to_csv()
    assert not add.isEnabled() and read_collection_rows("Lorcana") == [["001", "001", "normal", "1"]]
    scanner.undo_last_add()
    assert add.isEnabled() and read_collection_rows("Lorcana") == [["001", "001", "normal", "0"]]

    # The alternative match can be added without rescanning; the one already added can't.
    scanner.add_to_csv()
    scanner.next_match()
    assert add.isEnabled()
    scanner.prev_match()
    assert not add.isEnabled()
    scanner.forget_last_add()  # the collection was cleared: nothing is "already added"
    assert add.isEnabled()

    # After a failed scan the matches on screen are an earlier card: no Add, even when browsing.
    scanner.scan_failed("No match found.")
    scanner.next_match()
    assert not add.isEnabled()
    scanner.show_matches([("001-001.webp", 0.97)], "Lorcana")
    assert add.isEnabled()

    # A message is cleared by its own timeout, never by an earlier message's.
    scanner.flash_csv_status("first", timeout_ms=60)
    QTest.qWait(30)
    scanner.flash_csv_status("second", is_error=True, timeout_ms=5000)
    QTest.qWait(120)
    assert scanner.csv_status.text() == "second"
    scanner.flash_csv_status("third", timeout_ms=30)
    assert _wait_until(app, lambda: scanner.csv_status.text() == "")


def test_collection_view_clear_export_and_new_games(tmp_path, monkeypatch):
    from lorebook.core.csv_manager import read_collection_rows, update_cardlist
    from lorebook.ui import collection_view as cv

    QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Card_Images" / "Lorcana").mkdir(parents=True)
    monkeypatch.setattr(cv, "_export_folder", lambda: str(tmp_path / "Documents"))
    (tmp_path / "Documents").mkdir()
    update_cardlist("001-001.webp", False, 2, game="Lorcana")
    view = cv.CollectionView(initial_game="Lorcana")
    cleared = []
    view.cleared.connect(cleared.append)

    # Clear while the CSV is locked (open in Excel): a message, not a silent failure.
    def locked(game):
        raise PermissionError(13, "file is in use")

    monkeypatch.setattr(cv.QMessageBox, "warning", lambda *a, **k: cv.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(cv, "clear_collection", locked)
    view._clear_csv()
    assert "Not cleared" in view.summary_label.text() and "LorcanaList.csv" in view.summary_label.text()
    assert cleared == [] and read_collection_rows("Lorcana") == [["001", "001", "normal", "2"]]
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cv, "_export_folder", lambda: str(tmp_path / "Documents"))

    # Export suggests another folder than the collection file's own...
    suggested = []

    def accept_suggestion(parent, title, path, file_filter):
        suggested.append(path)
        return path, file_filter

    monkeypatch.setattr(cv.QFileDialog, "getSaveFileName", accept_suggestion)
    view._export_csv()
    exported = tmp_path / "Documents" / "LorcanaList.csv"
    assert (
        suggested == [str(exported)] and exported.read_bytes() == (tmp_path / "LorcanaList.csv").read_bytes()
    )
    assert view.summary_label.text() == f"Exported to {exported}"
    # ...and picking the collection file itself is refused with a clear message.
    monkeypatch.setattr(cv.QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "LorcanaList.csv"), ""))
    view._export_csv()
    assert "collection file itself" in view.summary_label.text()

    # A game folder created while the app runs shows up on the next refresh; the selection stays.
    (tmp_path / "Card_Images" / "Pokemon").mkdir()
    view.refresh()
    assert [view.game_combo.itemText(i) for i in range(view.game_combo.count())] == [
        "Lorcana",
        "Pokemon",
        "Riftbound",
    ]
    assert view.current_game() == "Lorcana" and view.table.rowCount() == 1


def test_settings_confidence_percent_round_trips(tmp_path, monkeypatch):
    from lorebook.core.settings import AppSettings
    from lorebook.ui.settings_window import SettingsWindow

    QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    # 100 * 0.57 is 56.99999999999999: the field used to show 56 and Apply saved 0.56.
    for percent in (29, 57, 58, 90, 100):
        dialog = SettingsWindow(AppSettings(confidence_threshold=percent / 100))
        assert dialog.confidence_input.text() == str(percent)
        assert dialog.result_settings().confidence_threshold == percent / 100
    assert SettingsWindow(AppSettings(confidence_threshold=0.905)).confidence_input.text() == "90.5"


def test_rebuild_during_a_build_runs_again_afterwards(tmp_path, monkeypatch):
    import lorebook.ui.main_window as mw
    from lorebook.core.build import BuildReport
    from lorebook.ui import workers

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Card_Images" / "Lorcana").mkdir(parents=True)
    builds = []

    def fake_build(games, **kwargs):  # runs on the build worker thread; no network
        builds.append(list(games))
        return BuildReport()

    monkeypatch.setattr(workers, "refresh_and_build", fake_build)
    start_build = mw.MainWindow.start_db_build_in_background
    monkeypatch.setattr(mw.MainWindow, "start_db_build_in_background", lambda self: None)  # no startup build
    window = mw.MainWindow()
    monkeypatch.setattr(mw.MainWindow, "start_db_build_in_background", start_build)

    # Rebuild pressed while a build is running: nothing starts yet...
    window.settings.selected_games = {"lorcana": False, "riftbound": True}
    window._build_running = True
    window._rebuild_database()
    assert window._rebuild_queued and builds == []
    # ...but when that build finishes, a new one runs and includes the folder Rebuild created.
    window._on_build_done(BuildReport())
    assert _wait_until(app, lambda: builds and not window._build_running)
    assert builds == [["Lorcana", "Riftbound"]] and not window._rebuild_queued

    # Cancelling drops the queued rebuild instead of starting over.
    window._build_running = True
    window._rebuild_database()
    window._cancel_db_build()
    window._on_build_done(BuildReport(cancelled=True))
    app.processEvents()
    assert len(builds) == 1 and not window._build_running

    window.close()
    assert not any(w.is_running() for w in window._workers())
    app.processEvents()


def test_worker_thread_keeps_thread_locals_between_slot_calls():
    # TensorFlow keeps its graph/eager context in thread-locals; a plain
    # moveToThread QThread wiped them between slot calls, so a model loaded by
    # warm_up crashed the next scan's predict (native access violation).
    import threading
    import time

    from PySide6.QtCore import QObject, Signal, Slot

    from lorebook.ui.threads import WorkerThread

    app = QApplication.instance() or QApplication([])
    main_thread = threading.get_ident()
    local = threading.local()
    seen: list = []

    class Worker(QObject):
        @Slot(int)
        def step(self, n: int) -> None:
            seen.append((threading.get_ident(), getattr(local, "value", None)))
            local.value = n

    class Sender(QObject):
        go = Signal(int)

    wt = WorkerThread(Worker(), "test-locals")
    sender = Sender()
    sender.go.connect(wt.worker.step)
    wt.start()
    try:
        for n in range(3):
            sender.go.emit(n)
        deadline = time.monotonic() + 5
        while len(seen) < 3 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
    finally:
        assert wt.stop()
    # On the main thread the locals would survive anyway and prove nothing.
    idents = {ident for ident, _ in seen}
    assert len(idents) == 1 and main_thread not in idents
    assert [value for _, value in seen] == [None, 0, 1]
