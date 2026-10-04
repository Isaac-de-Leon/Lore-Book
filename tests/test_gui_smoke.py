# tests/test_gui_smoke.py — headless Qt smoke test (offscreen platform).
#
# Runs in CI's GUI job (PySide6 installed, QT_QPA_PLATFORM=offscreen); the
# core test job has no Qt, so it skips. Guards the Settings regressions that
# once stopped scanning altogether.

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


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

    # A new game folder is selectable, keeps its case, and replaces the old one.
    dialog = window._make_settings_dialog()
    roots = {dialog.set_tree.topLevelItem(i).text(0): dialog.set_tree.topLevelItem(i)
             for i in range(dialog.set_tree.topLevelItemCount())}
    roots["MTG"].setCheckState(0, Qt.Checked)
    dialog.apply_settings()
    assert window.get_active_game() == "MTG"
    assert not window.settings.selected_games["lorcana"]

    # A scan runs on the scan worker thread and lands in the result panel,
    # tied to the game it was scanned for.
    import time

    import numpy as np

    import lorebook.core.features as features
    from lorebook.core import card_database

    vec = np.zeros(1280, np.float32)
    vec[0] = 1
    card_database._init_db(card_database._cache_path("MTG"))
    with card_database.sqlite3.connect(card_database._cache_path("MTG")) as conn:
        conn.execute("INSERT INTO features (filename, vector) VALUES (?, ?)", ("DOM-001.webp", vec.tobytes()))

    class FakeExtractor:
        def extract(self, img):
            return vec

        def ensure_ready(self):
            pass

    monkeypatch.setitem(features._extractors, "keras", FakeExtractor())
    window.last_frame = np.zeros((720, 1280, 3), np.uint8)
    window.capture_and_match()
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
