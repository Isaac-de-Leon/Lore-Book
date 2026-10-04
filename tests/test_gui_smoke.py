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
    # No network, no TensorFlow: skip the startup build and model warm-up.
    monkeypatch.setattr(mw.MainWindow, "start_db_build_in_background", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_warm_model_worker", lambda self: None)

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

    window.close()
    assert window._camera_state == "idle"
    app.processEvents()
