# tests/test_test_safety.py — the suite's own guard rails (tests/conftest.py).
#
# These protect the user's files from the tests: the collection CSVs sit in the
# checkout root, and Export suggests the real Documents folder.

import os

import pytest

from lorebook.core.csv_manager import update_cardlist
from lorebook.core.game_types import csv_for_game

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_test_starts_in_its_own_empty_folder(tmp_path):
    assert os.path.samefile(os.getcwd(), tmp_path) and os.listdir(tmp_path) == []
    assert "LOREBOOK_DATA_DIR" not in os.environ


def test_a_test_that_never_changes_directory_cannot_touch_the_real_collection(tmp_path):
    real_csv = os.path.join(REPO_ROOT, "LorcanaList.csv")
    before = open(real_csv, "rb").read() if os.path.exists(real_csv) else None

    update_cardlist("001-001.webp", False, game="Lorcana")  # no chdir, no tmp_path argument

    assert os.path.samefile(os.path.dirname(os.path.abspath(csv_for_game("Lorcana"))), tmp_path)
    assert (open(real_csv, "rb").read() if os.path.exists(real_csv) else None) == before


def test_undoing_monkeypatch_does_not_return_to_the_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "..")
    monkeypatch.undo()
    assert os.path.samefile(os.getcwd(), tmp_path)


def test_real_dialogs_fail_the_test_and_exports_stay_in_the_temp_folder(qt_app, tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from lorebook.ui import collection_view

    with pytest.raises(AssertionError, match="real dialog"):
        QMessageBox.warning(None, "title", "text")
    with pytest.raises(AssertionError, match="real dialog"):
        QFileDialog.getSaveFileName(None, "title", "path", "filter")
    assert collection_view._export_folder() == str(tmp_path / "Documents")
