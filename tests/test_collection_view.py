# tests/test_collection_view.py — the Collection page: table, Clear, Export, game list (headless Qt).

import json
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from lorebook.core.csv_manager import read_collection_rows, update_cardlist
from lorebook.ui import collection_view as cv

ROWS = [["001", "001", "normal", "2"], ["002", "001", "foil", "10"]]


@pytest.fixture
def collection(qt_app, tmp_path):
    """A CollectionView on a Lorcana collection of two rows; `view.cleared_games` records the signal."""
    (tmp_path / "Card_Images" / "Lorcana").mkdir(parents=True)
    (tmp_path / "Documents").mkdir()
    (tmp_path / "card_names_Lorcana.json").write_text(json.dumps({"1-1": "Ariel"}), encoding="utf-8")
    update_cardlist("001-001.webp", False, 2, game="Lorcana")
    update_cardlist("002-001.webp", True, 10, game="Lorcana")
    view = cv.CollectionView(initial_game="Lorcana")
    view.cleared_games = []
    view.cleared.connect(view.cleared_games.append)
    return view


def _cell(view, row, column):
    return view.table.item(row, column).text()


def _games(view):
    return [view.game_combo.itemText(i) for i in range(view.game_combo.count())]


# ---------------------------------------------------------------- the table


def test_table_lists_the_collection_with_names_and_totals(collection):
    assert collection.table.rowCount() == 2
    collection.table.sortItems(0, Qt.SortOrder.AscendingOrder)  # the table opens in Qt's default sort
    assert [_cell(collection, 0, c) for c in range(5)] == ["001", "001", "normal", "2", "Ariel"]
    assert [_cell(collection, 1, c) for c in range(5)] == ["002", "001", "foil", "10", ""]
    assert collection.summary_label.text() == "2 unique · 12 cards"


def test_count_column_sorts_as_numbers(collection):
    collection.table.sortItems(3, Qt.SortOrder.AscendingOrder)
    assert [_cell(collection, r, 3) for r in range(2)] == ["2", "10"]  # as text, "10" would come first


def test_a_non_numeric_count_shows_as_zero(collection, tmp_path):
    (tmp_path / "LorcanaList.csv").write_text(
        "Set Number,Card Number,Variant,Count\n001,001,normal,lots\n", encoding="utf-8"
    )
    collection.refresh()
    assert _cell(collection, 0, 3) == "0" and collection.summary_label.text() == "1 unique · 0 cards"


def test_unreadable_csv_is_reported_not_shown_half_read(collection, tmp_path):
    (tmp_path / "LorcanaList.csv").write_bytes(b"Set Number,Card Number,Variant,Count\r\n\xff\xfe broken\r\n")
    collection.refresh()
    assert collection.table.rowCount() == 0
    assert collection.summary_label.text() == "Couldn't read LorcanaList.csv — check the file."


def test_switching_game_shows_that_games_collection(collection):
    update_cardlist("OGN-001.webp", False, 1, game="Riftbound")
    collection.game_combo.setCurrentText("Riftbound")
    assert collection.table.rowCount() == 1 and _cell(collection, 0, 0) == "OGN"


def test_toolbar_icons_follow_the_theme(collection):
    collection.apply_icons("#E6EDF3", "#F85149")
    for button in (collection.refresh_btn, collection.export_btn, collection.clear_btn):
        assert not button.icon().isNull()


# ---------------------------------------------------------------- the game list


def test_game_folder_created_while_running_appears_on_refresh(collection, tmp_path):
    (tmp_path / "Card_Images" / "Pokemon").mkdir()
    collection.refresh()
    assert _games(collection) == ["Lorcana", "Pokemon", "Riftbound"]
    assert collection.current_game() == "Lorcana" and collection.table.rowCount() == 2  # selection kept


def test_removed_game_folder_drops_out_and_the_selection_moves_on(collection, tmp_path):
    (tmp_path / "Card_Images" / "Pokemon").mkdir()
    collection.refresh()
    collection.game_combo.setCurrentText("Pokemon")
    (tmp_path / "Card_Images" / "Pokemon").rmdir()
    collection.refresh()
    assert _games(collection) == ["Lorcana", "Riftbound"] and collection.current_game() == "Lorcana"


# ---------------------------------------------------------------- Clear


def test_clear_asks_first_and_empties_the_collection(collection, message_boxes):
    message_boxes.answer = QMessageBox.StandardButton.Yes
    collection.clear_btn.click()
    title, text = message_boxes[0]
    assert title == "Clear collection" and "Delete all 2 entries (12 cards) from LorcanaList.csv?" in text
    assert read_collection_rows("Lorcana") == []
    assert collection.cleared_games == ["Lorcana"]  # the Scanner page drops its Undo
    assert collection.table.rowCount() == 0 and collection.summary_label.text() == "Collection cleared."


def test_clear_cancelled_changes_nothing(collection, message_boxes):
    message_boxes.answer = QMessageBox.StandardButton.Cancel
    collection.clear_btn.click()
    assert len(message_boxes) == 1 and read_collection_rows("Lorcana") == ROWS
    assert collection.cleared_games == []


def test_clear_on_an_empty_collection_does_not_ask(collection):
    collection.game_combo.setCurrentText("Riftbound")
    collection.clear_btn.click()  # any dialog would fail the test
    assert collection.summary_label.text() == "Collection is already empty."


def test_clear_with_the_csv_locked_reports_it(collection, message_boxes, monkeypatch):
    def locked(game):
        raise PermissionError(13, "file is in use")

    message_boxes.answer = QMessageBox.StandardButton.Yes
    monkeypatch.setattr(cv, "clear_collection", locked)
    collection.clear_btn.click()
    assert collection.summary_label.text().startswith("Not cleared — LorcanaList.csv could not be updated")
    assert collection.cleared_games == [] and read_collection_rows("Lorcana") == ROWS


def test_clear_refuses_a_csv_it_cannot_read(collection, tmp_path):
    broken = b"Set Number,Card Number,Variant,Count\r\n\xff\xfe broken\r\n"
    (tmp_path / "LorcanaList.csv").write_bytes(broken)
    collection.clear_btn.click()  # no confirmation is asked for
    assert collection.summary_label.text() == "Couldn't read LorcanaList.csv — not clearing it."
    assert (tmp_path / "LorcanaList.csv").read_bytes() == broken


# ---------------------------------------------------------------- Export


def test_export_suggests_documents_and_saves_a_copy(collection, save_dialog, tmp_path):
    collection.export_btn.click()
    exported = tmp_path / "Documents" / "LorcanaList.csv"
    assert save_dialog == [str(exported)]  # not the collection file's own path
    assert exported.read_bytes() == (tmp_path / "LorcanaList.csv").read_bytes()
    assert collection.summary_label.text() == f"Exported to {exported}"


def test_export_onto_the_collection_file_itself_is_refused(collection, save_dialog, tmp_path):
    before = (tmp_path / "LorcanaList.csv").read_bytes()
    save_dialog.choose = str(tmp_path / "LorcanaList.csv")
    collection.export_btn.click()
    assert (
        collection.summary_label.text() == "That is the collection file itself — pick another name or folder."
    )
    assert (tmp_path / "LorcanaList.csv").read_bytes() == before


def test_export_cancelled_does_nothing(collection, save_dialog, tmp_path):
    save_dialog.choose = ""
    collection.export_btn.click()
    assert collection.summary_label.text() == "2 unique · 12 cards"
    assert os.listdir(tmp_path / "Documents") == []


def test_export_with_no_collection_file_says_so(collection, save_dialog):
    collection.game_combo.setCurrentText("Riftbound")
    collection.export_btn.click()
    assert save_dialog == []  # no dialog for nothing
    assert collection.summary_label.text() == "Nothing to export — no collection file yet."


def test_export_failure_is_reported(collection, save_dialog, tmp_path):
    save_dialog.choose = str(tmp_path / "no-such-folder" / "copy.csv")
    collection.export_btn.click()
    assert collection.summary_label.text().startswith("Export failed:")


def test_export_folder_is_an_absolute_native_path(qt_app):
    folder = cv._export_folder.real()  # the real lookup; tests otherwise get tmp_path/Documents
    assert os.path.isabs(folder) and folder == os.path.normpath(folder)  # no mixed slashes
