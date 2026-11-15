# tests/test_settings_window.py — the Settings dialog (headless Qt).

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from lorebook.core.settings import AppSettings
from lorebook.ui.settings_window import SettingsWindow


@pytest.fixture
def game_folders(qt_app, tmp_path):
    """Two game folders with images, so the dialog's game tree has sets to tick."""
    for game, files in (("Lorcana", ("001-001.webp", "002-001.webp")), ("MTG", ("DOM-001.webp",))):
        (tmp_path / "Card_Images" / game).mkdir(parents=True)
        for name in files:
            (tmp_path / "Card_Images" / game / name).touch()


def _roots(dialog):
    tree = dialog.set_tree
    return {tree.topLevelItem(i).text(0): tree.topLevelItem(i) for i in range(tree.topLevelItemCount())}


def test_confidence_percent_round_trips(game_folders):
    # 100 * 0.57 is 56.99999999999999: the field used to show 56 and Apply saved 0.56.
    for percent in (29, 57, 58, 90, 100):
        dialog = SettingsWindow(AppSettings(confidence_threshold=percent / 100))
        assert dialog.confidence_input.text() == str(percent)
        assert dialog.result_settings().confidence_threshold == percent / 100
    assert SettingsWindow(AppSettings(confidence_threshold=0.905)).confidence_input.text() == "90.5"


def test_thresholds_are_clamped_and_junk_falls_back_to_the_default(game_folders):
    dialog = SettingsWindow(AppSettings())
    dialog.confidence_input.setText("250")
    dialog.foil_threshold_input.setText("-5")
    result = dialog.result_settings()
    assert result.confidence_threshold == 1.0 and result.foil_threshold == 0.0

    dialog.confidence_input.setText("ninety")
    assert dialog.result_settings().confidence_threshold == AppSettings().confidence_threshold


def test_the_dialog_edits_a_copy_and_reports_the_result_on_apply(game_folders):
    original = AppSettings()
    dialog = SettingsWindow(original)
    applied = []
    dialog.applied.connect(applied.append)
    dialog.rotate_checkbox.setChecked(True)
    dialog.currency_combo.setCurrentIndex(dialog.currency_combo.findData("EUR"))
    dialog.theme_combo.setCurrentIndex(1)

    dialog.apply_settings()
    assert applied[0].rotate_display and applied[0].currency == "EUR" and applied[0].theme == "light"
    assert not original.rotate_display and original.currency == "USD"  # the caller's object is untouched


def test_only_one_game_can_be_selected(game_folders):
    dialog = SettingsWindow(AppSettings())
    roots = _roots(dialog)
    assert roots["Lorcana"].checkState(0) == Qt.CheckState.Checked

    roots["MTG"].child(0).setCheckState(0, Qt.CheckState.Checked)  # ticking a set selects its game
    result = dialog.result_settings()
    assert result.selected_games == {"lorcana": False, "mtg": True, "riftbound": False}


def test_every_set_ticked_is_saved_as_all_sets_and_a_subset_as_a_list(game_folders):
    dialog = SettingsWindow(AppSettings())
    assert dialog.result_settings().selected_sets["lorcana"] == []  # "all sets", including future ones

    _roots(dialog)["Lorcana"].child(1).setCheckState(0, Qt.CheckState.Unchecked)
    assert dialog.result_settings().selected_sets["lorcana"] == ["001"]


def test_rebuild_applies_the_settings_then_asks_for_a_rebuild(game_folders):
    dialog = SettingsWindow(AppSettings())
    events = []
    dialog.applied.connect(lambda settings: events.append("applied"))
    dialog.rebuild_requested.connect(lambda: events.append("rebuild"))
    dialog._rebuild_database()
    assert events == ["applied", "rebuild"]
