# tests/test_scanner_view.py — the Scanner page: results, foil lock, Add/Undo, shortcuts (headless Qt).

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest

from lorebook.core.card_prices import clear_price_cache
from lorebook.core.csv_manager import read_collection_rows
from lorebook.core.settings import AppSettings
from lorebook.ui import scanner_view
from lorebook.ui.scanner_view import ScannerView
from lorebook.ui.styles import theme_tokens

REGULAR = ("001-001.webp", 0.97)
ALTERNATIVE = ("002-001.webp", 0.90)
ENCHANTED = ("013-218.webp", 0.95)  # Lorcana card above 204: exists only as foil


@pytest.fixture
def scanner(qt_app, tmp_path, monkeypatch):
    """A ScannerView in an empty working folder; `scanner.changes` counts collection_changed."""
    monkeypatch.chdir(tmp_path)
    view = ScannerView(AppSettings())
    view.apply_theme(theme_tokens("dark"))
    view.changes = []
    view.collection_changed.connect(lambda: view.changes.append(1))
    return view


def _key(key, modifiers=Qt.KeyboardModifier.NoModifier):
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers)


# ---------------------------------------------------------------- showing results


def test_no_matches_says_so_and_keeps_add_off(scanner):
    scanner.show_matches([], "Lorcana")
    assert scanner.status_label.text() == "No match found."
    assert not scanner.add_csv_btn.isEnabled()


def test_match_panel_shows_code_score_and_position(scanner):
    scanner.show_matches([REGULAR, ALTERNATIVE], "Lorcana")
    assert scanner.match_name_label.text() == "001-001"  # no names file: the bare code
    assert scanner.match_detail_label.text() == "Set 001  ·  97.0% match"
    assert scanner.match_pos_label.text() == "1 / 2"
    assert scanner.status_label.text() == "0.970  001-001.webp"
    assert scanner.image_label.text() == "—"  # the reference image isn't on disk

    scanner.next_match()
    scanner.next_match()  # stepping past the end stays on the last match
    assert scanner.match_pos_label.text() == "2 / 2" and scanner.match_name_label.text() == "002-001"
    scanner.prev_match()
    scanner.prev_match()
    assert scanner.match_pos_label.text() == "1 / 2"


def test_match_panel_shows_name_price_and_thumbnail_when_available(scanner, tmp_path):
    (tmp_path / "card_names_Lorcana.json").write_text(json.dumps({"1-1": "Ariel"}), encoding="utf-8")
    (tmp_path / "card_prices_Lorcana.json").write_text(
        json.dumps(
            {"fetched_at": "2026-01-01T00:00:00+00:00", "prices": {"1-1": {"usd": 1.24, "usd_foil": 3.8}}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "Card_Images" / "Lorcana").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "Card_Images" / "Lorcana" / "001-001.png"), np.zeros((88, 63, 3), np.uint8))
    clear_price_cache()

    scanner.show_matches([("001-001.png", 0.97)], "Lorcana")
    assert scanner.match_name_label.text() == "Ariel  ·  001-001"
    assert scanner.match_price_label.text() == "$1.24 · foil $3.80"
    assert not scanner.image_label.pixmap().isNull()


def test_refresh_current_match_picks_up_new_prices(scanner, tmp_path):
    scanner.refresh_current_match()  # nothing shown yet: nothing to do
    scanner.show_matches([REGULAR], "Lorcana")
    assert scanner.match_price_label.text() == ""

    (tmp_path / "card_prices_Lorcana.json").write_text(
        json.dumps({"fetched_at": "2026-01-01T00:00:00+00:00", "prices": {"1-1": {"usd": 0.5}}}),
        encoding="utf-8",
    )
    clear_price_cache()  # what MainWindow does when a build refreshed the price file
    scanner.refresh_current_match()
    assert scanner.match_price_label.text() == "$0.50"


def test_close_scores_warn_before_adding(scanner):
    scanner.show_matches([("001-001.webp", 0.951), ("001-002.webp", 0.94)], "Lorcana")
    assert scanner.status_label.text().startswith("⚠ Close match (95.1% vs 94.0%)")
    scanner.show_matches([REGULAR, ALTERNATIVE], "Lorcana")  # seven points apart: no warning
    assert "Close match" not in scanner.status_label.text()


def test_only_the_best_twenty_matches_are_kept(scanner):
    scanner.show_matches([(f"001-{i:03d}.webp", 0.99 - i / 1000) for i in range(30)], "Lorcana")
    assert len(scanner.last_matches) == 20 and scanner.match_pos_label.text() == "1 / 20"


def test_preview_shows_frames_and_messages(scanner):
    scanner.preview_label.resize(640, 360)
    scanner.show_frame(np.zeros((72, 128, 3), np.uint8))
    assert not scanner.preview_label.pixmap().isNull()
    scanner.show_camera_message("Camera stopped")
    assert scanner.preview_label.text() == "Camera stopped"


def test_scan_button_shows_progress(scanner):
    scanner.set_scanning(True)
    assert not scanner.capture_btn.isEnabled() and scanner.capture_btn.text() == "SCANNING…"
    assert scanner.status_label.text() == "Scanning…"
    scanner.set_scanning(False)
    assert scanner.capture_btn.isEnabled() and scanner.capture_btn.text() == "SCAN CARD"


def test_fixed_size_controls_scale_with_the_window_within_limits(scanner):
    scanner.apply_scale(1000)
    assert scanner.result_panel.height() == 200
    scanner.apply_scale(100)  # tiny window: clamped, never collapsed
    assert scanner.result_panel.height() == 140 and scanner.capture_btn.minimumHeight() == 48
    scanner.apply_scale(5000)
    assert scanner.result_panel.height() == 200 and scanner.capture_btn.minimumHeight() == 76


# ---------------------------------------------------------------- foil-only cards


def test_foil_only_card_forces_foil_and_restores_the_choice_afterwards(scanner):
    scanner.show_matches([ENCHANTED, ("013-042.webp", 0.80)], "Lorcana")
    assert scanner.foil_check.isChecked() and not scanner.foil_check.isEnabled()

    scanner.next_match()  # a regular card: the box is the user's again, unticked as before
    assert not scanner.foil_check.isChecked() and scanner.foil_check.isEnabled()
    scanner.foil_check.setChecked(True)
    scanner.prev_match()  # locked again...
    scanner.next_match()  # ...and this time restored to ticked
    assert scanner.foil_check.isChecked() and scanner.foil_check.isEnabled()


def test_foil_only_card_is_recorded_as_foil(scanner):
    scanner.show_matches([ENCHANTED], "Lorcana")
    scanner.add_to_csv()
    assert read_collection_rows("Lorcana") == [["013", "218", "foil", "1"]]


def test_foil_lock_is_lorcana_only_and_released_by_a_new_scan_or_clear(scanner):
    scanner.show_matches([ENCHANTED], "Riftbound")  # the 204 rule is Lorcana's
    assert scanner.foil_check.isEnabled() and not scanner.foil_check.isChecked()

    scanner.show_matches([ENCHANTED], "Lorcana")
    scanner.prepare_scan(probably_foil=False)  # the next scan starts unlocked
    assert scanner.foil_check.isEnabled() and not scanner.foil_check.isChecked()
    scanner.prepare_scan(probably_foil=True)  # the foil detector pre-ticks the box
    assert scanner.foil_check.isChecked()

    scanner.foil_check.setChecked(False)
    scanner.show_matches([ENCHANTED], "Lorcana")
    scanner.clear_results()
    assert scanner.foil_check.isEnabled() and not scanner.foil_check.isChecked()
    assert scanner.match_name_label.text() == "—" and scanner.last_matches == []


# ---------------------------------------------------------------- settings


def test_new_settings_clear_results_from_another_game(scanner):
    scanner.show_matches([REGULAR], "Lorcana")
    scanner.apply_settings(AppSettings(selected_games={"lorcana": False, "riftbound": True}))
    assert scanner.last_matches == [] and scanner.status_label.text() == "Game changed — scan a card."
    assert not scanner.add_csv_btn.isEnabled()

    scanner.show_matches([REGULAR], "Riftbound")
    scanner.apply_settings(AppSettings(selected_games={"lorcana": False, "riftbound": True}))
    assert scanner.last_matches == [REGULAR]  # same game: results stay


def test_keep_foil_checked_setting_applies_now_or_after_a_foil_lock(scanner):
    scanner.apply_settings(AppSettings(keep_foil_checked=True))
    assert scanner.foil_check.isChecked()

    scanner.apply_settings(AppSettings(keep_foil_checked=False))
    scanner.show_matches([ENCHANTED, REGULAR], "Lorcana")  # locked on
    scanner.apply_settings(AppSettings(keep_foil_checked=True))
    assert not scanner.foil_check.isEnabled()  # still locked while the Enchanted is shown
    scanner.next_match()
    assert scanner.foil_check.isChecked() and scanner.foil_check.isEnabled()  # the new setting wins


# ---------------------------------------------------------------- Add and Undo


def test_add_then_undo(scanner):
    scanner.show_matches([REGULAR, ALTERNATIVE], "Lorcana")
    scanner.count_edit.setText("3")
    scanner.add_to_csv()
    assert read_collection_rows("Lorcana") == [["001", "001", "normal", "3"]]
    assert scanner.csv_status.text() == "Added 3× 001-001.webp to LorcanaList.csv"
    assert scanner.undo_btn.isEnabled() and scanner.changes == [1]

    scanner.undo_last_add()
    assert read_collection_rows("Lorcana") == [["001", "001", "normal", "0"]]
    assert scanner.csv_status.text() == "Removed 3× 001-001.webp"
    assert not scanner.undo_btn.isEnabled() and scanner.changes == [1, 1]
    scanner.undo_last_add()  # nothing left to undo
    assert scanner.changes == [1, 1]


def test_add_is_off_for_a_card_already_added_until_undone(scanner):
    scanner.show_matches([REGULAR, ALTERNATIVE], "Lorcana")
    add = scanner.add_csv_btn
    scanner.add_to_csv()
    assert not add.isEnabled()  # a second click can't double-count the card
    scanner.next_match()
    assert add.isEnabled()  # the alternative can be added without rescanning
    scanner.prev_match()
    assert not add.isEnabled()
    scanner.undo_last_add()
    assert add.isEnabled()

    scanner.add_to_csv()
    scanner.forget_last_add()  # the collection was cleared elsewhere
    assert add.isEnabled() and not scanner.undo_btn.isEnabled()


def test_add_stays_off_for_old_matches_after_a_failed_scan(scanner):
    scanner.show_matches([REGULAR, ALTERNATIVE], "Lorcana")
    scanner.scan_failed("No match found.")  # the matches on screen are an earlier card
    scanner.next_match()
    assert not scanner.add_csv_btn.isEnabled()
    scanner.show_matches([REGULAR], "Lorcana")
    assert scanner.add_csv_btn.isEnabled()


@pytest.mark.parametrize("typed", ["", "0", "-2", "1o"])
def test_invalid_count_adds_nothing(scanner, tmp_path, typed):
    scanner.show_matches([REGULAR], "Lorcana")
    scanner.count_edit.setText(typed)
    scanner.add_to_csv()
    assert scanner.csv_status.text() == "Invalid count — enter a number from 1 to 999."
    assert not (tmp_path / "LorcanaList.csv").exists()  # never a silent "1"
    assert scanner.add_csv_btn.isEnabled() and not scanner.undo_btn.isEnabled() and scanner.changes == []


def test_add_without_a_match_is_refused(scanner):
    scanner.add_to_csv()
    assert scanner.csv_status.text() == "No match to add."


def test_add_with_the_csv_locked_reports_it_and_can_be_retried(scanner, monkeypatch):
    def locked(*args, **kwargs):
        raise PermissionError(13, "file is in use")

    scanner.show_matches([REGULAR], "Lorcana")
    real_update = scanner_view.update_cardlist
    monkeypatch.setattr(scanner_view, "update_cardlist", locked)
    scanner.add_to_csv()
    assert scanner.csv_status.text().startswith("Not added — LorcanaList.csv could not be updated")
    assert scanner.add_csv_btn.isEnabled() and not scanner.undo_btn.isEnabled() and scanner.changes == []

    monkeypatch.setattr(scanner_view, "update_cardlist", real_update)  # the other program closes the file
    scanner.add_to_csv()
    assert read_collection_rows("Lorcana") == [["001", "001", "normal", "1"]]


def test_failed_undo_can_be_retried(scanner, monkeypatch):
    def locked(*args, **kwargs):
        raise PermissionError(13, "file is in use")

    scanner.show_matches([REGULAR], "Lorcana")
    scanner.add_to_csv()
    real_update = scanner_view.update_cardlist
    monkeypatch.setattr(scanner_view, "update_cardlist", locked)
    scanner.undo_last_add()
    assert scanner.csv_status.text().startswith("Undo failed")
    assert scanner.undo_btn.isEnabled() and not scanner.add_csv_btn.isEnabled()

    monkeypatch.setattr(scanner_view, "update_cardlist", real_update)
    scanner.undo_last_add()
    assert read_collection_rows("Lorcana") == [["001", "001", "normal", "0"]]


def test_add_uses_the_game_the_card_was_scanned_for(scanner, tmp_path):
    scanner.show_matches([REGULAR], "Riftbound")
    scanner.settings = AppSettings()  # Settings now say Lorcana, but the match is Riftbound's
    scanner.add_to_csv()
    assert (tmp_path / "RiftboundList.csv").exists() and not (tmp_path / "LorcanaList.csv").exists()


def test_status_message_is_cleared_by_its_own_timeout_only(scanner, wait_until):
    scanner.flash_csv_status("first", timeout_ms=60)
    QTest.qWait(30)
    scanner.flash_csv_status("second", is_error=True, timeout_ms=5000)
    QTest.qWait(120)  # the first message's 60 ms has long passed
    assert scanner.csv_status.text() == "second"
    scanner.flash_csv_status("third", timeout_ms=30)
    assert wait_until(lambda: scanner.csv_status.text() == "")
    scanner.flash_csv_status("stays", timeout_ms=0)  # 0 means until replaced
    QTest.qWait(60)
    assert scanner.csv_status.text() == "stays"


# ---------------------------------------------------------------- keyboard shortcuts


def test_c_requests_a_scan(scanner):
    requests = []
    scanner.scan_requested.connect(lambda: requests.append(1))
    assert scanner.handle_key(_key(Qt.Key.Key_C)) and requests == [1]


def test_a_and_d_step_through_matches(scanner):
    scanner.show_matches([REGULAR, ALTERNATIVE], "Lorcana")
    assert scanner.handle_key(_key(Qt.Key.Key_D)) and scanner.current_match_idx == 1
    assert scanner.handle_key(_key(Qt.Key.Key_A)) and scanner.current_match_idx == 0


def test_f_toggles_foil_unless_the_card_is_foil_only(scanner):
    assert scanner.handle_key(_key(Qt.Key.Key_F)) and scanner.foil_check.isChecked()
    scanner.handle_key(_key(Qt.Key.Key_F))
    assert not scanner.foil_check.isChecked()

    scanner.show_matches([ENCHANTED], "Lorcana")
    assert scanner.handle_key(_key(Qt.Key.Key_F))
    assert scanner.foil_check.isChecked()  # still forced on


@pytest.mark.parametrize("modifier", [Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier])
def test_ctrl_s_or_alt_s_adds_once_and_ctrl_z_undoes(scanner, modifier):
    scanner.show_matches([REGULAR], "Lorcana")
    assert scanner.handle_key(_key(Qt.Key.Key_S, modifier))
    assert scanner.handle_key(_key(Qt.Key.Key_S, modifier))  # Add is off now: no second row
    assert read_collection_rows("Lorcana") == [["001", "001", "normal", "1"]]

    assert scanner.handle_key(_key(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier))
    assert read_collection_rows("Lorcana") == [["001", "001", "normal", "0"]]


def test_other_keys_are_left_for_the_window(scanner):
    scanner.show_matches([REGULAR], "Lorcana")
    for event in (_key(Qt.Key.Key_S), _key(Qt.Key.Key_Z), _key(Qt.Key.Key_X), _key(Qt.Key.Key_Return)):
        assert scanner.handle_key(event) is False
    assert scanner.changes == []  # plain S and Z did nothing
