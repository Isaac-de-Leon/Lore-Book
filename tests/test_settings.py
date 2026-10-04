# tests/test_settings.py — AppSettings load/validate/migrate/save (no Qt).

import json

from lorebook.core.settings import AppSettings


def test_settings_round_trip_and_validation(tmp_path):
    path = str(tmp_path / "ui_settings.json")
    assert AppSettings.load(path) == AppSettings()  # missing file → defaults

    s = AppSettings(camera_index=2, theme="light", currency="CAD",
                    selected_games={"lorcana": False, "mtg": True}, selected_sets={"mtg": ["DOM"]})
    s.save(path)
    loaded = AppSettings.load(path)
    assert loaded == s
    assert loaded.active_game_key() == "mtg" and loaded.sets_for("MTG") == ["DOM"]
    copy = loaded.copy()
    copy.selected_sets["mtg"].append("WAR")
    assert loaded.sets_for("mtg") == ["DOM"]  # copies don't share mutable state

    # Bad values fall back per field; ranges are clamped; old files that
    # selected several games keep only the first.
    (tmp_path / "ui_settings.json").write_text(json.dumps({
        "camera_index": "two", "confidence_threshold": 7, "keep_foil_checked": 1,
        "theme": "Neon", "currency": "eur",
        "selected_games": {"Lorcana": True, "riftbound": True},
    }), encoding="utf-8-sig")
    loaded = AppSettings.load(path)
    assert loaded.camera_index == 0 and loaded.confidence_threshold == 1.0
    assert loaded.keep_foil_checked is False and loaded.theme == "dark" and loaded.currency == "EUR"
    assert loaded.selected_games == {"lorcana": True, "riftbound": False}

    for junk in ("[1, 2]", "{not json"):  # not an object / malformed → defaults
        (tmp_path / "ui_settings.json").write_text(junk, encoding="utf-8")
        assert AppSettings.load(path) == AppSettings()
