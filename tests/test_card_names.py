# tests/test_card_names.py — card-name loader (no network).

import json

from lorebook.core.card_names import load_card_names, name_for, normalize_key


def _write_names(tmp_path, game, mapping):
    (tmp_path / f"card_names_{game}.json").write_text(json.dumps(mapping), encoding="utf-8")


def test_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_names(tmp_path, "Lorcana", {"9-41": "Elsa", "010-018": "Scrooge"})
    _write_names(tmp_path, "Riftbound", {"OGN-23c": "Jinx"})

    assert name_for("009", "041", "Lorcana") == "Elsa"     # padded lookup, unpadded source
    assert name_for("10", "18", "Lorcana") == "Scrooge"    # unpadded lookup, padded source
    assert name_for("ogn", "23C", "Riftbound") == "Jinx"   # case-insensitive
    assert name_for("009", "999", "Lorcana") is None
    assert name_for("009", "041", "NoSuchGame") is None
    assert name_for("", "041", "Lorcana") is None

    assert normalize_key("000", "000") == "0-0"
    assert normalize_key("009", "041") == normalize_key("9", "41")

    (tmp_path / "card_names_Broken.json").write_text("{not json", encoding="utf-8")
    assert load_card_names("Broken") == {}
