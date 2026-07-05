# tests/test_card_names.py — card-name loader + fetch-script mappers (no network).

import importlib.util
import json
import os

from lorebook.core.card_names import load_card_names, name_for, normalize_key


def _write_names(tmp_path, game, mapping):
    (tmp_path / f"card_names_{game}.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )


class TestNameLookup:
    def test_padded_filename_codes_match_unpadded_source(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_names(tmp_path, "Lorcana", {"9-41": "Elsa - Spirit of Winter"})
        assert name_for("009", "041", "Lorcana") == "Elsa - Spirit of Winter"

    def test_padded_source_matches_unpadded_lookup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_names(tmp_path, "Lorcana", {"009-041": "Elsa - Spirit of Winter"})
        assert name_for("9", "41", "Lorcana") == "Elsa - Spirit of Winter"

    def test_alphanumeric_codes_case_insensitive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_names(tmp_path, "Riftbound", {"OGN-23c": "Jinx"})
        assert name_for("ogn", "23C", "Riftbound") == "Jinx"

    def test_unknown_card_and_missing_file_return_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_names(tmp_path, "Lorcana", {"9-41": "Elsa"})
        assert name_for("009", "999", "Lorcana") is None
        assert name_for("009", "041", "NoSuchGame") is None
        assert name_for("", "041", "Lorcana") is None

    def test_malformed_file_degrades_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "card_names_Lorcana.json").write_text("{not json", encoding="utf-8")
        assert load_card_names("Lorcana") == {}

    def test_normalize_key_zero_handling(self):
        assert normalize_key("000", "000") == "0-0"
        assert normalize_key("009", "041") == normalize_key("9", "41")


def _load_fetch_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fetch_card_names", os.path.join(root, "scripts", "fetch_card_names.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFetchMappers:
    def test_lorcana_mapper(self):
        mod = _load_fetch_module()
        data = {"cards": [
            {"setCode": "9", "number": 41, "fullName": "Elsa - Spirit of Winter"},
            {"setCode": "1", "number": 1, "name": "Ariel"},          # no fullName
            {"setCode": "", "number": 2, "fullName": "Skipped"},     # missing set
        ]}
        assert mod.lorcana_names(data) == {
            "9-41": "Elsa - Spirit of Winter",
            "1-1": "Ariel",
        }

    def test_riftbound_mapper_accepts_both_shapes_and_spellings(self):
        mod = _load_fetch_module()
        as_list = [{"set": "OGN", "number": "23c", "name": "Jinx"}]
        wrapped = {"cards": [{"setCode": "OGN", "collectorNumber": 24, "name": "Vi"}]}
        assert mod.riftbound_names(as_list) == {"OGN-23c": "Jinx"}
        assert mod.riftbound_names(wrapped) == {"OGN-24": "Vi"}

    def test_empty_sources_yield_empty(self):
        mod = _load_fetch_module()
        assert mod.lorcana_names({}) == {}
        assert mod.riftbound_names(None) == {}
