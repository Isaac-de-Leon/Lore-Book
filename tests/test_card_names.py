# tests/test_card_names.py — card-name loader + fetch-script mappers (no network).

import importlib.util
import json
import os

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

    # --- fetch-script mappers
    _fetch_mappers()


def _load_fetch_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fetch_card_names", os.path.join(root, "scripts", "fetch_card_names.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fetch_mappers():
    mod = _load_fetch_module()
    data = {"cards": [
        {"setCode": "9", "number": 41, "fullName": "Elsa - Spirit of Winter"},
        {"setCode": "1", "number": 1, "name": "Ariel"},          # no fullName
        {"setCode": "", "number": 2, "fullName": "Skipped"},     # missing set
    ]}
    assert mod.lorcana_names(data) == {"9-41": "Elsa - Spirit of Winter", "1-1": "Ariel"}

    # Promo printings share their home set's setCode/number in LorcanaJSON
    # (Zeus "18/P3" vs Scrooge "18/204"). The main-set name must win
    # regardless of order; promo-only keys are still kept.
    cards = [
        {"setCode": "10", "number": 18, "fullName": "Zeus", "fullIdentifier": "18/P3 · EN · 10"},
        {"setCode": "10", "number": 18, "fullName": "Scrooge", "fullIdentifier": "18/204 · EN · 10"},
        {"setCode": "10", "number": 99, "fullName": "Promo Only", "fullIdentifier": "99/P3 · EN · 10"},
    ]
    for order in (cards, cards[::-1]):
        names = mod.lorcana_names({"cards": order})
        assert names["10-18"] == "Scrooge"
        assert names["10-99"] == "Promo Only"

    jinx = {"set": "OGN", "number": "23c", "name": "Jinx", "image_url": "http://img/j.png"}
    assert mod.riftbound_names([jinx]) == {"OGN-23c": "Jinx"}
    wrapped = {"cards": [{"setCode": "OGN", "collectorNumber": 24, "name": "Vi", "image_url": "http://img/v.png"}]}
    assert mod.riftbound_names(wrapped) == {"OGN-24": "Vi"}

    # Real Radiance data: Ahri's special printing also has collector number 1,
    # and must not take over the name of RAD-001 (Amateur Demolitionist).
    def card(rid, number, name):
        return {"name": name, "riftbound_id": rid, "collector_number": number,
                "set": {"set_id": "RAD"}, "media": {"image_url": f"http://img/{rid}.png"}}

    cards = [card("rad-001-167", 1, "Amateur Demolitionist"), card("rad-sp1-005", 1, "Ahri, Confident"),
             card("rad-r06", 6, "Order Rune"), card("rad-006-167", 6, "Scorched Earth"),
             card("rad-023a-167", 23, "Ziggs, Short-Fused")]
    assert mod.riftbound_names({"items": cards}) == {
        "RAD-001": "Amateur Demolitionist", "RAD-sp1": "Ahri, Confident",
        "RAD-r06": "Order Rune", "RAD-006": "Scorched Earth", "RAD-023a": "Ziggs, Short-Fused",
    }
    # Every downloaded file finds its own name.
    from lorebook.core.image_fetcher import _safe_filename, riftbound_targets
    names = {normalize_key(*k.split("-", 1)): v for k, v in mod.riftbound_names(cards).items()}
    for set_code, number, _url in riftbound_targets(cards):
        stem = _safe_filename(set_code, number, "")
        assert names[normalize_key(*stem.split("-", 1))] is not None
    assert names[normalize_key("RAD", "001")] == "Amateur Demolitionist"
    assert mod.lorcana_names({}) == {}
    assert mod.riftbound_names(None) == {}
