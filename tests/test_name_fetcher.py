# tests/test_name_fetcher.py — card-name downloader and its fetch script (no network).

import importlib.util
import json
import os

from lorebook.core import name_fetcher
from lorebook.core.card_names import normalize_key
from lorebook.core.image_fetcher import _safe_filename, riftbound_targets
from lorebook.core.name_fetcher import download_card_names, lorcana_names, riftbound_names


def test_name_mappers():
    data = {"cards": [
        {"setCode": "9", "number": 41, "fullName": "Elsa - Spirit of Winter"},
        {"setCode": "1", "number": 1, "name": "Ariel"},          # no fullName
        {"setCode": "", "number": 2, "fullName": "Skipped"},     # missing set
    ]}
    assert lorcana_names(data) == {"9-41": "Elsa - Spirit of Winter", "1-1": "Ariel"}

    # Promo printings share their home set's setCode/number in LorcanaJSON
    # (Zeus "18/P3" vs Scrooge "18/204"). The main-set name must win
    # regardless of order; promo-only keys are still kept.
    cards = [
        {"setCode": "10", "number": 18, "fullName": "Zeus", "fullIdentifier": "18/P3 · EN · 10"},
        {"setCode": "10", "number": 18, "fullName": "Scrooge", "fullIdentifier": "18/204 · EN · 10"},
        {"setCode": "10", "number": 99, "fullName": "Promo Only", "fullIdentifier": "99/P3 · EN · 10"},
    ]
    for order in (cards, cards[::-1]):
        names = lorcana_names({"cards": order})
        assert names["10-18"] == "Scrooge"
        assert names["10-99"] == "Promo Only"

    jinx = {"set": "OGN", "number": "23c", "name": "Jinx", "image_url": "http://img/j.png"}
    assert riftbound_names([jinx]) == {"OGN-23c": "Jinx"}
    wrapped = {"cards": [{"setCode": "OGN", "collectorNumber": 24, "name": "Vi", "image_url": "http://img/v.png"}]}
    assert riftbound_names(wrapped) == {"OGN-24": "Vi"}

    # Real Radiance data: Ahri's special printing also has collector number 1,
    # and must not take over the name of RAD-001 (Amateur Demolitionist).
    def card(rid, number, name):
        return {"name": name, "riftbound_id": rid, "collector_number": number,
                "set": {"set_id": "RAD"}, "media": {"image_url": f"http://img/{rid}.png"}}

    cards = [card("rad-001-167", 1, "Amateur Demolitionist"), card("rad-sp1-005", 1, "Ahri, Confident"),
             card("rad-r06", 6, "Order Rune"), card("rad-006-167", 6, "Scorched Earth"),
             card("rad-023a-167", 23, "Ziggs, Short-Fused")]
    assert riftbound_names({"items": cards}) == {
        "RAD-001": "Amateur Demolitionist", "RAD-sp1": "Ahri, Confident",
        "RAD-r06": "Order Rune", "RAD-006": "Scorched Earth", "RAD-023a": "Ziggs, Short-Fused",
    }
    # Every downloaded file finds its own name.
    names = {normalize_key(*k.split("-", 1)): v for k, v in riftbound_names(cards).items()}
    for set_code, number, _url in riftbound_targets(cards):
        stem = _safe_filename(set_code, number, "")
        assert names[normalize_key(*stem.split("-", 1))] is not None
    assert names[normalize_key("RAD", "001")] == "Amateur Demolitionist"
    assert lorcana_names({}) == {}
    assert riftbound_names(None) == {}


def _stub_source(monkeypatch, data):
    monkeypatch.setitem(name_fetcher.GAMES, "lorcana", ("stub://cards", lambda url: data, lorcana_names))


def test_download_card_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_source(monkeypatch, {"cards": [{"setCode": "9", "number": 41, "fullName": "Elsa"}]})

    assert download_card_names("Lorcana") == 1
    assert json.loads((tmp_path / "card_names_Lorcana.json").read_text(encoding="utf-8")) == {"9-41": "Elsa"}
    assert sorted(os.listdir(tmp_path)) == ["card_names_Lorcana.json"]  # atomic: no temp file left

    assert download_card_names("NoSuchGame") is None  # no source registered: skipped

    # A source that parses to nothing must not replace a good file.
    _stub_source(monkeypatch, {"cards": []})
    assert download_card_names("Lorcana") == 0
    assert json.loads((tmp_path / "card_names_Lorcana.json").read_text(encoding="utf-8")) == {"9-41": "Elsa"}

    out = tmp_path / "elsewhere.json"
    _stub_source(monkeypatch, {"cards": [{"setCode": "1", "number": 1, "name": "Ariel"}]})
    assert download_card_names("Lorcana", out_path=str(out)) == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {"1-1": "Ariel"}


def _load_cli_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fetch_card_names", os.path.join(root, "scripts", "fetch_card_names.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_script(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    mod = _load_cli_module()
    assert mod.main(["--game", "NoSuchGame"]) == 1
    assert "Unsupported game" in capsys.readouterr().err

    _stub_source(monkeypatch, {"cards": [{"setCode": "9", "number": 41, "fullName": "Elsa"}]})
    assert mod.main(["--game", "lorcana"]) == 0
    assert "Wrote 1 card names to card_names_Lorcana.json" in capsys.readouterr().out
    assert (tmp_path / "card_names_Lorcana.json").exists()

    _stub_source(monkeypatch, {"cards": []})
    assert mod.main(["--game", "Lorcana"]) == 1
    assert "No cards parsed" in capsys.readouterr().err

    def offline(url):
        raise OSError("offline")

    monkeypatch.setitem(name_fetcher.GAMES, "lorcana", ("stub://cards", offline, lorcana_names))
    assert mod.main(["--game", "Lorcana"]) == 1
    assert "Download failed: offline" in capsys.readouterr().err
