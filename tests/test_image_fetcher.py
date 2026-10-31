# tests/test_image_fetcher.py — card-image downloader (no network).

import json
import threading

import numpy as np
import pytest

from lorebook.core import image_fetcher, net
from lorebook.core.image_fetcher import (
    RIFTBOUND_FALLBACK_URL,
    RIFTBOUND_RIOT_URL,
    FetchStats,
    _download,
    _fetch_riftbound,
    _norm,
    _pad,
    download_new_images,
    lorcana_targets,
    riftbound_targets,
)

PAGE1 = f"{RIFTBOUND_FALLBACK_URL}?size=100&page=1"


def _card(set_code, number, url="http://img/x.jpg"):
    return {"setCode": set_code, "number": number, "images": {"full": url}}


def _tiny_jpg() -> bytes:
    """A real, decodable JPG so the webp re-encode path can run."""
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _record(monkeypatch, responses):
    """Stub _fetch_json: log (url, headers) calls, reply from `responses`
    (a callable or a static value)."""
    calls = []

    def fake(url, headers=None):
        calls.append((url, headers or {}))
        return responses(url) if callable(responses) else responses

    monkeypatch.setattr(image_fetcher, "_fetch_json", fake)
    monkeypatch.setattr(image_fetcher, "_fetch_gallery_cards", lambda: [])
    return calls


def test_lorcana_targets_and_code_normalization():
    assert [_pad(c) for c in ("9", "41", "009", "P1")] == ["009", "041", "009", "P1"]
    assert _norm("009") == _norm("9") and _norm("000") == "0"

    data = {"cards": [
        _card("9", 41), _card("1", 1, "http://img/1.jpg"),
        {"setCode": "9", "number": 41},                         # no images
        {"setCode": "", "number": 1, "images": {"full": "u"}},  # no set
        {"setCode": "9", "images": {"full": "u"}},              # no number
    ]}
    assert list(lorcana_targets(data)) == [
        ("9", "41", "http://img/x.jpg"),
        ("1", "1", "http://img/1.jpg"),
    ]
    assert list(lorcana_targets(None)) == list(lorcana_targets({})) == []

    # Promos share their home set's setCode/number (Zeus "18/P3" vs Scrooge
    # "18/204", both setCode 10) — the image must be main-set art.
    promo = dict(_card("10", 18, "http://img/promo.jpg"), fullIdentifier="18/P3 · EN · 10")
    main = dict(_card("10", 18, "http://img/main.jpg"), fullIdentifier="18/204 · EN · 10")
    only_promo = dict(_card("10", 99, "http://img/only.jpg"), fullIdentifier="99/P3 · EN · 10")
    for cards in ([promo, main, only_promo], [main, promo, only_promo]):
        targets = {(s, n): u for s, n, u in lorcana_targets({"cards": cards})}
        assert targets[("10", "18")] == "http://img/main.jpg"
        assert targets[("10", "99")] == "http://img/only.jpg"  # promo-only key kept


def test_riftbound_targets():
    riot = {"sets": [{"id": "OGN", "cards": [
        {"set": "OGN", "collectorNumber": 1,
         "art": {"fullUrl": "http://img/full.jpg", "thumbnailUrl": "http://img/thumb.jpg"}},
        {"set": "OGN", "collectorNumber": 2, "art": {"thumbnailUrl": "http://img/thumb.jpg"}},
    ]}]}
    assert list(riftbound_targets(riot)) == [
        ("OGN", "1", "http://img/full.jpg"),
        ("OGN", "2", "http://img/thumb.jpg"),   # thumbnail only without full art
    ]

    # Riftcodex ships lowercase set ids; codes are uppercased to match OGN-001.
    cards = [
        {"set_id": "ogn", "collector_number": "23c", "media": {"image_url": "http://img/23c.png"}},
        {"set": "OGN", "number": 24, "image_url": "http://img/24.png"},
        {"set": "OGN", "number": 24, "image_url": "http://img/dupe.png"},  # first occurrence wins
        {"set": "OGN", "collectorNumber": 1},                        # no image
        {"collectorNumber": 2, "image_url": "u"},                    # no set
        {"set": "OGN", "image_url": "u"},                            # no number
        {"set": "OGN", "collectorNumber": 3, "image": {"x": "y"}},   # non-string url
        {"set": {"id": "OGN"}, "number": 4, "image_url": "u"},       # nested set object
        "not-a-dict",
    ]
    for data in (cards, {"cards": cards}, {"total": 2, "items": cards}):
        assert list(riftbound_targets(data)) == [
            ("OGN", "23c", "http://img/23c.png"),
            ("OGN", "24", "http://img/24.png"),
        ]
    # A nested set object is read by its code-like keys; riftbound_id is the fallback.
    nested = [{"set": {"set_id": "SFD", "label": "Spiritforged"}, "collector_number": 7,
               "media": {"image_url": "http://img/7.png"}}]
    assert list(riftbound_targets({"total": 1, "items": nested})) == [("SFD", "7", "http://img/7.png")]
    by_id = [{"riftbound_id": "unl-012-221", "collector_number": 12,
              "media": {"image_url": "http://img/12.png"}}]
    assert list(riftbound_targets(by_id)) == [("UNL", "012", "http://img/12.png")]

    for empty in (None, {}, {"sets": []}):
        assert list(riftbound_targets(empty)) == []


def test_registered_games_are_selectable_before_their_folder_exists(tmp_path):
    base = str(tmp_path / "images")
    (tmp_path / "images" / "Lorcana").mkdir(parents=True)
    (tmp_path / "images" / "MTG").mkdir()
    assert image_fetcher.selectable_games(base) == ["Lorcana", "MTG", "Riftbound"]

    (tmp_path / "images" / "riftbound").mkdir()  # an existing folder keeps its own case
    assert image_fetcher.selectable_games(base) == ["Lorcana", "MTG", "riftbound"]
    (tmp_path / "images" / "riftbound").rmdir()

    image_fetcher.ensure_game_folder("Riftbound", base)
    assert (tmp_path / "images" / "Riftbound").is_dir()
    image_fetcher.ensure_game_folder("Riftbound", base)  # already there: no error
    # Only registered games: a stale settings key must not create folders.
    for game in ("../outside", "Pokemon", "", None):
        image_fetcher.ensure_game_folder(game, base)
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == ["Lorcana", "MTG", "Riftbound"]
    assert not (tmp_path / "outside").exists()


def _riftcodex_card(riftbound_id, number, name, set_id, url):
    """The fields we read from a real api.riftcodex.com/cards item."""
    return {"id": riftbound_id, "name": name, "riftbound_id": riftbound_id, "collector_number": number,
            "set": {"set_id": set_id, "label": "x"}, "media": {"image_url": url}}


def test_riftbound_targets_keeps_each_printing():
    # Riftcodex's collector_number is a bare int shared by every printing;
    # the alternate art comes first here, as it does in the real response.
    page = {"total": 9, "page": 1, "size": 100, "pages": 1, "items": [
        _riftcodex_card("unl-116a-219", 116, "Poppy - Paragon (Alternate Art)", "UNL", "http://img/116a.png"),
        _riftcodex_card("unl-116-219", 116, "Poppy - Paragon", "UNL", "http://img/116.png"),
        _riftcodex_card("unl-229*-219", 229, "Vi - Piltover Enforcer (Signature)", "UNL", "http://img/229s.png"),
        _riftcodex_card("unl-229-219", 229, "Vi - Piltover Enforcer (Overnumbered)", "UNL", "http://img/229.png"),
        _riftcodex_card("ven-r06", 6, "Order Rune", "VEN", "http://img/r06.png"),
        # Same code, different home set, different card: both kept.
        _riftcodex_card("opp-086-221", 86, "World Atlas", "OPP", "http://img/atlas.png"),
        _riftcodex_card("opp-086-219", 86, "Zilean - Time Mage", "OPP", "http://img/zilean.png"),
        # Same id, same image (a real duplicate): one file, plain code.
        _riftcodex_card("ven-164-166", 164, "Sandswept Tomb", "VEN", "http://img/164.png"),
        _riftcodex_card("ven-164-166", 164, "Sandswept Tomb", "VEN", "http://img/164.png"),
    ]}
    assert list(riftbound_targets(page)) == [
        ("UNL", "116a", "http://img/116a.png"),
        ("UNL", "116", "http://img/116.png"),
        ("UNL", "229s", "http://img/229s.png"),
        ("UNL", "229", "http://img/229.png"),
        ("VEN", "r06", "http://img/r06.png"),
        ("OPP", "086-221", "http://img/atlas.png"),
        ("OPP", "086-219", "http://img/zilean.png"),
        ("VEN", "164", "http://img/164.png"),
    ]
    # Every code must make a filename split_filename can read back.
    assert image_fetcher._safe_filename("UNL", "229s", ".webp") == "UNL-229s.webp"
    assert image_fetcher._safe_filename("OPP", "086-221", ".webp") == "OPP-086-221.webp"


def _gallery_item(card_id, number, set_id, url):
    """The fields we read from a real card-gallery item."""
    return {"id": card_id, "collectorNumber": number, "publicCode": "x", "name": "x",
            "set": {"label": "Card Set", "value": {"id": set_id, "label": "x"}},
            "cardImage": {"type": "image", "url": url}}


def _gallery_page(items):
    data = {"props": {"pageProps": {"page": {"blades": [
        {"type": "header"}, {"type": "cards", "cards": {"items": items}},
    ]}}}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'


def test_gallery_cards_parse_into_the_usual_filenames():
    page = _gallery_page([
        _gallery_item("rad-175-167", 175, "RAD", "http://img/175.png"),
        _gallery_item("rad-sp1-005", 1, "RAD", "http://img/sp1.png"),
        _gallery_item("rad-r01a", 1, "RAD", "http://img/r01a.png"),
        _gallery_item("rad-153a-167", 153, "RAD", "http://img/153a.png"),
        _gallery_item("rad-200-star-167", 200, "RAD", "http://img/200s.png"),
    ])
    assert list(riftbound_targets(image_fetcher._gallery_cards(page))) == [
        ("RAD", "175", "http://img/175.png"),
        ("RAD", "sp1", "http://img/sp1.png"),
        ("RAD", "r01a", "http://img/r01a.png"),
        ("RAD", "153a", "http://img/153a.png"),
        ("RAD", "200s", "http://img/200s.png"),
    ]
    # A redesigned page is reported, not read as "no cards".
    for bad in ("<html>no data</html>", _gallery_page([{"id": "rad-1-1"}]),
                '<script id="__NEXT_DATA__">{"props": {}}</script>'):
        with pytest.raises(ValueError):
            image_fetcher._gallery_cards(bad)


def test_gallery_only_fills_sets_riftcodex_lacks(monkeypatch):
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    known = _riftcodex_card("ogn-001-298", 1, "x", "OGN", "http://img/riftcodex.png")
    _record(monkeypatch, {"total": 1, "items": [known]})
    gallery = [
        {"riftbound_id": "ogn-001-298", "collector_number": 1, "set": {"set_id": "OGN"},
         "media": {"image_url": "http://img/gallery.png"}},
        {"riftbound_id": "rad-001-167", "collector_number": 1, "set": {"set_id": "RAD"},
         "media": {"image_url": "http://img/rad.png"}},
    ]
    monkeypatch.setattr(image_fetcher, "_fetch_gallery_cards", lambda: gallery)
    assert list(riftbound_targets(_fetch_riftbound(RIFTBOUND_RIOT_URL))) == [
        ("OGN", "001", "http://img/riftcodex.png"),  # existing set: Riftcodex's image, same filename
        ("RAD", "001", "http://img/rad.png"),
    ]

    # The gallery is optional: losing it (or its format) leaves Riftcodex's cards.
    for error in (OSError("offline"), ValueError("page changed")):
        def fail(err=error):
            raise err
        monkeypatch.setattr(image_fetcher, "_fetch_gallery_cards", fail)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == [known]


def test_fetch_json_errors_name_the_url(monkeypatch):
    import urllib.error

    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(net.urllib.request, "urlopen", refuse)
    # Still an OSError: _fetch_riftbound's Riot → Riftcodex fallback catches those.
    with pytest.raises(OSError, match=r"404.*GET http://api\.example/cards"):
        net.fetch_json("http://api.example/cards")


def test_riftbound_source_selection(monkeypatch):
    riot_data = {"sets": [{"cards": []}]}
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
    calls = _record(monkeypatch, riot_data)
    assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == riot_data
    assert calls == [(RIFTBOUND_RIOT_URL, {"X-Riot-Token": "RGAPI-test"})]

    # A Riot URL override keeps the token...
    url = "https://europe.api.riotgames.com/riftbound/content/v1/contents?locale=en"
    calls = _record(monkeypatch, riot_data)
    _fetch_riftbound(url)
    assert calls == [(url, {"X-Riot-Token": "RGAPI-test"})]
    # ...a non-Riot override is paginated and never gets the token.
    calls = _record(monkeypatch, [{"id": 1}])
    _fetch_riftbound("https://mirror.example/cards")
    assert calls == [("https://mirror.example/cards?size=100&page=1", {})]

    # Riot failure falls back to Riftcodex.
    def riot_403(u):
        if u.startswith(RIFTBOUND_FALLBACK_URL):
            return [{"id": 1}]
        raise OSError("403 Forbidden")

    calls = _record(monkeypatch, riot_403)
    assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == [{"id": 1}]
    assert [c[0] for c in calls] == [RIFTBOUND_RIOT_URL, PAGE1]

    # No key: straight to Riftcodex.
    monkeypatch.delenv("RIOT_API_KEY")
    calls = _record(monkeypatch, [{"id": 1}])
    assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == [{"id": 1}]
    assert calls == [(PAGE1, {})]


def test_riftbound_pagination(monkeypatch):
    monkeypatch.delenv("RIOT_API_KEY", raising=False)

    def paged(pages, total=None):
        def responses(url):
            body = {"items": pages.get(int(url.rsplit("page=", 1)[1]), [])}
            if total is not None:
                body["total"] = total
            return body
        return responses

    # Collects until total; total (not page size) decides when to stop.
    page1 = [{"id": i} for i in range(100)]
    calls = _record(monkeypatch, paged({1: page1, 2: [{"id": 100}]}, total=101))
    assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == page1 + [{"id": 100}]
    assert len(calls) == 2
    calls = _record(monkeypatch, paged({1: [{"id": 1}, {"id": 2}], 2: [{"id": 3}]}, total=3))
    assert len(_fetch_riftbound(RIFTBOUND_RIOT_URL)) == 3 and len(calls) == 2

    # Server ignoring page=: the repeated page is detected and dropped.
    calls = _record(monkeypatch, {"total": 500, "items": page1})
    assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == page1
    assert len(calls) == 2

    # No total: an empty page ends it.
    calls = _record(monkeypatch, paged({1: [{"id": 1}]}))
    assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == [{"id": 1}]
    assert len(calls) == 2

    # Bare list and {"cards": ...} wrapper are single pages.
    for body in ([{"id": 1}], {"total": 1, "cards": [{"id": 1}]}):
        calls = _record(monkeypatch, body)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == [{"id": 1}]
        assert len(calls) == 1


def test_download_new_images(tmp_path, monkeypatch):
    jpg = _tiny_jpg()
    data = {"cards": [_card("9", 41), _card("9", 42), _card("10", 1)]}
    monkeypatch.setitem(image_fetcher.GAMES, "lorcana", ("stub://cards", lambda url: data, lorcana_targets))
    monkeypatch.setattr(image_fetcher, "_download", lambda url: jpg)

    def run(sub, **kw):
        out = tmp_path / sub
        out.mkdir()
        kw.setdefault("sets", ["009"])  # set filter is padding-insensitive
        return out, download_new_images("Lorcana", out_dir=str(out), delay=0, **kw)

    assert download_new_images("NotAGame", out_dir=str(tmp_path)) is None

    out, stats = run("webp")
    assert stats == FetchStats(downloaded=2, skipped=0, failed=0)
    assert sorted(p.name for p in out.iterdir()) == ["009-041.webp", "009-042.webp"]
    assert (out / "009-041.webp").read_bytes()[:4] == b"RIFF"

    out, stats = run("jpg", fmt="jpg", limit=1)
    assert stats.downloaded == 1
    assert [p.read_bytes() for p in out.iterdir()] == [jpg]  # source bytes kept

    out = tmp_path / "existing"
    out.mkdir()
    (out / "009-041.webp").write_bytes(b"already here")
    stats = download_new_images("Lorcana", out_dir=str(out), sets=["9"], delay=0)
    assert stats == FetchStats(downloaded=1, skipped=1, failed=0)
    assert (out / "009-041.webp").read_bytes() == b"already here"
    stats = download_new_images("Lorcana", out_dir=str(out), sets=["9"], force=True, delay=0)
    assert stats.downloaded == 2
    assert (out / "009-041.webp").read_bytes() != b"already here"

    # Codes from the downloaded card list become filenames: anything that
    # could escape the output folder is skipped, never written.
    data["cards"] += [_card("../../escaped", 1), _card("9", "/etc/x"), _card("..", 2)]
    out, stats = run("unsafe", sets=None)
    assert stats.failed == 3
    assert not (tmp_path.parent / "escaped-001.webp").exists()  # where ../../ would land
    assert "x" not in {p.stem.split("-")[-1] for p in out.iterdir()}
    del data["cards"][-3:]

    messages = []
    out, stats = run("dry", dry_run=True, progress_callback=messages.append)
    assert stats.downloaded == 2 and list(out.iterdir()) == []
    assert any("would fetch 009-041.webp" in m for m in messages)

    # Cancel set mid-flight takes effect before the next image; pre-set does nothing.
    cancel = threading.Event()

    def download_then_cancel(url):
        cancel.set()
        return jpg

    monkeypatch.setattr(image_fetcher, "_download", download_then_cancel)
    messages = []
    out, stats = run("cancel", cancel_event=cancel, progress_callback=messages.append)
    assert stats.downloaded == 1 and [p.name for p in out.iterdir()] == ["009-041.webp"]
    assert any("cancelled" in m.lower() for m in messages)
    out, stats = run("precancel", cancel_event=cancel)
    assert stats.downloaded == 0 and list(out.iterdir()) == []

    # A 0-byte image is a failed write and is fetched again; .part leftovers
    # from a killed run are cleaned up.
    monkeypatch.setattr(image_fetcher, "_download", lambda url: jpg)
    out = tmp_path / "repair"
    out.mkdir()
    (out / "009-041.webp").write_bytes(b"")
    (out / "009-042.webp.part").write_bytes(b"half")
    stats = download_new_images("Lorcana", out_dir=str(out), sets=["9"], delay=0)
    assert stats.downloaded == 2
    assert sorted(p.name for p in out.iterdir()) == ["009-041.webp", "009-042.webp"]
    assert (out / "009-041.webp").stat().st_size > 0

    # A write that fails midway leaves neither a truncated image nor a .part.
    monkeypatch.setattr(image_fetcher, "_download", lambda url: "not bytes")  # write() raises
    out, stats = run("midwrite", fmt="jpg")
    assert stats.failed == 2 and list(out.iterdir()) == []

    # A per-image failure is counted; a card-list failure raises.
    def boom(url):
        raise OSError("offline")

    monkeypatch.setattr(image_fetcher, "_download", boom)
    out, stats = run("fail")
    assert stats == FetchStats(downloaded=0, skipped=0, failed=2)
    monkeypatch.setitem(image_fetcher.GAMES, "lorcana", ("stub://cards", boom, lorcana_targets))
    with pytest.raises(OSError):
        run("listfail")

    # Riftbound shapes download with padded names too.
    rift = [
        {"set": "OGN", "collectorNumber": 1, "art": {"fullUrl": "http://img/1.jpg"}},
        {"set": "OGN", "collector_number": "23c", "media": {"image_url": "http://img/23c.jpg"}},
    ]
    monkeypatch.setitem(image_fetcher.GAMES, "riftbound", ("stub://cards", lambda url: rift, riftbound_targets))
    monkeypatch.setattr(image_fetcher, "_download", lambda url: jpg)
    out = tmp_path / "rift"
    stats = download_new_images("Riftbound", out_dir=str(out), delay=0)
    assert stats == FetchStats(downloaded=2, skipped=0, failed=0)
    assert sorted(p.name for p in out.iterdir()) == ["OGN-001.webp", "OGN-23c.webp"]

    # A 404 fails at once (no back-off); a 503 is retried.
    import urllib.error

    sleeps, calls = [], []
    monkeypatch.setattr(net.time, "sleep", sleeps.append)
    for code, expected_calls in ((404, 1), (503, 3)):
        calls.clear()

        def fail(req, timeout=None, code=code):
            calls.append(code)
            raise urllib.error.HTTPError("http://img/x.jpg", code, "err", {}, None)

        monkeypatch.setattr(net.urllib.request, "urlopen", fail)
        with pytest.raises(urllib.error.HTTPError):
            _download("http://img/x.jpg")
        assert len(calls) == expected_calls
    assert len(sleeps) == 2  # only the 503 backed off
