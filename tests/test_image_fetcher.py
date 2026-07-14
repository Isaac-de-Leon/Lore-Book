# tests/test_image_fetcher.py — card-image downloader (no network).

import numpy as np
import pytest

from lorebook.core import image_fetcher
from lorebook.core.image_fetcher import (
    RIFTBOUND_FALLBACK_URL,
    RIFTBOUND_RIOT_URL,
    FetchStats,
    _fetch_riftbound,
    _norm,
    _pad,
    download_new_images,
    lorcana_targets,
    riftbound_targets,
)


def _card(set_code, number, url="http://img/x.jpg"):
    return {"setCode": set_code, "number": number, "images": {"full": url}}


def _tiny_jpg() -> bytes:
    """A real, decodable JPG so the webp re-encode path can run."""
    import cv2

    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class TestCodeNormalization:
    def test_pad_zero_pads_numeric_codes(self):
        assert _pad("9") == "009"
        assert _pad("41") == "041"
        assert _pad("009") == "009"

    def test_pad_leaves_promo_codes_alone(self):
        assert _pad("P1") == "P1"
        assert _pad("Q1") == "Q1"

    def test_norm_makes_padding_insensitive(self):
        assert _norm("009") == _norm("9")
        assert _norm("000") == "0"


class TestLorcanaTargets:
    def test_parses_cards(self):
        data = {"cards": [_card("9", 41), _card("1", 1, "http://img/1.jpg")]}
        assert list(lorcana_targets(data)) == [
            ("9", "41", "http://img/x.jpg"),
            ("1", "1", "http://img/1.jpg"),
        ]

    def test_skips_incomplete_cards(self):
        data = {"cards": [
            {"setCode": "9", "number": 41},                      # no images
            {"setCode": "", "number": 1, "images": {"full": "u"}},  # no set
            {"setCode": "9", "images": {"full": "u"}},           # no number
        ]}
        assert list(lorcana_targets(data)) == []

    def test_tolerates_none_and_empty(self):
        assert list(lorcana_targets(None)) == []
        assert list(lorcana_targets({})) == []

    def test_main_set_printing_beats_promo_regardless_of_order(self):
        # Promos share their home set's setCode/number (Zeus "18/P3" vs
        # Scrooge "18/204", both setCode 10) — the image must be main-set art.
        promo = dict(_card("10", 18, "http://img/promo.jpg"),
                     fullIdentifier="18/P3 · EN · 10")
        main = dict(_card("10", 18, "http://img/main.jpg"),
                    fullIdentifier="18/204 · EN · 10")
        only_promo = dict(_card("10", 99, "http://img/only.jpg"),
                          fullIdentifier="99/P3 · EN · 10")
        for cards in ([promo, main, only_promo], [main, promo, only_promo]):
            targets = dict(((s, n), u) for s, n, u in lorcana_targets({"cards": cards}))
            assert targets[("10", "18")] == "http://img/main.jpg"
            assert targets[("10", "99")] == "http://img/only.jpg"  # promo-only key kept


class TestRiftboundTargets:
    def test_parses_riot_content_shape(self):
        data = {"sets": [{"id": "OGN", "cards": [
            {"set": "OGN", "collectorNumber": 1,
             "art": {"fullUrl": "http://img/full.jpg", "thumbnailUrl": "http://img/thumb.jpg"}},
        ]}]}
        assert list(riftbound_targets(data)) == [("OGN", "1", "http://img/full.jpg")]

    def test_riot_thumbnail_used_only_without_full(self):
        data = {"sets": [{"cards": [
            {"set": "OGN", "collectorNumber": 2, "art": {"thumbnailUrl": "http://img/thumb.jpg"}},
        ]}]}
        assert list(riftbound_targets(data)) == [("OGN", "2", "http://img/thumb.jpg")]

    def test_parses_riftcodex_shape(self):
        cards = [
            {"set_id": "OGN", "collector_number": "23c", "media": {"image_url": "http://img/23c.png"}},
            {"set": "OGN", "number": 24, "image_url": "http://img/24.png"},
        ]
        for data in (cards, {"cards": cards}):  # bare list or wrapper
            assert list(riftbound_targets(data)) == [
                ("OGN", "23c", "http://img/23c.png"),
                ("OGN", "24", "http://img/24.png"),
            ]

    def test_skips_incomplete_cards(self):
        data = [
            {"set": "OGN", "collectorNumber": 1},                        # no image
            {"collectorNumber": 2, "image_url": "u"},                    # no set
            {"set": "OGN", "image_url": "u"},                            # no number
            {"set": "OGN", "collectorNumber": 3, "image": {"x": "y"}},   # non-string url
            "not-a-dict",
        ]
        assert list(riftbound_targets(data)) == []

    def test_tolerates_none_and_empty(self):
        assert list(riftbound_targets(None)) == []
        assert list(riftbound_targets({})) == []
        assert list(riftbound_targets({"sets": []})) == []

    def test_dedupes_on_first_occurrence(self):
        data = [
            {"set": "OGN", "number": 1, "image_url": "http://img/first.png"},
            {"set": "OGN", "number": 1, "image_url": "http://img/second.png"},
        ]
        assert list(riftbound_targets(data)) == [("OGN", "1", "http://img/first.png")]


class TestFetchRiftbound:
    """Source selection and pagination, with _fetch_json stubbed (no network)."""

    def _record(self, monkeypatch, responses):
        """Stub _fetch_json: log (url, headers) calls, reply from `responses`
        (a callable or a static value)."""
        calls = []

        def fake(url, headers=None):
            calls.append((url, headers or {}))
            return responses(url) if callable(responses) else responses

        monkeypatch.setattr(image_fetcher, "_fetch_json", fake)
        return calls

    def test_key_set_uses_riot_endpoint_with_token(self, monkeypatch):
        monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
        riot_data = {"sets": [{"cards": []}]}
        calls = self._record(monkeypatch, riot_data)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == riot_data
        assert calls == [(RIFTBOUND_RIOT_URL, {"X-Riot-Token": "RGAPI-test"})]

    def test_no_key_falls_back_to_riftcodex(self, monkeypatch):
        monkeypatch.delenv("RIOT_API_KEY", raising=False)
        cards = [{"id": 1}]
        calls = self._record(monkeypatch, cards)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == cards
        assert calls == [(RIFTBOUND_FALLBACK_URL, {})]

    def test_riot_failure_falls_back_to_riftcodex(self, monkeypatch):
        monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
        cards = [{"id": 1}]

        def responses(url):
            if url.startswith(RIFTBOUND_FALLBACK_URL):
                return cards
            raise OSError("403 Forbidden")

        calls = self._record(monkeypatch, responses)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == cards
        assert [c[0] for c in calls] == [RIFTBOUND_RIOT_URL, RIFTBOUND_FALLBACK_URL]

    def test_url_override_is_fetched_as_is(self, monkeypatch):
        monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
        calls = self._record(monkeypatch, [{"id": 1}])
        _fetch_riftbound("https://mirror.example/cards")
        assert calls == [("https://mirror.example/cards", {})]  # no token leak off-Riot

    def test_riot_url_override_keeps_token(self, monkeypatch):
        monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
        data = {"sets": []}
        calls = self._record(monkeypatch, data)
        url = "https://europe.api.riotgames.com/riftbound/content/v1/contents?locale=en"
        assert _fetch_riftbound(url) == data
        assert calls == [(url, {"X-Riot-Token": "RGAPI-test"})]

    def test_pagination_assembles_full_pages(self, monkeypatch):
        monkeypatch.delenv("RIOT_API_KEY", raising=False)
        page1 = [{"id": i} for i in range(48)]
        page2 = [{"id": 48}]

        def responses(url):
            return page2 if "offset=48" in url else page1

        calls = self._record(monkeypatch, responses)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == page1 + page2
        assert len(calls) == 2

    def test_pagination_stops_when_offset_is_ignored(self, monkeypatch):
        monkeypatch.delenv("RIOT_API_KEY", raising=False)
        page = [{"id": i} for i in range(48)]  # same first id every time
        calls = self._record(monkeypatch, page)
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == page
        assert len(calls) == 2  # second (repeated) page detected and dropped

    def test_pagination_unwraps_cards_dict(self, monkeypatch):
        monkeypatch.delenv("RIOT_API_KEY", raising=False)
        calls = self._record(monkeypatch, {"cards": [{"id": 1}]})
        assert _fetch_riftbound(RIFTBOUND_RIOT_URL) == [{"id": 1}]
        assert len(calls) == 1


class TestDownloadNewImages:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        """Every test runs with the card list and image bytes stubbed."""
        self.jpg = _tiny_jpg()
        monkeypatch.setitem(
            image_fetcher.GAMES,
            "lorcana",
            ("stub://cards", lambda url: self.data, lorcana_targets),
        )
        monkeypatch.setattr(image_fetcher, "_download", lambda url: self.jpg)
        self.data = {"cards": [_card("9", 41), _card("9", 42)]}

    def test_unsupported_game_returns_none(self, tmp_path):
        assert download_new_images("NotAGame", out_dir=str(tmp_path)) is None

    def test_downloads_with_padded_names(self, tmp_path):
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), delay=0)
        assert stats == FetchStats(downloaded=2, skipped=0, failed=0)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["009-041.webp", "009-042.webp"]
        # webp files start with the RIFF magic
        assert (tmp_path / "009-041.webp").read_bytes()[:4] == b"RIFF"

    def test_jpg_format_keeps_source_bytes(self, tmp_path):
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), fmt="jpg", delay=0)
        assert stats.downloaded == 2
        assert (tmp_path / "009-041.jpg").read_bytes() == self.jpg

    def test_skips_existing_files(self, tmp_path):
        (tmp_path / "009-041.webp").write_bytes(b"already here")
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), delay=0)
        assert stats == FetchStats(downloaded=1, skipped=1, failed=0)
        assert (tmp_path / "009-041.webp").read_bytes() == b"already here"

    def test_force_redownloads(self, tmp_path):
        (tmp_path / "009-041.webp").write_bytes(b"stale")
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), force=True, delay=0)
        assert stats.downloaded == 2
        assert (tmp_path / "009-041.webp").read_bytes() != b"stale"

    def test_set_filter_is_padding_insensitive(self, tmp_path):
        self.data["cards"].append(_card("10", 1))
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), sets=["009"], delay=0)
        assert stats.downloaded == 2
        assert not (tmp_path / "010-001.webp").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        messages = []
        stats = download_new_images(
            "Lorcana", out_dir=str(tmp_path), dry_run=True, delay=0,
            progress_callback=messages.append,
        )
        assert stats.downloaded == 2
        assert list(tmp_path.iterdir()) == []
        assert any("would fetch 009-041.webp" in m for m in messages)

    def test_per_image_failure_is_counted_not_raised(self, tmp_path, monkeypatch):
        def boom(url):
            raise OSError("connection reset")
        monkeypatch.setattr(image_fetcher, "_download", boom)
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), delay=0)
        assert stats == FetchStats(downloaded=0, skipped=0, failed=2)
        assert list(tmp_path.iterdir()) == []

    def test_card_list_failure_raises(self, tmp_path, monkeypatch):
        def boom(url):
            raise OSError("offline")
        monkeypatch.setitem(
            image_fetcher.GAMES, "lorcana", ("stub://cards", boom, lorcana_targets)
        )
        with pytest.raises(OSError):
            download_new_images("Lorcana", out_dir=str(tmp_path), delay=0)

    def test_limit_caps_downloads(self, tmp_path):
        stats = download_new_images("Lorcana", out_dir=str(tmp_path), limit=1, delay=0)
        assert stats.downloaded == 1

    def test_riftbound_downloads_with_padded_names(self, tmp_path, monkeypatch):
        cards = [
            {"set": "OGN", "collectorNumber": 1, "art": {"fullUrl": "http://img/1.jpg"}},
            {"set": "OGN", "collector_number": "23c", "media": {"image_url": "http://img/23c.jpg"}},
        ]
        monkeypatch.setitem(
            image_fetcher.GAMES,
            "riftbound",
            ("stub://cards", lambda url: cards, riftbound_targets),
        )
        stats = download_new_images("Riftbound", out_dir=str(tmp_path), delay=0)
        assert stats == FetchStats(downloaded=2, skipped=0, failed=0)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["OGN-001.webp", "OGN-23c.webp"]
