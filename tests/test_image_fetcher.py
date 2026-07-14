# tests/test_image_fetcher.py — card-image downloader (no network).

import numpy as np
import pytest

from lorebook.core import image_fetcher
from lorebook.core.image_fetcher import (
    FetchStats,
    _norm,
    _pad,
    download_new_images,
    lorcana_targets,
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
        assert download_new_images("Riftbound", out_dir=str(tmp_path)) is None

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
