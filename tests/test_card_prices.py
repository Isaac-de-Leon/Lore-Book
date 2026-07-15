# tests/test_card_prices.py — price loader/formatter + price fetcher (no network).

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from lorebook.core import card_prices, price_fetcher
from lorebook.core.card_prices import (
    RATES_FILE,
    clear_price_cache,
    format_price,
    load_card_prices,
    price_for,
    prices_stale,
    rate_for,
)
from lorebook.core.price_fetcher import (
    download_card_prices,
    download_currency_rates,
    lorcana_prices,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    # The module cache is keyed by absolute path and never self-resets;
    # clear it so tests reusing filenames across tmp_paths can't collide.
    clear_price_cache()
    yield
    clear_price_cache()


def _iso_now(hours_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(
        timespec="seconds"
    )


def _write_prices(tmp_path, game, prices, fetched_at=None):
    (tmp_path / f"card_prices_{game}.json").write_text(
        json.dumps({"fetched_at": fetched_at or _iso_now(), "prices": prices}),
        encoding="utf-8",
    )


class TestPriceLookup:
    def test_padded_filename_codes_match_unpadded_source(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {"9-41": {"usd": 1.24, "usd_foil": 3.8}})
        assert price_for("009", "041", "Lorcana") == {"usd": 1.24, "usd_foil": 3.8}

    def test_padded_source_matches_unpadded_lookup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {"009-041": {"usd": 1.24}})
        assert price_for("9", "41", "Lorcana")["usd"] == 1.24

    def test_unknown_card_and_missing_file_return_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {"9-41": {"usd": 1.0}})
        assert price_for("009", "999", "Lorcana") is None
        assert price_for("009", "041", "NoSuchGame") is None
        assert price_for("", "041", "Lorcana") is None

    def test_malformed_file_degrades_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "card_prices_Lorcana.json").write_text("{not json", encoding="utf-8")
        assert load_card_prices("Lorcana") == {}

    def test_null_halves_survive_loading(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {"9-41": {"usd": None, "usd_foil": 3.8}})
        assert price_for("9", "41", "Lorcana") == {"usd": None, "usd_foil": 3.8}


class TestRateLookup:
    def test_usd_needs_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert rate_for("USD") == 1.0

    def test_rate_read_from_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / RATES_FILE).write_text(
            json.dumps({"fetched_at": _iso_now(), "base": "USD",
                        "rates": {"CAD": 1.37, "EUR": 0.92}}),
            encoding="utf-8",
        )
        assert rate_for("CAD") == 1.37
        assert rate_for("cad") == 1.37
        assert rate_for("GBP") is None

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert rate_for("CAD") is None


class TestFormatPrice:
    def test_both_prices(self):
        assert format_price({"usd": 1.24, "usd_foil": 3.8}) == "$1.24 · foil $3.80"

    def test_missing_halves_omitted(self):
        assert format_price({"usd": 0.05, "usd_foil": None}) == "$0.05"
        assert format_price({"usd": None, "usd_foil": 12.5}) == "foil $12.50"

    def test_no_data_renders_empty(self):
        assert format_price(None) == ""
        assert format_price({}) == ""
        assert format_price({"usd": None, "usd_foil": None}) == ""

    def test_currency_conversion(self):
        assert format_price({"usd": 1.24}, "CAD", 1.35) == "CA$1.67"
        assert format_price({"usd_foil": 10.0}, "EUR", 0.9) == "foil €9.00"

    def test_missing_rate_falls_back_to_usd(self):
        assert format_price({"usd": 1.24}, "CAD", None) == "$1.24"
        assert format_price({"usd": 1.24}, "XYZ", 2.0) == "$1.24"

    def test_thousands_separator(self):
        assert format_price({"usd_foil": 1267.58}) == "foil $1,267.58"


class TestPricesStale:
    def test_missing_file_is_stale(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert prices_stale("Lorcana") is True

    def test_fresh_file_is_not_stale(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {}, fetched_at=_iso_now())
        assert prices_stale("Lorcana") is False

    def test_old_file_is_stale(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {}, fetched_at=_iso_now(hours_ago=25))
        assert prices_stale("Lorcana") is True
        assert prices_stale("Lorcana", max_age_hours=48) is False

    def test_garbage_timestamp_is_stale(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {}, fetched_at="not-a-date")
        assert prices_stale("Lorcana") is True

    def test_rates_file_via_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert prices_stale(path=RATES_FILE) is True
        (tmp_path / RATES_FILE).write_text(
            json.dumps({"fetched_at": _iso_now(), "rates": {}}), encoding="utf-8"
        )
        assert prices_stale(path=RATES_FILE) is False


class TestLorcanaMapper:
    def test_maps_and_skips_defensively(self):
        cards = [
            {"set": {"code": "9"}, "collector_number": "41",
             "prices": {"usd": "1.24", "usd_foil": 3.8}},          # string price coerced
            {"set": "1", "collector_number": 207,
             "prices": {"usd": None, "usd_foil": 1267.58}},        # set as plain string
            {"set": {"code": "1"}, "number": "5", "prices": {"usd": 0.5}},  # number fallback
            {"set": {"code": "2"}, "collector_number": "7"},       # no prices → skipped
            {"set": {"code": "2"}, "collector_number": "8",
             "prices": {"usd": None, "usd_foil": None}},           # both null → skipped
            {"collector_number": "9", "prices": {"usd": 1.0}},     # no set → skipped
            "not a dict",                                          # junk → skipped
        ]
        assert lorcana_prices(cards) == {
            "9-41": {"usd": 1.24, "usd_foil": 3.8},
            "1-207": {"usd": None, "usd_foil": 1267.58},
            "1-5": {"usd": 0.5, "usd_foil": None},
        }

    def test_empty_source_yields_empty(self):
        assert lorcana_prices([]) == {}
        assert lorcana_prices(None) == {}


class TestDownloadCardPrices:
    def _lorcast_stub(self, responses):
        def fake(url):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return payload
            raise AssertionError(f"unexpected URL {url}")
        return fake

    def test_writes_file_and_evicts_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Seed a stale in-memory value that the refresh must replace
        _write_prices(tmp_path, "Lorcana", {"9-41": {"usd": 0.01}})
        assert price_for("9", "41", "Lorcana")["usd"] == 0.01

        monkeypatch.setattr(price_fetcher, "_fetch_json", self._lorcast_stub({
            "/sets": {"results": [{"code": "9"}]},
            "/sets/9/cards": [{"set": {"code": "9"}, "collector_number": "41",
                               "prices": {"usd": 1.24, "usd_foil": 3.8}}],
        }))
        assert download_card_prices("Lorcana") == 1

        written = json.loads((tmp_path / "card_prices_Lorcana.json").read_text("utf-8"))
        assert written["prices"] == {"9-41": {"usd": 1.24, "usd_foil": 3.8}}
        assert not prices_stale("Lorcana")
        assert price_for("9", "41", "Lorcana")["usd"] == 1.24  # cache was evicted

    def test_unregistered_game_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert download_card_prices("NoSuchGame") is None
        assert not os.path.exists(tmp_path / "card_prices_NoSuchGame.json")

    def test_per_set_failure_keeps_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_prices(tmp_path, "Lorcana", {"9-41": {"usd": 0.01}},
                      fetched_at="2020-01-01T00:00:00+00:00")

        def fake(url):
            if url.endswith("/sets"):
                return {"results": [{"code": "9"}, {"code": "10"}]}
            if url.endswith("/sets/9/cards"):
                return [{"set": {"code": "9"}, "collector_number": "41",
                         "prices": {"usd": 99.0}}]
            raise OSError("set 10 fetch failed")

        monkeypatch.setattr(price_fetcher, "_fetch_json", fake)
        with pytest.raises(OSError):
            download_card_prices("Lorcana")

        written = json.loads((tmp_path / "card_prices_Lorcana.json").read_text("utf-8"))
        assert written["prices"] == {"9-41": {"usd": 0.01}}  # old file untouched


class TestDownloadCurrencyRates:
    def test_writes_supported_rates_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(price_fetcher, "_fetch_json", lambda url: {
            "base": "USD",
            "rates": {"CAD": 1.37, "EUR": 0.92, "GBP": 0.79, "JPY": 155.0},
        })
        assert download_currency_rates() == 3
        written = json.loads((tmp_path / RATES_FILE).read_text("utf-8"))
        assert written["rates"] == {"CAD": 1.37, "EUR": 0.92, "GBP": 0.79}
        assert rate_for("CAD") == 1.37

    def test_fetch_failure_keeps_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / RATES_FILE).write_text(
            json.dumps({"fetched_at": _iso_now(), "rates": {"CAD": 1.3}}),
            encoding="utf-8",
        )

        def fail(url):
            raise OSError("offline")

        monkeypatch.setattr(price_fetcher, "_fetch_json", fail)
        with pytest.raises(OSError):
            download_currency_rates()
        assert rate_for("CAD") == 1.3


def _load_cli_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fetch_card_prices", os.path.join(root, "scripts", "fetch_card_prices.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFetchScript:
    def test_unsupported_game_fails_fast(self, capsys):
        mod = _load_cli_module()
        assert mod.main(["--game", "NoSuchGame"]) == 1
        assert "No price source" in capsys.readouterr().err

    def test_happy_path_writes_prices_and_rates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mod = _load_cli_module()

        def fake(url):
            if url.endswith("/sets"):
                return {"results": [{"code": "9"}]}
            if url.endswith("/sets/9/cards"):
                return [{"set": {"code": "9"}, "collector_number": "41",
                         "prices": {"usd": 1.24}}]
            return {"rates": {"CAD": 1.37}}

        monkeypatch.setattr(price_fetcher, "_fetch_json", fake)
        assert mod.main(["--game", "Lorcana"]) == 0
        assert (tmp_path / "card_prices_Lorcana.json").exists()
        assert (tmp_path / RATES_FILE).exists()
