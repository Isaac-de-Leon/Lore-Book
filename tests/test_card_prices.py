# tests/test_card_prices.py — price loader/formatter + price fetcher (no network).

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from lorebook.core import price_fetcher
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
def _fresh_cache(tmp_path, monkeypatch):
    # The module cache is keyed by absolute path and never self-resets;
    # clear it so tests reusing filenames across tmp_paths can't collide.
    monkeypatch.chdir(tmp_path)
    clear_price_cache()
    yield
    clear_price_cache()


def _iso_now(hours_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(timespec="seconds")


def _write_prices(tmp_path, game, prices, fetched_at=None):
    (tmp_path / f"card_prices_{game}.json").write_text(
        json.dumps({"fetched_at": fetched_at or _iso_now(), "prices": prices}),
        encoding="utf-8",
    )


def _write_rates(tmp_path, rates, fetched_at=None):
    (tmp_path / RATES_FILE).write_text(
        json.dumps({"fetched_at": fetched_at or _iso_now(), "base": "USD", "rates": rates}),
        encoding="utf-8",
    )


def _lorcast_stub(cards_by_set, rates=None):
    """Fake _fetch_json: /sets lists the sets, /sets/<code>/cards returns cards."""
    def fake(url):
        if url.endswith("/sets"):
            return {"results": [{"code": code} for code in cards_by_set]}
        for code, cards in cards_by_set.items():
            if url.endswith(f"/sets/{code}/cards"):
                if isinstance(cards, Exception):
                    raise cards
                return cards
        if rates is not None:
            return {"base": "USD", "rates": rates}
        raise AssertionError(f"unexpected URL {url}")
    return fake


def test_price_and_rate_lookup(tmp_path):
    assert rate_for("USD") == 1.0                       # needs no file
    assert rate_for("CAD") is None                      # missing file
    assert price_for("009", "041", "Lorcana") is None

    _write_prices(tmp_path, "Lorcana", {
        "9-41": {"usd": 1.24, "usd_foil": 3.8},
        "010-018": {"usd": None, "usd_foil": 2.0},     # null halves survive loading
    })
    clear_price_cache()
    assert price_for("009", "041", "Lorcana") == {"usd": 1.24, "usd_foil": 3.8}
    assert price_for("10", "18", "Lorcana") == {"usd": None, "usd_foil": 2.0}
    assert price_for("009", "999", "Lorcana") is None
    assert price_for("", "041", "Lorcana") is None

    (tmp_path / "card_prices_Broken.json").write_text("{not json", encoding="utf-8")
    assert load_card_prices("Broken") == {}

    _write_rates(tmp_path, {"CAD": 1.37, "EUR": 0.92})
    clear_price_cache()
    assert rate_for("cad") == 1.37
    assert rate_for("GBP") is None


def test_format_price():
    assert format_price({"usd": 1.24, "usd_foil": 3.8}) == "$1.24 · foil $3.80"
    assert format_price({"usd": 0.05, "usd_foil": None}) == "$0.05"
    assert format_price({"usd": None, "usd_foil": 12.5}) == "foil $12.50"
    assert format_price({"usd_foil": 1267.58}) == "foil $1,267.58"
    for empty in (None, {}, {"usd": None, "usd_foil": None}):
        assert format_price(empty) == ""
    assert format_price({"usd": 1.24}, "CAD", 1.35) == "CA$1.67"
    assert format_price({"usd_foil": 10.0}, "EUR", 0.9) == "foil €9.00"
    assert format_price({"usd": 1.24}, "CAD", None) == "$1.24"   # missing rate → USD
    assert format_price({"usd": 1.24}, "XYZ", 2.0) == "$1.24"    # unknown currency → USD


def test_prices_stale(tmp_path):
    assert prices_stale("Lorcana") is True                        # missing file
    _write_prices(tmp_path, "Lorcana", {})
    assert prices_stale("Lorcana") is False
    _write_prices(tmp_path, "Lorcana", {}, fetched_at=_iso_now(hours_ago=25))
    assert prices_stale("Lorcana") is True
    assert prices_stale("Lorcana", max_age_hours=48) is False
    _write_prices(tmp_path, "Lorcana", {}, fetched_at="not-a-date")
    assert prices_stale("Lorcana") is True

    assert prices_stale(path=RATES_FILE) is True
    _write_rates(tmp_path, {})
    assert prices_stale(path=RATES_FILE) is False


def test_lorcana_mapper():
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
    assert lorcana_prices([]) == {}
    assert lorcana_prices(None) == {}


def test_downloads(tmp_path, monkeypatch):
    # A stale in-memory value must be evicted by the refresh.
    _write_prices(tmp_path, "Lorcana", {"9-41": {"usd": 0.01}})
    assert price_for("9", "41", "Lorcana")["usd"] == 0.01

    card = {"set": {"code": "9"}, "collector_number": "41", "prices": {"usd": 1.24, "usd_foil": 3.8}}
    monkeypatch.setattr(price_fetcher, "_fetch_json", _lorcast_stub({"9": [card]}))
    assert download_card_prices("Lorcana") == 1
    written = json.loads((tmp_path / "card_prices_Lorcana.json").read_text("utf-8"))
    assert written["prices"] == {"9-41": {"usd": 1.24, "usd_foil": 3.8}}
    assert not prices_stale("Lorcana")
    assert price_for("9", "41", "Lorcana")["usd"] == 1.24

    assert download_card_prices("NoSuchGame") is None
    assert not os.path.exists(tmp_path / "card_prices_NoSuchGame.json")

    # One failing set aborts the refresh and keeps the existing file.
    monkeypatch.setattr(price_fetcher, "_fetch_json",
                        _lorcast_stub({"9": [card], "10": OSError("set 10 failed")}))
    with pytest.raises(OSError):
        download_card_prices("Lorcana")
    assert json.loads((tmp_path / "card_prices_Lorcana.json").read_text("utf-8")) == written

    # Currency rates: only supported currencies are kept.
    monkeypatch.setattr(price_fetcher, "_fetch_json", lambda url: {
        "base": "USD", "rates": {"CAD": 1.37, "EUR": 0.92, "GBP": 0.79, "JPY": 155.0},
    })
    assert download_currency_rates() == 3
    assert json.loads((tmp_path / RATES_FILE).read_text("utf-8"))["rates"] == {
        "CAD": 1.37, "EUR": 0.92, "GBP": 0.79,
    }
    assert rate_for("CAD") == 1.37

    def offline(url):
        raise OSError("offline")

    monkeypatch.setattr(price_fetcher, "_fetch_json", offline)
    with pytest.raises(OSError):
        download_currency_rates()
    assert rate_for("CAD") == 1.37                      # existing file kept


def _load_cli_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fetch_card_prices", os.path.join(root, "scripts", "fetch_card_prices.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_script(tmp_path, monkeypatch, capsys):
    mod = _load_cli_module()
    assert mod.main(["--game", "NoSuchGame"]) == 1
    assert "No price source" in capsys.readouterr().err

    card = {"set": {"code": "9"}, "collector_number": "41", "prices": {"usd": 1.24}}
    monkeypatch.setattr(price_fetcher, "_fetch_json", _lorcast_stub({"9": [card]}, rates={"CAD": 1.37}))
    assert mod.main(["--game", "Lorcana"]) == 0
    assert (tmp_path / "card_prices_Lorcana.json").exists()
    assert (tmp_path / RATES_FILE).exists()
