# price_fetcher.py — download card market prices into card_prices_<Game>.json.
#
# Core, no Qt. Used by scripts/fetch_card_prices.py (CLI) and by the GUI's
# background DB-build worker, which refreshes any prices file older than a day
# (lorebook.core.card_prices.prices_stale) so scans show current values. A
# fetch failure must never block the build or replace a good file — writes are
# atomic and only happen after every set fetched cleanly.
#
# Sources (pass url= if a format changes):
#   Lorcana — Lorcast API, TCGplayer-sourced USD prices   https://lorcast.com/docs/api
#   Rates   — Frankfurter, ECB daily USD exchange rates   https://frankfurter.dev
#
# Games without an entry in GAMES are skipped silently (same contract as
# image_fetcher): their scans simply show no price line.

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Optional

from lorebook.core.card_prices import (
    RATES_FILE,
    SUPPORTED_CURRENCIES,
    clear_price_cache,
    prices_file_for,
)

LORCANA_URL = "https://api.lorcast.com/v0"
RATES_URL = "https://api.frankfurter.dev/v1/latest?base=USD"

_UA = "Lore-Book-price-fetcher/1.0 (+https://github.com/; personal collection tool)"


def _fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _as_price(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lorcana_prices(cards) -> dict:
    """
    Map Lorcast card objects → {"<set>-<number>": {"usd", "usd_foil"}}.
    Cards without a set/number or without any price are skipped, so the file
    only carries entries that can actually render.
    """
    out: dict = {}
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        set_obj = card.get("set")
        set_code = str(set_obj.get("code", "") if isinstance(set_obj, dict)
                       else set_obj or "").strip()
        number = str(card.get("collector_number") or card.get("number") or "").strip()
        if not (set_code and number):
            continue
        prices = card.get("prices")
        prices = prices if isinstance(prices, dict) else {}
        usd = _as_price(prices.get("usd"))
        usd_foil = _as_price(prices.get("usd_foil"))
        if usd is None and usd_foil is None:
            continue
        out[f"{set_code}-{number}"] = {"usd": usd, "usd_foil": usd_foil}
    return out


def _fetch_lorcana(base_url: str, progress_callback: Optional[Callable[[str], None]] = None) -> list:
    """
    Fetch every card from the Lorcast API, set by set. Any per-set failure
    propagates so a partial result never replaces a complete prices file
    under a fresh timestamp.
    """
    base = base_url.rstrip("/")
    sets_payload = _fetch_json(f"{base}/sets")
    sets = sets_payload.get("results") if isinstance(sets_payload, dict) else sets_payload

    cards: list = []
    for entry in sets or []:
        code = str(entry.get("code", "")).strip() if isinstance(entry, dict) else ""
        if not code:
            continue
        if progress_callback:
            progress_callback(f"Fetching prices for set {code}…")
        page = _fetch_json(f"{base}/sets/{code}/cards")
        if isinstance(page, dict):
            page = page.get("results") or page.get("cards") or []
        cards.extend(page or [])
    return cards


# game (lowercase) -> (default base URL, fetcher(url, progress_callback) -> cards,
#                      mapper(cards) -> {"<set>-<number>": {"usd", "usd_foil"}})
GAMES = {
    "lorcana": (LORCANA_URL, _fetch_lorcana, lorcana_prices),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: str, payload: dict) -> None:
    # The GUI's main thread may read these files while the build worker
    # writes them — write-then-replace so readers never see a torn file.
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)
    clear_price_cache()


def download_card_prices(game: str, out_path: Optional[str] = None,
                         url: Optional[str] = None,
                         progress_callback: Optional[Callable[[str], None]] = None
                         ) -> Optional[int]:
    """
    Fetch current market prices for a game and write card_prices_<Game>.json.
    Returns the number of priced cards, or None for games without a
    registered source. Fetch failures raise — callers catch and keep the
    previous file.
    """
    key = str(game).strip().lower()
    if key not in GAMES:
        logging.info(f"No price source registered for {game!r} — skipping price fetch.")
        return None
    default_url, fetcher, mapper = GAMES[key]

    cards = fetcher(url or default_url, progress_callback)
    prices = mapper(cards)
    out = out_path or prices_file_for(str(game).strip())
    _write_json_atomic(out, {"fetched_at": _now_iso(), "prices": prices})
    logging.info(f"Wrote {len(prices)} {game} price entries to {out}")
    return len(prices)


def download_currency_rates(out_path: Optional[str] = None,
                            url: Optional[str] = None) -> int:
    """
    Fetch daily USD exchange rates for the supported display currencies and
    write currency_rates.json. Returns the number of rates written; raises on
    fetch failure (callers catch and keep the previous file — USD display
    never needs this file at all).
    """
    data = _fetch_json(url or RATES_URL)
    raw = data.get("rates") if isinstance(data, dict) else None
    raw = raw if isinstance(raw, dict) else {}

    rates = {}
    for code in SUPPORTED_CURRENCIES:
        if code == "USD":
            continue
        rate = _as_price(raw.get(code))
        if rate is not None:
            rates[code] = rate

    out = out_path or RATES_FILE
    _write_json_atomic(out, {"fetched_at": _now_iso(), "base": "USD", "rates": rates})
    logging.info(f"Wrote {len(rates)} currency rates to {out}")
    return len(rates)
