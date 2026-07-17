# card_prices.py — optional card market-price lookup (display only).
#
# Prices come from a local card_prices_<Game>.json written by
# lorebook.core.price_fetcher ({"fetched_at": ..., "prices": {"<set>-<number>":
# {"usd": 1.24, "usd_foil": 3.8}}}). The GUI refreshes it in the background
# when it's missing or older than a day; scripts/fetch_card_prices.py does the
# same from the command line. The file is optional and gitignored: every
# lookup degrades to None when it's absent. Prices never enter the collection
# CSV — that file must stay 4-column for Dreamborn.ink bulk import.
#
# Source prices are USD (TCGplayer, via the Lorcast API). Other display
# currencies are converted with the daily USD rates cached in
# currency_rates.json (Frankfurter/ECB), falling back to USD when no rate is
# available.

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

from lorebook.core.card_names import normalize_key

RATES_FILE = "currency_rates.json"

# Display currencies offered in Settings; all are converted from the USD
# source prices via the cached rates file. Order = dropdown order.
SUPPORTED_CURRENCIES = ("USD", "CAD", "EUR", "GBP")
_SYMBOLS = {"USD": "$", "CAD": "CA$", "EUR": "€", "GBP": "£"}


def prices_file_for(game: str) -> str:
    """Path of the (optional) prices file for a game folder name."""
    return f"card_prices_{game}.json"


# Loaded price/rate maps keyed by the file's absolute path (so tests that
# chdir, and games sharing a cwd, never collide). Unlike card names, prices
# change while the app runs — clear_price_cache() evicts after a refresh.
_cache: Dict[str, dict] = {}


def clear_price_cache() -> None:
    """Drop all cached price/rate maps so the next lookup re-reads the files."""
    _cache.clear()


def _read_json(file: str) -> Optional[dict]:
    try:
        with open(file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else None
    except Exception as e:
        logging.warning(f"Could not load {file}: {e}")
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_card_prices(game: str, path: Optional[str] = None) -> Dict[str, dict]:
    """
    Load (and cache) the normalized {key: {"usd", "usd_foil"}} map for a game.
    Returns {} when the file is missing or unreadable.
    """
    file = os.path.abspath(path or prices_file_for(game))
    if file in _cache:
        return _cache[file]

    prices: Dict[str, dict] = {}
    if os.path.exists(file):
        raw = _read_json(file) or {}
        entries = raw.get("prices")
        for key, entry in (entries if isinstance(entries, dict) else {}).items():
            set_code, sep, card_code = str(key).partition("-")
            if sep and isinstance(entry, dict):
                prices[normalize_key(set_code, card_code)] = {
                    "usd": _as_float(entry.get("usd")),
                    "usd_foil": _as_float(entry.get("usd_foil")),
                }

    _cache[file] = prices
    return prices


def price_for(set_code: str, card_code: str, game: str) -> Optional[dict]:
    """Return {"usd": float|None, "usd_foil": float|None}, or None when unknown."""
    if not set_code or not game:
        return None
    return load_card_prices(game).get(normalize_key(set_code, card_code))


def rate_for(currency: str, path: Optional[str] = None) -> Optional[float]:
    """USD→currency rate from the cached rates file; 1.0 for USD, None when unknown."""
    currency = (currency or "USD").upper()
    if currency == "USD":
        return 1.0
    file = os.path.abspath(path or RATES_FILE)
    if file not in _cache:
        rates: Dict[str, float] = {}
        if os.path.exists(file):
            raw = _read_json(file) or {}
            entries = raw.get("rates")
            for code, value in (entries if isinstance(entries, dict) else {}).items():
                rate = _as_float(value)
                if rate is not None:
                    rates[str(code).upper()] = rate
        _cache[file] = rates
    return _cache[file].get(currency)


def format_price(entry: Optional[dict], currency: str = "USD",
                 rate: Optional[float] = None) -> str:
    """
    Render a price entry as e.g. "$1.24 · foil $3.80" (missing halves omitted,
    "" when there's nothing to show). Stored prices are USD; a non-USD
    currency renders converted values when a rate is known and falls back to
    USD otherwise, so an unfetched rates file never hides prices.
    """
    if not entry:
        return ""
    currency = (currency or "USD").upper()
    if currency not in _SYMBOLS or rate is None or rate <= 0:
        currency, rate = "USD", 1.0
    symbol = _SYMBOLS[currency]

    parts = []
    usd = _as_float(entry.get("usd"))
    usd_foil = _as_float(entry.get("usd_foil"))
    if usd is not None:
        parts.append(f"{symbol}{usd * rate:,.2f}")
    if usd_foil is not None:
        parts.append(f"foil {symbol}{usd_foil * rate:,.2f}")
    return " · ".join(parts)


def prices_stale(game: Optional[str] = None, path: Optional[str] = None,
                 max_age_hours: float = 24.0) -> bool:
    """
    True when the prices file (or, with path=RATES_FILE, the rates file) is
    missing or its fetched_at is older than max_age_hours or unparseable.
    Reads the file directly — no cache — so it's safe from worker threads.
    """
    file = os.path.abspath(path or prices_file_for(game))
    try:
        with open(file, "r", encoding="utf-8") as f:
            fetched_at = json.load(f).get("fetched_at")
        fetched_ts = datetime.fromisoformat(str(fetched_at)).timestamp()
    except Exception:
        return True
    return (time.time() - fetched_ts) > max_age_hours * 3600
