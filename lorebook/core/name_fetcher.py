"""Download card names into card_names_<Game>.json.

Core, no Qt. Used by scripts/fetch_card_names.py (CLI). The output
({"<set>-<number>": "Card Name"}) is display-only data read by
lorebook.core.card_names; it is gitignored and safe to delete and regenerate.

Sources (community-maintained; pass url= if a format changes):
  Lorcana   — LorcanaJSON bulk data   https://lorcanajson.org
  Riftbound — the same sources and numbering as the card images
              (lorebook.core.image_fetcher), so names always match filenames

Games without an entry in GAMES are skipped (same contract as image_fetcher
and price_fetcher): their scans simply show the card code without a name.
"""

import logging
from collections.abc import Callable
from typing import Any

from lorebook.core.card_names import names_file_for
from lorebook.core.fileio import atomic_write_json
from lorebook.core.image_fetcher import (
    LORCANA_URL,
    RIFTBOUND_FALLBACK_URL,
    fetch_riftbound_open_sources,
    is_promo_printing,
    riftbound_printings,
)
from lorebook.core.net import fetch_json

logger = logging.getLogger(__name__)


def lorcana_names(data: Any) -> dict[str, str]:
    """
    Map LorcanaJSON allCards.json → {"<setCode>-<number>": fullName}.

    Main-set printings win setCode-number collisions with promos (a promo name
    is only kept when no main printing shares its key).
    """
    names: dict[str, str] = {}
    is_promo: dict[str, bool] = {}
    for card in (data or {}).get("cards", []):
        set_code = str(card.get("setCode", "")).strip()
        number = str(card.get("number", "")).strip()
        name = card.get("fullName") or card.get("name")
        if not (set_code and number and name):
            continue
        key = f"{set_code}-{number}"
        promo = is_promo_printing(card)
        if key not in names or (is_promo[key] and not promo):
            names[key] = str(name)
            is_promo[key] = promo
    return names


def riftbound_names(data: Any) -> dict[str, str]:
    """
    Map Riftbound card data → {"<set>-<number>": name}, numbered exactly like
    the image files. A bare collector number won't do: rune, token and special
    printings reuse the numbers of unrelated cards in their set.
    """
    return {
        f"{set_code}-{number}": str(card["name"])
        for set_code, number, _url, card in riftbound_printings(data)
        if card.get("name")
    }


# game (lowercase) -> (default URL, fetcher(url) -> card data, mapper(data) -> {"<set>-<number>": name})
GAMES: dict[str, tuple[str, Callable[[str], Any], Callable[[Any], dict[str, str]]]] = {
    "lorcana": (LORCANA_URL, fetch_json, lorcana_names),
    "riftbound": (RIFTBOUND_FALLBACK_URL, fetch_riftbound_open_sources, riftbound_names),
}


def download_card_names(game: str, out_path: str | None = None, url: str | None = None) -> int | None:
    """
    Fetch a game's card names and write card_names_<Game>.json.

    Returns the number of names written, or None for games without a
    registered source. Returns 0, leaving any existing file untouched, when
    the source yields no names (its format may have changed). Fetch failures
    raise OSError or ValueError — callers catch and keep the previous file.
    """
    key = str(game).strip().lower()
    if key not in GAMES:
        logger.info("No name source registered for %r — skipping name fetch.", game)
        return None
    default_url, fetcher, mapper = GAMES[key]

    names = mapper(fetcher(url or default_url))
    if not names:
        return 0
    out = out_path or names_file_for(str(game).strip())
    atomic_write_json(out, names, ensure_ascii=False, indent=1, sort_keys=True)
    logger.info("Wrote %s %s card names to %s", len(names), game, out)
    return len(names)
