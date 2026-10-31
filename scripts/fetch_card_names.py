#!/usr/bin/env python3
# fetch_card_names.py — download card names and write card_names_<Game>.json.
#
# The output ({"<set>-<number>": "Card Name"}) is display-only data read by
# lorebook.core.card_names; it is gitignored and safe to delete/regenerate.
# Run once per game, and again when new sets release:
#
#   python scripts/fetch_card_names.py --game Lorcana
#   python scripts/fetch_card_names.py --game Riftbound
#
# Sources (community-maintained; override with --url if a format changes):
#   Lorcana   — LorcanaJSON bulk data   https://lorcanajson.org
#   Riftbound — the same sources and numbering as the card images
#               (lorebook.core.image_fetcher), so names always match filenames

import argparse
import json
import os
import sys

try:
    from lorebook.core.card_names import names_file_for
    from lorebook.core.image_fetcher import (
        RIFTBOUND_FALLBACK_URL,
        fetch_riftbound_open_sources,
        is_promo_printing,
        riftbound_printings,
    )
    from lorebook.core.net import fetch_json as _fetch_json
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lorebook.core.card_names import names_file_for
    from lorebook.core.image_fetcher import (
        RIFTBOUND_FALLBACK_URL,
        fetch_riftbound_open_sources,
        is_promo_printing,
        riftbound_printings,
    )
    from lorebook.core.net import fetch_json as _fetch_json

LORCANA_URL = "https://lorcanajson.org/files/current/en/allCards.json"


def lorcana_names(data) -> dict:
    """
    Map LorcanaJSON allCards.json → {"<setCode>-<number>": fullName}.

    Main-set printings win setCode-number collisions with promos (a promo name
    is only kept when no main printing shares its key).
    """
    names, is_promo = {}, {}
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


def riftbound_names(data) -> dict:
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


GAMES = {
    "lorcana": (LORCANA_URL, _fetch_json, lorcana_names),
    "riftbound": (RIFTBOUND_FALLBACK_URL, fetch_riftbound_open_sources, riftbound_names),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch card names into card_names_<Game>.json.")
    p.add_argument("--game", required=True, help="Lorcana or Riftbound.")
    p.add_argument("--url", help="Override the source URL (e.g. a mirror or newer endpoint).")
    p.add_argument("--out", help="Output path (default: card_names_<Game>.json where the app reads it).")
    args = p.parse_args(argv)

    key = args.game.strip().lower()
    if key not in GAMES:
        print(f"Unsupported game {args.game!r} (supported: {', '.join(sorted(GAMES))}).",
              file=sys.stderr)
        return 1

    default_url, fetcher, mapper = GAMES[key]
    url = args.url or default_url
    print(f"Fetching {url} …")
    try:
        data = fetcher(url)
    except (OSError, ValueError) as e:  # network/HTTP error or bad JSON
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    names = mapper(data)
    if not names:
        print("No cards parsed — the source format may have changed. "
              "Try --url with an alternative source.", file=sys.stderr)
        return 1

    out = args.out or names_file_for(args.game.strip().capitalize())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"Wrote {len(names)} card names to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
