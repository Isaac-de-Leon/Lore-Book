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
#   Riftbound — RiftScribe card API     https://riftscribe.gg

import argparse
import json
import sys
import urllib.request

LORCANA_URL = "https://lorcanajson.org/files/current/en/allCards.json"
RIFTBOUND_URL = "https://riftscribe.gg/api/cards"


def _fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def _fetch_riftbound(url: str):
    """
    Fetch all RiftScribe cards. The API returns fixed-size pages (48 cards)
    and pages with an `offset` query param, so keep requesting until a page
    comes back short or repeats.
    """
    cards, offset, last_first_id = [], 0, None
    for _ in range(200):  # hard cap: never loop forever on a misbehaving API
        sep = "&" if "?" in url else "?"
        page = _fetch_json(f"{url}{sep}offset={offset}" if offset else url)
        if isinstance(page, dict):
            page = page.get("cards", [])
        if not page:
            break
        first_id = page[0].get("id") if isinstance(page[0], dict) else None
        if first_id is not None and first_id == last_first_id:
            break  # API ignored the offset; stop rather than duplicate
        last_first_id = first_id
        cards.extend(page)
        if len(page) < 48:
            break
        offset += len(page)
    return cards


def lorcana_names(data) -> dict:
    """Map LorcanaJSON allCards.json → {"<setCode>-<number>": fullName}."""
    names = {}
    for card in (data or {}).get("cards", []):
        set_code = str(card.get("setCode", "")).strip()
        number = str(card.get("number", "")).strip()
        name = card.get("fullName") or card.get("name")
        if set_code and number and name:
            names[f"{set_code}-{number}"] = str(name)
    return names


def riftbound_names(data) -> dict:
    """
    Map RiftScribe card data → {"<set>-<number>": name}.

    Tolerates a top-level list or a {"cards": [...]} wrapper, and the common
    field spellings for set code and collector number.
    """
    cards = data.get("cards", data) if isinstance(data, dict) else data
    names = {}
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        set_code = str(
            card.get("set") or card.get("setCode") or card.get("set_code")
            or card.get("set_id") or ""
        ).strip()
        number = str(
            card.get("number") or card.get("collectorNumber") or card.get("collector_number") or ""
        ).strip()
        name = card.get("name")
        if set_code and number and name:
            names[f"{set_code}-{number}"] = str(name)
    return names


GAMES = {
    "lorcana": (LORCANA_URL, _fetch_json, lorcana_names),
    "riftbound": (RIFTBOUND_URL, _fetch_riftbound, riftbound_names),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch card names into card_names_<Game>.json.")
    p.add_argument("--game", required=True, help="Lorcana or Riftbound.")
    p.add_argument("--url", help="Override the source URL (e.g. a mirror or newer endpoint).")
    p.add_argument("--out", help="Output path (default: card_names_<Game>.json in the cwd).")
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
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    names = mapper(data)
    if not names:
        print("No cards parsed — the source format may have changed. "
              "Try --url with an alternative source.", file=sys.stderr)
        return 1

    out = args.out or f"card_names_{args.game.strip().capitalize()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"Wrote {len(names)} card names to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
