#!/usr/bin/env python3
"""CLI for downloading card names into card_names_<Game>.json.

Thin wrapper around lorebook.core.name_fetcher. The output is display-only
data read by lorebook.core.card_names; it is gitignored and safe to
delete/regenerate. Run once per game, and again when new sets release:

  python scripts/fetch_card_names.py --game Lorcana
  python scripts/fetch_card_names.py --game Riftbound
"""

import argparse
import os
import sys

try:
    from lorebook.core.card_names import names_file_for
    from lorebook.core.name_fetcher import GAMES, download_card_names
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lorebook.core.card_names import names_file_for
    from lorebook.core.name_fetcher import GAMES, download_card_names


def main(argv=None) -> int:
    """Parse arguments, download the game's card names, and return the exit code."""
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

    out = args.out or names_file_for(args.game.strip().capitalize())
    print(f"Fetching {args.url or GAMES[key][0]} …")
    try:
        count = download_card_names(args.game, out_path=out, url=args.url)
    except (OSError, ValueError) as e:  # network/HTTP error or bad JSON
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    if not count:
        print("No cards parsed — the source format may have changed. "
              "Try --url with an alternative source.", file=sys.stderr)
        return 1
    print(f"Wrote {count} card names to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
