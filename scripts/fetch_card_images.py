#!/usr/bin/env python3
# fetch_card_images.py — CLI for downloading card images into Card_Images/<Game>/.
#
# Thin wrapper around lorebook.core.image_fetcher.download_new_images(), which
# is also called automatically by the GUI's Rebuild Database button. Re-run when
# a new set releases — existing files are skipped, so only new cards download.
#
#   python scripts/fetch_card_images.py --game Lorcana
#   python scripts/fetch_card_images.py --game Lorcana --set 9      # just set 9
#   python scripts/fetch_card_images.py --game Lorcana --format jpg # keep source JPG
#
# After it finishes, fold the new images into the feature DB:
#   python PhotoMatching.py --game Lorcana --build

import argparse
import os
import sys

try:
    from lorebook.core.image_fetcher import GAMES, download_new_images
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lorebook.core.image_fetcher import GAMES, download_new_images


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download card images into Card_Images/<Game>/.")
    p.add_argument("--game", default="Lorcana", help="Game folder name (default: Lorcana).")
    p.add_argument("--url", help="Override the source bulk-data URL.")
    p.add_argument("--out", help="Output folder (default: Card_Images/<Game>).")
    p.add_argument("--set", action="append", dest="sets", metavar="CODE",
                   help="Only download these set codes (repeatable; '9' and '009' both match).")
    p.add_argument("--format", choices=("webp", "jpg"), default="webp",
                   help="webp = re-encode via OpenCV (default); jpg = keep the source file.")
    p.add_argument("--quality", type=int, default=95, help="WebP quality 1-100 (default: 95).")
    p.add_argument("--limit", type=int, help="Stop after N downloads (for testing).")
    p.add_argument("--force", action="store_true", help="Re-download images that already exist.")
    p.add_argument("--delay", type=float, default=0.05,
                   help="Seconds to pause between downloads (be polite; default: 0.05).")
    p.add_argument("--dry-run", action="store_true", help="List what would download; fetch nothing.")
    args = p.parse_args(argv)

    if args.game.strip().lower() not in GAMES:
        print(f"Unsupported game {args.game!r} (supported: {', '.join(sorted(GAMES))}).",
              file=sys.stderr)
        return 1

    try:
        stats = download_new_images(
            args.game,
            out_dir=args.out,
            progress_callback=print,
            url=args.url,
            sets=args.sets,
            fmt=args.format,
            quality=args.quality,
            limit=args.limit,
            force=args.force,
            delay=args.delay,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"Download of card list failed: {e}", file=sys.stderr)
        return 1

    if stats.downloaded == stats.skipped == stats.failed == 0:
        return 1  # nothing matched / source format changed (message already reported)
    if stats.downloaded and not args.dry_run:
        print(f"Next: python PhotoMatching.py --game {args.game.strip().capitalize()} --build")
    return 1 if stats.failed and not stats.downloaded else 0


if __name__ == "__main__":
    sys.exit(main())
