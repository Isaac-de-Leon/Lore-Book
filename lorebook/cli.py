"""Command-line entry point: build a game's feature cache and/or match one image.

    python -m lorebook --game Lorcana --build
    python -m lorebook --game Lorcana --match path/to/photo.jpg

The headless sorter has its own entry point: ``python -m lorebook.sorter``.
"""

import argparse
import os
import sys

from lorebook.core.card_database import build_feature_database, load_cache
from lorebook.core.features import extract_features
from lorebook.core.game_types import CARD_IMAGES_DIR
from lorebook.core.matching import find_best_matches

MAX_MATCHES_SHOWN = 10


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the parser for the build/match command line."""
    parser = argparse.ArgumentParser(
        prog="python -m lorebook",
        description="Build a game's feature cache and/or match an image against it.",
        epilog="For the headless sorter run `python -m lorebook.sorter --help`.",
    )
    parser.add_argument("--game", default="Lorcana", help="Game folder inside Card_Images/.")
    parser.add_argument("--build", action="store_true", help="Build/update the feature cache.")
    parser.add_argument("--match", metavar="IMAGE", help="Path to an input image to match.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested build and/or match; return the process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (args.build or args.match):
        parser.error("nothing to do: pass --build and/or --match IMAGE")

    image_dir = os.path.join(CARD_IMAGES_DIR, args.game)
    os.makedirs(image_dir, exist_ok=True)

    if args.build:
        vectors = build_feature_database(
            image_dir, progress_callback=lambda pct, name: print(f"{pct:3d}% - {name or ''}")
        )
        print(f"DB entries: {len(vectors)}")

    if args.match:
        features = extract_features(args.match)
        if features is None:
            print(f"Could not read or extract features from {args.match}", file=sys.stderr)
            return 1
        matches = find_best_matches(features, load_cache(image_dir), threshold=args.threshold)
        for filename, score in matches[:MAX_MATCHES_SHOWN]:
            print(f"{score:.4f}  {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
