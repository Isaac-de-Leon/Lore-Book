#!/usr/bin/env python3
"""Verify the tflite extractor matches the keras extractor.

Both backends should produce near-identical 1280-dim vectors, so an existing
DBCardCache_*.db stays valid regardless of which backend built it. Run on a
desktop after generating the .tflite model.

    python scripts/convert_to_tflite.py
    python scripts/check_parity.py path/to/card1.jpg path/to/card2.webp
"""
import argparse
import sys

import numpy as np

from lorebook.core.features import get_extractor

MIN_COSINE = 0.99


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Card images to compare across backends.")
    parser.add_argument("--min-cosine", type=float, default=MIN_COSINE)
    args = parser.parse_args()

    keras = get_extractor("keras")
    tflite = get_extractor("tflite")

    worst = 1.0
    for path in args.images:
        a = keras.extract(path)
        b = tflite.extract(path)
        if a is None or b is None:
            print(f"FAIL  {path}: an extractor returned None")
            return 1
        cos = float(np.dot(a, b))  # both are L2-normalized
        worst = min(worst, cos)
        status = "ok " if cos >= args.min_cosine else "LOW"
        print(f"{status} {cos:.5f}  {path}")

    print(f"\nworst cosine: {worst:.5f} (threshold {args.min_cosine})")
    return 0 if worst >= args.min_cosine else 1


if __name__ == "__main__":
    sys.exit(main())
