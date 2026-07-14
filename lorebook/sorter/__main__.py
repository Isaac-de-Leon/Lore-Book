# __main__.py — `python -m lorebook.sorter` headless sorter runner.
#
# Wires a camera source (live or a folder of images), a MockTransport, the
# pluggable feature extractor, the per-game feature cache, and a rules config
# into a SortPipeline, then runs the loop. Motion is always mocked here — this
# is the dry-run / bring-up entry point that proves the software loop before
# any sorting mechanism exists.

import argparse
import logging
import os
import sys

from lorebook.core.card_database import load_cache, set_database_path
from lorebook.core.features import get_extractor
from lorebook.hardware.camera import MockCameraSource, OpenCVCameraSource
from lorebook.hardware.transport import MockTransport
from lorebook.sorter.pipeline import SortPipeline
from lorebook.sorter.rules import SortRules, load_rules


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lorebook.sorter",
        description="Headless card sorter: capture → match → decide bin → (mock) route.",
    )
    p.add_argument("--game", default="Lorcana", help="Game folder inside Card_Images/.")
    p.add_argument("--backend", default="keras", choices=["keras", "tflite"],
                   help="Feature extractor backend.")
    p.add_argument("--rules", help="Path to a JSON sort-rules config. Omit to send everything to reject_bin.")
    p.add_argument("--source", default="camera",
                   help="'camera' for a live feed, or a path to an image file/folder for a dry replay.")
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (with --source camera).")
    p.add_argument("--threshold", type=float, default=0.70, help="Minimum cosine match score.")
    p.add_argument("--foil-threshold", type=float, default=None, help="Override foil-detection threshold.")
    p.add_argument("--max-cards", type=int, default=None, help="Stop after N cards.")
    crop = p.add_mutually_exclusive_group()
    crop.add_argument("--crop", dest="crop", action="store_true", default=None,
                      help="Crop frames to the centered card box before matching "
                           "(default with --source camera).")
    crop.add_argument("--no-crop", dest="crop", action="store_false",
                      help="Match the full frame (default when replaying image files).")
    dry = p.add_mutually_exclusive_group()
    dry.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                     help="Do not write the CSV (default).")
    dry.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                     help="Write matched cards to the CSV via update_cardlist.")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("lorebook.sorter")

    # Reference feature cache for the chosen game.
    set_database_path(args.game)
    feature_db = load_cache()
    if not feature_db:
        log.error("No feature database for %s. Build it first: "
                  "python PhotoMatching.py --game %s --build", args.game, args.game)
        return 1

    # Rules: load from file, or default to an all-reject config.
    rules = load_rules(args.rules) if args.rules else SortRules()
    if not args.rules:
        log.warning("No --rules given; every card routes to '%s'.", rules.reject_bin)

    # Camera: live or a folder/file replay.
    if args.source == "camera":
        try:
            camera = OpenCVCameraSource(args.camera_index)
        except Exception as e:
            log.error("Could not open camera %d: %s", args.camera_index, e)
            return 1
    else:
        if not os.path.exists(args.source):
            log.error("Source path not found: %s", args.source)
            return 1
        camera = MockCameraSource(args.source)

    # Fail fast (and warm the model) instead of silently rejecting every card
    # when e.g. the .tflite model hasn't been generated yet.
    extractor = get_extractor(args.backend)
    try:
        extractor.ensure_ready()
    except Exception as e:
        log.error("Feature extractor (%s) is not usable: %s", args.backend, e)
        camera.release()
        return 1

    transport = MockTransport()

    # Live frames have the card centered against background — crop like the
    # GUI does. Replayed reference images are already just the card.
    crop_to_focus = args.crop if args.crop is not None else (args.source == "camera")

    pipeline = SortPipeline(
        camera=camera,
        transport=transport,
        extractor=extractor,
        feature_db=feature_db,
        rules=rules,
        game=args.game,
        threshold=args.threshold,
        foil_threshold=args.foil_threshold,
        dry_run=args.dry_run,
        crop_to_focus=crop_to_focus,
    )

    outcomes = pipeline.run(max_cards=args.max_cards)

    log.info("Done. %d card(s) processed. Per-bin totals:", len(outcomes))
    for bin_id, count in sorted(transport.routed.items()):
        log.info("  %-12s %d", bin_id, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
