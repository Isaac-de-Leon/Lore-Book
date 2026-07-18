# PhotoMatching.py — backward-compatible re-export shim.
#
# All logic has been moved to focused modules under lorebook/core/:
#   game_types.py    — GameType enum, get_game_type, constants
#   image_utils.py   — ensure_valid_image, foil_score, is_probably_foil
#   matching.py      — _l2_normalize, _cosine_score, find_best_matches
#   features.py      — extract_features, visualize_activation_overlay
#   card_database.py — load_cache, build_feature_database, set_database_path
#   csv_manager.py   — update_cardlist, get_available_sets, CSV helpers
#
# Existing imports of the form `from PhotoMatching import ...` continue to work.

from lorebook.core.card_database import (
    build_feature_database,
    clear_from_cache,
    databasePath,
    get_database_path,
    load_cache,
    set_database_path,
)
from lorebook.core.csv_manager import (
    _normalize_existing_rows,
    _split_filename,
    split_filename,
    _write_rows_4col,
    clear_collection,
    get_available_sets,
    read_collection_rows,
    update_cardlist,
    update_cardlist_batch,
)
from lorebook.core.features import (
    extract_features,
    get_extractor,
    visualize_activation_overlay,
)
from lorebook.core.game_types import (
    BASE_DATABASE_PATH,
    LORCANA_CSV,
    LORCANA_MAX_NORMAL_CARD,
    RIFTBOUND_CSV,
    SUPPORTED_EXTS,
    GameType,
    csv_for_game,
    game_type_from_name,
    get_game_type,
    is_foil_only_card,
)
from lorebook.core.image_utils import ensure_valid_image, foil_score, is_probably_foil
from lorebook.core.matching import MatchIndex, _cosine_score, _l2_normalize, find_best_matches

# Legacy name aliases
baseDatabasePath = BASE_DATABASE_PATH
LORCANA_FILE = LORCANA_CSV
RIFTBOUND_FILE = RIFTBOUND_CSV

# ---- CLI (python PhotoMatching.py --build / --match / --sort) --------------
if __name__ == "__main__":
    import argparse
    import sys

    # --sort delegates everything after it to the headless sorter, which has
    # its own argument set (it does not compose with --build/--match).
    if "--sort" in sys.argv[1:]:
        from lorebook.sorter.__main__ import main as sorter_main

        rest = [a for a in sys.argv[1:] if a != "--sort"]
        raise SystemExit(sorter_main(rest))

    parser = argparse.ArgumentParser(
        description="Build feature DB and/or match an image.",
        epilog="For the headless sorter run `python PhotoMatching.py --sort [sorter options]` "
               "(equivalent to `python -m lorebook.sorter`; --sort does not combine with the "
               "flags above — see `python -m lorebook.sorter --help`).",
    )
    parser.add_argument("--game", default="Lorcana", help="Game folder inside Card_Images/")
    parser.add_argument("--build", action="store_true", help="Build/update feature DB.")
    parser.add_argument("--match", help="Path to an input image to match.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold.")
    args = parser.parse_args()

    set_database_path(args.game)

    if args.build:
        db = build_feature_database(lambda p, k: print(f"{p:3d}% - {k or ''}"))
        print(f"DB entries: {len(db)}")

    if args.match:
        db = load_cache()
        feat = extract_features(args.match)
        for fname, score in find_best_matches(feat, db, threshold=args.threshold)[:10]:
            print(f"{score:.4f}  {fname}")
