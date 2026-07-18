# game_types.py — Game type enum, detection, and shared constants.

import logging
from enum import Enum
from pathlib import Path

SUPPORTED_EXTS = (".webp", ".jpg", ".jpeg", ".png")
BASE_DATABASE_PATH = "Card_Images"
LORCANA_CSV = "LorcanaList.csv"
RIFTBOUND_CSV = "RiftboundList.csv"


class GameType(Enum):
    LORCANA = "lorcana"
    RIFTBOUND = "riftbound"
    UNKNOWN = "unknown"


def game_type_from_name(name: str) -> "GameType":
    """Resolve a game name (e.g. "Lorcana", "riftbound") to a GameType."""
    try:
        return GameType((name or "").strip().lower())
    except ValueError:
        return GameType.UNKNOWN


def csv_for_game(game_name: str) -> str:
    """
    Return the collection CSV filename for a game folder name.

    Follows the `Card_Images/<Game>/` → `<Game>List.csv` convention. Lorcana
    and Riftbound resolve to their legacy constants regardless of case; any
    other game derives its CSV from the folder name verbatim, so new games
    work without code changes.
    """
    name = (game_name or "").strip().strip("/\\")
    if not name:
        raise ValueError("game name is required")
    game_type = game_type_from_name(name)
    if game_type == GameType.LORCANA:
        return LORCANA_CSV
    if game_type == GameType.RIFTBOUND:
        return RIFTBOUND_CSV
    return f"{name}List.csv"


# Lorcana main sets have exactly 204 regular cards; anything numbered above
# that (Enchanted / Epic / Iconic) is printed only as foil.
LORCANA_MAX_NORMAL_CARD = 204


def is_foil_only_card(set_code: str, card_code: str, game_type: "GameType") -> bool:
    """
    True when a card exists only in foil, so its collection variant must be
    "foil" (Dreamborn.ink rejects a "normal" row for these).
    """
    if game_type != GameType.LORCANA:
        return False
    code = (card_code or "").strip()
    return code.isdigit() and int(code) > LORCANA_MAX_NORMAL_CARD


def get_game_type(filepath: str) -> "GameType":
    """Determine game type based on image file location."""
    try:
        abs_path = str(Path(filepath).resolve())
        path_parts = [p.lower() for p in Path(abs_path).parts]
        if "riftbound" in path_parts:
            return GameType.RIFTBOUND
        elif "lorcana" in path_parts:
            return GameType.LORCANA
        return GameType.UNKNOWN
    except Exception as e:
        logging.error(f"Error determining game type for {filepath}: {e}")
        return GameType.UNKNOWN
