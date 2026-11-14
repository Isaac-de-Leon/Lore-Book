"""Game type enum, detection, and shared constants."""

import os
from enum import Enum
from pathlib import Path

from lorebook.core.paths import data_path

SUPPORTED_EXTS = (".webp", ".jpg", ".jpeg", ".png")
CARD_IMAGES_DIR = data_path("Card_Images")
LORCANA_CSV = "LorcanaList.csv"
RIFTBOUND_CSV = "RiftboundList.csv"


class GameType(Enum):
    """The games with built-in rules (foil-only cards, legacy CSV names); anything else is UNKNOWN."""

    LORCANA = "lorcana"
    RIFTBOUND = "riftbound"
    UNKNOWN = "unknown"


def game_type_from_name(name: str | None) -> "GameType":
    """Resolve a game name (e.g. "Lorcana", "riftbound") to a GameType."""
    try:
        return GameType((name or "").strip().lower())
    except ValueError:
        return GameType.UNKNOWN


def csv_for_game(game_name: str | None) -> str:
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
        return data_path(LORCANA_CSV)
    if game_type == GameType.RIFTBOUND:
        return data_path(RIFTBOUND_CSV)
    return data_path(f"{name}List.csv")


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
    """
    Determine game type from an image's game folder: the folder right after
    Card_Images/ in the path, else the file's own parent folder. Ancestors
    above that are ignored, so a checkout living under e.g. ~/Riftbound/
    can't make every Lorcana card look like Riftbound.
    """
    if not filepath:
        return GameType.UNKNOWN
    parts = list(Path(filepath).parts)
    lowered = [p.lower() for p in parts]
    # Compare the folder *name*: CARD_IMAGES_DIR is a full per-user path
    # in installed builds (paths.data_path), never a single path part.
    base = os.path.basename(CARD_IMAGES_DIR).lower()
    if base in lowered and lowered.index(base) + 1 < len(parts) - 1:
        folder = parts[lowered.index(base) + 1]
    elif len(parts) >= 2:
        folder = parts[-2]
    else:
        return GameType.UNKNOWN
    return game_type_from_name(folder)
