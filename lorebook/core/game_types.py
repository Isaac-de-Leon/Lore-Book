# game_types.py — Game type enum, detection, and shared constants.

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

SUPPORTED_EXTS = (".webp", ".jpg", ".jpeg", ".png")
BASE_DATABASE_PATH = "Card_Images"
LORCANA_CSV = "LorcanaList.csv"
RIFTBOUND_CSV = "RiftboundList.csv"


class GameType(Enum):
    LORCANA = "lorcana"
    RIFTBOUND = "riftbound"
    UNKNOWN = "unknown"


def game_folders(base: str = BASE_DATABASE_PATH) -> List[str]:
    """Game folder names under Card_Images/, sorted (hidden/cache folders skipped)."""
    try:
        names = os.listdir(base)
    except OSError:
        return []
    return sorted(
        n for n in names
        if os.path.isdir(os.path.join(base, n))
        and not n.startswith((".", "__"))
        and n != "logs"
    )


def resolve_game_folder(key: str, base: str = BASE_DATABASE_PATH) -> Optional[str]:
    """
    Map a settings key (lowercased folder name, e.g. "mtg") back to the real
    folder name ("MTG"), or None when no such folder exists. Never rebuild
    the name with .capitalize() — that breaks folders like "MTG" on
    case-sensitive filesystems and points at the wrong <Game>List.csv.
    """
    key = (key or "").strip().lower()
    for name in game_folders(base):
        if name.lower() == key:
            return name
    return None


def sets_to_store(checked: Iterable[str], available: Iterable[str]) -> List[str]:
    """
    Set filter to persist for a selected game. Every set ticked is stored as
    [] ("all sets"), so sets released later are included automatically;
    otherwise the ticked sets in folder order.
    """
    available = list(available)
    checked = set(checked)
    if not available or all(s in checked for s in available):
        return []
    return [s for s in available if s in checked]


def sets_to_display(stored: Iterable[str], available: Iterable[str], game_selected: bool) -> List[str]:
    """
    Inverse of sets_to_store for the Settings tree: which sets to show ticked.
    A selected game with [] ("all sets") shows every set ticked — the tree
    derives the game's own box from its sets, so an all-unticked tree would
    read as "game off" and deselect it on Apply.
    """
    if not game_selected:
        return []
    stored = set(stored)
    available = list(available)
    # A stale filter (none of its sets on disk any more) falls back to "all"
    # rather than showing the game as off.
    return [s for s in available if s in stored] or available


def game_type_from_name(name: Optional[str]) -> "GameType":
    """Resolve a game name (e.g. "Lorcana", "riftbound") to a GameType."""
    try:
        return GameType((name or "").strip().lower())
    except ValueError:
        return GameType.UNKNOWN


def csv_for_game(game_name: Optional[str]) -> str:
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
    """
    Determine game type from an image's game folder: the folder right after
    Card_Images/ in the path, else the file's own parent folder. Ancestors
    above that are ignored, so a checkout living under e.g. ~/Riftbound/
    can't make every Lorcana card look like Riftbound.
    """
    try:
        parts = list(Path(filepath).parts)
        lowered = [p.lower() for p in parts]
        base = BASE_DATABASE_PATH.lower()
        if base in lowered and lowered.index(base) + 1 < len(parts) - 1:
            folder = parts[lowered.index(base) + 1]
        elif len(parts) >= 2:
            folder = parts[-2]
        else:
            return GameType.UNKNOWN
        return game_type_from_name(folder)
    except Exception as e:
        logging.error(f"Error determining game type for {filepath}: {e}")
        return GameType.UNKNOWN
