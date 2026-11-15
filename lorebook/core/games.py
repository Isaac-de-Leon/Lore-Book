"""Game folders and the card images inside them.

Each game is a folder under Card_Images/ (CARD_IMAGES_DIR) holding one image
per card, named ``<SetCode>-<CardCode>.<ext>``. This module finds the game
folders, lists their images and reads set and card codes out of filenames.
"""

import os

from lorebook.core.game_types import CARD_IMAGES_DIR, SUPPORTED_EXTS


def game_folders(base: str = CARD_IMAGES_DIR) -> list[str]:
    """Game folder names under Card_Images/, sorted (hidden/cache folders skipped)."""
    try:
        names = os.listdir(base)
    except OSError:
        return []
    return sorted(
        n
        for n in names
        if os.path.isdir(os.path.join(base, n)) and not n.startswith((".", "__")) and n != "logs"
    )


def resolve_game_folder(key: str, base: str = CARD_IMAGES_DIR) -> str | None:
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


def list_image_files(folder: str) -> list[str]:
    """Return sorted list of supported image filenames in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTS))


def split_filename(matched_filename: str) -> tuple[str, str]:
    """
    Split a card image filename into (set_code, card_code).

    Examples:
        "001-042.webp"      → ("001", "042")
        "ONG-23c-alt.jpg"   → ("ONG", "23c-alt")
        "nosetcode.png"     → ("",    "nosetcode")
    """
    base = os.path.splitext(os.path.basename(matched_filename))[0]
    parts = base.split("-", maxsplit=2)
    if len(parts) < 2:
        return "", base
    return parts[0], "-".join(parts[1:])


def available_sets(image_dir: str) -> list[str]:
    """Return the sorted set codes of every card image in a game's image folder."""
    set_codes = {split_filename(filename)[0] for filename in list_image_files(image_dir)}
    return sorted(code for code in set_codes if code)
