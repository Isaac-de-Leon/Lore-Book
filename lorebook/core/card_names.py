# card_names.py — optional card-name lookup (display only).
#
# Names come from a local card_names_<Game>.json generated once by
# scripts/fetch_card_names.py ({"<set>-<number>": "Card Name", ...}). The file
# is optional and gitignored: every lookup degrades to None when it's absent,
# so nothing depends on it at runtime. Names never enter the collection CSV —
# that file must stay 4-column for Dreamborn.ink bulk import.

import json
import logging
import os
from typing import Dict, Optional


def names_file_for(game: str) -> str:
    """Path of the (optional) names file for a game folder name."""
    return f"card_names_{game}.json"


def normalize_key(set_code: str, card_code: str) -> str:
    """
    Zero-pad- and case-insensitive lookup key: '9'/'41' and '009'/'041' both
    map to '9-41', so LorcanaJSON's unpadded codes match our padded filenames.
    """
    def norm(part: str) -> str:
        stripped = part.lstrip("0")
        return stripped if stripped else ("0" if part else "")
    return f"{norm(set_code)}-{norm(card_code)}".lower()


# Loaded name maps keyed by the names file's absolute path (so tests that
# chdir, and games sharing a cwd, never collide).
_cache: Dict[str, Dict[str, str]] = {}


def load_card_names(game: str, path: Optional[str] = None) -> Dict[str, str]:
    """
    Load (and cache) the normalized {key: name} map for a game.
    Returns {} when the file is missing or unreadable.
    """
    file = os.path.abspath(path or names_file_for(game))
    if file in _cache:
        return _cache[file]

    names: Dict[str, str] = {}
    if os.path.exists(file):
        try:
            with open(file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key, name in raw.items():
                set_code, sep, card_code = str(key).partition("-")
                if sep:
                    names[normalize_key(set_code, card_code)] = str(name)
        except Exception as e:
            logging.warning(f"Could not load card names from {file}: {e}")

    _cache[file] = names
    return names


def name_for(set_code: str, card_code: str, game: str) -> Optional[str]:
    """Return the card's display name, or None when unknown/no names file."""
    if not set_code or not game:
        return None
    return load_card_names(game).get(normalize_key(set_code, card_code))
