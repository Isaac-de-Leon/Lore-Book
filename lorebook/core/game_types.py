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
