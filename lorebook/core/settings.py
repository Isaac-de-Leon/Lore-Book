# settings.py — the app's persisted preferences (ui_settings.json), Qt-free.
#
# AppSettings is the single owner of what the settings are, their defaults,
# how a file on disk is validated/migrated, and how it is saved (atomically).
# The GUI edits a copy (SettingsWindow) and hands back the result; nothing
# else reads or writes ui_settings.json.

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from lorebook.core.card_prices import SUPPORTED_CURRENCIES
from lorebook.core.fileio import atomic_write_json
from lorebook.core.game_types import resolve_game_folder
from lorebook.core.paths import data_path

logger = logging.getLogger(__name__)

SETTINGS_FILE = data_path("ui_settings.json")

# Theme names the UI ships (lorebook.ui.styles.THEMES must match).
THEME_NAMES = ("dark", "light")
DEFAULT_THEME = "dark"


@dataclass
class AppSettings:
    """Every persisted preference, with its default."""

    camera_index: int = 0
    keep_foil_checked: bool = False
    confidence_threshold: float = 0.90
    foil_threshold: float = 0.08
    debug_mode: bool = False
    # Keyed by lowercased Card_Images folder name; one game is active.
    selected_games: dict[str, bool] = field(default_factory=lambda: {"lorcana": True, "riftbound": False})
    # Set codes to search per game; [] means all sets (including future ones).
    selected_sets: dict[str, list[str]] = field(default_factory=lambda: {"lorcana": [], "riftbound": []})
    rotate_display: bool = False
    crop_to_focus: bool = True
    auto_scan: bool = False
    currency: str = "USD"
    theme: str = DEFAULT_THEME

    # ------------------------------------------------------------ queries

    def active_game_key(self) -> str | None:
        """Settings key (lowercased folder name) of the selected game, or None."""
        return next((key for key, on in self.selected_games.items() if on), None)

    def active_game(self) -> str | None:
        """Card_Images folder name of the selected game (real case), or None."""
        key = self.active_game_key()
        if key is None:
            return None
        return resolve_game_folder(key) or key.capitalize()

    def sets_for(self, game: str) -> list[str]:
        """Selected set codes for a game ([] = all sets)."""
        return list(self.selected_sets.get(game.lower(), []))

    def copy(self) -> "AppSettings":
        return replace(
            self,
            selected_games=dict(self.selected_games),
            selected_sets={k: list(v) for k, v in self.selected_sets.items()},
        )

    # ------------------------------------------------------------ (de)serialization

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppSettings":
        """Validate a loaded mapping field by field; bad values fall back to defaults."""
        defaults = cls()
        values: dict[str, Any] = {}
        for f in fields(cls):
            default = getattr(defaults, f.name)
            value = raw.get(f.name, default)
            try:
                values[f.name] = _coerce(f.name, value, default)
            except (TypeError, ValueError) as e:
                logger.error("Error loading setting %s: %s", f.name, e)
                values[f.name] = default
        settings = cls(**values)
        # Older files could select several games; scanning only ever used the
        # first, so keep just that one.
        first = settings.active_game_key()
        settings.selected_games = {k: (k == first) for k in settings.selected_games}
        return settings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str = SETTINGS_FILE) -> "AppSettings":
        """Load from disk; a missing or unreadable file gives the defaults."""
        if not os.path.exists(path):
            logger.info("No settings file found — using defaults")
            return cls()
        try:
            # utf-8-sig tolerates a BOM (editors/PowerShell often add one)
            with open(path, encoding="utf-8-sig") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("settings file is not a JSON object")
        except (OSError, ValueError) as e:  # unreadable, bad JSON, or not an object
            logger.error("Error loading settings file: %s", e)
            return cls()
        return cls.from_dict(raw)

    def save(self, path: str = SETTINGS_FILE) -> None:
        """Persist atomically: a crash mid-save can't truncate the file
        (which would silently reset every preference)."""
        try:
            atomic_write_json(path, self.to_dict(), indent=2)
        except OSError as e:
            logger.error("Error saving settings: %s", e)


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Type-check one loaded value against its default; clamp/normalize it."""
    # bool before int: isinstance(True, int) is True.
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, (int, float)) and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return default
    if name in ("confidence_threshold", "foil_threshold"):
        return max(0.0, min(1.0, float(value)))
    if name == "camera_index":
        return max(0, int(value))
    if name == "selected_games":
        if not isinstance(value, dict):
            return default
        return {str(k).lower(): bool(v) for k, v in value.items()}
    if name == "selected_sets":
        if not isinstance(value, dict):
            return default
        return {str(k).lower(): [str(s) for s in v] for k, v in value.items() if isinstance(v, list)}
    if name == "currency":
        value = str(value).upper()
        return value if value in SUPPORTED_CURRENCIES else default
    if name == "theme":
        value = str(value).lower()
        return value if value in THEME_NAMES else default
    return value
