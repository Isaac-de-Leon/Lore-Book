"""Lore-Book — desktop trading-card scanner package.

Subpackages:
    lorebook.core — game-agnostic logic (matching, features, database, CSV).
    lorebook.ui   — PySide6 widgets (MainWindow, SettingsWindow, styles).
"""

# Single source of truth for the app version: pyproject.toml reads it via
# [tool.setuptools.dynamic], the UI shows it in the window title, and the
# release workflow checks it against the git tag. Keep this module free of
# imports — CI and build scripts import it before any dependencies exist.
__version__ = "0.2.0"
