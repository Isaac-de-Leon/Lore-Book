# paths.py — writable-data root resolution.
#
# From source (not frozen, no override) every path stays exactly as it always
# was: a bare CWD-relative name. Frozen (installed) builds write to
# %LOCALAPPDATA%\LoreBook on Windows / $XDG_DATA_HOME/lorebook elsewhere, so
# the app works when installed to a read-only program directory.
#
# LOREBOOK_DATA_DIR overrides both. Set it before launch — module constants
# like BASE_DATABASE_PATH resolve at import time.
#
# This module must not import anything else from lorebook: it sits at the
# bottom of the package import graph.

import os
import sys

APP_NAME = "LoreBook"


def is_frozen() -> bool:
    """True when running from a PyInstaller (or similar) frozen bundle."""
    return bool(getattr(sys, "frozen", False))


def data_dir() -> str:
    """Root directory for all writable app data ("." when running from source)."""
    override = os.environ.get("LOREBOOK_DATA_DIR")
    if override:
        return override
    if not is_frozen():
        return "."
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, APP_NAME)
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "lorebook")


def data_path(*parts: str) -> str:
    """
    Join parts onto the data root.

    Returns the bare relative join when the root is "." so from-source
    behavior (and test string equality on filenames) is byte-identical to
    the pre-refactor CWD-relative layout.
    """
    root = data_dir()
    if root == ".":
        return os.path.join(*parts)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, *parts)


def configure_frozen_environment() -> None:
    """
    Point the Keras weights cache into the per-user data dir for frozen
    builds, so the MobileNetV2 download persists across app updates.
    Call before anything imports TensorFlow/Keras.
    """
    if is_frozen():
        os.environ.setdefault("KERAS_HOME", os.path.join(data_dir(), "keras"))
