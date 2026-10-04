# Tests for lorebook.core.paths — the writable-data root resolver.

import os
import sys

import lorebook
from lorebook.core.game_types import csv_for_game
from lorebook.core.paths import configure_frozen_environment, data_dir, data_path, is_frozen


def test_data_paths(monkeypatch, tmp_path):
    # From source (not frozen, no override): byte-identical to the legacy
    # CWD-relative layout, and KERAS_HOME is left alone.
    monkeypatch.delenv("LOREBOOK_DATA_DIR", raising=False)
    monkeypatch.delenv("KERAS_HOME", raising=False)
    assert is_frozen() is False and data_dir() == "."
    assert data_path("ui_settings.json") == "ui_settings.json"
    assert data_path("logs", "x.log") == os.path.join("logs", "x.log")
    configure_frozen_environment()
    assert "KERAS_HOME" not in os.environ

    # LOREBOOK_DATA_DIR overrides the root (created on demand); collection
    # CSVs follow it at call time.
    root = tmp_path / "lorebook-data"
    monkeypatch.setenv("LOREBOOK_DATA_DIR", str(root))
    assert data_path("LorcanaList.csv") == str(root / "LorcanaList.csv") and root.is_dir()
    assert csv_for_game("Lorcana") == str(root / "LorcanaList.csv")

    # Frozen: KERAS_HOME moves into the data dir unless already set...
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    configure_frozen_environment()
    assert os.environ["KERAS_HOME"] == os.path.join(str(root), "keras")
    monkeypatch.setenv("KERAS_HOME", "/already/set")
    configure_frozen_environment()
    assert os.environ["KERAS_HOME"] == "/already/set"
    # ...and without an override the per-user OS location is used.
    monkeypatch.delenv("LOREBOOK_DATA_DIR")
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert data_dir() == os.path.join(str(tmp_path), "LoreBook")
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert data_dir() == os.path.join(str(tmp_path), "lorebook")

    parts = lorebook.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
