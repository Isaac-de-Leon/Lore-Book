# Tests for lorebook.core.paths — the writable-data root resolver.

import os
import sys

from lorebook.core.game_types import csv_for_game
from lorebook.core.paths import (
    configure_frozen_environment,
    data_dir,
    data_path,
    is_frozen,
)


class TestFromSource:
    """Not frozen, no override: byte-identical to the legacy CWD-relative layout."""

    def test_data_dir_is_cwd(self, monkeypatch):
        monkeypatch.delenv("LOREBOOK_DATA_DIR", raising=False)
        assert data_dir() == "."

    def test_data_path_returns_bare_name(self, monkeypatch):
        monkeypatch.delenv("LOREBOOK_DATA_DIR", raising=False)
        assert data_path("ui_settings.json") == "ui_settings.json"
        assert data_path("Card_Images") == "Card_Images"
        assert data_path("logs", "x.log") == os.path.join("logs", "x.log")

    def test_not_frozen(self):
        assert is_frozen() is False

    def test_configure_frozen_environment_noop(self, monkeypatch):
        monkeypatch.delenv("KERAS_HOME", raising=False)
        configure_frozen_environment()
        assert "KERAS_HOME" not in os.environ


class TestEnvOverride:
    def test_data_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOREBOOK_DATA_DIR", str(tmp_path))
        assert data_dir() == str(tmp_path)

    def test_data_path_prefixes_and_creates_root(self, monkeypatch, tmp_path):
        root = tmp_path / "lorebook-data"
        monkeypatch.setenv("LOREBOOK_DATA_DIR", str(root))
        assert data_path("LorcanaList.csv") == str(root / "LorcanaList.csv")
        assert root.is_dir()

    def test_csv_for_game_respects_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOREBOOK_DATA_DIR", str(tmp_path))
        assert csv_for_game("Lorcana") == str(tmp_path / "LorcanaList.csv")


class TestFrozen:
    def test_frozen_linux_uses_xdg(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LOREBOOK_DATA_DIR", raising=False)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        if sys.platform == "win32":
            monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
            assert data_dir() == os.path.join(str(tmp_path), "LoreBook")
        else:
            monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
            assert data_dir() == os.path.join(str(tmp_path), "lorebook")

    def test_frozen_sets_keras_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KERAS_HOME", raising=False)
        monkeypatch.setenv("LOREBOOK_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        configure_frozen_environment()
        assert os.environ["KERAS_HOME"] == os.path.join(str(tmp_path), "keras")

    def test_frozen_keeps_existing_keras_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KERAS_HOME", "/already/set")
        monkeypatch.setenv("LOREBOOK_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        configure_frozen_environment()
        assert os.environ["KERAS_HOME"] == "/already/set"


class TestVersion:
    def test_version_is_importable_and_sane(self):
        import lorebook

        parts = lorebook.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)
