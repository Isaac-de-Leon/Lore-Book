# tests/test_cli.py — `python -m lorebook` build/match command line (no model, no camera).

import cv2
import numpy as np
import pytest

import lorebook.core.features as features
from lorebook import cli


class UnitVectorExtractor:
    """Stub extractor: every image maps to the same unit vector."""

    def extract_batch(self, images):
        return [np.array([1.0, 0.0, 0.0, 0.0], np.float32) for _ in images]


def _make_images(folder, count):
    folder.mkdir(parents=True)
    for i in range(count):
        cv2.imwrite(str(folder / f"001-{i:03d}.png"), np.zeros((4, 4, 3), np.uint8))


def test_build_then_match(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _make_images(tmp_path / "Card_Images" / "Lorcana", 3)
    monkeypatch.setattr(features, "get_extractor", lambda backend=None: UnitVectorExtractor())

    assert cli.main(["--game", "Lorcana", "--build"]) == 0
    assert "DB entries: 3" in capsys.readouterr().out

    query = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    monkeypatch.setattr(cli, "extract_features", lambda path: query)
    assert cli.main(["--game", "Lorcana", "--match", "photo.jpg"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"1.0000  001-{i:03d}.png" for i in range(3)]

    # A threshold above every score prints nothing; an unreadable image is an error.
    assert cli.main(["--match", "photo.jpg", "--threshold", "1.5"]) == 0
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(cli, "extract_features", lambda path: None)
    assert cli.main(["--match", "missing.jpg"]) == 1
    assert "missing.jpg" in capsys.readouterr().err


def test_no_action_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "--build" in capsys.readouterr().err
