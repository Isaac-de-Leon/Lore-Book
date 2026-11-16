# tests/test_sorter_cli.py — `python -m lorebook.sorter` wiring (no camera, no model).

import csv
import json
import logging
import runpy
import sys

import cv2
import numpy as np
import pytest

from lorebook.core import feature_cache
from lorebook.sorter import __main__ as sorter_cli

VEC = np.array([1.0, 0.0, 0.0, 0.0], np.float32)


class FakeExtractor:
    """Every frame is card 001-001."""

    def ensure_ready(self):
        pass

    def extract(self, img):
        return VEC


@pytest.fixture
def sorter_setup(tmp_path, monkeypatch, caplog):
    """A working folder with two card images, a feature cache, a rules file and a fake model."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)
    images = tmp_path / "replay"
    images.mkdir()
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(images / name), np.zeros((4, 4, 3), np.uint8))

    cache_file = feature_cache._cache_path("Lorcana")
    feature_cache._init_db(cache_file)
    with feature_cache.sqlite3.connect(cache_file) as conn:
        conn.execute("INSERT INTO features (filename, vector) VALUES (?, ?)", ("001-001.webp", VEC.tobytes()))

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{"set_code": "001", "bin": "bin-1"}]}), encoding="utf-8")
    monkeypatch.setattr(sorter_cli, "get_extractor", lambda backend: FakeExtractor())
    return ["--game", "Lorcana", "--source", str(images), "--rules", str(rules)]


def test_replay_is_a_dry_run_by_default(sorter_setup, tmp_path, caplog):
    assert sorter_cli.main(sorter_setup) == 0
    assert "2 card(s) processed" in caplog.text and "bin-1" in caplog.text
    assert not (tmp_path / "LorcanaList.csv").exists()


def test_no_dry_run_records_the_cards(sorter_setup, tmp_path):
    assert sorter_cli.main([*sorter_setup, "--no-dry-run", "--max-cards", "1"]) == 0
    with open(tmp_path / "LorcanaList.csv", newline="", encoding="utf-8") as f:
        assert ["001", "001", "normal", "1"] in list(csv.reader(f))


def test_without_rules_everything_goes_to_the_reject_bin(sorter_setup, caplog):
    args = sorter_setup[:4]  # no --rules
    assert sorter_cli.main(args) == 0
    assert "every card routes to 'reject'" in caplog.text


def test_missing_feature_cache_explains_how_to_build_it(sorter_setup, caplog):
    assert sorter_cli.main(["--game", "Riftbound", "--source", sorter_setup[3]]) == 1
    assert "python -m lorebook --game Riftbound --build" in caplog.text


def test_missing_source_path_is_an_error(sorter_setup, caplog):
    assert sorter_cli.main(["--game", "Lorcana", "--source", "no/such/folder"]) == 1
    assert "Source path not found" in caplog.text


def test_unusable_model_fails_fast_and_releases_the_camera(sorter_setup, monkeypatch, caplog):
    class Broken(FakeExtractor):
        def ensure_ready(self):
            raise FileNotFoundError("tflite model not found")

    released = []
    real_source = sorter_cli.MockCameraSource

    class TrackedSource(real_source):
        def release(self):
            released.append(True)
            super().release()

    monkeypatch.setattr(sorter_cli, "MockCameraSource", TrackedSource)
    monkeypatch.setattr(sorter_cli, "get_extractor", lambda backend: Broken())
    assert sorter_cli.main(sorter_setup) == 1
    assert "is not usable: tflite model not found" in caplog.text and released == [True]


def test_camera_that_will_not_open_is_an_error(sorter_setup, monkeypatch, caplog):
    def no_camera(index):
        raise RuntimeError("Failed to open camera 2 with any backend")

    monkeypatch.setattr(sorter_cli, "OpenCVCameraSource", no_camera)
    assert sorter_cli.main(["--game", "Lorcana", "--source", "camera", "--camera-index", "2"]) == 1
    assert "Could not open camera 2" in caplog.text


def test_python_dash_m_lorebook_runs_the_build_and_match_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lorebook"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("lorebook", run_name="__main__")
    assert exit_info.value.code == 2 and "--build" in capsys.readouterr().err
