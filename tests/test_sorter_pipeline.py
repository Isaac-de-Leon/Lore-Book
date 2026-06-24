# tests/test_sorter_pipeline.py
# End-to-end tests for the headless sort pipeline using mock hardware and a
# fake extractor. No camera, no motors, no TensorFlow.

import csv
import os
import sys
import unittest.mock

import numpy as np
import pytest

# Stub TF/Keras before any lorebook import that might transitively touch them.
_tf_mock = unittest.mock.MagicMock()
for _mod in ["tensorflow", "keras", "keras.applications",
             "keras.applications.mobilenet_v2", "keras.models"]:
    sys.modules.setdefault(_mod, _tf_mock)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorebook.hardware.camera import CameraSource, MockCameraSource
from lorebook.hardware.transport import MockTransport
from lorebook.sorter.pipeline import SortPipeline
from lorebook.sorter.rules import Rule, SortRules


def _unit(idx: int) -> np.ndarray:
    """A 1280-dim L2-normalized basis vector (matching the real feature shape)."""
    v = np.zeros(1280, dtype=np.float32)
    v[idx] = 1.0
    return v


# Reference DB: each card filename maps to a distinct orthogonal unit vector.
FEATURE_DB = {
    "001-001.webp": _unit(0),
    "008-002.webp": _unit(1),
}


class ArrayCameraSource(CameraSource):
    """Yields a preset list of BGR frames, then ends the feed."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame


class SequenceExtractor:
    """Returns preset vectors in call order (None means extraction failure)."""

    def __init__(self, vectors):
        self._vectors = list(vectors)
        self._i = 0

    def extract(self, _img):
        v = self._vectors[self._i]
        self._i += 1
        return v


@pytest.fixture(autouse=True)
def _foil_off(monkeypatch):
    """Default foil detection to False so tests control it explicitly."""
    monkeypatch.setattr("lorebook.sorter.pipeline.is_probably_foil", lambda *a, **k: False)


def _multi_bin_rules():
    return SortRules(
        rules=[
            Rule(bin="bin-1", set_code="001"),
            Rule(bin="bin-8", set_code="008"),
        ],
        reject_bin="reject",
    )


def _make_pipeline(camera, extractor, transport, **kwargs):
    return SortPipeline(
        camera=camera,
        transport=transport,
        extractor=extractor,
        feature_db=FEATURE_DB,
        rules=_multi_bin_rules(),
        game="Lorcana",
        threshold=0.7,
        **kwargs,
    )


class TestPipelineRouting:
    def test_each_card_routed_to_expected_bin(self):
        frames = [np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8)]
        camera = ArrayCameraSource(frames)
        extractor = SequenceExtractor([_unit(0), _unit(1)])  # → 001-001, 008-002
        transport = MockTransport()

        outcomes = _make_pipeline(camera, extractor, transport).run()

        assert [o.bin for o in outcomes] == ["bin-1", "bin-8"]
        assert [o.filename for o in outcomes] == ["001-001.webp", "008-002.webp"]
        assert all(o.matched for o in outcomes)
        assert transport.history == ["bin-1", "bin-8"]
        assert transport.routed["bin-1"] == 1 and transport.routed["bin-8"] == 1
        assert transport.cards_advanced == 2
        assert transport.homed is True

    def test_no_match_routes_to_reject(self):
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(500)])  # orthogonal to all refs
        transport = MockTransport()

        outcomes = _make_pipeline(camera, extractor, transport).run()

        assert outcomes[0].bin == "reject"
        assert outcomes[0].matched is False
        assert outcomes[0].filename is None

    def test_extraction_failure_routes_to_reject(self):
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([None])
        transport = MockTransport()

        outcomes = _make_pipeline(camera, extractor, transport).run()

        assert outcomes[0].bin == "reject"
        assert outcomes[0].matched is False

    def test_foil_rule_applied(self, monkeypatch):
        monkeypatch.setattr("lorebook.sorter.pipeline.is_probably_foil", lambda *a, **k: True)
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(0)])
        transport = MockTransport()
        rules = SortRules(rules=[Rule(bin="foils", foil=True)], reject_bin="reject")

        pipeline = SortPipeline(
            camera=camera, transport=transport, extractor=extractor,
            feature_db=FEATURE_DB, rules=rules, game="Lorcana", threshold=0.7,
        )
        outcomes = pipeline.run()
        assert outcomes[0].bin == "foils"
        assert outcomes[0].is_foil is True

    def test_max_cards_limits_run(self):
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(5)]
        camera = ArrayCameraSource(frames)
        extractor = SequenceExtractor([_unit(0)] * 5)
        transport = MockTransport()

        outcomes = _make_pipeline(camera, extractor, transport).run(max_cards=2)
        assert len(outcomes) == 2


class TestPipelineCsv:
    def test_dry_run_does_not_write_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(0)])
        transport = MockTransport()

        _make_pipeline(camera, extractor, transport, dry_run=True).run()
        assert not os.path.exists(tmp_path / "LorcanaList.csv")

    def test_no_dry_run_writes_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(0)])  # → 001-001.webp
        transport = MockTransport()

        _make_pipeline(camera, extractor, transport, dry_run=False).run()

        csv_path = tmp_path / "LorcanaList.csv"
        assert csv_path.exists()
        rows = list(csv.reader(csv_path.open(encoding="utf-8")))
        assert rows[0] == ["Set Number", "Card Number", "Variant", "Count"]
        assert ["001", "001", "normal", "1"] in rows


class TestMockCameraSource:
    def test_reads_images_from_folder(self, tmp_path):
        import cv2

        for name in ("a.png", "b.png"):
            cv2.imwrite(str(tmp_path / name), np.zeros((4, 4, 3), np.uint8))
        cam = MockCameraSource(str(tmp_path))

        assert cam.read() is not None
        assert cam.read() is not None
        assert cam.read() is None  # feed exhausted
