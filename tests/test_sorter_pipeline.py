# tests/test_sorter_pipeline.py
# End-to-end tests for the headless sort pipeline using mock hardware and a
# fake extractor. No camera, no motors, no TensorFlow.

import csv
import os

import numpy as np
import pytest

# TF/Keras stubbing and sys.path setup happen in tests/conftest.py.

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

    def test_unmatched_card_not_captured_by_foil_rule(self, monkeypatch):
        """An unidentified card must go to reject even if a foil-only rule would match it."""
        monkeypatch.setattr("lorebook.sorter.pipeline.is_probably_foil", lambda *a, **k: True)
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(500)])  # orthogonal to all refs → no match
        transport = MockTransport()
        rules = SortRules(rules=[Rule(bin="foils", foil=True)], reject_bin="reject")

        pipeline = SortPipeline(
            camera=camera, transport=transport, extractor=extractor,
            feature_db=FEATURE_DB, rules=rules, game="Lorcana", threshold=0.7,
        )
        outcomes = pipeline.run()

        assert outcomes[0].matched is False
        assert outcomes[0].bin == "reject"  # not "foils"

    def test_unmatched_card_not_captured_by_game_rule(self):
        """A game-only catch-all rule must not swallow unidentified cards."""
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(500)])
        transport = MockTransport()
        rules = SortRules(rules=[Rule(bin="keep", game="Lorcana")], reject_bin="reject")

        pipeline = SortPipeline(
            camera=camera, transport=transport, extractor=extractor,
            feature_db=FEATURE_DB, rules=rules, game="Lorcana", threshold=0.7,
        )
        assert pipeline.run()[0].bin == "reject"

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

    def test_repeat_cards_accumulate_in_one_row(self, tmp_path, monkeypatch):
        """Batched CSV writes still record every card, merged into one row."""
        monkeypatch.chdir(tmp_path)
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
        camera = ArrayCameraSource(frames)
        extractor = SequenceExtractor([_unit(0)] * 3)  # same card three times
        transport = MockTransport()

        _make_pipeline(camera, extractor, transport, dry_run=False).run()

        rows = list(csv.reader((tmp_path / "LorcanaList.csv").open(encoding="utf-8")))
        assert ["001", "001", "normal", "3"] in rows

    def test_flush_interval_writes_mid_run(self, tmp_path, monkeypatch):
        """The CSV is flushed every csv_flush_interval cards, not only at the end."""
        monkeypatch.chdir(tmp_path)
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8) for _ in range(2)])
        extractor = SequenceExtractor([_unit(0), _unit(1)])
        transport = MockTransport()
        pipeline = _make_pipeline(
            camera, extractor, transport, dry_run=False, csv_flush_interval=1
        )

        pipeline.process_one(camera.read())
        assert (tmp_path / "LorcanaList.csv").exists()  # flushed after 1 card

    def test_new_game_gets_its_own_csv(self, tmp_path, monkeypatch):
        """A game beyond Lorcana/Riftbound writes <Game>List.csv, not LorcanaList.csv."""
        monkeypatch.chdir(tmp_path)
        camera = ArrayCameraSource([np.zeros((2, 2, 3), np.uint8)])
        extractor = SequenceExtractor([_unit(0)])
        transport = MockTransport()

        pipeline = SortPipeline(
            camera=camera, transport=transport, extractor=extractor,
            feature_db=FEATURE_DB, rules=_multi_bin_rules(), game="Pokemon",
            threshold=0.7, dry_run=False,
        )
        outcomes = pipeline.run()

        assert outcomes[0].bin == "bin-1"
        assert not os.path.exists(tmp_path / "LorcanaList.csv")
        rows = list(csv.reader((tmp_path / "PokemonList.csv").open(encoding="utf-8")))
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

    def test_skips_unreadable_files_without_recursion(self, tmp_path):
        import cv2

        # Many corrupt "images" followed by one valid one: must skip them all
        # iteratively (the old recursive skip would blow the stack at scale).
        paths = []
        for i in range(50):
            bad = tmp_path / f"bad{i:03d}.png"
            bad.write_text("not an image")
            paths.append(str(bad))
        good = tmp_path / "zzz-good.png"
        cv2.imwrite(str(good), np.zeros((4, 4, 3), np.uint8))
        paths.append(str(good))

        cam = MockCameraSource(paths)
        assert cam.read() is not None  # the good frame, after skipping 50
        assert cam.read() is None

    def test_all_unreadable_with_loop_terminates(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_text("not an image")

        cam = MockCameraSource([str(bad)], loop=True)
        assert cam.read() is None  # one full pass, then clean end — no infinite recursion
