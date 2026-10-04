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


class ShapeExtractor:
    """Records the shape of each frame it sees; always matches card 001-001."""

    def __init__(self):
        self.shapes = []

    def extract(self, img):
        self.shapes.append(img.shape)
        return _unit(0)


def _frames(n=1, shape=(2, 2, 3)):
    """n blank BGR frames."""
    return [np.zeros(shape, np.uint8) for _ in range(n)]


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


def _make_pipeline(camera, extractor, transport, *, rules=None, game="Lorcana", **kwargs):
    return SortPipeline(
        camera=camera,
        transport=transport,
        extractor=extractor,
        feature_db=FEATURE_DB,
        rules=rules if rules is not None else _multi_bin_rules(),
        game=game,
        threshold=0.7,
        **kwargs,
    )


def test_routing():
    camera = ArrayCameraSource(_frames(4))
    # 001-001, 008-002, then no match and an extraction failure
    extractor = SequenceExtractor([_unit(0), _unit(1), _unit(500), None])
    transport = MockTransport()

    outcomes = _make_pipeline(camera, extractor, transport).run()

    assert [o.bin for o in outcomes] == ["bin-1", "bin-8", "reject", "reject"]
    assert [o.filename for o in outcomes] == ["001-001.webp", "008-002.webp", None, None]
    assert [o.matched for o in outcomes] == [True, True, False, False]
    assert transport.history == ["bin-1", "bin-8", "reject", "reject"]
    assert transport.routed["bin-1"] == 1 and transport.routed["reject"] == 2
    assert transport.cards_advanced == 4
    assert transport.homed is True

    camera = ArrayCameraSource(_frames(5))
    outcomes = _make_pipeline(camera, SequenceExtractor([_unit(0)] * 5), MockTransport()).run(max_cards=2)
    assert len(outcomes) == 2


def test_foil_and_broad_rules(monkeypatch):
    """Foil rules apply to matched cards; an unidentified card goes to reject
    even when a broad rule (foil-only, game-only) would otherwise match it."""
    monkeypatch.setattr("lorebook.sorter.pipeline.is_probably_foil", lambda *a, **k: True)
    for rule in (Rule(bin="foils", foil=True), Rule(bin="foils", game="Lorcana")):
        rules = SortRules(rules=[rule], reject_bin="reject")
        camera = ArrayCameraSource(_frames(2))
        extractor = SequenceExtractor([_unit(0), _unit(500)])
        outcomes = _make_pipeline(camera, extractor, MockTransport(), rules=rules).run()
        assert [(o.bin, o.matched, o.is_foil) for o in outcomes][0] == ("foils", True, True)
        assert (outcomes[1].bin, outcomes[1].matched) == ("reject", False)


def test_csv_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run(vectors, **kwargs):
        camera = ArrayCameraSource(_frames(len(vectors)))
        return _make_pipeline(camera, SequenceExtractor(vectors), MockTransport(), **kwargs).run()

    run([_unit(0)], dry_run=True)
    assert not os.path.exists("LorcanaList.csv")

    # Batched CSV writes still record every card, merged into one row.
    run([_unit(0)] * 3, dry_run=False)
    rows = list(csv.reader(open("LorcanaList.csv", encoding="utf-8")))
    assert rows[0] == ["Set Number", "Card Number", "Variant", "Count"]
    assert ["001", "001", "normal", "3"] in rows

    # A game beyond Lorcana/Riftbound writes <Game>List.csv.
    outcomes = run([_unit(0)], game="Pokemon", dry_run=False)
    assert outcomes[0].bin == "bin-1"
    assert ["001", "001", "normal", "1"] in list(csv.reader(open("PokemonList.csv", encoding="utf-8")))

    # The CSV is flushed every csv_flush_interval cards, not only at the end.
    camera = ArrayCameraSource(_frames(2))
    pipeline = _make_pipeline(
        camera, SequenceExtractor([_unit(0), _unit(1)]), MockTransport(),
        game="Riftbound", dry_run=False, csv_flush_interval=1,
    )
    pipeline.process_one(camera.read())
    assert os.path.exists("RiftboundList.csv")


def test_cameras_and_crop(tmp_path):
    import cv2

    from lorebook.core.image_utils import focus_rect

    _, _, fw, fh = focus_rect(720, 1280)
    for crop, expected in ((True, (fh, fw, 3)), (False, (720, 1280, 3))):
        extractor = ShapeExtractor()
        camera = ArrayCameraSource(_frames(shape=(720, 1280, 3)))
        _make_pipeline(camera, extractor, MockTransport(), crop_to_focus=crop).run()
        assert extractor.shapes == [expected]

    # MockCameraSource skips many unreadable files iteratively (the old
    # recursive skip would blow the stack at scale), then ends the feed.
    for i in range(50):
        (tmp_path / f"bad{i:03d}.png").write_text("not an image")
    cv2.imwrite(str(tmp_path / "zzz-good.png"), np.zeros((4, 4, 3), np.uint8))
    for source in (str(tmp_path), sorted(str(p) for p in tmp_path.iterdir())):
        cam = MockCameraSource(source)
        assert cam.read() is not None
        assert cam.read() is None

    # All unreadable with loop=True: one full pass, then a clean end.
    cam = MockCameraSource([str(tmp_path / "bad000.png")], loop=True)
    assert cam.read() is None
