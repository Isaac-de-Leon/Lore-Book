# tests/test_camera.py — open_capture warm-up and failure behavior (fake VideoCapture).

import numpy as np
import pytest

import lorebook.hardware.camera as camera_mod
from lorebook.hardware.camera import open_capture


class FakeCapture:
    """Stand-in for cv2.VideoCapture that counts reads."""

    opens = True

    def __init__(self, *args, **kwargs):
        self.reads = 0
        self.released = False
        self.props = {}

    def isOpened(self):
        return self.opens

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        self.reads += 1
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class NeverOpensCapture(FakeCapture):
    opens = False


def test_open_capture(monkeypatch):
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeCapture)
    # 1 validity read + 5 warm-up reads — the warm-up must not bail on the
    # first successful frame (auto-exposure needs the full pass).
    cap = open_capture(0, warmup_frames=5)
    assert cap.reads == 6 and not cap.released
    assert open_capture(0, warmup_frames=0).reads == 1

    # "Auto exposure" means 3 on V4L2 (1 is manual there); other backends keep 1.
    for system, expected in (("Linux", 3), ("Windows", 1)):
        monkeypatch.setattr(camera_mod.platform, "system", lambda s=system: s)
        props = open_capture(0, warmup_frames=0).props
        assert props[camera_mod.cv2.CAP_PROP_AUTO_EXPOSURE] == expected

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", NeverOpensCapture)
    with pytest.raises(RuntimeError):
        open_capture(0, warmup_frames=0)


class ScriptedCapture:
    """Stand-in for cv2.VideoCapture whose read() and grab() follow a script."""

    def __init__(self, reads, grab_error_after=None):
        self.reads = list(reads)
        self.grabs = 0
        self.grab_error_after = grab_error_after
        self.released = 0

    def read(self):
        frame = self.reads.pop(0) if self.reads else None
        return frame is not None, frame

    def grab(self):
        if self.grab_error_after is not None and self.grabs >= self.grab_error_after:
            raise camera_mod.cv2.error("device gone")
        self.grabs += 1
        return True

    def release(self):
        self.released += 1


FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


def _live_source(monkeypatch, capture, **kwargs):
    monkeypatch.setattr(camera_mod, "open_capture", lambda index: capture)
    monkeypatch.setattr(camera_mod.time, "sleep", lambda seconds: None)
    return camera_mod.OpenCVCameraSource(0, **kwargs)


def test_live_camera_tolerates_dropped_frames(monkeypatch):
    source = _live_source(monkeypatch, ScriptedCapture([None, None, FRAME]), max_read_failures=2)
    assert source.read() is FRAME  # two dropped frames, then a good one


def test_live_camera_reports_a_dead_feed_and_releases_once(monkeypatch):
    capture = ScriptedCapture([])
    source = _live_source(monkeypatch, capture, max_read_failures=2)
    assert source.read() is None
    source.release()
    source.release()  # safe to call twice
    assert capture.released == 1
    assert source.read() is None
    source.flush()  # nothing to flush once released


def test_live_camera_flush_discards_buffered_frames(monkeypatch):
    capture = ScriptedCapture([FRAME])
    source = _live_source(monkeypatch, capture)
    source.flush()
    assert capture.grabs == camera_mod.OpenCVCameraSource._FLUSH_FRAMES

    capture = ScriptedCapture([FRAME], grab_error_after=1)
    _live_source(monkeypatch, capture).flush()  # a camera error mid-flush is not fatal
    assert capture.grabs == 1


def test_replay_source_skips_unreadable_files_and_can_loop(tmp_path):
    import cv2

    good = tmp_path / "001-001.png"
    cv2.imwrite(str(good), FRAME)
    bad = tmp_path / "002-001.png"
    bad.write_text("not an image")

    source = camera_mod.MockCameraSource([str(bad), str(good)])
    assert source.read().shape == FRAME.shape  # the unreadable file is skipped
    assert source.read() is None  # then the feed ends

    looping = camera_mod.MockCameraSource(str(good), loop=True)  # a single file path
    assert looping.read() is not None and looping.read() is not None
    looping.release()
    assert looping.read() is None

    # A looping feed with nothing readable ends instead of spinning forever.
    assert camera_mod.MockCameraSource([str(bad)], loop=True).read() is None
    # A folder is replayed in name order, supported image types only.
    (tmp_path / "notes.txt").write_text("ignored")
    assert camera_mod.MockCameraSource(str(tmp_path))._paths == [str(good), str(bad)]
