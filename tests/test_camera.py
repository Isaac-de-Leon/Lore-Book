# tests/test_camera.py — open_capture warm-up and failure behavior (fake VideoCapture).

import numpy as np
import pytest

import lorebook.hardware.camera as camera_mod
from lorebook.hardware.camera import open_capture


class FakeCapture:
    """Stand-in for cv2.VideoCapture that counts reads."""

    def __init__(self, *args, **kwargs):
        self.reads = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        self.reads += 1
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class NeverOpensCapture(FakeCapture):
    def isOpened(self):
        return False


class TestOpenCapture:
    def test_warmup_reads_all_frames(self, monkeypatch):
        created = []

        def factory(*args, **kwargs):
            cap = FakeCapture()
            created.append(cap)
            return cap

        monkeypatch.setattr(camera_mod.cv2, "VideoCapture", factory)
        cap = open_capture(0, warmup_frames=5)
        # 1 validity read + 5 warm-up reads — the warm-up must not bail on
        # the first successful frame (auto-exposure needs the full pass).
        assert cap.reads == 6
        assert not cap.released

    def test_zero_warmup_frames(self, monkeypatch):
        monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeCapture)
        cap = open_capture(0, warmup_frames=0)
        assert cap.reads == 1

    def test_raises_when_no_backend_opens(self, monkeypatch):
        monkeypatch.setattr(camera_mod.cv2, "VideoCapture", NeverOpensCapture)
        with pytest.raises(RuntimeError):
            open_capture(0, warmup_frames=0)
