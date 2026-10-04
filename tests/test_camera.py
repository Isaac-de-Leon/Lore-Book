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

    def isOpened(self):
        return self.opens

    def set(self, prop, value):
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

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", NeverOpensCapture)
    with pytest.raises(RuntimeError):
        open_capture(0, warmup_frames=0)
