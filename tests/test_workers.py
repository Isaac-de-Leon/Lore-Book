# tests/test_workers.py — the camera, scan and build workers, called directly (no threads).
#
# Each worker's slots are plain methods, so they are exercised on the test
# thread with fakes for the camera, the model and the build job. Running them
# on a real WorkerThread is covered by tests/test_gui_smoke.py.

import os
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
from PySide6.QtWidgets import QApplication

from lorebook.core import feature_cache
from lorebook.core.build import BuildReport
from lorebook.ui import workers
from lorebook.ui.workers import BuildWorker, CameraState, CameraWorker, ScanResult, ScanWorker

FRAME = np.zeros((4, 4, 3), np.uint8)


@pytest.fixture(autouse=True)
def _qt_app():
    return QApplication.instance() or QApplication([])


class FakeCapture:
    """Stands in for cv2.VideoCapture: replays a list of read() results."""

    def __init__(self, reads):
        self.reads = list(reads)
        self.released = False

    def read(self):
        result = self.reads.pop(0) if self.reads else None
        if isinstance(result, Exception):
            raise result
        return result is not None, result

    def release(self):
        self.released = True


def _camera(monkeypatch, capture):
    """A CameraWorker wired to `capture`, with every signal recorded."""
    monkeypatch.setattr(workers, "open_capture", lambda index: capture)
    worker = CameraWorker()
    seen = {"states": [], "frames": [], "failures": []}
    worker.state_changed.connect(seen["states"].append)
    worker.frame_ready.connect(seen["frames"].append)
    worker.failed.connect(seen["failures"].append)
    return worker, seen


def test_camera_sends_one_frame_at_a_time(monkeypatch):
    capture = FakeCapture([FRAME, FRAME, FRAME])
    worker, seen = _camera(monkeypatch, capture)

    worker.start(0)
    assert seen["states"] == [CameraState.STARTING, CameraState.RUNNING]
    worker._read_frame()
    worker._read_frame()  # the GUI has not taken the first frame yet: dropped, not queued
    assert len(seen["frames"]) == 1
    worker.frame_consumed()
    worker._read_frame()
    assert len(seen["frames"]) == 2

    worker.stop()
    assert seen["states"][-1] == CameraState.IDLE and capture.released
    worker._read_frame()  # a late timer tick after stop does nothing
    worker.release()  # safe to call again once the thread has stopped
    assert len(seen["frames"]) == 2 and seen["failures"] == []


def test_camera_gives_up_after_too_many_failed_reads(monkeypatch):
    monkeypatch.setattr(CameraWorker, "MAX_READ_FAILS", 2)
    capture = FakeCapture([None, cv2.error("driver glitch"), FRAME, None, None, None])
    worker, seen = _camera(monkeypatch, capture)
    worker.start(0)

    worker._read_frame()
    worker._read_frame()
    worker._read_frame()  # a good frame resets the failure count
    assert len(seen["frames"]) == 1 and seen["failures"] == []
    for _ in range(3):
        worker._read_frame()
    assert seen["failures"] == ["Camera not delivering frames."]
    assert seen["states"][-1] == CameraState.IDLE and capture.released


def test_camera_reports_a_camera_that_will_not_open(monkeypatch):
    def no_camera(index):
        raise RuntimeError("Failed to open camera 3 with any backend")

    monkeypatch.setattr(workers, "open_capture", no_camera)
    worker = CameraWorker()
    states, failures = [], []
    worker.state_changed.connect(states.append)
    worker.failed.connect(failures.append)
    worker.start(3)
    assert states == [CameraState.STARTING, CameraState.IDLE]  # the GUI never stays in "Starting…"
    assert failures == ["Failed to open camera 3 with any backend"]


def test_build_worker_forwards_progress_and_always_reports_done(monkeypatch):
    def fake_build(games, *, cancel_event, status, progress):
        status("Building Lorcana database…")
        progress(40)
        return BuildReport(prices_refreshed=True)

    monkeypatch.setattr(workers, "refresh_and_build", fake_build)
    worker = BuildWorker()
    statuses, progress, reports = [], [], []
    worker.status.connect(statuses.append)
    worker.progress.connect(progress.append)
    worker.done.connect(reports.append)
    worker.run(["Lorcana"], threading.Event())
    assert statuses == ["Building Lorcana database…"] and progress == [40]
    assert reports[0].ok and reports[0].prices_refreshed

    # The job blowing up still ends with done(): every game failed, cancel flag preserved.
    def broken(games, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(workers, "refresh_and_build", broken)
    cancel = threading.Event()
    cancel.set()
    worker.run(["Lorcana", "Riftbound"], cancel)
    assert reports[1].failed == ["Lorcana", "Riftbound"] and reports[1].cancelled


def _seed_cache(game, vectors):
    cache_file = feature_cache._cache_path(game)
    feature_cache._init_db(cache_file)
    with feature_cache.sqlite3.connect(cache_file) as conn:
        conn.executemany(
            "INSERT INTO features (filename, vector) VALUES (?, ?)",
            [(name, vec.tobytes()) for name, vec in vectors.items()],
        )


def test_scan_worker_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vec = np.zeros(8, np.float32)
    vec[0] = 1.0
    worker = ScanWorker()
    results = []
    worker.done.connect(lambda token, result: results.append((token, result)))

    # No cache for the game yet.
    monkeypatch.setattr(workers, "extract_features", lambda img: vec)
    worker.scan(1, FRAME, 0.5, "Lorcana")
    assert results[-1] == (1, ScanResult("Lorcana", None, no_database=True))

    # After a build the worker is told to reload, then finds the card.
    _seed_cache("Lorcana", {"001-001.webp": vec})
    worker.invalidate()
    worker.scan(2, FRAME, 0.5, "Lorcana")
    token, result = results[-1]
    assert token == 2 and result.matches == [("001-001.webp", pytest.approx(1.0))]
    assert result.error is None and not result.no_database

    # The image can't be turned into a vector: matches is None, not an empty list.
    monkeypatch.setattr(workers, "extract_features", lambda img: None)
    worker.scan(3, FRAME, 0.5, "Lorcana")
    assert results[-1][1].matches is None and results[-1][1].error is None

    # An unexpected error still produces a result, so the GUI leaves "Scanning…".
    def crash(img):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(workers, "extract_features", crash)
    worker.scan(4, FRAME, 0.5, "Lorcana")
    assert results[-1][0] == 4 and results[-1][1].error == "model exploded"


def test_scan_worker_warm_up_and_heatmap_never_raise(monkeypatch):
    class BrokenExtractor:
        def ensure_ready(self):
            raise OSError("weights download failed")

    monkeypatch.setattr(workers, "get_extractor", lambda: BrokenExtractor())
    worker = ScanWorker()
    worker.warm_up()  # optional: a failure is logged and retried on the first scan

    heatmaps = []
    worker.heatmap_ready.connect(heatmaps.append)
    tinted = np.full((4, 4, 3), 7, np.uint8)
    monkeypatch.setattr(workers, "visualize_activation_overlay", lambda frame: tinted)
    worker.heatmap(FRAME)
    assert heatmaps[-1] is tinted

    def crash(frame):
        raise RuntimeError("no model")

    monkeypatch.setattr(workers, "visualize_activation_overlay", crash)
    worker.heatmap(FRAME)
    assert heatmaps[-1] is FRAME  # the plain frame is shown instead
