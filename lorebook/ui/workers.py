# workers.py — QObject workers that run on their own QThread (ui/threads.py).
#
# Each worker exposes slots the GUI calls through queued signals, and reports
# back only through its own signals (delivered on the GUI thread). Slots are
# boundaries: whatever happens inside, the GUI must hear back, so each one
# ends by emitting its result even on an unexpected error.

import contextlib
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from lorebook.core.build import BuildReport, refresh_and_build
from lorebook.core.features import extract_features, get_extractor
from lorebook.core.matching import MatchIndex
from lorebook.hardware.camera import open_capture

logger = logging.getLogger(__name__)


class BuildWorker(QObject):
    """Runs core.build.refresh_and_build for a list of games.

    Cancellation is a threading.Event the GUI sets directly (thread-safe);
    a queued "cancel" signal couldn't be delivered while the job is running.
    """

    status = Signal(str)
    progress = Signal(int)       # 0–100, or -1 for indeterminate
    done = Signal(object)        # BuildReport

    @Slot(object, object)
    def run(self, games: list[str], cancel_event: threading.Event) -> None:
        try:
            report = refresh_and_build(
                games, cancel_event=cancel_event, status=self.status.emit, progress=self.progress.emit
            )
        except Exception:  # noqa: BLE001 — boundary: the GUI must always get done()
            logger.exception("Database build job failed")
            report = BuildReport(failed=list(games), cancelled=cancel_event.is_set())
        self.done.emit(report)


class CameraWorker(QObject):
    """Owns the camera: opens it and reads frames on its own thread.

    Commands arrive as queued slot calls (start/stop), so they are handled in
    order between frame reads — no generation tokens needed. Frames go out via
    frame_ready one at a time: the next is only sent after the GUI calls
    frame_consumed(), so a slow GUI drops frames instead of queueing a backlog.

    States (state_changed): "starting" → "running" → "idle".
    """

    state_changed = Signal(str)
    frame_ready = Signal(object)   # np.ndarray (BGR); never mutated after emit
    failed = Signal(str)

    MAX_READ_FAILS = 20            # consecutive bad reads before giving up
    READ_INTERVAL_MS = 5           # read() itself blocks until the next frame

    def __init__(self) -> None:
        super().__init__()
        self._cap: cv2.VideoCapture | None = None
        self._timer: QTimer | None = None
        self._fails = 0
        # The only state shared with the GUI thread (Event is thread-safe).
        self._frame_pending = threading.Event()

    def frame_consumed(self) -> None:
        """GUI thread: the last frame was handled; the next one may be sent."""
        self._frame_pending.clear()

    @Slot(int)
    def start(self, camera_index: int) -> None:
        self._close()
        self.state_changed.emit("starting")
        try:
            cap = open_capture(camera_index)
        # Boundary: whatever happens, the GUI must hear back (it sits in
        # "Starting…" until state_changed/failed arrives).
        except Exception as e:  # noqa: BLE001
            self.state_changed.emit("idle")
            self.failed.emit(str(e))
            return
        self._cap = cap
        self._fails = 0
        self._frame_pending.clear()
        if self._timer is None:  # created here so it lives on this thread
            self._timer = QTimer(self)
            self._timer.setInterval(self.READ_INTERVAL_MS)
            self._timer.timeout.connect(self._read_frame)
        self._timer.start()
        self.state_changed.emit("running")

    @Slot()
    def stop(self) -> None:
        self._close()
        self.state_changed.emit("idle")

    def release(self) -> None:
        """Free the camera after the thread has stopped (shutdown path).

        Called from the GUI thread once this worker's thread is finished, so it
        must not touch the timer (a QTimer may only be used from its thread).
        """
        self._release_capture()

    def _close(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._release_capture()

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                with contextlib.suppress(cv2.error):  # a dead camera may refuse
                    self._cap.release()
            finally:
                self._cap = None

    def _read_frame(self) -> None:
        if self._cap is None:
            return
        try:
            ok, frame = self._cap.read()
        except cv2.error as e:
            logger.debug("Camera read error: %s", e)
            ok, frame = False, None
        if not ok or frame is None or frame.size == 0:
            self._fails += 1
            if self._fails > self.MAX_READ_FAILS:
                self._close()
                self.state_changed.emit("idle")
                self.failed.emit("Camera not delivering frames.")
            return
        self._fails = 0
        if not self._frame_pending.is_set():
            self._frame_pending.set()
            self.frame_ready.emit(frame)


@dataclass
class ScanResult:
    game: str
    matches: list[tuple[str, float]] | None   # None: extraction failed
    error: str | None = None


class ScanWorker(QObject):
    """Feature extraction + matching on a long-lived thread.

    Requests carry a token; the GUI drops results whose token is stale (a
    newer scan started). Model use is additionally serialized by
    features._model_lock, so a scan during a DB build waits for the current
    batch instead of racing it.
    """

    done = Signal(int, object)     # (token, ScanResult)

    @Slot()
    def warm_up(self) -> None:
        """Load the model now so the first scan doesn't stall (optional)."""
        try:
            get_extractor("keras").ensure_ready()
        except Exception as e:  # noqa: BLE001 — optional warm-up; the scan reports real errors
            logger.warning("Background model load failed (will retry on scan): %s", e)

    @Slot(int, object, object, float, str)
    def scan(self, token: int, img_bgr: np.ndarray, index: MatchIndex, threshold: float, game: str) -> None:
        """Uses the index it was handed, so a game switch mid-scan can't mix games."""
        try:
            features = extract_features(img_bgr)
            matches = index.find(features, threshold=threshold) if features is not None else None
            result = ScanResult(game, matches)
        # Boundary: the GUI stays in "Scanning…" until done() arrives.
        except Exception as e:  # noqa: BLE001
            logger.exception("Scan failed")
            result = ScanResult(game, None, str(e))
        self.done.emit(token, result)
