# camera.py — camera abstraction for the headless sorter.
#
# `open_capture()` holds the platform-specific OpenCV backend-probe + warm-up
# logic, factored verbatim out of lorebook/ui/main_window.py's start_camera so
# the GUI and the headless sorter share one camera-open code path. The
# CameraSource hierarchy then wraps that for the sorter: OpenCVCameraSource is
# the real path (a USB cam on the Pi via V4L2 works today), and
# MockCameraSource replays images from disk for dry runs and tests — no camera,
# no Qt.

import logging
import os
import platform
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import cv2
import numpy as np

from lorebook.core.game_types import SUPPORTED_EXTS

logger = logging.getLogger(__name__)


def _backend_candidates() -> List[tuple]:
    """Return (flag, name) camera backends to try, in priority order per OS."""
    system = platform.system()
    if system == "Windows":
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Default"),
        ]
    if system == "Linux":
        return [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Default"),
        ]
    # macOS and others
    return [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Default"),
    ]


def open_capture(camera_index: int = 0, warmup_frames: int = 30) -> cv2.VideoCapture:
    """
    Open a camera, trying platform-appropriate backends in order.

    Applies the same capture properties (buffer size, 1280x720@30, autofocus,
    auto-exposure) and warm-up read loop the GUI uses, and returns a
    VideoCapture that is delivering valid frames. Raises RuntimeError if no
    backend can open the camera or it never delivers a frame.
    """
    last_error: Optional[str] = None

    for backend_flag, backend_name in _backend_candidates():
        cap = None
        try:
            logger.info("Trying camera %s with %s…", camera_index, backend_name)
            cap = (
                cv2.VideoCapture(camera_index)
                if backend_flag == cv2.CAP_ANY
                else cv2.VideoCapture(camera_index + backend_flag)
            )
            if not cap.isOpened():
                raise RuntimeError(f"{backend_name} failed to open camera")

            for prop, value in [
                (cv2.CAP_PROP_BUFFERSIZE, 1),
                (cv2.CAP_PROP_FRAME_WIDTH, 1280),
                (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                (cv2.CAP_PROP_FPS, 30),
                (cv2.CAP_PROP_AUTOFOCUS, 1),
                (cv2.CAP_PROP_AUTO_EXPOSURE, 1),
            ]:
                try:
                    cap.set(prop, value)
                except Exception:
                    pass

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                raise RuntimeError("Camera not providing valid frames")

            # Warm-up: let auto-exposure/focus settle before the caller reads.
            # Every warmup frame is read and discarded; sleep only after a
            # failed read so a healthy camera settles at its native frame rate.
            for _ in range(max(0, warmup_frames)):
                try:
                    ret, frm = cap.read()
                    if ret and frm is not None and frm.size > 0:
                        continue
                except Exception:
                    pass
                time.sleep(0.1)

            logger.info("Camera opened with %s", backend_name)
            return cap

        except Exception as e:
            last_error = str(e)
            logger.warning("Failed with %s: %s", backend_name, e)
            if cap is not None:
                cap.release()

    raise RuntimeError(
        f"Failed to open camera {camera_index} with any backend"
        + (f" (last error: {last_error})" if last_error else "")
    )


class CameraSource(ABC):
    """A source of card frames for the sorter pipeline."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None when the feed is exhausted/failed."""

    def release(self) -> None:
        """Release any underlying resources. Safe to call multiple times."""


class OpenCVCameraSource(CameraSource):
    """
    Live camera via OpenCV (USB/V4L2 today; real path on the Pi).

    Tolerates up to max_read_failures consecutive bad reads (dropped/black
    frames are routine on USB cams) before reporting the feed as dead —
    mirroring the GUI's read-fail tolerance so one glitch doesn't end a
    whole sort run.
    """

    def __init__(self, camera_index: int = 0, max_read_failures: int = 20):
        self.camera_index = camera_index
        self.max_read_failures = max_read_failures
        self._cap = open_capture(camera_index)

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        failures = 0
        while failures <= self.max_read_failures:
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
            failures += 1
            time.sleep(0.05)
        logger.error(
            "Camera %d gave %d consecutive failed reads; treating feed as dead.",
            self.camera_index, failures,
        )
        return None

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            finally:
                self._cap = None


class MockCameraSource(CameraSource):
    """
    Replay frames from a list of image paths or a folder, then end the feed.

    Used for desktop dry runs and tests — no camera needed. Unreadable
    files are skipped with a warning.
    """

    def __init__(self, images: Sequence, loop: bool = False):
        self._paths: List[str] = self._resolve(images)
        self._loop = loop
        self._idx = 0

    @staticmethod
    def _resolve(images: Sequence) -> List[str]:
        if isinstance(images, str):
            if os.path.isdir(images):
                return sorted(
                    os.path.join(images, f)
                    for f in os.listdir(images)
                    if f.lower().endswith(SUPPORTED_EXTS)
                )
            return [images]
        return list(images)

    def read(self) -> Optional[np.ndarray]:
        # Iterative skip of unreadable files, bounded to one full pass so a
        # looping feed of all-bad paths ends cleanly instead of spinning.
        skipped = 0
        while skipped < max(len(self._paths), 1):
            if self._idx >= len(self._paths):
                if not self._loop or not self._paths:
                    return None
                self._idx = 0
            path = self._paths[self._idx]
            self._idx += 1
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            logger.warning("MockCameraSource could not read %s; skipping.", path)
            skipped += 1
        return None

    def release(self) -> None:
        self._paths = []
