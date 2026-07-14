# lorebook.hardware — hardware abstraction for the card sorter.
#
# This package isolates everything physical (camera, card transport/motion)
# behind small interfaces so the headless sorter pipeline in lorebook.sorter
# can run unchanged against either real hardware or desktop mocks. Heavy /
# platform-specific imports (cv2, picamera2, gpio, serial) are done lazily
# inside the concrete implementations, so importing this package on a dev
# laptop never pulls in hardware drivers.

from lorebook.hardware.camera import (
    CameraSource,
    MockCameraSource,
    OpenCVCameraSource,
    open_capture,
)
from lorebook.hardware.transport import MockTransport, Transport

__all__ = [
    "CameraSource",
    "OpenCVCameraSource",
    "MockCameraSource",
    "open_capture",
    "Transport",
    "MockTransport",
]
