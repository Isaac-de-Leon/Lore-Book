# app_logging.py — application-wide logging and crash diagnostics.

import faulthandler
import logging
import logging.handlers
import os
import sys

from lorebook.core.paths import data_path

logger = logging.getLogger(__name__)

# Held at module level so the faulthandler target file is never GC-closed.
_crash_log_file = None


def setup_logging(log_file: str = "card_scanner.log") -> None:
    """Configure application-wide logging with rotating file + console handlers."""
    log_dir = data_path("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    ch.setLevel(logging.WARNING)

    root.addHandler(fh)
    root.addHandler(ch)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)

    # Crash diagnostics. Native faults (e.g. access violations inside Qt or
    # OpenCV) kill the process with no Python traceback — faulthandler dumps
    # every thread's Python stack to logs/crash_native.log at fault time so
    # the crash site is identifiable afterwards.
    global _crash_log_file
    try:
        _crash_log_file = open(  # must stay open for the process lifetime
            os.path.join(log_dir, "crash_native.log"), "a", encoding="utf-8"
        )
        faulthandler.enable(file=_crash_log_file)
    except OSError as e:
        logger.warning("Could not enable native crash logging: %s", e)

    # Uncaught Python exceptions land in the main log too (PySide6 prints
    # them to stderr, which is invisible when launched outside a terminal).
    def _log_excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_excepthook

    logger.info("Logging initialized — %s", log_path)


class GuiStallWatchdog:
    """Writes every thread's stack to logs/hang_dump.log when the GUI thread
    stops processing events for stall_seconds.

    Uses faulthandler.dump_traceback_later, whose timer is a C thread that
    does not need the GIL — so the dump still happens in a true deadlock
    where some thread holds the GIL forever (a Python-level watchdog thread
    would be stuck too). A QTimer on the GUI thread keeps re-arming the
    timer; if the GUI thread blocks, re-arming stops and the dump fires once.
    """

    def __init__(self, parent, stall_seconds: float = 8.0) -> None:
        from PySide6.QtCore import QTimer

        self._stall = stall_seconds
        path = os.path.join(data_path("logs"), "hang_dump.log")
        # Kept open for the process lifetime (the C timer writes to its fd).
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._timer = QTimer(parent)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._rearm)
        self._timer.start()
        self._rearm()

    def _rearm(self) -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(self._stall, repeat=False, file=self._file)

    def stop(self) -> None:
        self._timer.stop()
        faulthandler.cancel_dump_traceback_later()
