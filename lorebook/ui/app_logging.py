# app_logging.py — application-wide logging and crash diagnostics.

import faulthandler
import logging
import logging.handlers
import os
import sys
import threading
import time

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
    """Writes every thread's Python stack to logs/hang_dump.log when the GUI
    thread stops processing events for stall_seconds.

    A frozen window ("Not Responding") leaves no traceback on its own; this
    records where each thread was stuck. A QTimer on the GUI thread updates a
    heartbeat; a plain daemon thread (deliberately not Qt — it must keep
    running while the GUI thread is blocked) checks it. One dump per stall.
    """

    def __init__(self, parent, stall_seconds: float = 5.0) -> None:
        from PySide6.QtCore import QTimer

        self._stall = stall_seconds
        self._beat = time.monotonic()
        self._timer = QTimer(parent)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._heartbeat)
        self._timer.start()
        threading.Thread(target=self._watch, name="lorebook-stall-watchdog", daemon=True).start()

    def _heartbeat(self) -> None:
        self._beat = time.monotonic()

    def _watch(self) -> None:
        dumped = False
        while True:
            time.sleep(1.0)
            stalled = time.monotonic() - self._beat
            if stalled < self._stall:
                dumped = False
            elif not dumped:
                dumped = True
                self._dump(stalled)

    def _dump(self, stalled: float) -> None:
        path = os.path.join(data_path("logs"), "hang_dump.log")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n=== GUI thread unresponsive for {stalled:.1f}s at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} — all thread stacks ===\n")
                f.flush()
                faulthandler.dump_traceback(file=f, all_threads=True)
            logger.error("GUI thread unresponsive for %.1fs — thread stacks written to %s", stalled, path)
        except OSError as e:
            logger.error("GUI stalled for %.1fs; could not write %s: %s", stalled, path, e)
