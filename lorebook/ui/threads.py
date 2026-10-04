# threads.py — one consistent way to run a worker QObject on its own QThread.
#
# Pattern (the one Qt recommends): a plain QObject worker is moved onto a
# QThread; the GUI talks to it only through queued signals/slots, and the
# worker talks back only through its own signals. Never subclass QThread and
# never touch a worker's attributes from the GUI thread while it runs.
#
# stop() asks the thread to finish (requestInterruption + quit) and waits.
# A worker blocked in a long call (a network request, a camera open) can't
# be interrupted mid-call, so stop() may time out; the app then exits without
# destroying the still-running QThread (see lorebook.ui.app.main), which is
# what would otherwise abort with "QThread: Destroyed while thread is still
# running".

import logging

from PySide6.QtCore import QObject, QThread

logger = logging.getLogger(__name__)


class WorkerThread:
    """Owns one QThread and the worker QObject that lives on it."""

    def __init__(self, worker: QObject, name: str):
        self.worker = worker
        self.thread = QThread()
        self.thread.setObjectName(name)
        worker.moveToThread(self.thread)

    @property
    def name(self) -> str:
        return self.thread.objectName()

    def start(self) -> None:
        if not self.thread.isRunning():
            self.thread.start()

    def is_running(self) -> bool:
        return self.thread.isRunning()

    def stop(self, timeout_ms: int = 3000) -> bool:
        """Stop the thread's event loop and wait. True if it finished in time."""
        if not self.thread.isRunning():
            return True
        self.thread.requestInterruption()
        self.thread.quit()
        finished = self.thread.wait(timeout_ms)
        if not finished:
            logger.warning("Thread %s did not stop within %s ms", self.name, timeout_ms)
        return bool(finished)
