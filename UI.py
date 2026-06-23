# UI.py — Application entry point.
#
# All UI logic lives in lorebook/ui/:
#   main_window.py    — MainWindow (camera, matching, CSV export)
#   settings_window.py — SettingsWindow dialog

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lorebook.ui.main_window import MainWindow, setup_logging

setup_logging()

if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.start_camera)
    app.exec()
