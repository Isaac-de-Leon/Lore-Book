# app.py — GUI application entry point (also exposed as the `lorebook-ui` script).

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lorebook.ui.main_window import MainWindow, setup_logging


def main() -> int:
    setup_logging()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.start_camera)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
