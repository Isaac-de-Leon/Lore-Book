# app.py — GUI application entry point (also exposed as the `lorebook-ui` script).

import os
import sys

# Must run before anything imports TensorFlow/Keras (main_window pulls in the
# feature pipeline): frozen builds keep the Keras weights cache in the
# per-user data dir instead of the ephemeral bundle.
from lorebook.core.paths import configure_frozen_environment

configure_frozen_environment()

from lorebook import __version__  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from lorebook.ui.main_window import MainWindow, setup_logging  # noqa: E402

_ICON_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")


def main() -> int:
    setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("LoreBook")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("LoreBook")
    if os.path.exists(_ICON_PNG):
        app.setWindowIcon(QIcon(_ICON_PNG))
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.start_camera)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
