# UI.py — Application entry point (thin wrapper; also available as `lorebook-ui`).
#
# All UI logic lives in lorebook/ui/:
#   app.py            — QApplication bootstrap (main())
#   main_window.py    — MainWindow (camera, matching, CSV export)
#   settings_window.py — SettingsWindow dialog

import sys

from lorebook.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
