# progress_dialog.py — modeless popup showing DB build progress with a Cancel button.

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from lorebook.ui.styles import APP_STYLESHEET


class BuildProgressDialog(QDialog):
    """Progress popup for the background database build.

    Modeless (shown with show(), never exec()) so the main window stays
    usable while a build runs — including the startup build, which begins
    before the main window is shown. Cancel (button, Esc, or the window X)
    emits cancel_requested; the dialog stays open in a "Cancelling…" state
    until the worker notices the flag and MainWindow hides it on build_done.
    """

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Building Card Database")
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowModality(Qt.NonModal)
        self.setFixedWidth(440)

        self.status_label = QLabel("Preparing…")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setObjectName("buildProgressBar")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self.status_label)
        layout.addWidget(self.bar)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self._cancelling = False

    # ------------------------------------------------------------------ API

    def set_status(self, text: str) -> None:
        if not self._cancelling:
            self.status_label.setText(text)

    def set_progress(self, pct: int) -> None:
        """Update the bar; pct == -1 switches to indeterminate (busy) mode."""
        if pct < 0:
            if self.bar.maximum() != 0:
                self.bar.setRange(0, 0)
            return
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 100)
        self.bar.setValue(min(pct, 100))

    def reset_for_new_build(self) -> None:
        self._cancelling = False
        self.cancel_btn.setEnabled(True)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.status_label.setText("Preparing…")

    def enter_cancelling_state(self) -> None:
        self._cancelling = True
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancelling…")

    # ------------------------------------------------------------------ events

    def _on_cancel_clicked(self) -> None:
        # Don't close: the worker needs up to one batch/image/request to
        # unwind. MainWindow hides the dialog when build_done arrives.
        self.enter_cancelling_state()
        self.cancel_requested.emit()

    def closeEvent(self, event) -> None:
        # Closing the only progress surface means cancel (QProgressDialog
        # convention); the hide is allowed so the user isn't trapped.
        if not self._cancelling:
            self.enter_cancelling_state()
            self.cancel_requested.emit()
        super().closeEvent(event)

    def reject(self) -> None:  # Esc key
        if not self._cancelling:
            self.enter_cancelling_state()
            self.cancel_requested.emit()
        super().reject()
