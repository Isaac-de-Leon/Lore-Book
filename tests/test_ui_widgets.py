# tests/test_ui_widgets.py — small Qt pieces: progress dialog, image rendering, icons, styles, logging.

import logging
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QLabel

from lorebook.core.settings import THEME_NAMES
from lorebook.ui import app_logging
from lorebook.ui.icons import get_icon
from lorebook.ui.progress_dialog import BuildProgressDialog
from lorebook.ui.qt_images import crop_to_fill, show_bgr
from lorebook.ui.styles import THEMES, build_stylesheet, theme_tokens


@pytest.fixture(autouse=True)
def _qt_app():
    return QApplication.instance() or QApplication([])


def test_progress_dialog_bar_modes():
    dialog = BuildProgressDialog()
    dialog.set_progress(-1)  # a phase that only reports text: busy indicator
    assert dialog.bar.maximum() == 0
    dialog.set_progress(40)
    assert (dialog.bar.minimum(), dialog.bar.maximum(), dialog.bar.value()) == (0, 100, 40)
    dialog.set_progress(250)
    assert dialog.bar.value() == 100


def test_progress_dialog_cancel_is_requested_once_and_stays_open():
    dialog = BuildProgressDialog()
    requests = []
    dialog.cancel_requested.connect(lambda: requests.append(1))

    dialog.cancel_btn.click()
    assert requests == [1] and not dialog.cancel_btn.isEnabled()
    assert dialog.status_label.text() == "Cancelling…"
    dialog.set_status("Processing 001-001.webp…")  # late progress must not hide "Cancelling…"
    assert dialog.status_label.text() == "Cancelling…"
    dialog.reject()  # Esc while already cancelling: no second request
    assert requests == [1]

    dialog.reset_for_new_build()
    assert dialog.cancel_btn.isEnabled() and dialog.status_label.text() == "Preparing…"
    dialog.set_status("Building Lorcana database…")
    assert dialog.status_label.text() == "Building Lorcana database…"


@pytest.mark.parametrize("dismiss", ["close", "reject"])
def test_progress_dialog_closing_it_means_cancel(dismiss):
    dialog = BuildProgressDialog()
    requests = []
    dialog.cancel_requested.connect(lambda: requests.append(1))
    dialog.show()
    getattr(dialog, dismiss)()  # the window X, or Esc
    assert requests == [1] and dialog.status_label.text() == "Cancelling…"


def test_crop_to_fill_keeps_the_centre_at_the_target_aspect():
    wide = np.zeros((100, 400, 3), np.uint8)
    wide[:, 150:250] = 255
    cropped = crop_to_fill(wide, 100, 100)
    assert cropped.shape == (100, 100, 3) and cropped.min() == 255 and cropped.flags["C_CONTIGUOUS"]
    assert crop_to_fill(np.zeros((400, 100, 3), np.uint8), 100, 100).shape == (100, 100, 3)
    # Nothing sensible to crop to: the image comes back unchanged.
    assert crop_to_fill(wide, 0, 100).shape == wide.shape
    assert crop_to_fill(np.zeros((0, 0, 3), np.uint8), 10, 10).size == 0


def test_show_bgr_accepts_every_image_kind():
    label = QLabel()
    label.resize(80, 60)
    show_bgr(label, None)
    show_bgr(label, np.zeros((0, 0, 3), np.uint8))
    assert label.pixmap().isNull()  # nothing to draw: the label is left alone

    for image in (
        np.zeros((30, 20, 3), np.uint8),  # BGR
        np.zeros((30, 20), np.uint8),  # gray
        np.zeros((30, 20, 4), np.uint8),  # BGRA
        np.zeros((30, 40, 3), np.float64)[:, ::2],  # wrong dtype, not contiguous
    ):
        show_bgr(label, image)
        pixmap = label.pixmap()
        assert not pixmap.isNull()
        assert (pixmap.width(), pixmap.height()) == (40, 60)  # 20x30 scaled to fit, aspect kept

    show_bgr(label, np.zeros((30, 20, 3), np.uint8), fill=True)
    assert (label.pixmap().width(), label.pixmap().height()) == (80, 60)  # cropped to cover the label


def test_icons_are_tinted_and_cached():
    icon = get_icon("book", "#F97316")
    assert not icon.isNull() and get_icon("book", "#F97316") is icon
    assert get_icon("book", "#F97316", checked_color="#FFFFFF") is not icon
    with pytest.raises(KeyError):
        get_icon("no-such-icon", "#000000")


def test_stylesheet_fills_every_token_for_every_theme():
    assert set(THEMES) == set(THEME_NAMES)
    for theme in THEME_NAMES:
        qss = build_stylesheet(theme)
        assert "$" not in qss  # an unknown token would be left as-is by a safe substitute
        assert THEMES[theme]["accent"] in qss
    assert theme_tokens("no-such-theme") is theme_tokens()  # unknown theme: the default palette


class _FaultHandlerSpy:
    """Stands in for the faulthandler module so tests don't touch the real, process-wide one."""

    def __init__(self):
        self.calls = []

    def enable(self, file):
        self.calls.append(("enable", file.name))

    def dump_traceback_later(self, timeout, repeat, file):
        self.calls.append(("dump_later", timeout, repeat))

    def cancel_dump_traceback_later(self):
        self.calls.append(("cancel",))


@pytest.fixture
def clean_logging(monkeypatch):
    """Undo what setup_logging does to the root logger, sys.excepthook and its open files."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    spy = _FaultHandlerSpy()
    monkeypatch.setattr(app_logging, "faulthandler", spy)
    yield spy
    for handler in [h for h in root.handlers if h not in handlers]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    if app_logging._crash_log_file is not None:
        app_logging._crash_log_file.close()
        app_logging._crash_log_file = None


def test_setup_logging_writes_a_log_file_and_logs_uncaught_errors(
    tmp_path, monkeypatch, clean_logging, capsys
):
    monkeypatch.chdir(tmp_path)
    app_logging.setup_logging("test.log")
    log_file = tmp_path / "logs" / "test.log"

    logging.getLogger("lorebook.test").info("hello from the test")
    try:
        raise ValueError("nobody caught this")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = log_file.read_text(encoding="utf-8")
    assert "hello from the test" in text
    assert "Uncaught exception" in text and "nobody caught this" in text
    assert "nobody caught this" in capsys.readouterr().err  # Python's own report still appears
    assert clean_logging.calls == [("enable", os.path.join("logs", "crash_native.log"))]


def test_stall_watchdog_keeps_pushing_the_dump_deadline_back(tmp_path, monkeypatch, clean_logging):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    parent = QObject()  # owns the watchdog's timer, as the QApplication does in the app
    watchdog = app_logging.GuiStallWatchdog(parent, stall_seconds=8.0)
    try:
        assert clean_logging.calls == [("cancel",), ("dump_later", 8.0, False)]
        watchdog._rearm()  # what the one-second timer does while the GUI is responsive
        assert clean_logging.calls[-2:] == [("cancel",), ("dump_later", 8.0, False)]
        watchdog.stop()
        assert clean_logging.calls[-1] == ("cancel",) and not watchdog._timer.isActive()
    finally:
        watchdog._file.close()
