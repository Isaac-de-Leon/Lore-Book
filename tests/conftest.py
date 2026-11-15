# tests/conftest.py — shared test setup.
#
# Stubs TensorFlow/Keras in sys.modules before any test module imports
# lorebook code, so the suite runs without those heavy packages installed
# (no model download, no GPU). Also puts the repo root on sys.path so
# `import lorebook` works from any pytest invocation dir without installing.
#
# The Qt fixtures below skip their tests when PySide6 isn't installed (the
# core CI job). Qt test modules still start with pytest.importorskip("PySide6")
# because they import Qt names at module level.

import os
import sys
import time
import unittest.mock

import numpy as np
import pytest

_tf_mock = unittest.mock.MagicMock()
for _mod in [
    "tensorflow",
    "tf",
    "keras",
    "keras.applications",
    "keras.applications.mobilenet_v2",
    "keras.models",
]:
    sys.modules.setdefault(_mod, _tf_mock)

# The suite's model paths exercise the (stubbed) keras extractor; without
# this, a dev venv with onnxruntime installed would default to "onnx" and
# build the real model. ONNX tests name their backend explicitly.
os.environ.setdefault("LOREBOOK_BACKEND", "keras")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed, none opened

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _keep_tests_away_from_real_files_and_dialogs(tmp_path):
    """Make it impossible for a test to touch the user's files or open a real dialog.

    The app's data files (collection CSVs, settings, caches) are relative to
    the working directory, so every test starts in its own empty tmp_path:
    a test that forgets to change directory can never write to the checkout's
    real LorcanaList.csv. This uses its own MonkeyPatch, so a test calling
    monkeypatch.undo() does not land back in the checkout either.

    With Qt installed, a message box or file dialog would hang the run and
    Export suggests the real Documents folder. Every dialog fails the test
    unless the test replaces it (see the message_boxes and save_dialog
    fixtures), and exports are suggested into tmp_path.
    """
    with pytest.MonkeyPatch.context() as guard:
        guard.chdir(tmp_path)
        guard.delenv("LOREBOOK_DATA_DIR", raising=False)
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
        except ImportError:  # no Qt installed: no dialogs to guard
            yield
            return
        from lorebook.ui import collection_view

        def unexpected(name):
            def fail(*args, **kwargs):
                raise AssertionError(
                    f"{name} would open a real dialog; use the message_boxes/save_dialog fixture"
                )

            return fail

        for name in ("warning", "information", "critical", "question"):
            guard.setattr(QMessageBox, name, unexpected(f"QMessageBox.{name}"))
        for name in ("getSaveFileName", "getOpenFileName", "getExistingDirectory"):
            guard.setattr(QFileDialog, name, unexpected(f"QFileDialog.{name}"))

        def export_folder():
            return str(tmp_path / "Documents")

        export_folder.real = collection_view._export_folder  # for the one test that checks it
        guard.setattr(collection_view, "_export_folder", export_folder)
        yield


@pytest.fixture
def qt_app():
    """The QApplication (offscreen), created once for the test run."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def wait_until(qt_app):
    """wait_until(condition, timeout=5.0): run the Qt event loop until condition() is true."""

    def wait(condition, timeout=5.0):
        deadline = time.time() + timeout
        while not condition() and time.time() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        return bool(condition())

    return wait


@pytest.fixture
def message_boxes(monkeypatch):
    """Record QMessageBox.warning calls as (title, text) instead of showing them.

    Set `.answer` to choose what the box returns (default: Ok).
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox

    class Shown(list):
        answer = QMessageBox.StandardButton.Ok

    shown = Shown()

    def warning(parent, title, text, *args, **kwargs):
        shown.append((title, text))
        return shown.answer

    monkeypatch.setattr(QMessageBox, "warning", warning)
    return shown


@pytest.fixture
def save_dialog(monkeypatch):
    """Record the path each save dialog suggests; answer with `.choose`.

    `.choose` is None (accept the suggestion, the default), "" (the user
    cancels) or a path to pick instead.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QFileDialog

    class Suggested(list):
        choose = None

    suggested = Suggested()

    def get_save_file_name(parent, title, path, file_filter):
        suggested.append(path)
        return (path if suggested.choose is None else suggested.choose), file_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", get_save_file_name)
    return suggested


class FakeCapture:
    """Stands in for cv2.VideoCapture: an endless feed of small black frames."""

    def __init__(self):
        self.released = False

    def read(self):
        return True, np.zeros((72, 128, 3), np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    """A MainWindow over two game folders (Lorcana, MTG) with no camera, network or model.

    - The camera is a FakeCapture, so Start/Stop run the real worker thread.
    - Database builds are counted in `window.builds`, not run. After
      `window.use_real_builds()` they go through the real build worker, whose
      job is a fake that records its game list in `window.build_jobs`.
    The window is closed (and its threads stopped) after the test.
    """
    import lorebook.ui.main_window as mw
    from lorebook.core.build import BuildReport
    from lorebook.ui import workers

    monkeypatch.chdir(tmp_path)
    for game, files in (("Lorcana", ("001-001.webp", "002-001.webp")), ("MTG", ("DOM-001.webp",))):
        (tmp_path / "Card_Images" / game).mkdir(parents=True)
        for name in files:
            (tmp_path / "Card_Images" / game / name).touch()

    builds, build_jobs = [], []

    def fake_build_job(games, **kwargs):  # runs on the build worker thread
        build_jobs.append(list(games))
        return BuildReport()

    real_start_build = mw.MainWindow.start_db_build_in_background
    monkeypatch.setattr(workers, "open_capture", lambda index: FakeCapture())
    monkeypatch.setattr(workers, "refresh_and_build", fake_build_job)
    monkeypatch.setattr(mw.MainWindow, "start_db_build_in_background", lambda self: builds.append(1))

    window = mw.MainWindow()
    qt_app.processEvents()  # let the deferred startup build request arrive
    builds.clear()
    window.builds = builds
    window.build_jobs = build_jobs
    window.use_real_builds = lambda: monkeypatch.setattr(
        mw.MainWindow, "start_db_build_in_background", real_start_build
    )
    yield window
    window.close()
    qt_app.processEvents()
