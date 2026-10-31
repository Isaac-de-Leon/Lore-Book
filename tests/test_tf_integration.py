# tests/test_tf_integration.py — real TensorFlow on the GUI's worker thread (opt-in).
#
# Skipped unless LOREBOOK_TF_TESTS=1 (needs TensorFlow + PySide6, and the
# MobileNetV2 weights — downloaded on first run, ~14 MB):
#
#   LOREBOOK_TF_TESTS=1 python -m pytest tests/test_tf_integration.py -v
#
# conftest.py stubs TensorFlow for the whole pytest process, so the scenario
# runs in a child interpreter — which also turns a native crash into a test
# failure instead of killing pytest.
#
# Guards the scan crash where ScanWorker.warm_up loaded the model in one slot
# call and the next slot's inference ran on torn-down TF thread-local state
# (access violation / hang): warm up, then scan in a later slot, on the real
# WorkerThread + ScanWorker.

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LOREBOOK_TF_TESTS") != "1",
    reason="real-TensorFlow test; set LOREBOOK_TF_TESTS=1 to run",
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCENARIO = textwrap.dedent(
    """
    import importlib.util
    import sys

    # Check for TensorFlow without importing it: the app first imports it on
    # the scan thread (warm_up), and only that order reproduces the crash.
    # Importing it here on the main thread made the scenario pass even with
    # the bug.
    try:
        if importlib.util.find_spec("tensorflow") is None:
            raise ImportError("No module named 'tensorflow'")
        from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal, Slot
    except ImportError as e:
        print("SKIP", e, flush=True)
        sys.exit(0)

    import numpy as np

    from lorebook.core.features import extract_features
    from lorebook.ui.threads import WorkerThread
    from lorebook.ui.workers import ScanWorker

    class Probe(QObject):
        @Slot()
        def scan(self):
            rng = np.random.default_rng(0)
            for i in range(3):
                v = extract_features(rng.integers(0, 255, (500, 350, 3), dtype=np.uint8))
                print("SCAN", i, None if v is None else v.shape, flush=True)
            app.quit()

    class Requests(QObject):
        warm = Signal()
        scan = Signal()

    app = QCoreApplication(sys.argv)
    worker, probe = ScanWorker(), Probe()
    wt = WorkerThread(worker, "test-scan")
    probe.moveToThread(wt.thread)  # same thread, separate slot calls
    req = Requests()
    req.warm.connect(worker.warm_up)
    req.scan.connect(probe.scan)
    wt.start()
    req.warm.emit()
    req.scan.emit()
    QTimer.singleShot(240_000, app.quit)  # safety net if a scan hangs
    app.exec()
    print("STOPPED", wt.stop(), flush=True)
    """
)


def test_warm_up_then_scan_on_worker_thread():
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", _SCENARIO],
            cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"scan scenario hung (>300 s)\n{e.stdout}\n{e.stderr}")

    out = proc.stdout
    if out.startswith("SKIP"):
        pytest.skip(out.strip())
    detail = f"exit code {proc.returncode}\n--- stdout\n{out}\n--- stderr (tail)\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0, detail
    for i in range(3):
        assert f"SCAN {i} (1280,)" in out, detail
    assert "STOPPED True" in out, detail
