# tests/test_build.py — the refresh-and-build job (no network, no TF, no Qt).

import threading

import cv2
import numpy as np

from lorebook.core import build


class Extractor:
    def extract_batch(self, images):
        v = np.zeros(4, np.float32)
        v[0] = 1.0
        return [v.copy() for _ in images]


def test_refresh_and_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("download_new_images", "download_card_prices", "download_currency_rates"):
        monkeypatch.setattr(build, name, lambda *a, **k: None)
    monkeypatch.setattr(build, "prices_stale", lambda *a, **k: True)
    (tmp_path / "Lorcana").mkdir()
    cv2.imwrite(str(tmp_path / "Lorcana" / "001-001.png"), np.zeros((4, 4, 3), np.uint8))
    (tmp_path / "Empty").mkdir()
    (tmp_path / "Broken").mkdir()
    (tmp_path / "Broken" / "001-001.png").write_text("not an image")

    statuses, progress = [], []
    report = build.refresh_and_build(
        ["Lorcana", "Empty", "Broken", "Missing"], cancel_event=threading.Event(),
        status=statuses.append, progress=progress.append, base=str(tmp_path), extractor=Extractor(),
    )
    # Empty folder: skipped, not a failure. Unreadable images / no folder: failed.
    assert report.failed == ["Broken", "Missing"] and not report.cancelled and not report.ok
    assert "No images for Empty — skipped" in statuses
    assert build.INDETERMINATE in progress and 100 in progress

    # A network step blowing up never stops the build.
    def offline(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(build, "download_new_images", offline)
    monkeypatch.setattr(build, "download_currency_rates", offline)
    assert build.refresh_and_build(["Lorcana"], cancel_event=threading.Event(), base=str(tmp_path),
                                   extractor=Extractor()).ok

    # Cancelled before starting: nothing built, reported as cancelled.
    cancel = threading.Event()
    cancel.set()
    report = build.refresh_and_build(["Lorcana"], cancel_event=cancel, base=str(tmp_path))
    assert report.cancelled and not report.failed
