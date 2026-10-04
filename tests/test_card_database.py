# tests/test_card_database.py — feature-cache build/prune behavior (no TF, no camera).

import sqlite3
import threading

import cv2
import numpy as np

import lorebook.core.card_database as card_database
from lorebook.core.card_database import (
    _cache_path,
    _init_db,
    build_feature_database,
    load_cache,
)


class RecordingExtractor:
    """Stub extract_batch: records chunk sizes, returns a unit vector per image."""

    def __init__(self, dtype=np.float32):
        self.batch_sizes = []
        self.dtype = dtype

    def extract_batch(self, images):
        self.batch_sizes.append(len(images))
        vec = np.zeros(4, self.dtype)
        vec[0] = 1.0
        return [vec.copy() for _ in images]


def _seed_cache(cache_file, names, dim=4):
    _init_db(cache_file)
    with sqlite3.connect(cache_file) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO features (filename, vector) VALUES (?, ?)",
            [(n, np.ones(dim, dtype=np.float32).tobytes()) for n in names],
        )


def _make_images(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"001-{i:03d}.png"), np.zeros((4, 4, 3), np.uint8))


def test_batched_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(card_database, "_BATCH_SIZE", 2)
    images = tmp_path / "Lorcana"
    _make_images(images, 5)
    (images / "001-999.png").write_text("not an image")  # unreadable: skipped, not fatal

    # Regression: float64 vectors used to be saved as float64 bytes but re-read
    # as float32, turning every cached vector into 2x-length NaN garbage.
    extractor = RecordingExtractor(dtype=np.float64)
    progress = []
    db = build_feature_database(
        progress_callback=lambda pct, fname: progress.append(pct),
        db_path=str(images),
        extractor=extractor,
    )

    assert set(db) == {f"001-{i:03d}.png" for i in range(5)}
    assert extractor.batch_sizes == [2, 2, 1]
    assert progress[-1] == 100
    reloaded = load_cache(str(images))
    assert set(reloaded) == set(db)
    vec = reloaded["001-000.png"]
    assert vec.dtype == np.float32 and vec.shape == (4,)
    np.testing.assert_allclose(vec, [1.0, 0.0, 0.0, 0.0])


def test_cancel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(card_database, "_BATCH_SIZE", 2)
    images = tmp_path / "Lorcana"
    _make_images(images, 6)

    # Cancel after batch 1: partial progress is committed so the next build resumes.
    cancel = threading.Event()
    extractor = RecordingExtractor()
    db = build_feature_database(
        progress_callback=lambda pct, fname: cancel.set(),
        db_path=str(images),
        extractor=extractor,
        cancel_event=cancel,
    )
    assert extractor.batch_sizes == [2]
    assert len(db) == 2
    assert set(load_cache(str(images))) == set(db)

    # Pre-set cancel processes nothing and keeps what's cached.
    extractor = RecordingExtractor()
    db2 = build_feature_database(db_path=str(images), extractor=extractor, cancel_event=cancel)
    assert extractor.batch_sizes == []
    assert set(db2) == set(db)


def test_stale_cache_pruning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    images = tmp_path / "Lorcana"
    images.mkdir()
    (images / "001-001.webp").write_bytes(b"not-a-real-image")
    _seed_cache(_cache_path(str(images)), ["001-001.webp", "999-999.webp"])

    db = build_feature_database(db_path=str(images))
    assert set(db) == {"001-001.webp"}              # deleted image pruned, survivor kept
    assert set(load_cache(str(images))) == {"001-001.webp"}

    # A typo'd/unmounted db_path must be a no-op, not "all images deleted".
    missing = tmp_path / "Riftbound"
    _seed_cache(_cache_path(str(missing)), ["001-001.webp", "001-002.webp"])
    assert set(build_feature_database(db_path=str(missing))) == {"001-001.webp", "001-002.webp"}
    assert set(load_cache(str(missing))) == {"001-001.webp", "001-002.webp"}
