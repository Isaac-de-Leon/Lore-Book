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

    def __init__(self):
        self.batch_sizes = []

    def extract_batch(self, images):
        self.batch_sizes.append(len(images))
        vec = np.zeros(4, np.float32)
        vec[0] = 1.0
        return [vec.copy() for _ in images]


def _seed_cache(cache_file, names, dim=4):
    _init_db(cache_file)
    with sqlite3.connect(cache_file) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO features (filename, vector) VALUES (?, ?)",
            [(n, np.ones(dim, dtype=np.float32).tobytes()) for n in names],
        )


class TestBatchedBuild:
    def _make_images(self, folder, n):
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(folder / f"001-{i:03d}.png"), np.zeros((4, 4, 3), np.uint8))

    def test_build_extracts_in_chunks_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(card_database, "_BATCH_SIZE", 2)
        images = tmp_path / "Lorcana"
        self._make_images(images, 5)

        extractor = RecordingExtractor()
        progress = []
        db = build_feature_database(
            progress_callback=lambda pct, fname: progress.append(pct),
            db_path=str(images),
            extractor=extractor,
        )

        assert len(db) == 5
        assert extractor.batch_sizes == [2, 2, 1]
        assert progress[-1] == 100
        assert set(load_cache(str(images))) == set(db)

    def test_float64_vectors_round_trip_as_float32(self, tmp_path, monkeypatch):
        # Regression: an extractor yielding float64 (e.g. sklearn-normalized)
        # used to be saved as float64 bytes but re-read as float32, turning
        # every cached vector into 2x-length NaN garbage.
        monkeypatch.chdir(tmp_path)
        images = tmp_path / "Lorcana"
        self._make_images(images, 1)

        class Float64Extractor:
            def extract_batch(self, imgs):
                v = np.zeros(4, np.float64)
                v[0] = 1.0
                return [v.copy() for _ in imgs]

        build_feature_database(db_path=str(images), extractor=Float64Extractor())
        reloaded = load_cache(str(images))

        (vec,) = reloaded.values()
        assert vec.dtype == np.float32
        assert vec.shape == (4,)
        assert np.isfinite(vec).all()
        np.testing.assert_allclose(vec, [1.0, 0.0, 0.0, 0.0])

    def test_unreadable_image_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        images = tmp_path / "Lorcana"
        self._make_images(images, 1)
        (images / "001-999.png").write_text("not an image")

        db = build_feature_database(db_path=str(images), extractor=RecordingExtractor())

        assert "001-000.webp" not in db  # sanity: names come from real files
        assert set(db) == {"001-000.png"}

    def test_cancel_stops_between_batches_and_commits_partial(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(card_database, "_BATCH_SIZE", 2)
        images = tmp_path / "Lorcana"
        self._make_images(images, 6)

        cancel = threading.Event()
        extractor = RecordingExtractor()
        db = build_feature_database(
            progress_callback=lambda pct, fname: cancel.set(),  # cancel after batch 1
            db_path=str(images),
            extractor=extractor,
            cancel_event=cancel,
        )

        assert extractor.batch_sizes == [2]
        assert len(db) == 2
        # Partial progress is committed so the next build resumes from it.
        assert set(load_cache(str(images))) == set(db)

    def test_preset_cancel_processes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        images = tmp_path / "Lorcana"
        self._make_images(images, 3)
        cache_file = _cache_path(str(images))
        _seed_cache(cache_file, ["001-000.png"])

        cancel = threading.Event()
        cancel.set()
        extractor = RecordingExtractor()
        db = build_feature_database(
            db_path=str(images), extractor=extractor, cancel_event=cancel
        )

        assert extractor.batch_sizes == []
        assert set(db) == {"001-000.png"}  # pre-seeded entry survives
        assert set(load_cache(str(images))) == {"001-000.png"}


class TestStaleCachePruning:
    def test_build_prunes_entries_for_deleted_images(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        images = tmp_path / "Lorcana"
        images.mkdir()
        (images / "001-001.webp").write_bytes(b"not-a-real-image")

        cache_file = _cache_path(str(images))
        _seed_cache(cache_file, ["001-001.webp", "999-999.webp"])

        db = build_feature_database(db_path=str(images))

        assert "999-999.webp" not in db          # stale entry pruned from the dict
        assert "001-001.webp" in db              # surviving entry kept
        reloaded = load_cache(str(images))
        assert set(reloaded) == {"001-001.webp"}  # and from the SQLite cache

    def test_missing_folder_does_not_wipe_cache(self, tmp_path, monkeypatch):
        # A typo'd/unmounted db_path must be a no-op, not "all images deleted".
        monkeypatch.chdir(tmp_path)
        missing = tmp_path / "Lorcana"  # never created
        cache_file = _cache_path(str(missing))
        _seed_cache(cache_file, ["001-001.webp", "001-002.webp"])

        db = build_feature_database(db_path=str(missing))

        assert set(db) == {"001-001.webp", "001-002.webp"}
        assert set(load_cache(str(missing))) == {"001-001.webp", "001-002.webp"}

    def test_build_without_stale_entries_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        images = tmp_path / "Lorcana"
        images.mkdir()
        (images / "001-001.webp").write_bytes(b"x")

        cache_file = _cache_path(str(images))
        _seed_cache(cache_file, ["001-001.webp"])

        db = build_feature_database(db_path=str(images))
        assert set(db) == {"001-001.webp"}
