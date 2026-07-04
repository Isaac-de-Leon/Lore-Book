# tests/test_card_database.py — feature-cache build/prune behavior (no TF, no camera).

import sqlite3

import numpy as np

from lorebook.core.card_database import (
    _cache_path,
    _init_db,
    build_feature_database,
    load_cache,
)


def _seed_cache(cache_file, names, dim=4):
    _init_db(cache_file)
    with sqlite3.connect(cache_file) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO features (filename, vector) VALUES (?, ?)",
            [(n, np.ones(dim, dtype=np.float32).tobytes()) for n in names],
        )


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

    def test_build_without_stale_entries_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        images = tmp_path / "Lorcana"
        images.mkdir()
        (images / "001-001.webp").write_bytes(b"x")

        cache_file = _cache_path(str(images))
        _seed_cache(cache_file, ["001-001.webp"])

        db = build_feature_database(db_path=str(images))
        assert set(db) == {"001-001.webp"}
