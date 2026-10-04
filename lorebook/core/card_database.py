# card_database.py — Feature cache management and database build pipeline.

import concurrent.futures
import logging
import os
import sqlite3
import threading
from collections.abc import Callable

import cv2
import numpy as np

from lorebook.core.game_types import BASE_DATABASE_PATH, SUPPORTED_EXTS

logger = logging.getLogger(__name__)

# Current active database path — changed via set_database_path().
databasePath: str = os.path.join(BASE_DATABASE_PATH, "Lorcana")


def set_database_path(game_name: str) -> None:
    """Switch the active database folder to a different game."""
    global databasePath
    databasePath = os.path.join(BASE_DATABASE_PATH, game_name)
    os.makedirs(databasePath, exist_ok=True)


def get_database_path() -> str:
    """Return the current active database path."""
    return databasePath


def _cache_path(db_path: str) -> str:
    # Strip trailing separators so "Card_Images/Lorcana/" (e.g. from shell
    # tab-completion) resolves to "Lorcana", not "".
    game = os.path.basename(db_path.rstrip("/\\")) or "default"
    return f"DBCardCache_{game}.db"


def _init_db(path: str) -> None:
    """Create the features table if needed, migrating older caches in place.

    mtime_ns/size record the image file each vector was extracted from, so a
    replaced image (same name, new bytes) is re-extracted. Caches written
    before these columns existed get NULLs, which are trusted and backfilled
    on the next build rather than forcing a full rebuild.
    """
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS features "
            "(filename TEXT PRIMARY KEY, vector BLOB NOT NULL, mtime_ns INTEGER, size INTEGER)"
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(features)")}
        for column in ("mtime_ns", "size"):
            if column not in columns:
                conn.execute(f"ALTER TABLE features ADD COLUMN {column} INTEGER")


Stamp = tuple[int, int]  # (st_mtime_ns, st_size) of a source image


def _file_stamp(path: str) -> Stamp | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _load_stamps(cache_file: str) -> dict[str, Stamp | None]:
    """{filename: (mtime_ns, size) or None for legacy rows} from the cache."""
    if not os.path.exists(cache_file):
        return {}
    try:
        _init_db(cache_file)
        with sqlite3.connect(cache_file) as conn:
            return {
                name: (mtime, size) if mtime is not None and size is not None else None
                for name, mtime, size in conn.execute("SELECT filename, mtime_ns, size FROM features")
            }
    except sqlite3.Error as e:
        logger.error("Error reading cache stamps from %s: %s", cache_file, e)
        return {}


def _save_stamps(cache_file: str, stamps: dict[str, Stamp]) -> None:
    """Record stamps for existing rows (backfills legacy NULLs)."""
    if not stamps or not os.path.exists(cache_file):
        return
    try:
        with sqlite3.connect(cache_file) as conn:
            conn.executemany(
                "UPDATE features SET mtime_ns = ?, size = ? WHERE filename = ?",
                [(m, sz, name) for name, (m, sz) in stamps.items()],
            )
    except sqlite3.Error as e:
        logger.error("Error saving cache stamps to %s: %s", cache_file, e)


def load_cache(db_path: str | None = None) -> dict[str, np.ndarray]:
    """Load the feature cache from the SQLite database for the given (or current) database path."""
    path = _cache_path(db_path or databasePath)
    if not os.path.exists(path):
        return {}
    try:
        result: dict[str, np.ndarray] = {}
        with sqlite3.connect(path) as conn:
            for filename, blob in conn.execute("SELECT filename, vector FROM features"):
                result[filename] = np.frombuffer(blob, dtype=np.float32).copy()
        return result
    except (sqlite3.Error, ValueError) as e:  # ValueError: a corrupt (odd-length) vector blob
        logger.error("Could not load cache from %s: %s", path, e)
        return {}


def _remove_cache_entries(cache_file: str, filenames: list[str]) -> None:
    """Delete cache rows whose source images no longer exist on disk."""
    if not filenames or not os.path.exists(cache_file):
        return
    try:
        with sqlite3.connect(cache_file) as conn:
            conn.executemany(
                "DELETE FROM features WHERE filename = ?", [(n,) for n in filenames]
            )
    except sqlite3.Error as e:
        logger.error("Error pruning stale cache entries from %s: %s", cache_file, e)


def list_image_files(folder: str) -> list[str]:
    """Return sorted list of supported image filenames in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTS))


# Backward-compat alias (was private; csv_manager and the GUI need it).
_list_image_files = list_image_files


# Images per inference batch during builds. Reads are parallelized within a
# chunk; inference runs once per chunk on the calling thread (Keras predict is
# not guaranteed thread-safe, and its per-call overhead dominates per-image).
_BATCH_SIZE = 32


def _read_image(filename: str, db_path: str) -> tuple[str, np.ndarray | None]:
    """Load one image for the build pipeline (thread-safe: no globals, no TF)."""
    return filename, cv2.imread(os.path.join(db_path, filename), cv2.IMREAD_COLOR)


def build_feature_database(
    progress_callback: Callable[[int, str | None], None] | None = None,
    max_workers: int | None = None,
    db_path: str | None = None,
    extractor=None,
    cancel_event: threading.Event | None = None,
) -> dict[str, np.ndarray]:
    """
    Build or update the feature database for db_path (defaults to databasePath).

    Processes only images not already in the cache, updates the cache file, and
    returns the full feature dict. Image reads run in a thread pool; feature
    extraction runs in batches through extractor.extract_batch (defaults to the
    keras backend, resolved lazily so importing this module never pulls in TF).

    cancel_event: when set, the build stops between batches. Vectors extracted
    so far are still written to the cache, so the next build resumes from them.

    Unreadable images and failed extractions are logged, counted and skipped;
    anything unexpected propagates to the caller (it is not swallowed into a
    silently partial result).
    """
    resolved = db_path or databasePath
    featureDB: dict[str, np.ndarray] = load_cache(resolved)
    logger.info("Loaded %s cached entries from %s", len(featureDB), resolved)

    current_files = list_image_files(resolved)

    # Prune cache entries for deleted/renamed images so they can't keep
    # matching against cards that no longer exist. Only when the folder
    # itself exists — a missing folder (typo'd path, unmounted drive)
    # must not be read as "every image was deleted" and wipe the cache.
    stale = sorted(set(featureDB) - set(current_files)) if os.path.isdir(resolved) else []
    if stale:
        for name in stale:
            featureDB.pop(name, None)
        _remove_cache_entries(_cache_path(resolved), stale)
        logger.info("Pruned %s stale cache entries from %s", len(stale), resolved)

    # Re-extract images replaced under the same name (e.g. re-downloaded
    # with --force): their recorded size/mtime no longer match the file.
    # Legacy rows without a stamp are trusted and backfilled below.
    cache_file = _cache_path(resolved)
    stored = _load_stamps(cache_file)
    disk = {f: _file_stamp(os.path.join(resolved, f)) for f in current_files}
    changed = sorted(
        f for f in current_files
        if f in featureDB and stored.get(f) is not None and stored[f] != disk[f]
    )
    if changed:
        for name in changed:
            featureDB.pop(name, None)
        _remove_cache_entries(cache_file, changed)
        logger.info("Re-extracting %s changed images in %s", len(changed), resolved)
    _save_stamps(cache_file, {
        f: stamp for f in featureDB
        if stored.get(f) is None and (stamp := disk.get(f)) is not None
    })

    new_files = [f for f in current_files if f not in featureDB]
    total = len(new_files)
    logger.info("Processing %s new files in %s", total, resolved)

    if total == 0:
        if progress_callback:
            progress_callback(100, None)
        return featureDB

    if extractor is None:
        from lorebook.core.features import get_extractor  # lazy TF import
        extractor = get_extractor("keras")

    successful = failed = done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, total, _BATCH_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Build cancelled after %s/%s files in %s", done, total, resolved)
                break
            chunk = new_files[start:start + _BATCH_SIZE]
            loaded = list(pool.map(lambda f: _read_image(f, resolved), chunk))

            readable = []
            for fname, img in loaded:
                if img is None:
                    failed += 1
                    logger.warning("Could not read image %s", fname)
                else:
                    readable.append((fname, img))

            vectors = extractor.extract_batch([img for _, img in readable]) if readable else []
            for (fname, _), vec in zip(readable, vectors):
                if vec is not None:
                    featureDB[fname] = vec
                    successful += 1
                else:
                    failed += 1
                    logger.warning("Feature extraction failed for %s", fname)

            done += len(chunk)
            if progress_callback:
                progress_callback(int(done / total * 100), chunk[-1])

    try:
        _init_db(cache_file)
        with sqlite3.connect(cache_file) as conn:
            # Cast defensively: load_cache reads blobs as float32, so any
            # other dtype here would corrupt the round-trip.
            rows = []
            for k, v in featureDB.items():
                m, sz = disk.get(k) or (None, None)
                rows.append((k, np.asarray(v, dtype=np.float32).tobytes(), m, sz))
            conn.executemany(
                "INSERT OR REPLACE INTO features (filename, vector, mtime_ns, size) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.info("Saved %s entries to %s", len(featureDB), cache_file)
    except (sqlite3.Error, OSError) as e:
        logger.error("Error saving cache to %s: %s", cache_file, e)

    logger.info("Build complete: %s ok, %s failed", successful, failed)
    return featureDB



def clear_from_cache(filename: str, db_path: str | None = None) -> None:
    """Remove a single entry from the SQLite cache (useful after deleting an image)."""
    path = _cache_path(db_path or databasePath)
    if not os.path.exists(path):
        return
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM features WHERE filename = ?", (filename,))
        logger.info("Cleared %s from %s", filename, path)
    except sqlite3.Error as e:
        logger.error("Error clearing cache entry %s: %s", filename, e)
