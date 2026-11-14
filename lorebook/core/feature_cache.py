"""Feature cache management and database build pipeline."""

import concurrent.futures
import logging
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from lorebook.core.game_types import CARD_IMAGES_DIR
from lorebook.core.games import list_image_files
from lorebook.core.matching import MatchIndex
from lorebook.core.paths import data_path

logger = logging.getLogger(__name__)


def _cache_path(image_dir: str) -> str:
    """Return the SQLite cache file for a game's image folder."""
    # Strip trailing separators so "Card_Images/Lorcana/" (e.g. from shell
    # tab-completion) resolves to "Lorcana", not "".
    game = os.path.basename(image_dir.rstrip("/\\")) or "default"
    return data_path(f"DBCardCache_{game}.db")


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
    """Return a file's (mtime_ns, size) stamp, or None if it can't be read."""
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


def load_cache(image_dir: str) -> dict[str, np.ndarray]:
    """Load the cached {filename: vector} map for a game's image folder ({} if none)."""
    path = _cache_path(image_dir)
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


# Images per inference batch during builds. Reads are parallelized within a
# chunk; inference runs once per chunk on the calling thread (Keras predict is
# not guaranteed thread-safe, and its per-call overhead dominates per-image).
_BATCH_SIZE = 32


def _read_image(filename: str, image_dir: str) -> tuple[str, np.ndarray | None]:
    """Load one image for the build pipeline (thread-safe: no globals, no TF)."""
    return filename, cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_COLOR)


def _prune_missing(feature_db: dict[str, np.ndarray], current_files: list[str], image_dir: str) -> None:
    """Drop cached vectors whose image was deleted or renamed, so they can't keep matching.

    Only when the folder itself exists: a missing folder (typo'd path,
    unmounted drive) must not be read as "every image was deleted" and wipe
    the cache.
    """
    if not os.path.isdir(image_dir):
        return
    stale = sorted(set(feature_db) - set(current_files))
    if not stale:
        return
    for name in stale:
        feature_db.pop(name, None)
    _remove_cache_entries(_cache_path(image_dir), stale)
    logger.info("Pruned %s stale cache entries from %s", len(stale), image_dir)


def _drop_replaced(feature_db: dict[str, np.ndarray], disk: dict[str, Stamp | None], image_dir: str) -> None:
    """Drop vectors whose image was replaced under the same name, so it is re-extracted.

    A replaced image (e.g. re-downloaded with --force) no longer matches the
    size/mtime recorded with its vector. Legacy rows without a stamp are
    trusted and backfilled with the file's current stamp.
    """
    cache_file = _cache_path(image_dir)
    stored = _load_stamps(cache_file)
    changed = sorted(
        f for f in disk
        if f in feature_db and stored.get(f) is not None and stored[f] != disk[f]
    )
    if changed:
        for name in changed:
            feature_db.pop(name, None)
        _remove_cache_entries(cache_file, changed)
        logger.info("Re-extracting %s changed images in %s", len(changed), image_dir)
    _save_stamps(cache_file, {
        f: stamp for f in feature_db
        if stored.get(f) is None and (stamp := disk.get(f)) is not None
    })


def _extract_chunk(
    chunk: list[str], image_dir: str, extractor, pool: concurrent.futures.Executor
) -> dict[str, np.ndarray]:
    """Read one batch of images in parallel and extract their vectors in one call.

    Returns {filename: vector} for the images that worked; unreadable images
    and failed extractions are logged and left out.
    """
    readable = []
    for filename, image in pool.map(lambda f: _read_image(f, image_dir), chunk):
        if image is None:
            logger.warning("Could not read image %s", filename)
        else:
            readable.append((filename, image))

    vectors: dict[str, np.ndarray] = {}
    extracted = extractor.extract_batch([image for _, image in readable]) if readable else []
    for (filename, _), vector in zip(readable, extracted):
        if vector is None:
            logger.warning("Feature extraction failed for %s", filename)
        else:
            vectors[filename] = vector
    return vectors


def _extract_in_batches(
    new_files: list[str],
    image_dir: str,
    extractor,
    max_workers: int | None,
    cancel_event: threading.Event | None,
) -> Iterator[tuple[list[str], dict[str, np.ndarray]]]:
    """Yield (filenames, {filename: vector}) for each batch of new images.

    Stops before the next batch once cancel_event is set, so a cancelled
    build keeps everything extracted so far.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, len(new_files), _BATCH_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Build cancelled after %s/%s files in %s", start, len(new_files), image_dir)
                return
            chunk = new_files[start:start + _BATCH_SIZE]
            yield chunk, _extract_chunk(chunk, image_dir, extractor, pool)


def _save_cache(cache_file: str, feature_db: dict[str, np.ndarray], disk: dict[str, Stamp | None]) -> None:
    """Write every vector, with its image's stamp, to the SQLite cache (errors are logged)."""
    try:
        _init_db(cache_file)
        with sqlite3.connect(cache_file) as conn:
            # Cast defensively: load_cache reads blobs as float32, so any
            # other dtype here would corrupt the round-trip.
            rows = []
            for filename, vector in feature_db.items():
                mtime_ns, size = disk.get(filename) or (None, None)
                rows.append((filename, np.asarray(vector, dtype=np.float32).tobytes(), mtime_ns, size))
            conn.executemany(
                "INSERT OR REPLACE INTO features (filename, vector, mtime_ns, size) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.info("Saved %s entries to %s", len(feature_db), cache_file)
    except (sqlite3.Error, OSError) as e:
        logger.error("Error saving cache to %s: %s", cache_file, e)


def build_feature_database(
    image_dir: str,
    progress_callback: Callable[[int, str | None], None] | None = None,
    max_workers: int | None = None,
    extractor=None,
    cancel_event: threading.Event | None = None,
) -> dict[str, np.ndarray]:
    """
    Build or update the feature cache for the images in image_dir.

    Processes only images not already in the cache, updates the cache file, and
    returns the full feature dict. Image reads run in a thread pool; feature
    extraction runs in batches through extractor.extract_batch (defaults to the
    default backend, resolved lazily so importing this module never pulls in a
    model runtime).

    cancel_event: when set, the build stops between batches. Vectors extracted
    so far are still written to the cache, so the next build resumes from them.

    Unreadable images and failed extractions are logged, counted and skipped;
    anything unexpected propagates to the caller (it is not swallowed into a
    silently partial result).
    """
    feature_db: dict[str, np.ndarray] = load_cache(image_dir)
    logger.info("Loaded %s cached entries from %s", len(feature_db), image_dir)

    current_files = list_image_files(image_dir)
    _prune_missing(feature_db, current_files, image_dir)
    disk = {f: _file_stamp(os.path.join(image_dir, f)) for f in current_files}
    _drop_replaced(feature_db, disk, image_dir)

    new_files = [f for f in current_files if f not in feature_db]
    total = len(new_files)
    logger.info("Processing %s new files in %s", total, image_dir)
    if total == 0:
        if progress_callback:
            progress_callback(100, None)
        return feature_db

    if extractor is None:
        from lorebook.core.features import get_extractor  # lazy model-runtime import
        extractor = get_extractor()

    extracted = done = 0
    for chunk, vectors in _extract_in_batches(new_files, image_dir, extractor, max_workers, cancel_event):
        feature_db.update(vectors)
        extracted += len(vectors)
        done += len(chunk)
        if progress_callback:
            progress_callback(int(done / total * 100), chunk[-1])

    _save_cache(_cache_path(image_dir), feature_db, disk)
    logger.info("Build complete: %s ok, %s failed", extracted, done - extracted)
    return feature_db


def clear_from_cache(filename: str, image_dir: str) -> None:
    """Remove a single entry from the SQLite cache (useful after deleting an image)."""
    path = _cache_path(image_dir)
    if not os.path.exists(path):
        return
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM features WHERE filename = ?", (filename,))
        logger.info("Cleared %s from %s", filename, path)
    except sqlite3.Error as e:
        logger.error("Error clearing cache entry %s: %s", filename, e)


class GameDatabase:
    """
    The active game's reference vectors and match index, loaded together.

    Replaces the old pattern of a module-global "current database path" plus
    loose feature_db/index/loaded-game fields on the window: one object owns
    the invariant that the index always matches the loaded vectors. Not
    thread-safe — the owner (GUI main thread) mutates it; workers only ever
    get the immutable MatchIndex it hands out.
    """

    def __init__(self, base: str = CARD_IMAGES_DIR):
        """Create an empty database that loads games from folders under base."""
        self._base = base
        self.game: str | None = None
        self.vectors: dict[str, np.ndarray] = {}
        self.index = MatchIndex({})

    def __len__(self) -> int:
        """Return the number of loaded reference vectors."""
        return len(self.vectors)

    def load(self, game: str) -> bool:
        """Load game's cache (replacing whatever was loaded). True if non-empty."""
        self.vectors = load_cache(os.path.join(self._base, game))
        self.index = MatchIndex(self.vectors)
        self.game = game if self.vectors else None
        logger.info("Loaded %s entries for %s", len(self.vectors), game)
        return bool(self.vectors)

    def ensure(self, game: str) -> bool:
        """Load game unless it's already loaded (and non-empty). True if non-empty."""
        if self.game != game or not self.vectors:
            return self.load(game)
        return True

    def invalidate(self) -> None:
        """Force the next ensure() to re-read from disk (e.g. after a rebuild)."""
        self.game = None
