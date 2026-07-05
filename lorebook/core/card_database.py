# card_database.py — Feature cache management and database build pipeline.

import concurrent.futures
import logging
import os
import sqlite3
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lorebook.core.game_types import BASE_DATABASE_PATH, SUPPORTED_EXTS

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
    """Create the features table if it doesn't already exist."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS features "
            "(filename TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )


def load_cache(db_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load the feature cache from the SQLite database for the given (or current) database path."""
    path = _cache_path(db_path or databasePath)
    if not os.path.exists(path):
        return {}
    try:
        result: Dict[str, np.ndarray] = {}
        with sqlite3.connect(path) as conn:
            for filename, blob in conn.execute("SELECT filename, vector FROM features"):
                result[filename] = np.frombuffer(blob, dtype=np.float32).copy()
        return result
    except Exception as e:
        logging.error(f"Unexpected error loading cache from {path}: {e}")
        return {}


def _remove_cache_entries(cache_file: str, filenames: List[str]) -> None:
    """Delete cache rows whose source images no longer exist on disk."""
    if not filenames or not os.path.exists(cache_file):
        return
    try:
        with sqlite3.connect(cache_file) as conn:
            conn.executemany(
                "DELETE FROM features WHERE filename = ?", [(n,) for n in filenames]
            )
    except Exception as e:
        logging.error(f"Error pruning stale cache entries from {cache_file}: {e}")


def _list_image_files(folder: str) -> List[str]:
    """Return sorted list of supported image filenames in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTS))


# Images per inference batch during builds. Reads are parallelized within a
# chunk; inference runs once per chunk on the calling thread (Keras predict is
# not guaranteed thread-safe, and its per-call overhead dominates per-image).
_BATCH_SIZE = 32


def _read_image(filename: str, db_path: str) -> Tuple[str, Optional[np.ndarray]]:
    """Load one image for the build pipeline (thread-safe: no globals, no TF)."""
    return filename, cv2.imread(os.path.join(db_path, filename), cv2.IMREAD_COLOR)


def build_feature_database(
    progress_callback: Optional[Callable[[int, Optional[str]], None]] = None,
    max_workers: Optional[int] = None,
    db_path: Optional[str] = None,
    extractor=None,
) -> Dict[str, np.ndarray]:
    """
    Build or update the feature database for db_path (defaults to databasePath).

    Processes only images not already in the cache, updates the cache file, and
    returns the full feature dict. Image reads run in a thread pool; feature
    extraction runs in batches through extractor.extract_batch (defaults to the
    keras backend, resolved lazily so importing this module never pulls in TF).
    """
    resolved = db_path or databasePath
    featureDB: Dict[str, np.ndarray] = {}
    try:
        featureDB = load_cache(resolved)
        logging.info(f"Loaded {len(featureDB)} cached entries from {resolved}")

        current_files = _list_image_files(resolved)

        # Prune cache entries for deleted/renamed images so they can't keep
        # matching against cards that no longer exist. Only when the folder
        # itself exists — a missing folder (typo'd path, unmounted drive)
        # must not be read as "every image was deleted" and wipe the cache.
        stale = sorted(set(featureDB) - set(current_files)) if os.path.isdir(resolved) else []
        if stale:
            for name in stale:
                featureDB.pop(name, None)
            _remove_cache_entries(_cache_path(resolved), stale)
            logging.info(f"Pruned {len(stale)} stale cache entries from {resolved}")

        new_files = [f for f in current_files if f not in featureDB]
        total = len(new_files)
        logging.info(f"Processing {total} new files in {resolved}")

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
                chunk = new_files[start:start + _BATCH_SIZE]
                loaded = list(pool.map(lambda f: _read_image(f, resolved), chunk))

                readable = []
                for fname, img in loaded:
                    if img is None:
                        failed += 1
                        logging.warning(f"Could not read image {fname}")
                    else:
                        readable.append((fname, img))

                vectors = extractor.extract_batch([img for _, img in readable]) if readable else []
                for (fname, _), vec in zip(readable, vectors):
                    if vec is not None:
                        featureDB[fname] = vec
                        successful += 1
                    else:
                        failed += 1
                        logging.warning(f"Feature extraction failed for {fname}")

                done += len(chunk)
                if progress_callback:
                    progress_callback(int(done / total * 100), chunk[-1])

        cache_file = _cache_path(resolved)
        try:
            _init_db(cache_file)
            with sqlite3.connect(cache_file) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO features (filename, vector) VALUES (?, ?)",
                    [(k, v.tobytes()) for k, v in featureDB.items()],
                )
            logging.info(f"Saved {len(featureDB)} entries to {cache_file}")
        except Exception as e:
            logging.error(f"Error saving cache to {cache_file}: {e}")

        logging.info(f"Build complete: {successful} ok, {failed} failed")
        return featureDB

    except Exception as e:
        logging.error(f"Error building feature database: {e}")
        return featureDB


def clear_from_cache(filename: str, db_path: Optional[str] = None) -> None:
    """Remove a single entry from the SQLite cache (useful after deleting an image)."""
    path = _cache_path(db_path or databasePath)
    if not os.path.exists(path):
        return
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM features WHERE filename = ?", (filename,))
        logging.info(f"Cleared {filename} from {path}")
    except Exception as e:
        logging.error(f"Error clearing cache entry {filename}: {e}")
