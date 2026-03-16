# card_database.py — Feature cache management and database build pipeline.

import concurrent.futures
import logging
import os
import sqlite3
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from game_types import BASE_DATABASE_PATH, SUPPORTED_EXTS

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
    game = os.path.basename(db_path) or "default"
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


def _list_image_files(folder: str) -> List[str]:
    """Return sorted list of supported image filenames in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTS))


def _process_image(filename: str, db_path: str) -> Tuple[str, Optional[np.ndarray]]:
    """
    Extract features for a single image. db_path is passed explicitly so this
    function is safe to call from multiple threads without relying on the global.
    """
    from features import extract_features  # lazy: avoids importing TF at module level
    img = cv2.imread(os.path.join(db_path, filename), cv2.IMREAD_COLOR)
    return (filename, extract_features(img)) if img is not None else (filename, None)


def build_feature_database(
    progress_callback: Optional[Callable[[int, Optional[str]], None]] = None,
    max_workers: Optional[int] = None,
    db_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Build or update the feature database for db_path (defaults to databasePath).

    Processes only images not already in the cache, updates the cache file, and
    returns the full feature dict. Uses a thread pool for parallel extraction.
    """
    resolved = db_path or databasePath
    featureDB: Dict[str, np.ndarray] = {}
    try:
        featureDB = load_cache(resolved)
        logging.info(f"Loaded {len(featureDB)} cached entries from {resolved}")

        new_files = [f for f in _list_image_files(resolved) if f not in featureDB]
        total = len(new_files)
        logging.info(f"Processing {total} new files in {resolved}")

        if total == 0:
            if progress_callback:
                progress_callback(100, None)
            return featureDB

        successful = failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_image, f, resolved): f for f in new_files}
            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                fname = futures[future]
                try:
                    k, v = future.result()
                    if v is not None:
                        featureDB[k] = v
                        successful += 1
                    else:
                        failed += 1
                        logging.warning(f"Feature extraction failed for {fname}")
                except Exception as e:
                    failed += 1
                    logging.error(f"Error processing {fname}: {e}")
                if progress_callback:
                    progress_callback(int(idx / total * 100), fname)

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
