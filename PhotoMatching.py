# PhotoMatching.py
# Feature DB + cosine match + heatmap + simple foil detector
# CSV writer uses ONLY CardList.csv (4 columns: Set Number, Card Number, Variant, Count)

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import csv
import sqlite3
import cv2
import numpy as np
import concurrent.futures
import warnings
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import tensorflow as tf

# Suppress various warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---- Optional sklearn fallbacks ------------------------------------------------
try:
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    sk_normalize = None

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
except Exception:
    sk_cosine = None

# ---- TensorFlow / Keras (use tf.keras for broad compatibility) ----------------
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model

# ---- Game Types & Constants -----------------------------------------------------
from enum import Enum
from pathlib import Path

class GameType(Enum):
    LORCANA = "lorcana"
    RIFTBOUND = "riftbound"
    UNKNOWN = "unknown"

def get_game_type(filepath: str) -> GameType:
    """
    Determine game type based on image file location.
    Resolves path components and checks if the path contains game-specific folders.

    Args:
        filepath: Path to check, can be relative or absolute

    Returns:
        GameType enum value indicating detected game or UNKNOWN
    """
    try:
        # Resolve any .. path components and normalize path
        abs_path = str(Path(filepath).resolve())
        path_parts = [p.lower() for p in Path(abs_path).parts]
        
        if "riftbound" in path_parts:
            return GameType.RIFTBOUND
        elif "lorcana" in path_parts:
            return GameType.LORCANA
        else:
            return GameType.UNKNOWN
    except Exception as e:
        logging.error(f"Error determining game type for {filepath}: {e}")
        return GameType.UNKNOWN

# ---- Globals & paths -----------------------------------------------------------
SUPPORTED_EXTS = (".webp", ".jpg", ".jpeg", ".png")
baseDatabasePath = "Card_Images"
databasePath = os.path.join(baseDatabasePath, "Lorcana")  # default game

# CSV file constants
LORCANA_FILE = "LorcanaList.csv"
RIFTBOUND_FILE = "RiftboundList.csv"

# Lazy-loaded models
_base_model: Optional[Model] = None
_feat_model: Optional[Model] = None         # 1280-dim pooled features
_act_model: Optional[Model] = None          # last conv (for heatmaps)

def _cache_path() -> str:
    # Returns the path to the SQLite feature cache for the current game.
    game = os.path.basename(databasePath) or "default"
    return f"DBCardCache_{game}.db"

def set_database_path(game_name: str) -> None:
    """Switch the database folder to a different game (e.g., 'Lorcana', 'Pokémon')."""
    global databasePath
    databasePath = os.path.join(baseDatabasePath, game_name)
    os.makedirs(databasePath, exist_ok=True)

def _get_models() -> Tuple[Model, Model]:
    """Return (feature_model, activation_model), lazy-initialized."""
    global _base_model, _feat_model, _act_model
    if _feat_model is None or _act_model is None:
        # Load MobileNetV2 and create two models: one for features, one for activations
        _base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        _feat_model = Model(inputs=_base_model.input, outputs=_base_model.output)  # (1280,)
        # Last conv activation (7x7x1280) used for heatmaps
        _act_model = Model(inputs=_base_model.input, outputs=_base_model.layers[-2].output)
    return _feat_model, _act_model

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    # L2-normalize a vector, using sklearn if available.
    if sk_normalize is not None:
        return sk_normalize([vec])[0]
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v

def extract_features(img_or_path: Union[np.ndarray, str]) -> Optional[np.ndarray]:
    """
    Extract a normalized 1280-dim feature vector from an image or image path.
    
    Uses MobileNetV2 pretrained on ImageNet to extract features from the
    average pooling layer, resulting in a 1280-dimensional vector that is
    then L2-normalized.
    
    Args:
        img_or_path: Either a BGR image array or path to an image file
        
    Returns:
        1280-dim normalized feature vector, or None if extraction fails
    """
    try:
        # Handle string path input
        if isinstance(img_or_path, str):
            try:
                img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
            except Exception as e:
                logging.error(f"Error reading image file {img_or_path}: {e}")
                return None
        else:
            img = img_or_path
            
        # Validate and normalize to BGR
        img = ensure_valid_image(img)
        if img is None:
            return None

        # Convert BGR -> RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Prepare batch and preprocess
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        processed = preprocess_input(batch)

        # Extract features
        feat_model, _ = _get_models()
        features = feat_model.predict(processed, verbose=0).flatten().astype(np.float32)
        
        # Validate output
        if features.size != 1280:
            logging.error(f"Unexpected feature dimension: {features.size}")
            return None
            
        # Normalize and return
        return _l2_normalize(features)
        
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def visualize_activation_overlay(img_bgr: np.ndarray, model: Optional[Model] = None) -> np.ndarray:
    """Return an image with a jet heatmap overlay of average last-conv activations."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    _, act_model = _get_models()
    use_model = model or act_model

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    processed = preprocess_input(batch)

    activations = use_model.predict(processed, verbose=0)[0]  # (7,7,1280)
    heatmap = activations.mean(axis=-1)  # (7,7)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0,255]
    hmin, hmax = float(np.min(heatmap)), float(np.max(heatmap))
    if hmax > hmin:
        heatmap = (255 * (heatmap - hmin) / (hmax - hmin)).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

def _init_db(path: str) -> None:
    """Create the features table if it doesn't already exist."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS features "
            "(filename TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )


def load_cache() -> Dict[str, np.ndarray]:
    """Load the feature cache from the SQLite database, if present."""
    path = _cache_path()
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
    # List all supported image files in a folder.
    if not os.path.isdir(folder):
        return []
    return sorted([n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTS)])

def process_image(filename: str) -> Tuple[str, Optional[np.ndarray]]:
    # Helper for parallel feature extraction.
    img_path = os.path.join(databasePath, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return (filename, extract_features(img)) if img is not None else (filename, None)

def build_feature_database(
    progress_callback: Optional[Callable[[int, Optional[str]], None]] = None,
    max_workers: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Build or update the feature database for the current game.
    
    Processes all images in the current databasePath, extracting features
    for new images and updating the cache file. Uses parallel processing
    for better performance.
    
    Args:
        progress_callback: Optional function to report progress (0-100) and current file
        max_workers: Optional limit on number of parallel workers
        
    Returns:
        Dictionary mapping filenames to feature vectors
    """
    try:
        # Load existing cache
        featureDB = load_cache()
        logging.info(f"Loaded {len(featureDB)} existing entries from cache")
        
        # Find new files to process
        imageFiles = _list_image_files(databasePath)
        files_to_process = [f for f in imageFiles if f not in featureDB]
        total = len(files_to_process)
        
        logging.info(f"Found {total} new files to process in {databasePath}")

        if total == 0:
            if progress_callback:
                progress_callback(100, None)
            return featureDB

        # Process new images in parallel
        successful = 0
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_image, f): f for f in files_to_process
            }
            
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                fname = future_to_file[future]
                try:
                    k, v = future.result()
                    if v is not None:
                        featureDB[k] = v
                        successful += 1
                    else:
                        failed += 1
                        logging.warning(f"Failed to extract features from {fname}")
                except Exception as e:
                    failed += 1
                    logging.error(f"Error processing {fname}: {e}")
                
                if progress_callback:
                    progress_callback(int(idx / total * 100), fname)

        # Save updated cache to SQLite
        cache_path = _cache_path()
        try:
            _init_db(cache_path)
            with sqlite3.connect(cache_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO features (filename, vector) VALUES (?, ?)",
                    [(k, v.tobytes()) for k, v in featureDB.items()],
                )
            logging.info(f"Saved {len(featureDB)} entries to {cache_path}")
        except Exception as e:
            logging.error(f"Error saving cache to {cache_path}: {e}")
            
        # Log summary
        logging.info(f"Database build complete: {successful} succeeded, {failed} failed")
        return featureDB
        
    except Exception as e:
        logging.error(f"Error building feature database: {e}")
        return featureDB

def clear_from_cache(filename: str) -> None:
    """Remove a single filename from the SQLite cache (helpful if you delete the image)."""
    path = _cache_path()
    if not os.path.exists(path):
        return
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM features WHERE filename = ?", (filename,))
        logging.info(f"Cleared {filename} from cache {path}")
    except Exception as e:
        logging.error(f"Unexpected error while clearing cache entry {filename}: {e}")

def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity for l2-normalized vectors.
    Both input vectors must be l2-normalized; otherwise, results may be incorrect.
    """
    def is_l2_normalized(v: np.ndarray, tol: float = 1e-3) -> bool:
        norm = np.linalg.norm(v)
        return abs(norm - 1.0) < tol

    if not is_l2_normalized(a) or not is_l2_normalized(b):
        raise ValueError("Both input vectors must be l2-normalized.")

    if sk_cosine is not None:
        return float(sk_cosine([a], [b])[0][0])
    return float(np.dot(a, b))  # both are l2-normalized

def find_best_matches(inputFeatures: np.ndarray, featureDB: Dict[str, np.ndarray],
                      threshold: float = 0.70) -> List[Tuple[str, float]]:
    # Find all DB entries with cosine similarity above threshold, sorted descending.
    if inputFeatures is None or inputFeatures.size == 0 or not featureDB:
        return []
    q = _l2_normalize(inputFeatures)
    scored: List[Tuple[str, float]] = []
    for fname, vec in featureDB.items():
        if vec is None or np.size(vec) == 0:
            continue
        score = _cosine_score(q, vec)
        if score >= threshold:
            scored.append((fname, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

# ---------- CardList.csv writer (4 columns) ------------------------------------

def _normalize_existing_rows(csv_path: str) -> List[List[str]]:
    """
    Read an existing CSV that might be 4- or 5-columns and normalize to 4 columns.
    
    Normalizes to format: [Set Number, Card Number, Variant, Count]
    
    Args:
        csv_path: Path to CSV file to read
        
    Returns:
        List of normalized 4-column rows. Empty list if file doesn't exist
        or can't be read.
        
    Handles:
    - Files with or without headers
    - 3-column format (adds count=0)
    - 4-column format 
    - 5-column format (ignores 5th "Tag" column)
    - Empty/missing files
    """
    rows: List[List[str]] = []
    if not os.path.exists(csv_path):
        return rows
        
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first = True
            for row_num, row in enumerate(reader, 1):
                try:
                    if first:
                        first = False
                        if row and any("set" in c.lower() for c in row):
                            continue  # skip header
                    if not row:
                        continue
                    if len(row) >= 4:
                        count = row[4] if len(row) > 4 else row[3]
                        count = "0" if count == "" else count
                        rows.append([row[0], row[1], row[2], count])
                    elif len(row) == 3:
                        rows.append([row[0], row[1], row[2], "0"])
                except Exception as e:
                    logging.warning(f"Error processing row {row_num} in {csv_path}: {e}")
                    continue
        return rows
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        return rows

def _write_rows_4col(csv_path: str, rows: List[List[str]]) -> None:
    # Write rows to CSV with 4 columns and header.
    header = ["Set Number", "Card Number", "Variant", "Count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3]])

def _split_filename(matchedFilename: str) -> Tuple[str, str]:
    """
    Split filename into set code and card code.
    
    Handles various formats:
    - Numeric (001-001)
    - Alphanumeric (ONG-23c)
    - Complex variants (ONG-23c-alt)
    
    Args:
        matchedFilename: Filename (with or without path) to parse
        
    Returns:
        Tuple of (set_code, card_code). If no set code can be determined,
        returns ("", original_basename)
        
    Examples:
        >>> _split_filename("ONG-23c-alt.jpg")
        ('ONG', '23c-alt')
        >>> _split_filename("001-001.png")
        ('001', '001')
    """
    try:
        base = os.path.splitext(os.path.basename(matchedFilename))[0]
        splitCard = base.split("-", maxsplit=2)  # Split only on first hyphen
        if len(splitCard) < 2:
            return "", base
            
        # Keep set code and handle the rest as card code
        set_code = splitCard[0]
        
        # Join remaining parts to preserve suffixes
        card_code = "-".join(splitCard[1:])
            
        return set_code, card_code
    except Exception as e:
        logging.error(f"Error splitting filename {matchedFilename}: {e}")
        return "", matchedFilename

def update_cardlist(matchedFilename: str, is_foil: bool, count: int = 1) -> None:
    """
    Append/increment a row in game-specific CSV (4 columns):
    Set Number, Card Number, Variant, Count
    """
    try:
        count = int(count)
    except Exception:
        count = 1
    if count < 1:
        return

    game_type = get_game_type(matchedFilename)
    target_file = RIFTBOUND_FILE if game_type == GameType.RIFTBOUND else LORCANA_FILE

    set_code, card_code = _split_filename(matchedFilename)
    variant = "foil" if is_foil else "normal"

    existing = _normalize_existing_rows(target_file)
    for r in existing:
        if r[0] == set_code and r[1] == card_code and r[2] == variant:
            try:
                r[3] = str(int(r[3]) + count)
            except Exception:
                r[3] = str(count)
            break
    else:
        existing.append([set_code, card_code, variant, str(count)])

    _write_rows_4col(target_file, existing)

def get_available_sets(game_type: Optional[GameType] = None) -> List[str]:
    """Return a sorted list of all set codes found in the database folder."""
    sets = set()
    
    # If no game type specified, use current databasePath's game type
    if game_type is None:
        game_type = get_game_type(databasePath)
    
    # Get files from the correct folder
    files = _list_image_files(databasePath)
    
    for fname in files:
        set_code, _ = _split_filename(fname)
        if set_code:  # Only add if we got a valid set code
            if game_type == GameType.LORCANA and set_code.isdigit():
                sets.add(set_code)
            elif game_type == GameType.RIFTBOUND:
                sets.add(set_code)
    return sorted(sets)

# --- Image Processing Utilities ------------------------------------------------------

def ensure_valid_image(img_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Validate and normalize an input image to 3-channel BGR format.
    
    Args:
        img_bgr: Input image array or None
        
    Returns:
        Normalized 3-channel BGR image or None if input is invalid
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
        
    # Convert grayscale to BGR
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
    # Strip alpha channel if present
    if img_bgr.shape[-1] == 4:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        
    # Already BGR
    if img_bgr.shape[-1] == 3:
        return img_bgr
        
    return None

def foil_score(img_bgr: np.ndarray) -> float:
    """
    Calculate a 0..1 score indicating how likely an image is to be a foil card.
    
    Uses a combination of:
    - Bright spots ratio (specular highlights)
    - Local contrast (textural detail)
    
    Args:
        img_bgr: BGR format image array
        
    Returns:
        Score between 0.0 and 1.0, higher means more likely to be foil
    """
    img = ensure_valid_image(img_bgr)
    if img is None:
        return 0.0
        
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate bright spots ratio
        bright = (gray > 240).astype(np.uint8)
        bright_ratio = float(bright.mean())
        
        # Calculate local contrast
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        contrast = float(np.mean(np.abs(lap))) / 255.0
        
        # Combine metrics (70% brightness, 30% contrast)
        return float(np.clip(0.7 * bright_ratio + 0.3 * contrast, 0.0, 1.0))
        
    except Exception as e:
        logging.error(f"Error calculating foil score: {e}")
        return 0.0

def is_probably_foil(img_bgr: np.ndarray, threshold: float = 0.08) -> bool:
    """
    Determine if an image is likely to be a foil card.
    
    Args:
        img_bgr: BGR format image array
        threshold: Score threshold, default 0.08 tuned empirically
        
    Returns:
        True if the image appears to be a foil card
    """
    return foil_score(img_bgr) >= threshold

# ---- CLI (optional) ------------------------------------------------------------
if __name__ == "__main__":
    # Optional command-line interface for building DB and matching images.
    import argparse
    parser = argparse.ArgumentParser(description="Build feature DB and/or match an image.")
    parser.add_argument("--game", type=str, default="Lorcana", help="Game folder inside Card_Images/")
    parser.add_argument("--build", action="store_true", help="Build/update feature DB.")
    parser.add_argument("--match", type=str, help="Path to an input image to match.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold.")
    args = parser.parse_args()

    set_database_path(args.game)

    if args.build:
        def cb(p, k): print(f"{p:3d}% - {k or ''}")
        db = build_feature_database(cb)
        print(f"DB entries: {len(db)}")

    if args.match:
        db = load_cache()
        feat = extract_features(args.match)
        matches = find_best_matches(feat, db, threshold=args.threshold)
        for fname, score in matches[:10]:
            print(f"{score:.4f}  {fname}")
