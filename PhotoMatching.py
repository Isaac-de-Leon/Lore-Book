# PhotoMatching.py
# Feature DB + cosine match + heatmap + simple foil detector
# CSV writer uses ONLY CardList.csv (4 columns: Set Number, Card Number, Variant, Count)

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import csv
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

# ---- Globals & paths -----------------------------------------------------------
SUPPORTED_EXTS = (".webp", ".jpg", ".jpeg", ".png")
baseDatabasePath = "Card_Images"
databasePath = os.path.join(baseDatabasePath, "Lorcana")  # default game

# The ONLY CSV we write to:
CARDLIST_FILE = "CardList.csv"

# Lazy-loaded models
_base_model: Optional[Model] = None
_feat_model: Optional[Model] = None         # 1280-dim pooled features
_act_model: Optional[Model] = None          # last conv (for heatmaps)

def _cache_path() -> str:
    # Returns the path to the feature cache file for the current game.
    game = os.path.basename(databasePath) or "default"
    return f"DBCardCache_{game}.json"

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
    """Extract a normalized 1280-dim feature vector from an image or image path."""
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
    else:
        img = img_or_path
    if img is None or img.size == 0:
        return None

    # Ensure 3-channel RGB for MobileNetV2
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # BGR -> RGB, resize, preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    processed = preprocess_input(batch)

    feat_model, _ = _get_models()
    features = feat_model.predict(processed, verbose=0).flatten().astype(np.float32)
    if features.size == 0:
        return None
    return _l2_normalize(features)

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

def load_cache() -> Dict[str, np.ndarray]:
    # Load the feature cache from disk, if present.
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    except Exception:
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

def build_feature_database(progress_callback: Optional[Callable[[int, Optional[str]], None]] = None
                           ) -> Dict[str, np.ndarray]:
    """Build/update per-game feature DB from images in `databasePath`."""
    featureDB = load_cache()
    imageFiles = _list_image_files(databasePath)
    files_to_process = [f for f in imageFiles if f not in featureDB]
    total = len(files_to_process)

    if total == 0:
        if progress_callback:
            progress_callback(100, None)
        return featureDB

    # Extract features in parallel for new images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, (k, v) in enumerate(executor.map(process_image, files_to_process), 1):
            if v is not None:
                featureDB[k] = v
            if progress_callback:
                progress_callback(int(idx / total * 100), k)

    # Save cache
    try:
        with open(_cache_path(), "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in featureDB.items()}, f, indent=2)
    except Exception:
        pass
    return featureDB

def clear_from_cache(filename: str) -> None:
    """Remove a single filename from the cache file (helpful if you delete the image)."""
    path = _cache_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if filename in data:
            del data[filename]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except Exception:
        pass

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
    Read an existing CSV that might be 4- or 5-columns and return rows as
    [Set Number, Card Number, Variant, Count].
    """
    rows: List[List[str]] = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if first:
                first = False
                if row and any("set" in c.lower() for c in row):
                    continue  # skip header
            if not row:
                continue
            if len(row) >= 4:
                count = row[4] if len(row) > 4 else row[3]  # ignore a 5th "Tag" if present
                if count == "":
                    count = "0"
                rows.append([row[0], row[1], row[2], count])
            elif len(row) == 3:
                rows.append([row[0], row[1], row[2], "0"])
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
    # Split filename into set code and card code.
    base = os.path.splitext(os.path.basename(matchedFilename))[0]
    splitCard = base.split("-")
    if len(splitCard) < 2:
        return "", base
    return splitCard[0], splitCard[1]

def update_cardlist(matchedFilename: str, is_foil: bool, count: int = 1) -> None:
    """
    Append/increment a row in CardList.csv (4 columns):
    Set Number, Card Number, Variant, Count
    """
    try:
        count = int(count)
    except Exception:
        count = 1
    if count < 1:
        return

    set_code, card_code = _split_filename(matchedFilename)
    variant = "foil" if is_foil else "normal"

    existing = _normalize_existing_rows(CARDLIST_FILE)
    for r in existing:
        if r[0] == set_code and r[1] == card_code and r[2] == variant:
            try:
                r[3] = str(int(r[3]) + count)
            except Exception:
                r[3] = str(count)
            break
    else:
        existing.append([set_code, card_code, variant, str(count)])

    _write_rows_4col(CARDLIST_FILE, existing)

def get_available_sets() -> List[str]:
    # Return a sorted list of all set codes found in the database folder.
    sets = set()
    for fname in _list_image_files(databasePath):
        if len(fname) >= 3 and fname[:3].isdigit():
            sets.add(fname[:3])
    return sorted(sets)

# --- Simple foil heuristic ------------------------------------------------------

def foil_score(img_bgr: np.ndarray) -> float:
    """Return a 0..1 score of 'foil-likeness' based on specular highlights and local contrast."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bright = (gray > 240).astype(np.uint8)
    bright_ratio = float(bright.mean())
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    contrast = float(np.mean(np.abs(lap))) / 255.0
    return float(np.clip(0.7 * bright_ratio + 0.3 * contrast, 0.0, 1.0))

def is_probably_foil(img_bgr: np.ndarray, threshold: float = 0.08) -> bool:
    # Returns True if the image is likely a foil card.
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
