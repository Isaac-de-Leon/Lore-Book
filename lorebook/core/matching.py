# matching.py — Vector normalization and cosine-similarity matching.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    sk_normalize = None

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
except Exception:
    sk_cosine = None


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, using sklearn if available."""
    if sk_normalize is not None:
        return sk_normalize([vec])[0]
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity for L2-normalized vectors.
    Raises ValueError if either vector is not L2-normalized.
    """
    def _is_normalized(v: np.ndarray, tol: float = 1e-3) -> bool:
        return abs(np.linalg.norm(v) - 1.0) < tol

    if not _is_normalized(a) or not _is_normalized(b):
        raise ValueError("Both input vectors must be l2-normalized.")
    if sk_cosine is not None:
        return float(sk_cosine([a], [b])[0][0])
    return float(np.dot(a, b))


def find_best_matches(
    inputFeatures: np.ndarray,
    featureDB: Dict[str, np.ndarray],
    threshold: float = 0.70,
) -> List[Tuple[str, float]]:
    """Return all DB entries with cosine similarity >= threshold, sorted descending."""
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
