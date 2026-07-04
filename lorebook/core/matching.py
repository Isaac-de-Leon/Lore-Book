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


class MatchIndex:
    """
    Vectorized matcher: the feature DB stacked into one (N, dim) matrix.

    Build once from a {filename: vector} dict, then score each query with a
    single matrix-vector product instead of a per-entry Python loop — the
    per-card matching cost that matters on a Raspberry Pi. Results are
    identical in contract to find_best_matches: [(filename, score), ...]
    sorted descending, filtered by threshold.
    """

    def __init__(self, featureDB: Dict[str, np.ndarray]):
        names: List[str] = []
        vectors: List[np.ndarray] = []
        dim: Optional[int] = None
        for fname, vec in (featureDB or {}).items():
            if vec is None or np.size(vec) == 0:
                continue
            v = np.asarray(vec, dtype=np.float32).ravel()
            if dim is None:
                dim = v.size
            elif v.size != dim:
                logging.warning(
                    f"MatchIndex: skipping {fname} (dim {v.size} != {dim})"
                )
                continue
            names.append(fname)
            vectors.append(v)
        self._names = names
        self._matrix = np.vstack(vectors) if vectors else np.empty((0, dim or 0), np.float32)

    def __len__(self) -> int:
        return len(self._names)

    def find(self, inputFeatures: np.ndarray, threshold: float = 0.70) -> List[Tuple[str, float]]:
        """Return all entries with cosine similarity >= threshold, sorted descending."""
        if inputFeatures is None or np.size(inputFeatures) == 0 or not self._names:
            return []
        q = _l2_normalize(np.asarray(inputFeatures, dtype=np.float32).ravel())
        if q.size != self._matrix.shape[1]:
            logging.error(
                f"MatchIndex: query dim {q.size} != index dim {self._matrix.shape[1]}"
            )
            return []
        scores = self._matrix @ q
        hits = np.flatnonzero(scores >= threshold)
        order = hits[np.argsort(scores[hits])[::-1]]
        return [(self._names[i], float(scores[i])) for i in order]
