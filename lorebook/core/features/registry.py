"""Choose a feature backend and hand out its (cached) extractor."""

import importlib.util
import os

import numpy as np

from lorebook.core.features.keras_backend import KerasExtractor
from lorebook.core.features.onnx_backend import OnnxExtractor
from lorebook.core.features.tflite_backend import TFLiteExtractor

BACKENDS = ("onnx", "keras", "tflite")


def default_backend() -> str:
    """
    The extractor used when none is named: LOREBOOK_BACKEND if set, else
    "onnx" when onnxruntime is installed, else "keras".
    """
    override = os.environ.get("LOREBOOK_BACKEND")
    if override:
        return override.lower()
    return "onnx" if importlib.util.find_spec("onnxruntime") is not None else "keras"


_extractors: dict = {}


def get_extractor(backend: str | None = None):
    """
    Return a cached feature extractor for the given backend (default_backend() if None).

    The returned object exposes `.extract(img_or_path) -> Optional[np.ndarray]`,
    yielding a 1280-dim L2-normalized vector. Supported backends: see BACKENDS.
    """
    key = (backend or default_backend()).lower()
    if key not in _extractors:
        if key == "onnx":
            _extractors[key] = OnnxExtractor()
        elif key == "keras":
            _extractors[key] = KerasExtractor()
        elif key == "tflite":
            _extractors[key] = TFLiteExtractor()
        else:
            source = "" if backend else " (from LOREBOOK_BACKEND)"
            raise ValueError(
                f"Unknown extractor backend: {key!r}{source} (use one of {', '.join(BACKENDS)})."
            )
    return _extractors[key]


def extract_features(img_or_path: np.ndarray | str) -> np.ndarray | None:
    """
    Extract a normalized 1280-dim feature vector from an image or file path.

    Uses default_backend(); the model is loaded lazily on first call. For a
    specific backend use get_extractor(backend).extract(...).
    """
    return get_extractor().extract(img_or_path)
