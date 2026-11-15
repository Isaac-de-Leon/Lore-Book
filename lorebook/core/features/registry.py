"""Choose a feature backend and hand out its (cached) extractor."""

import importlib.util
import os
from collections.abc import Callable
from enum import StrEnum

import numpy as np

from lorebook.core.features.extractor import FeatureExtractor, ImageInput
from lorebook.core.features.keras_backend import KerasExtractor
from lorebook.core.features.onnx_backend import OnnxExtractor
from lorebook.core.features.tflite_backend import TFLiteExtractor


class Backend(StrEnum):
    """The feature backends, by the name used on the command line and in LOREBOOK_BACKEND."""

    ONNX = "onnx"
    KERAS = "keras"
    TFLITE = "tflite"


BACKENDS = tuple(backend.value for backend in Backend)

_FACTORIES: dict[str, Callable[[], FeatureExtractor]] = {
    Backend.ONNX: OnnxExtractor,
    Backend.KERAS: KerasExtractor,
    Backend.TFLITE: TFLiteExtractor,
}


def default_backend() -> str:
    """
    The extractor used when none is named: LOREBOOK_BACKEND if set, else
    "onnx" when onnxruntime is installed, else "keras".
    """
    override = os.environ.get("LOREBOOK_BACKEND")
    if override:
        return override.lower()
    return Backend.ONNX if importlib.util.find_spec("onnxruntime") is not None else Backend.KERAS


_extractors: dict[str, FeatureExtractor] = {}


def get_extractor(backend: str | None = None) -> FeatureExtractor:
    """
    Return a cached feature extractor for the given backend (default_backend() if None).

    The returned object exposes `.extract(img_or_path) -> Optional[np.ndarray]`,
    yielding a 1280-dim L2-normalized vector. Supported backends: see BACKENDS.
    """
    key = (backend or default_backend()).lower()
    if key not in _extractors:
        factory = _FACTORIES.get(key)
        if factory is None:
            source = "" if backend else " (from LOREBOOK_BACKEND)"
            raise ValueError(
                f"Unknown extractor backend: {key!r}{source} (use one of {', '.join(BACKENDS)})."
            )
        _extractors[key] = factory()
    return _extractors[key]


def extract_features(img_or_path: ImageInput) -> np.ndarray | None:
    """
    Extract a normalized 1280-dim feature vector from an image or file path.

    Uses default_backend(); the model is loaded lazily on first call. For a
    specific backend use get_extractor(backend).extract(...).
    """
    return get_extractor().extract(img_or_path)
