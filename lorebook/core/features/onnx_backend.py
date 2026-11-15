"""MobileNetV2 feature extraction on ONNX Runtime (the desktop default).

The model is built from the Keras weights without TensorFlow
(lorebook/core/onnx_model.py): Windows Smart App Control blocks TensorFlow's
unsigned DLL but allows onnxruntime.
"""

import logging
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from lorebook.core.features.extractor import ImageInput
from lorebook.core.features.preprocessing import finalize_vector, preprocess_to_batch

logger = logging.getLogger(__name__)


class OnnxExtractor:
    """MobileNetV2 feature extractor on ONNX Runtime (model built on first use).

    InferenceSession.run is thread-safe, so inference needs no lock; only the
    one-time model build + session load is serialized.
    """

    def __init__(self, model_path: str | None = None):
        """Remember the model path; the model is built and loaded on first use."""
        self._explicit_model_path = model_path
        self._session = None
        self._load_lock = threading.Lock()

    @property
    def model_path(self) -> str:
        """Return the .onnx file to load: the explicit path, else the default location."""
        from lorebook.core.onnx_model import default_model_path

        return self._explicit_model_path or default_model_path()

    def loaded(self) -> bool:
        """True once the session is in memory (no load is triggered)."""
        return self._session is not None

    def ensure_ready(self) -> None:
        """Build/load the model and run one warm-up inference, raising on failure."""
        self._run(np.zeros((1, 224, 224, 3), np.float32))

    def _ensure_session(self) -> Any:
        """Return the inference session, building the model and loading it on first use."""
        if self._session is not None:
            return self._session
        with self._load_lock:
            if self._session is None:
                import onnxruntime as ort

                from lorebook.core.onnx_model import ensure_model

                path = ensure_model(self.model_path)
                opts = ort.SessionOptions()
                opts.log_severity_level = 3  # errors only
                self._session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        return self._session

    def _run(self, batch: np.ndarray, output: str = "features") -> np.ndarray:
        """Run a preprocessed batch through the model and return the named output."""
        return self._ensure_session().run([output], {"input": batch})[0]

    def activations(self, batch: np.ndarray) -> np.ndarray:
        """(N, 7, 7, 1280) last-conv activations for a preprocessed batch."""
        return self._run(batch, "activations")

    def extract(self, img_or_path: ImageInput) -> np.ndarray | None:
        """Return the image's normalized 1280-dim vector, or None if it can't be extracted."""
        try:
            batch = preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            return finalize_vector(self._run(batch))
        except Exception as e:  # noqa: BLE001 — boundary, see KerasExtractor.extract()
            logger.error("Error extracting features (onnx): %s", e)
            return None

    def extract_batch(self, images: Sequence[ImageInput]) -> list[np.ndarray | None]:
        """One session run per call; same contract as KerasExtractor.extract_batch."""
        results: list[np.ndarray | None] = [None] * len(images)
        rows, positions = [], []
        for i, img in enumerate(images):
            batch = preprocess_to_batch(img)
            if batch is not None:
                rows.append(batch[0])
                positions.append(i)
        if not rows:
            return results
        try:
            features = self._run(np.stack(rows))
            for pos, row in zip(positions, features):
                results[pos] = finalize_vector(row)
        except Exception as e:  # noqa: BLE001 — boundary, see extract()
            logger.error("Error extracting batch features (onnx): %s", e)
        return results
