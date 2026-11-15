"""MobileNetV2 feature extraction on TensorFlow/Keras, and the lock that serializes it.

The fallback desktop backend, and the one that builds the .tflite model.
"""

import logging
import threading
import time
import warnings
from collections.abc import Sequence

import numpy as np

from lorebook.core.features.preprocessing import finalize_vector, preprocess_to_batch

logger = logging.getLogger(__name__)

# TensorFlow/Keras are imported lazily (only when the "keras" backend or the
# heatmap overlay is actually used) so importing this module — and the "tflite"
# backend — works on a Raspberry Pi that has only tflite-runtime installed.
_base_model = None
_feat_model = None  # 1280-dim pooled features
_act_model = None  # last conv activations (for heatmaps)

# Serializes everything that touches the Keras models: the one-time load and
# every inference. The GUI scans while the DB-build worker extracts batches;
# concurrent model construction or inference on a shared model is not
# thread-safe, so both sides queue here. Reentrant because inference paths
# call get_models() while holding it.
model_lock = threading.RLock()

# Python locks aren't fair: a build looping batch after batch can re-take
# model_lock before a waiting scan ever gets it, leaving the scan stuck for
# the whole build. Interactive work (a scan, the debug heatmap) raises this
# count while it waits; batch work yields to it before each batch.
_interactive_waiters = 0
_waiters_guard = threading.Lock()


class _InteractivePriority:
    """Context manager: hold model_lock with priority over batch extraction."""

    def __enter__(self) -> None:
        """Take model_lock, counting as a waiter until it is acquired so batch work yields."""
        global _interactive_waiters
        with _waiters_guard:
            _interactive_waiters += 1
        try:
            model_lock.acquire()
        finally:
            with _waiters_guard:
                _interactive_waiters -= 1

    def __exit__(self, *exc) -> None:
        """Release model_lock."""
        model_lock.release()


def _yield_to_interactive(max_wait_s: float = 30.0) -> None:
    """Batch work: let any waiting interactive caller go first."""
    deadline = time.monotonic() + max_wait_s
    while _interactive_waiters and time.monotonic() < deadline:
        time.sleep(0.005)


def get_models():
    """Return (feature_model, activation_model), lazy-initialized on first call.

    Imports TensorFlow/Keras on first use only (keras backend / heatmap overlay).
    """
    global _base_model, _feat_model, _act_model
    if _feat_model is not None and _act_model is not None:
        return _feat_model, _act_model
    with model_lock:
        if _feat_model is not None and _act_model is not None:
            return _feat_model, _act_model  # another thread finished loading first
        # Silence TF/Keras import-time UserWarnings here only — never as a
        # process-wide filter at module import.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            import tensorflow as tf
            from keras.applications import MobileNetV2
            from keras.models import Model

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        tf.get_logger().setLevel("ERROR")

        base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        feat = Model(inputs=base.input, outputs=base.output)
        act = Model(inputs=base.input, outputs=base.layers[-2].output)
        # Publish only fully built models (the unlocked fast path reads these).
        _base_model, _act_model, _feat_model = base, act, feat
        return _feat_model, _act_model


def models_loaded() -> bool:
    """True once the Keras models are in memory (no load is triggered)."""
    return _feat_model is not None and _act_model is not None


def infer(model, batch: np.ndarray) -> np.ndarray:
    """
    Run one small batch through a Keras model and return a NumPy array.

    Calls the model directly instead of predict(): predict() builds a tf.data
    pipeline and traces a tf.function on every call, which is slow for one
    frame and is where a scan crashed natively on a worker thread. Batch
    extraction (DB builds) keeps predict(), whose overhead is paid per batch.
    Callers hold model_lock.
    """
    return np.asarray(model(batch, training=False))


class KerasExtractor:
    """MobileNetV2 feature extractor backed by TensorFlow/Keras."""

    def ensure_ready(self) -> None:
        """Load the model and run one warm-up inference now, raising on
        failure, so callers can fail fast and the first real scan doesn't pay
        TensorFlow's first-call setup."""
        with model_lock:
            feat_model, _ = get_models()
            infer(feat_model, np.zeros((1, 224, 224, 3), np.float32))

    def extract(self, img_or_path: np.ndarray | str) -> np.ndarray | None:
        """Return the image's normalized 1280-dim vector, or None if it can't be extracted."""
        try:
            batch = preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            with _InteractivePriority():
                feat_model, _ = get_models()
                features = infer(feat_model, batch)
            return finalize_vector(features)
        # Boundary: TF/Keras raise many error types; the contract is "None on
        # failure" so one bad frame never aborts a scan or sort run.
        except Exception as e:  # noqa: BLE001
            logger.error("Error extracting features (keras): %s", e)
            return None

    def extract_batch(self, images: Sequence) -> list[np.ndarray | None]:
        """
        Extract features for many images with a single predict call.

        Returns one vector (or None for unreadable/invalid inputs) per input,
        in order. This is the DB-build fast path: Keras predict overhead is
        paid once per batch instead of once per card, and inference stays on
        one thread (concurrent predicts on a shared model are not guaranteed
        thread-safe).
        """
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
            _yield_to_interactive()
            with model_lock:
                feat_model, _ = get_models()
                features = feat_model.predict(np.stack(rows), verbose=0)
            for pos, row in zip(positions, features):
                results[pos] = finalize_vector(row)
        except Exception as e:  # noqa: BLE001 — boundary, see extract()
            logger.error("Error extracting batch features (keras): %s", e)
        return results
